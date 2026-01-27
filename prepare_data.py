#!/usr/bin/env python3
"""
Prepare Data from Bruker PvDatasets
====================================

Extract reference images, T2* maps, and perfusion data from Bruker PvDatasets
for use in BoldPy MLCO analysis workflow.

This is Step 0 of the complete workflow.

Uses existing boldpy loaders that handle .PvDatasets ZIP archives.
"""

import numpy as np
import argparse
import json
import zipfile
import re  # For metadata reading
from pathlib import Path
import sys
from typing import Dict, Optional, Tuple, List
from datetime import datetime

# Add src directory to path
script_dir = Path(__file__).parent
src_dir = script_dir / 'src'
if src_dir.exists():
    sys.path.insert(0, str(src_dir))

# Import existing loaders
try:
    from boldpy.loaders.pvdataset import load_pvdataset
    from boldpy.fitting.t2star_fitter import fit_t2star_map, compute_r2star_map
    from boldpy.analysis.perfusion_analysis import load_bruker_perfusion
    LOADERS_AVAILABLE = True
except ImportError as e:
    print("="*70)
    print("ERROR: Could not import boldpy loaders")
    print("="*70)
    print(f"\nError: {e}")
    print("\nMake sure you've installed boldpy:")
    print("  cd boldpy_v2.1.0_mlco/")
    print("  pip install -e .")
    print("="*70)
    sys.exit(1)

# Import working functions from fit_t2star.py
fit_t2star_module = script_dir / 'fit_t2star.py'
if fit_t2star_module.exists():
    import importlib.util
    spec = importlib.util.spec_from_file_location("fit_t2star_module", fit_t2star_module)
    fit_t2star = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fit_t2star)
    
    # Get the working functions
    extract_echo_times_from_method = fit_t2star.extract_echo_times_from_method
    load_echo_data_from_zip = fit_t2star.load_echo_data
    WORKING_LOADERS = True
else:
    print("⚠️  Warning: fit_t2star.py not found, some features may not work")
    WORKING_LOADERS = False


# ============================================================================
# TIER ED T2* FRAME DETECTION - NEW FUNCTIONALITY
# ============================================================================

def read_bruker_param_from_zip(pvdatasets_path: Path, param_name: str, pdata_id: int = 2) -> Optional[List]:
    """
    Read Bruker parameter from visu_pars inside ZIP
    
    Parameters:
    -----------
    pvdatasets_path : Path
        Path to .PvDatasets ZIP file
    param_name : str
        Parameter name (e.g., 'VisuCoreFrameType')
    pdata_id : int
        pdata directory number (default: 2)
    
    Returns:
    --------
    value : list or None
    """
    try:
        with zipfile.ZipFile(pvdatasets_path, 'r') as zf:
            # Find visu_pars
            visu_pars_path = None
            for name in zf.namelist():
                if name.endswith(f'pdata/{pdata_id}/visu_pars'):
                    visu_pars_path = name
                    break
            
            if not visu_pars_path:
                return None
            
            # Read file
            content = zf.read(visu_pars_path).decode('latin-1')
            
            # Parse parameter
            pattern = rf'##\${param_name}=(.+?)(?=##\$|\Z)'
            match = re.search(pattern, content, re.DOTALL)
            
            if not match:
                return None
            
            value_str = match.group(1).strip()
            
            # Check if array
            if value_str.startswith('('):
                lines = value_str.split('\n')
                value_lines = []
                in_values = False
                
                for line in lines:
                    if line.strip().startswith('(') and line.strip().endswith(')'):
                        in_values = True
                        continue
                    if in_values and line.strip():
                        line = line.strip().strip('<>').strip()
                        if line:
                            value_lines.append(line)
                
                # Angle-bracketed strings
                if '<' in value_str:
                    return value_lines
                else:
                    # Space-separated numeric
                    values = []
                    for line in value_lines:
                        values.extend(line.split())
                    return values
            else:
                return [value_str]
    
    except Exception as e:
        return None


def identify_t2star_from_metadata(pvdatasets_path: Path, n_frames: int) -> Tuple[Optional[int], Optional[int], Optional[str]]:
    """
    TIER 1: Identify T2* from Bruker metadata
    
    Returns:
    --------
    t2_frame : int or None (0-indexed)
    r2_frame : int or None (0-indexed)
    method : str or None
    """
    # Try VisuCoreFrameType
    frame_types = read_bruker_param_from_zip(pvdatasets_path, 'VisuCoreFrameType', pdata_id=2)
    
    if frame_types and isinstance(frame_types, list):
        t2_frame = None
        r2_frame = None
        
        for i, ftype in enumerate(frame_types):
            ftype_lower = ftype.lower()
            
            # Check for T2/T2* indicators
            if 't2' in ftype_lower and ('map' in ftype_lower or 'star' in ftype_lower or 'relax' in ftype_lower):
                if t2_frame is None:
                    t2_frame = i
            
            # Check for R2/R2*
            if 'r2' in ftype_lower and ('map' in ftype_lower or 'star' in ftype_lower):
                if r2_frame is None:
                    r2_frame = i
        
        if t2_frame is not None:
            return t2_frame, r2_frame, 'VisuCoreFrameType'
    
    return None, None, None


def score_t2star_candidate(frame: np.ndarray, visu_range: Optional[Tuple[float, float]] = None) -> Tuple[float, List[str]]:
    """
    TIER 2: Score frame likelihood to be T2* (0-100 points)
    """
    nonzero_mask = frame > 0
    nonzero_vals = frame[nonzero_mask]
    
    if len(nonzero_vals) == 0:
        return 0, []
    
    mean = np.mean(nonzero_vals)
    std = np.std(nonzero_vals)
    max_val = np.max(nonzero_vals)
    pct_nonzero = 100 * len(nonzero_vals) / frame.size
    
    score = 0
    details = []
    
    # Criterion 1: Mean value (40 pts)
    if 10 <= mean <= 30:
        score += 40
        details.append(f"Mean {mean:.1f} ms: 40/40 pts")
    elif 5 <= mean <= 50:
        score += 30
        details.append(f"Mean {mean:.1f} ms: 30/40 pts")
    elif 3 <= mean <= 100:
        score += 15
        details.append(f"Mean {mean:.1f} ms: 15/40 pts")
    
    # Criterion 2: Max value (20 pts)
    if max_val <= 100:
        score += 20
        details.append(f"Max {max_val:.1f} ms: 20/20 pts")
    elif max_val <= 200:
        score += 15
    elif max_val <= 500:
        score += 10
    
    # Criterion 3: visu_pars range (15 pts)
    if visu_range:
        visu_min, visu_max = visu_range
        if 0 <= visu_max <= 200:
            score += 15
            details.append(f"Range [0, {visu_max:.1f}]: 15/15 pts")
    
    # Criterion 4: Distribution (15 pts)
    cv = std / mean if mean > 0 else 0
    if 0.2 <= cv <= 0.8:
        score += 15
        details.append(f"Distribution: 15/15 pts")
    elif 0.1 <= cv <= 1.0:
        score += 10
    else:
        score += 5
    
    # Criterion 5: Coverage (10 pts)
    if pct_nonzero > 10:
        score += 10
        details.append(f"Coverage {pct_nonzero:.1f}%: 10/10 pts")
    elif pct_nonzero > 5:
        score += 5
    
    return score, details


def identify_t2star_from_heuristic(data: np.ndarray, pvdatasets_path: Path) -> Tuple[Optional[int], Optional[int], List[float], List[List[str]]]:
    """
    TIER 2: Enhanced heuristic scoring
    
    Returns:
    --------
    t2_frame : int or None (0-indexed)
    r2_frame : int or None (0-indexed)
    scores : list of floats
    all_details : list of lists
    """
    n_frames = data.shape[0]
    
    # Read visu_pars ranges
    visu_min_vals = read_bruker_param_from_zip(pvdatasets_path, 'VisuCoreDataMin', pdata_id=2)
    visu_max_vals = read_bruker_param_from_zip(pvdatasets_path, 'VisuCoreDataMax', pdata_id=2)
    
    visu_ranges = None
    if visu_min_vals and visu_max_vals:
        try:
            visu_ranges = [(float(visu_min_vals[i]), float(visu_max_vals[i])) for i in range(len(visu_min_vals))]
        except:
            pass
    
    # Score each frame
    scores = []
    all_details = []
    
    for i in range(n_frames):
        visu_range = visu_ranges[i] if visu_ranges and i < len(visu_ranges) else None
        score, details = score_t2star_candidate(data[i], visu_range)
        scores.append(score)
        all_details.append(details)
    
    # Select best frame
    if max(scores) > 0:
        best_idx = np.argmax(scores)
        
        # Check if next frame might be R2*
        r2_frame = None
        if best_idx + 1 < n_frames:
            next_frame = data[best_idx + 1]
            next_mean = np.mean(next_frame[next_frame > 0])
            if 10 <= next_mean <= 200:  # R2* range
                r2_frame = best_idx + 1
        
        return best_idx, r2_frame, scores, all_details
    
    return None, None, scores, all_details


def identify_t2star_frame_tiered(data: np.ndarray, pvdatasets_path: Path, manual_frame: Optional[int] = None) -> Tuple[int, Optional[int], str, str]:
    """
    Tiered T2* frame identification
    
    TIER 1: Manual override
    TIER 2: Metadata (VisuCoreFrameType)
    TIER 3: Enhanced heuristic scoring
    
    Parameters:
    -----------
    data : ndarray
        Shape (n_frames, height, width)
    pvdatasets_path : Path
        Path to .PvDatasets ZIP file
    manual_frame : int, optional
        Manually specified frame (1-indexed)
    
    Returns:
    --------
    t2_frame : int (0-indexed)
    r2_frame : int or None (0-indexed)
    method : str
    confidence : str
    """
    n_frames = data.shape[0]
    
    # TIER 1: Manual override
    if manual_frame is not None:
        t2_frame = manual_frame - 1  # Convert to 0-indexed
        print(f"  ✓ Using manually specified frame {manual_frame} for T2*")
        return t2_frame, None, 'manual', 'MANUAL'
    
    # TIER 2: Metadata
    print(f"  Tier 1: Checking Bruker metadata...")
    t2_frame, r2_frame, method = identify_t2star_from_metadata(pvdatasets_path, n_frames)
    
    if t2_frame is not None:
        print(f"  ✓ Frame {t2_frame+1} identified as T2* via {method}")
        if r2_frame is not None:
            print(f"  ✓ Frame {r2_frame+1} identified as R2* via {method}")
        return t2_frame, r2_frame, method, 'HIGH'
    else:
        print(f"  ⚠️  Metadata uninformative (frames labeled generically)")
    
    # TIER 3: Enhanced heuristic
    print(f"  Tier 2: Using enhanced scoring heuristic...")
    t2_frame, r2_frame, scores, all_details = identify_t2star_from_heuristic(data, pvdatasets_path)
    
    if t2_frame is not None:
        best_score = scores[t2_frame]
        
        # Print results
        print(f"     Frame scores:")
        for i in range(n_frames):
            marker = "★" if i == t2_frame else " "
            print(f"       {marker} Frame {i+1}: {scores[i]:.0f}/100 pts")
            if i == t2_frame and all_details[i]:
                for detail in all_details[i][:3]:
                    print(f"           {detail}")
        
        # Determine confidence
        if best_score >= 70:
            confidence = "HIGH"
        elif best_score >= 50:
            confidence = "MEDIUM"
        else:
            confidence = "LOW"
        
        print(f"  ✓ Selected frame {t2_frame+1} (score: {best_score:.0f}/100, confidence: {confidence})")
        
        if r2_frame is not None:
            print(f"  ✓ Frame {r2_frame+1} appears to be R2*")
        
        if best_score < 50:
            print(f"  ⚠️  WARNING: Low confidence detection!")
            print(f"     Verify T2* values are reasonable")
            print(f"     Re-run with --t2-frame N if incorrect")
        
        return t2_frame, r2_frame, 'heuristic', confidence
    
    # Failed
    raise ValueError(
        f"Could not identify T2* frame in {n_frames} frames.\n"
        f"Please specify manually using --t2-frame option."
    )


# ============================================================================
# EXISTING FUNCTIONS (PRESERVED)
# ============================================================================

def extract_reference_image(
    pvdatasets_path: Path,
    output_2d: bool = True,
    slice_index: Optional[int] = None
) -> Tuple[np.ndarray, Dict]:
    """
    Extract reference anatomical image from PvDatasets
    
    Uses the working load_echo_data function from fit_t2star.py
    
    Parameters:
    -----------
    pvdatasets_path : Path
        Path to .PvDatasets file
    output_2d : bool
        If True, return 2D image (auto-selected slice)
    slice_index : int, optional
        Specific slice to extract
    
    Returns:
    --------
    reference : ndarray
        Reference image
    info : dict
        Extraction info
    """
    info = {'method': None, 'shape': None, 'slice': None, 'reco_id': 1}
    
    if not WORKING_LOADERS:
        raise ImportError("Working loaders not available (fit_t2star.py not found)")
    
    # Load echo data directly from ZIP
    echo_data, data_slope = load_echo_data_from_zip(pvdatasets_path, reco_id=1)
    
    # Take first echo (best anatomical contrast)
    if echo_data.ndim == 3:  # (n_echoes, height, width)
        reference = echo_data[0]
        info['method'] = 'first_echo_pdata1'
        info['shape'] = '2D'
        info['dimensions'] = reference.shape
        print(f"  ✓ Extracted first echo from pdata/1")
        
    elif echo_data.ndim == 2:  # Already 2D
        reference = echo_data
        info['method'] = 'image_pdata1'
        info['shape'] = '2D'
        info['dimensions'] = reference.shape
        print(f"  ✓ Extracted image from pdata/1")
        
    else:
        raise ValueError(f"Unexpected echo_data shape: {echo_data.shape}")
    
    return reference.astype(np.float32), info



def extract_bruker_t2star(
    pvdatasets_path: Path,
    manual_frame: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract Bruker's pre-fitted T2* map from pdata/2
    
    Uses tiered detection: metadata → enhanced heuristic → manual override
    
    Parameters:
    -----------
    pvdatasets_path : Path
        Path to .PvDatasets file
    manual_frame : int, optional
        Manually specify T2* frame (1-indexed)
    
    Returns:
    --------
    t2star_map : ndarray
        T2* in milliseconds
    r2star_map : ndarray
        R2* in Hz
    """
    if not WORKING_LOADERS:
        raise ImportError("Working loaders not available (fit_t2star.py not found)")
    
    # Load from pdata/2 (processed/fitted data)
    try:
        t2star_data, data_slope = load_echo_data_from_zip(pvdatasets_path, reco_id=2)
    except Exception as e:
        raise ValueError(
            f"Could not load Bruker T2* from pdata/2.\n"
            f"Error: {e}\n"
            f"This might mean pdata/2 doesn't exist or doesn't contain T2* maps."
        )
    
    # Check what we loaded
    if t2star_data.ndim == 3:
        # (n_frames, height, width)
        n_frames = t2star_data.shape[0]
        print(f"  ⚠️  pdata/2 contains {n_frames} frames - using tiered detection")
        
        # Use tiered detection
        t2_frame, r2_frame, method, confidence = identify_t2star_frame_tiered(
            t2star_data,
            pvdatasets_path,
            manual_frame=manual_frame
        )
        
        # Extract T2* map
        t2star_map = t2star_data[t2_frame]
        
        # Get or calculate R2* map
        if r2_frame is not None:
            r2star_map = t2star_data[r2_frame]
            print(f"  ✓ Using frame {r2_frame+1} for R2* map")
        else:
            # Calculate from T2*
            r2star_map = np.zeros_like(t2star_map, dtype=np.float32)
            mask = t2star_map > 0
            r2star_map[mask] = 1000.0 / t2star_map[mask]
            print(f"  ✓ Calculated R2* from T2* (R2* = 1000/T2*)")
    else:
        # Already 2D
        t2star_map = t2star_data
        print(f"  ✓ Loaded T2* map from pdata/2")
        
        # Compute R2*
        r2star_map = np.zeros_like(t2star_map, dtype=np.float32)
        mask = t2star_map > 0
        r2star_map[mask] = 1000.0 / t2star_map[mask]
    
    mean_val = t2star_map[t2star_map > 0].mean() if np.any(t2star_map > 0) else 0
    
    # Warn if values are suspicious
    if mean_val > 200 or mean_val < 5:
        print(f"  ⚠️  WARNING: Suspicious T2* values (mean={mean_val:.1f} ms)")
        print(f"     Normal kidney T2* is 10-50 ms at high field")
        print(f"     This may not be a T2* map - recommend using --custom-t2star")
    else:
        print(f"    T2* range: {t2star_map.min():.1f} - {t2star_map.max():.1f} ms")
        print(f"    T2* mean: {mean_val:.1f} ms")
    
    return t2star_map.astype(np.float32), r2star_map


def extract_custom_t2star(
    pvdatasets_path: Path,
    bounds: Tuple[float, float] = (5.0, 2000.0),  # Match fit_t2star.py defaults
    min_signal: float = 1.0
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Fit custom T2* from raw echoes in pdata/1
    
    Uses working functions from fit_t2star.py
    
    Parameters:
    -----------
    pvdatasets_path : Path
        Path to .PvDatasets file
    bounds : tuple
        (min, max) for T2* in ms
    min_signal : float
        Minimum signal threshold for fitting
    
    Returns:
    --------
    t2star_map : ndarray
    r2star_map : ndarray
    fit_info : dict
    """
    if not WORKING_LOADERS:
        raise ImportError("Working loaders not available (fit_t2star.py not found)")
    
    # Extract echo times using working function
    echo_times = extract_echo_times_from_method(pvdatasets_path)
    echo_times = np.array(echo_times)  # CRITICAL: Convert to numpy array!
    
    # Load echo data using working function
    echo_data, data_slope = load_echo_data_from_zip(pvdatasets_path, reco_id=1)
    
    print(f"  ✓ Loaded {echo_data.shape[0]} echoes")
    print(f"  ✓ Echo times: {[f'{t:.2f}' for t in echo_times]} ms")
    print(f"  ✓ Image shape: {echo_data.shape}")
    print(f"  ✓ Signal range: {echo_data.min():.1f} - {echo_data.max():.1f}")
    
    # Verify number of echoes matches
    if len(echo_times) != echo_data.shape[0]:
        print(f"  ⚠️  Warning: {len(echo_times)} echo times but {echo_data.shape[0]} echo images")
        # Use the minimum
        n_echoes = min(len(echo_times), echo_data.shape[0])
        echo_times = echo_times[:n_echoes]
        echo_data = echo_data[:n_echoes]
        print(f"  ✓ Using first {n_echoes} echoes")
    
    echo_times = np.array(echo_times)  # Convert again after any trimming
    
    # Fit T2* map - NOTE: returns a dict!
    print(f"  ⏳ Fitting T2* map (this may take a minute)...")
    
    # Convert bounds format: (min, max) -> ((s0_min, t2_min), (s0_max, t2_max))
    fit_bounds = ((0, bounds[0]), (np.inf, bounds[1]))
    
    print(f"  ✓ Bounds: T2* = [{bounds[0]}, {bounds[1]}] ms")
    print(f"  ✓ Min signal threshold: {min_signal}")
    
    results = fit_t2star_map(
        echo_data,
        echo_times,
        bounds=fit_bounds,
        min_signal=min_signal,
        show_progress=True
    )
    
    # Extract the actual T2* map from results dict
    t2star_map = results['t2star_map']
    success_map = results['success_map']
    
    # Check if any pixels were fitted
    n_fitted = np.sum(success_map)
    if n_fitted == 0:
        raise ValueError(
            f"T2* fitting failed for all pixels!\n"
            f"Signal range: {echo_data.min():.1f} - {echo_data.max():.1f}\n"
            f"Min signal threshold: {min_signal}\n"
            f"Possible causes:\n"
            f"  - Signal too low (try --min-signal {echo_data.min()/10:.2f})\n"
            f"  - Wrong bounds (current: {bounds})\n"
            f"  - Data quality issues"
        )
    
    # Compute R2*
    r2star_map = compute_r2star_map(t2star_map)
    
    # Fit quality metrics
    fit_info = {
        'n_echoes': len(echo_times),
        'echo_times': echo_times.tolist(),  # Convert numpy array to list for JSON
        'bounds': bounds,
        't2star_mean': float(np.mean(t2star_map[success_map])) if n_fitted > 0 else 0.0,
        't2star_std': float(np.std(t2star_map[success_map])) if n_fitted > 0 else 0.0,
        'n_valid_pixels': int(n_fitted),
        'n_total_pixels': int(t2star_map.size),
        'success_rate': float(n_fitted / t2star_map.size),
        'min_signal': min_signal
    }
    
    return t2star_map, r2star_map, fit_info


def resample_to_match(source: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    """
    Resample source image to match target shape using nearest neighbor
    
    Parameters:
    -----------
    source : ndarray
        Source image to resample
    target_shape : tuple
        Target (height, width)
        
    Returns:
    --------
    resampled : ndarray
        Resampled image
    """
    from scipy import ndimage
    
    if source.shape == target_shape:
        return source
    
    print(f"  ⏳ Resampling from {source.shape} to {target_shape}...")
    
    zoom_factors = (target_shape[0] / source.shape[0],
                    target_shape[1] / source.shape[1])
    
    resampled = ndimage.zoom(source, zoom_factors, order=0)  # order=0 = nearest neighbor
    
    print(f"  ✓ Resampled to {resampled.shape}")
    
    return resampled


def extract_scan_metadata(pvdatasets_path: Path) -> Dict:
    """
    Extract metadata using working loaders
    
    Returns:
    --------
    metadata : dict
    """
    try:
        if WORKING_LOADERS:
            # Use working echo time extraction
            echo_times = extract_echo_times_from_method(pvdatasets_path)
            echo_data, _ = load_echo_data_from_zip(pvdatasets_path, reco_id=1)
            
            # Check if pdata/2 exists
            has_pdata_2 = False
            try:
                load_echo_data_from_zip(pvdatasets_path, reco_id=2)
                has_pdata_2 = True
            except:
                pass
            
            return {
                'scan_file': str(pvdatasets_path),
                'extracted_at': datetime.now().isoformat(),
                'echo_times': echo_times,
                'n_echoes': len(echo_times),
                'image_shape': echo_data.shape,
                'format': 'pvdataset',
                'backend': 'direct_zip_extraction',
                'has_pdata_1': True,
                'has_pdata_2': has_pdata_2,
            }
        else:
            # Fallback
            return {
                'scan_file': str(pvdatasets_path),
                'extracted_at': datetime.now().isoformat(),
                'error': 'Working loaders not available'
            }
        
    except Exception as e:
        print(f"  Warning: Could not extract full metadata: {e}")
        return {
            'scan_file': str(pvdatasets_path),
            'extracted_at': datetime.now().isoformat(),
            'error': str(e)
        }


def prepare_single_scan(
    pvdatasets_path: Path,
    output_dir: Path,
    sample_name: Optional[str] = None,
    extract_bruker: bool = True,
    extract_custom: bool = False,
    extract_perfusion_flag: bool = False,
    output_2d: bool = True,
    slice_index: Optional[int] = None
,
    manual_frame: Optional[int] = None
) -> Dict:
    """
    Prepare data from a single PvDatasets file
    
    Parameters:
    -----------
    pvdatasets_path : Path
        Input .PvDatasets file
    output_dir : Path
        Output directory
    sample_name : str, optional
        Sample name for output files
    extract_bruker : bool
        Extract Bruker T2* from pdata/2
    extract_custom : bool
        Fit custom T2* from pdata/1
    extract_perfusion_flag : bool
        Extract perfusion if available
    output_2d : bool
        Output 2D reference (middle slice)
    slice_index : int, optional
        Specific slice for 2D
    
    Returns:
    --------
    results : dict
        Extraction results
    """
    pvdatasets_path = Path(pvdatasets_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine sample name
    if sample_name is None:
        sample_name = pvdatasets_path.stem.replace('.PvDatasets', '')
    
    print(f"\n{'='*70}")
    print(f"Processing: {pvdatasets_path.name}")
    print(f"Sample: {sample_name}")
    print(f"{'='*70}")
    
    results = {
        'sample_name': sample_name,
        'input_file': str(pvdatasets_path),
        'output_dir': str(output_dir),
        'files_created': [],
        'metadata': {}
    }
    
    # Extract metadata
    print("\n1. Extracting metadata...")
    metadata = extract_scan_metadata(pvdatasets_path)
    results['metadata'].update(metadata)
    if 'echo_times' in metadata:
        print(f"  ✓ Found {len(metadata['echo_times'])} echoes")
    if 'has_pdata_2' in metadata:
        if metadata['has_pdata_2']:
            print(f"  ✓ Has pdata/2 (Bruker fitted maps available)")
        else:
            print(f"  ⚠️  No pdata/2 found (Bruker fitted maps not available)")
    
    # Extract reference image (ALWAYS)
    print("\n2. Extracting reference image...")
    try:
        reference, ref_info = extract_reference_image(
            pvdatasets_path,
            output_2d=output_2d,
            slice_index=slice_index
        )
        
        ref_filename = output_dir / f"{sample_name}_reference.npy"
        np.save(ref_filename, reference)
        results['files_created'].append(str(ref_filename))
        results['metadata']['reference'] = ref_info
        
        print(f"  ✓ Saved: {ref_filename.name}")
        print(f"    Shape: {reference.shape}")
        print(f"    Method: {ref_info['method']}")
        
    except Exception as e:
        print(f"  ✗ Failed to extract reference: {e}")
        import traceback
        traceback.print_exc()
        return results
    
    # Extract Bruker T2*
    if extract_bruker:
        print("\n3. Extracting Bruker fitted T2* map...")
        try:
            t2star, r2star = extract_bruker_t2star(pvdatasets_path, manual_frame=manual_frame)
            
            t2_filename = output_dir / f"{sample_name}_t2star_bruker.npy"
            r2_filename = output_dir / f"{sample_name}_r2star_bruker.npy"
            
            np.save(t2_filename, t2star)
            np.save(r2_filename, r2star)
            
            results['files_created'].extend([str(t2_filename), str(r2_filename)])
            results['metadata']['bruker_t2star'] = {
                'mean': float(np.mean(t2star[t2star > 0])),
                'std': float(np.std(t2star[t2star > 0]))
            }
            
            print(f"  ✓ Saved: {t2_filename.name}")
            print(f"  ✓ Saved: {r2_filename.name}")
            print(f"    T2* mean: {results['metadata']['bruker_t2star']['mean']:.2f} ms")
            
        except ValueError as e:
            # pdata/2 doesn't exist or can't be loaded
            print(f"  ⚠️  {e}")
        except Exception as e:
            print(f"  ✗ Failed to extract Bruker T2*: {e}")
    
    # Fit custom T2*
    if extract_custom:
        print("\n4. Fitting custom T2* map...")
        try:
            t2star, r2star, fit_info = extract_custom_t2star(pvdatasets_path)
            
            t2_filename = output_dir / f"{sample_name}_t2star_custom.npy"
            r2_filename = output_dir / f"{sample_name}_r2star_custom.npy"
            
            np.save(t2_filename, t2star)
            np.save(r2_filename, r2star)
            
            results['files_created'].extend([str(t2_filename), str(r2_filename)])
            results['metadata']['custom_t2star'] = fit_info
            
            print(f"  ✓ Saved: {t2_filename.name}")
            print(f"  ✓ Saved: {r2_filename.name}")
            print(f"    T2* mean: {fit_info['t2star_mean']:.2f} ms")
            
        except Exception as e:
            print(f"  ✗ Failed to fit custom T2*: {e}")
            import traceback
            traceback.print_exc()
    
    # Extract perfusion
    if extract_perfusion_flag:
        print("\n5. Extracting perfusion map...")
        try:
            perfusion = load_bruker_perfusion(pvdatasets_path)
            
            if perfusion is not None:
                # Check if perfusion needs resampling to match reference
                if reference.shape != perfusion.shape:
                    print(f"  ⚠️  Perfusion shape {perfusion.shape} doesn't match reference {reference.shape}")
                    perfusion = resample_to_match(perfusion, reference.shape)
                
                # Save perfusion map
                perf_filename = output_dir / f"{sample_name}_perfusion.npy"
                np.save(perf_filename, perfusion)
                results['files_created'].append(str(perf_filename))
                
                # Store stats
                valid_mask = (perfusion > 0) & (perfusion < 200)  # Reasonable perfusion range
                if np.any(valid_mask):
                    results['metadata']['perfusion'] = {
                        'mean': float(np.mean(perfusion[valid_mask])),
                        'std': float(np.std(perfusion[valid_mask])),
                        'range': [float(perfusion[valid_mask].min()), float(perfusion[valid_mask].max())],
                        'valid_pixels': int(np.sum(valid_mask)),
                        'units': 'relative %'
                    }
                
                print(f"  ✓ Saved: {perf_filename.name}")
                if np.any(valid_mask):
                    print(f"    Mean perfusion: {results['metadata']['perfusion']['mean']:.2f} %")
                    print(f"    Valid pixels: {results['metadata']['perfusion']['valid_pixels']:,}")
            else:
                print(f"  ⚠️  Could not extract perfusion map")
                print(f"     This might not be a perfusion scan or pdata/2 is unavailable")
                
        except Exception as e:
            print(f"  ✗ Failed to extract perfusion: {e}")
            import traceback
            traceback.print_exc()
    
    # Save metadata
    metadata_filename = output_dir / f"{sample_name}_metadata.json"
    with open(metadata_filename, 'w') as f:
        json.dump(results['metadata'], f, indent=2)
    results['files_created'].append(str(metadata_filename))
    
    print(f"\n✓ Processing complete!")
    print(f"  Created {len(results['files_created'])} files in {output_dir}")
    
    return results


def prepare_batch(
    input_dir: Path,
    output_dir: Path,
    pattern: str = "*.PvDatasets",
    **kwargs
) -> List[Dict]:
    """
    Batch process multiple PvDatasets files
    """
    input_dir = Path(input_dir)
    files = sorted(input_dir.glob(pattern))
    
    if not files:
        print(f"⚠️  No files matching '{pattern}' found in {input_dir}")
        return []
    
    print(f"\n{'='*70}")
    print(f"BATCH PROCESSING: {len(files)} scans")
    print(f"{'='*70}")
    
    all_results = []
    for i, file_path in enumerate(files, 1):
        print(f"\n[{i}/{len(files)}]")
        try:
            results = prepare_single_scan(file_path, output_dir, **kwargs,
            manual_frame=args.t2_frame
        )
            all_results.append(results)
        except Exception as e:
            print(f"✗ Failed to process {file_path.name}: {e}")
            continue
    
    print(f"\n{'='*70}")
    print(f"BATCH COMPLETE: {len(all_results)}/{len(files)} successful")
    print(f"{'='*70}")
    
    return all_results


def main():
    parser = argparse.ArgumentParser(
        description='Prepare data from Bruker PvDatasets for BoldPy analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default (reference + Bruker T2* if available)
  python prepare_data.py --input scan.PvDatasets --output-dir prepared/
  
  # With sample name
  python prepare_data.py --input scan.PvDatasets --output-dir prepared/ --sample-name M1_air
  
  # Custom T2* fitting (recommended for best quality)
  python prepare_data.py --input scan.PvDatasets --output-dir prepared/ --custom-t2star --no-bruker
  
  # Both Bruker and custom T2* (for comparison)
  python prepare_data.py --input scan.PvDatasets --output-dir prepared/ --both-t2star
  
  # Extract perfusion (for ASL/FAIR-EPI scans)
  python prepare_data.py --input perfusion.PvDatasets --output-dir prepared/ --extract-perfusion
  
  # Complete preparation (T2* + perfusion)
  python prepare_data.py --input scan.PvDatasets --output-dir prepared/ --both-t2star --extract-perfusion
  
  # 3D output
  python prepare_data.py --input scan.PvDatasets --output-dir prepared/ --3d --custom-t2star
  
  # Batch process (T2* scans)
  python prepare_data.py --input-dir bold/ --output-dir prepared/ --pattern "*air*.PvDatasets" --both-t2star
  
  # Batch process (perfusion scans)
  python prepare_data.py --input-dir perfusion/ --output-dir prepared/ --pattern "*perfusion*.PvDatasets" --extract-perfusion
  
Note: Bruker T2* extraction requires pdata/2 in the .PvDatasets file.
      Custom T2* fitting works from raw echoes in pdata/1 and usually gives better quality.
      Perfusion extraction loads from pdata/2, Frame 5 (Bruker ASL/FAIR-EPI processing).
        """
    )
    
    # Input/output
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--input', type=Path,
                            help='Input .PvDatasets file')
    input_group.add_argument('--input-dir', type=Path,
                            help='Directory with .PvDatasets files (batch mode)')
    
    parser.add_argument('--output-dir', type=Path, default=Path('prepared_data'),
                       help='Output directory (default: prepared_data/)')
    parser.add_argument('--sample-name', type=str,
                       help='Sample name for output files (default: scan filename)')
    
    # T2* extraction options
    t2star_group = parser.add_argument_group('T2* extraction')
    t2star_group.add_argument('--no-bruker', action='store_true',
                             help='Skip Bruker fitted T2* (pdata/2) - currently required')
    t2star_group.add_argument('--custom-t2star', action='store_true',
                             help='Fit custom T2* from raw echoes (recommended)')
    t2star_group.add_argument('--both-t2star', action='store_true',
                             help='Extract both Bruker and custom T2* (Bruker not yet implemented)')
    t2star_group.add_argument('--t2-frame', type=int,
                             help='Manually specify T2* frame number (1-indexed) for Bruker extraction')
    
    # Other data
    parser.add_argument('--extract-perfusion', action='store_true',
                       help='Extract perfusion map from Bruker pdata/2 (ASL/FAIR-EPI scans)')
    
    # Image format
    format_group = parser.add_argument_group('Image format')
    format_group.add_argument('--3d', dest='output_3d', action='store_true',
                             help='Output 3D volume (default: 2D middle slice)')
    format_group.add_argument('--slice', type=int,
                             help='Specific slice index for 2D output')
    
    # Batch options
    batch_group = parser.add_argument_group('Batch processing')
    batch_group.add_argument('--pattern', default='*.PvDatasets',
                            help='File pattern for batch mode (default: *.PvDatasets)')
    
    args = parser.parse_args()
    
    # Parse T2* options
    extract_bruker = not args.no_bruker
    extract_custom = args.custom_t2star or args.both_t2star
    
    # Single scan or batch
    if args.input:
        results = prepare_single_scan(
            pvdatasets_path=args.input,
            output_dir=args.output_dir,
            sample_name=args.sample_name,
            extract_bruker=extract_bruker,
            extract_custom=extract_custom,
            extract_perfusion_flag=args.extract_perfusion,
            output_2d=not args.output_3d,
            slice_index=args.slice,
            manual_frame=args.t2_frame
        )
    else:
        results = prepare_batch(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            pattern=args.pattern,
            extract_bruker=extract_bruker,
            extract_custom=extract_custom,
            extract_perfusion_flag=args.extract_perfusion,
            output_2d=not args.output_3d,
            slice_index=args.slice
        )


if __name__ == '__main__':
    main()
