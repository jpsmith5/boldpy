#!/usr/bin/env python3
"""
Fit T2* Maps from Bruker Multi-Echo Data
=========================================

Proper T2* fitting using actual echo times from method files.
Processes pdata/1 (raw 8-echo magnitude images).
"""

import numpy as np
import zipfile
from pathlib import Path
import sys
import argparse
import json

# Add src directory to path (relative to this script)
script_dir = Path(__file__).parent
src_dir = script_dir / 'src'
if src_dir.exists():
    sys.path.insert(0, str(src_dir))

from boldpy.fitting.t2star_fitter import fit_t2star_map, compute_r2star_map, validate_fit_quality


def extract_echo_times_from_method(pvdatasets_path: Path) -> list:
    """
    Extract actual echo times from Bruker method file
    
    Parameters:
    -----------
    pvdatasets_path : Path
        Path to .PvDatasets file
        
    Returns:
    --------
    echo_times : list
        Echo times in milliseconds
    """
    with zipfile.ZipFile(pvdatasets_path, 'r') as zf:
        # Find method file
        method_files = [f for f in zf.namelist() if f.endswith('/method')]
        
        if not method_files:
            raise ValueError(f"No method file found in {pvdatasets_path}")
        
        # Read method file
        with zf.open(method_files[0]) as f:
            method_content = f.read().decode('utf-8', errors='ignore')
        
        # Extract EffectiveTE
        lines = method_content.split('\n')
        for i, line in enumerate(lines):
            if '##$EffectiveTE=' in line:
                # Extract expected number of echo times
                # Line format: ##$EffectiveTE=( N )
                n_echoes = None
                if '(' in line and ')' in line:
                    try:
                        n_echoes = int(line.split('(')[1].split(')')[0].strip())
                    except:
                        pass
                
                # Collect echo times from following lines
                echo_times = []
                j = i + 1
                
                # Read subsequent lines until we have all echo times or hit a new parameter
                while j < len(lines) and len(echo_times) < (n_echoes or 20):
                    next_line = lines[j].strip()
                    
                    # Stop if we hit a new parameter
                    if next_line.startswith('##$') or next_line.startswith('$$'):
                        break
                    
                    # Try to parse numbers from this line
                    parts = next_line.rstrip('\\').split()
                    for part in parts:
                        try:
                            val = float(part)
                            echo_times.append(val)
                        except:
                            pass
                    
                    j += 1
                
                if echo_times:
                    print(f"\n✓ Extracted {len(echo_times)} echo times: {echo_times} ms")
                    return echo_times
    
    # Fallback to standard MGE echo times if extraction fails
    print("\n⚠️  Could not extract echo times from method file")
    print("   Using standard MGE echo times")
    return [3.0, 7.18, 11.36, 15.54, 19.71, 23.89, 28.07, 32.25]


def load_echo_data(pvdatasets_path: Path, reco_id: int = 1) -> tuple:
    """
    Load multi-echo data from PvDatasets file
    
    Parameters:
    -----------
    pvdatasets_path : Path
        Path to .PvDatasets file
    reco_id : int
        Reconstruction ID (1 = raw echoes, 2 = processed maps)
        
    Returns:
    --------
    echo_data : ndarray
        Multi-echo data, shape (n_echoes, height, width)
    data_slope : float
        Scaling factor from visu_pars
    """
    with zipfile.ZipFile(pvdatasets_path, 'r') as zf:
        # Find 2dseq file for specified reco
        seq_files = [f for f in zf.namelist() if f.endswith(f'/pdata/{reco_id}/2dseq')]
        
        if not seq_files:
            raise ValueError(f"No 2dseq file found for reco {reco_id}")
        
        # Read 2dseq
        with zf.open(seq_files[0]) as f:
            data_raw = f.read()
        
        # Find visu_pars to get dimensions and scaling
        visu_files = [f for f in zf.namelist() if f.endswith(f'/pdata/{reco_id}/visu_pars')]
        
        if not visu_files:
            raise ValueError(f"No visu_pars found for reco {reco_id}")
        
        with zf.open(visu_files[0]) as f:
            visu_content = f.read().decode('utf-8', errors='ignore')
        
        # Extract parameters
        frame_count = None
        core_size = []
        data_slope = None
        word_type = None
        
        for line in visu_content.split('\n'):
            if '##$VisuCoreFrameCount=' in line:
                frame_count = int(line.split('=')[1])
            elif '##$VisuCoreSize=' in line and not core_size:
                # Next line has dimensions
                idx = visu_content.index(line)
                size_line = visu_content[idx:].split('\n')[1]
                core_size = [int(x) for x in size_line.split()]
            elif '##$VisuCoreDataSlope=' in line:
                # Next line(s) have slope values
                idx = visu_content.index(line)
                slope_lines = visu_content[idx:].split('\n')[1:3]
                slope_text = ' '.join(slope_lines)
                slopes = [float(x) for x in slope_text.split() if x.replace('.', '').replace('-', '').isdigit()]
                if slopes:
                    data_slope = slopes[0]  # Use first slope
            elif '##$VisuCoreWordType=' in line:
                word_type = line.split('=')[1].strip()
    
    # Determine dtype
    if '_16BIT_SGN_INT' in word_type:
        dtype = '<i2'  # 16-bit signed integer, little endian
    elif '_32BIT_FLOAT' in word_type:
        dtype = '<f4'  # 32-bit float, little endian
    else:
        raise ValueError(f"Unknown word type: {word_type}")
    
    # Parse data
    data = np.frombuffer(data_raw, dtype=dtype)
    
    # Reshape
    if len(core_size) == 2:
        height, width = core_size
        expected_size = frame_count * height * width
        
        if data.size != expected_size:
            raise ValueError(f"Data size mismatch: got {data.size}, expected {expected_size}")
        
        data = data.reshape((frame_count, height, width))
    else:
        raise ValueError(f"Unexpected core size: {core_size}")
    
    # Apply scaling for integer data
    if dtype == '<i2' and data_slope is not None:
        data = data.astype(np.float32) * data_slope
    
    print(f"\n✓ Loaded {frame_count} echoes, size {height}×{width}")
    print(f"  Data type: {word_type}")
    if data_slope:
        print(f"  Scaling factor: {data_slope:.4f}")
    
    return data, data_slope


def fit_scan(pvdatasets_path: str,
             output_dir: str,
             scan_label: str,
             bounds: tuple = ((0, 5), (np.inf, 2000))) -> dict:
    """
    Fit T2* for a single scan
    
    Parameters:
    -----------
    pvdatasets_path : str
        Path to .PvDatasets file
    output_dir : str
        Output directory for results
    scan_label : str
        Label for this scan (e.g., 'M1_E19_air')
    bounds : tuple
        Fitting bounds for (S0, T2*)
        
    Returns:
    --------
    results : dict
        Fitting results and quality metrics
    """
    pvdatasets_path = Path(pvdatasets_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"FITTING T2* MAP: {scan_label}")
    print(f"{'='*70}")
    print(f"Input: {pvdatasets_path.name}")
    
    # Extract echo times from method file
    echo_times = extract_echo_times_from_method(pvdatasets_path)
    echo_times = np.array(echo_times)
    
    # Load echo data (pdata/1 = raw echoes)
    echo_data, data_slope = load_echo_data(pvdatasets_path, reco_id=1)
    n_echoes = echo_data.shape[0]
    
    # Ensure echo times match number of echoes
    if len(echo_times) < n_echoes:
        print(f"\n⚠️  Warning: Only {len(echo_times)} echo times extracted, but {n_echoes} echoes in data")
        print(f"   Cannot proceed without all echo times")
        raise ValueError(f"Need {n_echoes} echo times, only got {len(echo_times)}")
    elif len(echo_times) > n_echoes:
        print(f"\n✓ Using first {n_echoes} echo times (extracted {len(echo_times)})")
        echo_times = echo_times[:n_echoes]
    
    echo_times = np.array(echo_times)
    
    # Fit T2* map
    print(f"\nFitting T2* with echo times: {echo_times.tolist()} ms")
    print(f"Bounds: S0=[{bounds[0][0]}, {bounds[1][0]}], T2*=[{bounds[0][1]}, {bounds[1][1]}] ms")
    
    fit_results = fit_t2star_map(
        echo_data=echo_data,
        echo_times=echo_times,
        bounds=bounds,
        initial_t2star=30.0,
        min_signal=10.0,
        show_progress=True
    )
    
    # Compute R2* map
    r2star_map = compute_r2star_map(fit_results['t2star_map'])
    
    # Validate quality
    quality = validate_fit_quality(fit_results)
    
    # Print summary
    print(f"\n{'─'*70}")
    print("FIT QUALITY SUMMARY:")
    print(f"{'─'*70}")
    print(f"Pixels fitted: {quality['n_fitted_pixels']:,} / {quality['n_total_pixels']:,} "
          f"({quality['pct_fitted']:.1f}%)")
    print(f"\nT2* Statistics:")
    print(f"  Mean:   {quality['t2_mean']:.1f} ms")
    print(f"  Median: {quality['t2_median']:.1f} ms")
    print(f"  Std:    {quality['t2_std']:.1f} ms")
    print(f"  Range:  [{quality['t2_min']:.1f}, {quality['t2_max']:.1f}] ms")
    print(f"\nFit Quality (R²):")
    print(f"  Mean R²: {quality['r2_mean']:.3f}")
    print(f"  Good fits (R²>0.7): {quality['pct_good_fits']:.1f}%")
    print(f"\nArtifact Detection:")
    print(f"  At floor (≤10ms): {quality['pct_at_floor']:.1f}%")
    print(f"  At ceiling (≥1950ms): {quality['pct_at_ceiling']:.1f}%")
    
    # Save results
    print(f"\n{'─'*70}")
    print("SAVING RESULTS:")
    print(f"{'─'*70}")
    
    # Save maps
    np.save(output_dir / f'{scan_label}_t2star.npy', fit_results['t2star_map'])
    np.save(output_dir / f'{scan_label}_r2star.npy', r2star_map)
    np.save(output_dir / f'{scan_label}_s0.npy', fit_results['s0_map'])
    np.save(output_dir / f'{scan_label}_r2_goodness.npy', fit_results['r2_map'])
    np.save(output_dir / f'{scan_label}_success.npy', fit_results['success_map'])
    
    print(f"✓ Saved: {scan_label}_t2star.npy")
    print(f"✓ Saved: {scan_label}_r2star.npy")
    print(f"✓ Saved: {scan_label}_s0.npy")
    print(f"✓ Saved: {scan_label}_r2_goodness.npy")
    print(f"✓ Saved: {scan_label}_success.npy")
    
    # Save metadata
    metadata = {
        'scan_label': scan_label,
        'input_file': str(pvdatasets_path),
        'echo_times_ms': echo_times.tolist(),
        'n_echoes': int(n_echoes),
        'fitting_bounds': {
            's0': [float(bounds[0][0]), float(bounds[1][0])],
            't2star_ms': [float(bounds[0][1]), float(bounds[1][1])]
        },
        'quality_metrics': {k: float(v) if isinstance(v, (int, float, np.number)) else v 
                           for k, v in quality.items()},
        'data_slope': float(data_slope) if data_slope else None
    }
    
    with open(output_dir / f'{scan_label}_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✓ Saved: {scan_label}_metadata.json")
    
    return {
        'fit_results': fit_results,
        'r2star_map': r2star_map,
        'quality': quality,
        'metadata': metadata
    }


def main():
    parser = argparse.ArgumentParser(
        description='Fit T2* maps from Bruker multi-echo data'
    )
    parser.add_argument('pvdatasets', help='Path to .PvDatasets file')
    parser.add_argument('--output', '-o', required=True, help='Output directory')
    parser.add_argument('--label', '-l', required=True, help='Scan label (e.g., M1_E19_air)')
    parser.add_argument('--t2-min', type=float, default=5.0, help='Minimum T2* (ms)')
    parser.add_argument('--t2-max', type=float, default=2000.0, help='Maximum T2* (ms)')
    
    args = parser.parse_args()
    
    # Fit scan
    bounds = ((0, args.t2_min), (np.inf, args.t2_max))
    results = fit_scan(args.pvdatasets, args.output, args.label, bounds)
    
    print(f"\n{'='*70}")
    print("✓ FITTING COMPLETE!")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
