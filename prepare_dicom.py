#!/usr/bin/env python3
"""
Prepare Data from DICOM Exports
=================================

Extract T2* maps, R2* maps, and perfusion data from Bruker DICOM exports
for use in BoldPy MLCO analysis workflow.

This is Step 0 of the complete workflow (DICOM variant of prepare_data.py).

Expected DICOM file naming convention:
    Lucas_Ferreira_{session}_{type}_{scanID}_{condition}_{scan}_{proc}_EnIm1.dcm

File types handled:
    - BOLD_{id}_{pre|20%|post}_E*_P1       Raw multi-echo BOLD
    - Parameter_maps_T2_relaxation_*_E*_P2  Pre-computed T2* parameter maps
    - Lucas_Perfusion_{id}_E*_P1            Raw perfusion images
    - Parameter_maps_T1_*_E*_P2             T1 IR parameter maps
    - Parameter_maps_T1_*_E*_P3             rCBF perfusion maps
    - T2_{id}_E*_P1                         T2 anatomical reference
"""

import numpy as np
import argparse
import json
import re
from pathlib import Path
import sys
from typing import Dict, Optional, Tuple, List
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# Add src directory to path
script_dir = Path(__file__).parent
src_dir = script_dir / 'src'
if src_dir.exists():
    sys.path.insert(0, str(src_dir))

# Import DICOM parametric loader
try:
    from boldpy.loaders.dicom_parametric import load_bruker_parametric_dicom
    LOADER_AVAILABLE = True
except ImportError as e:
    print("=" * 70)
    print("ERROR: Could not import boldpy DICOM loader")
    print("=" * 70)
    print(f"\nError: {e}")
    print("\nMake sure you've installed boldpy and pydicom:")
    print('  pip install -e ".[dicom]"')
    print("=" * 70)
    LOADER_AVAILABLE = False

# Condition mapping: DICOM filename condition -> pipeline condition name
CONDITION_MAP = {
    'pre': 'oxygen_1',
    '20%': 'air',
    'post': 'oxygen_2',
}


# ============================================================================
# DICOM FILE CLASSIFICATION
# ============================================================================

def classify_dicom_files(sample_dir: Path) -> Dict:
    """
    Scan a sample directory and classify DICOM files by type.

    Parameters
    ----------
    sample_dir : Path
        Directory containing DICOM files for one sample.

    Returns
    -------
    classified : dict
        Dictionary with keys:
        - 'bold': {condition: Path} for raw BOLD files
        - 't2_param_maps': {condition: Path} for T2* parameter map files
        - 'perfusion_raw': Path or None
        - 'perfusion_param_p2': Path or None (T1 IR P2)
        - 'perfusion_param_p3': Path or None (rCBF P3)
        - 't2_anatomical': Path or None
        - 'unclassified': [Path, ...]
        - 'session': str (session ID from filename prefix)
        - 'warnings': [str, ...]
    """
    sample_dir = Path(sample_dir)
    dcm_files = sorted(sample_dir.glob('*.dcm'))

    if not dcm_files:
        raise FileNotFoundError(f"No .dcm files found in {sample_dir}")

    classified = {
        'bold': {},
        't2_param_maps': {},
        'perfusion_raw': None,
        'perfusion_param_p2': None,
        'perfusion_param_p3': None,
        't2_anatomical': None,
        'unclassified': [],
        'session': None,
        'warnings': [],
        'experiment_map': {},  # E-number -> condition mapping
    }

    # Regex patterns for file classification
    # Format: Lucas_Ferreira_{session}_{type_and_rest}_EnIm1.dcm
    # The session part may contain letters (e.g., "173979repeat")
    bold_pattern = re.compile(
        r'_BOLD_(\w+)_(pre|20%|post)_E(\d+)_P1_', re.IGNORECASE
    )
    t2_param_pattern = re.compile(
        r'_Parameter_maps_T2_relaxation.*?_E(\d+)_P2_', re.IGNORECASE
    )
    perfusion_raw_pattern = re.compile(
        r'_(?:Lucas_)?Perfusion_(\w+)_E(\d+)_P1_', re.IGNORECASE
    )
    t1_param_p2_pattern = re.compile(
        r'_Parameter_maps_T1_inversion_recovery.*?_E(\d+)_P2_', re.IGNORECASE
    )
    t1_param_p3_pattern = re.compile(
        r'_Parameter_maps_T1_inversion_recovery.*?_E(\d+)_P3_', re.IGNORECASE
    )
    t2_anat_pattern = re.compile(
        r'_T2_(\w+)_E(\d+)_P1_', re.IGNORECASE
    )
    # Extract session from the filename prefix
    session_pattern = re.compile(
        r'^Lucas_Ferreira_(\w+?)_', re.IGNORECASE
    )

    for dcm_path in dcm_files:
        fname = dcm_path.name
        matched = False

        # Extract session ID
        if classified['session'] is None:
            session_match = session_pattern.match(fname)
            if session_match:
                classified['session'] = session_match.group(1)

        # BOLD raw
        m = bold_pattern.search(fname)
        if m:
            scan_id, condition, exp_num = m.group(1), m.group(2), m.group(3)
            classified['bold'][condition] = dcm_path
            classified['experiment_map'][f'E{exp_num}'] = condition
            matched = True
            continue

        # T2 parameter maps
        m = t2_param_pattern.search(fname)
        if m:
            exp_num = m.group(1)
            # Store by experiment number first; we'll map to conditions later
            classified['t2_param_maps'][f'E{exp_num}'] = dcm_path
            matched = True
            continue

        # Perfusion raw
        m = perfusion_raw_pattern.search(fname)
        if m:
            classified['perfusion_raw'] = dcm_path
            matched = True
            continue

        # T1 IR P3 (rCBF) - check before P2 since P3 is more specific
        m = t1_param_p3_pattern.search(fname)
        if m:
            classified['perfusion_param_p3'] = dcm_path
            matched = True
            continue

        # T1 IR P2
        m = t1_param_p2_pattern.search(fname)
        if m:
            classified['perfusion_param_p2'] = dcm_path
            matched = True
            continue

        # T2 anatomical
        m = t2_anat_pattern.search(fname)
        if m:
            classified['t2_anatomical'] = dcm_path
            matched = True
            continue

        if not matched:
            classified['unclassified'].append(dcm_path)

    # Map T2 param maps from E-numbers to conditions using BOLD experiment map
    t2_by_condition = {}
    for e_key, t2_path in classified['t2_param_maps'].items():
        condition = classified['experiment_map'].get(e_key)
        if condition:
            t2_by_condition[condition] = t2_path
        else:
            # E-number doesn't match any BOLD condition
            classified['warnings'].append(
                f"T2 param map {t2_path.name} (experiment {e_key}) "
                f"has no matching BOLD condition"
            )
            # Still keep it with the E-number key as fallback
            t2_by_condition[e_key] = t2_path
    classified['t2_param_maps'] = t2_by_condition

    # Validate: check for cross-referenced scan IDs (like sample 174004)
    dir_name = sample_dir.name
    if classified['session'] and classified['session'] != dir_name:
        # Session name in files doesn't match directory - could have suffix like "repeat"
        if not classified['session'].startswith(dir_name):
            classified['warnings'].append(
                f"Session ID in filenames ({classified['session']}) "
                f"differs from directory name ({dir_name})"
            )

    # Check for BOLD scan IDs that differ from session
    for condition, bold_path in classified['bold'].items():
        m = bold_pattern.search(bold_path.name)
        if m:
            bold_scan_id = m.group(1)
            session = classified['session'] or dir_name
            # Strip common suffixes for comparison
            session_base = re.sub(r'repeat$', '', session)
            if bold_scan_id != session_base and bold_scan_id != dir_name:
                classified['warnings'].append(
                    f"BOLD {condition} references scan ID {bold_scan_id} "
                    f"(directory: {dir_name}, session: {session})"
                )

    return classified


def print_classification_summary(classified: Dict, sample_dir: Path) -> None:
    """Print a human-readable summary of classified DICOM files."""
    print(f"\n  Session: {classified.get('session', 'unknown')}")

    # BOLD files
    print(f"\n  BOLD raw ({len(classified['bold'])} conditions):")
    for condition in ['pre', '20%', 'post']:
        if condition in classified['bold']:
            print(f"    {condition:6s} -> {classified['bold'][condition].name}")
        else:
            print(f"    {condition:6s} -> MISSING")

    # T2 parameter maps
    print(f"\n  T2* parameter maps ({len(classified['t2_param_maps'])} conditions):")
    for condition in ['pre', '20%', 'post']:
        if condition in classified['t2_param_maps']:
            print(f"    {condition:6s} -> {classified['t2_param_maps'][condition].name}")
    # Show any unmapped E-number entries
    for key, path in classified['t2_param_maps'].items():
        if key.startswith('E'):
            print(f"    {key:6s} -> {path.name}  (unmapped)")

    # Perfusion
    print(f"\n  Perfusion:")
    if classified['perfusion_raw']:
        print(f"    Raw:  {classified['perfusion_raw'].name}")
    else:
        print(f"    Raw:  not found")
    if classified['perfusion_param_p3']:
        print(f"    rCBF: {classified['perfusion_param_p3'].name}")
    else:
        print(f"    rCBF: not found")
    if classified['perfusion_param_p2']:
        print(f"    T1IR: {classified['perfusion_param_p2'].name}")

    # Reference
    print(f"\n  Reference:")
    if classified['t2_anatomical']:
        print(f"    T2:   {classified['t2_anatomical'].name}")
    else:
        print(f"    T2:   not found")

    # Unclassified
    if classified['unclassified']:
        print(f"\n  Unclassified ({len(classified['unclassified'])}):")
        for p in classified['unclassified']:
            print(f"    ?     {p.name}")

    # Warnings
    if classified['warnings']:
        print(f"\n  Warnings:")
        for w in classified['warnings']:
            print(f"    !  {w}")


# ============================================================================
# DATA EXTRACTION
# ============================================================================

def extract_t2star_from_dicom(
    dcm_path: Path,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract T2* and R2* maps from a Bruker T2 parameter map DICOM.

    Parameters
    ----------
    dcm_path : Path
        Path to T2 relaxation parameter map DICOM (Enhanced, 6 frames).

    Returns
    -------
    t2star_map : ndarray (height, width), float32
        T2* relaxation time in ms.
    r2star_map : ndarray (height, width), float32
        R2* = 1000/T2* in Hz.
    """
    if not LOADER_AVAILABLE:
        raise ImportError("DICOM loader not available. Install pydicom.")

    data = load_bruker_parametric_dicom(str(dcm_path), extract_uncertainty=True)

    t2star_map = data['map'].astype(np.float32)

    # Validate T2* range
    valid_mask = t2star_map > 0
    if np.any(valid_mask):
        mean_val = float(np.mean(t2star_map[valid_mask]))
        max_val = float(np.max(t2star_map[valid_mask]))
        logger.info(f"T2* stats: mean={mean_val:.1f} ms, max={max_val:.1f} ms")

        if mean_val > 200 or mean_val < 1:
            logger.warning(
                f"Suspicious T2* values (mean={mean_val:.1f} ms). "
                f"Expected 5-100 ms for kidney at high field."
            )

    # Compute R2* = 1000 / T2*
    r2star_map = np.zeros_like(t2star_map, dtype=np.float32)
    mask = t2star_map > 0
    r2star_map[mask] = 1000.0 / t2star_map[mask]

    return t2star_map, r2star_map


def extract_perfusion_from_dicom(dcm_path: Path) -> np.ndarray:
    """
    Extract perfusion (rCBF) map from a Bruker T1 IR parameter map DICOM (P3).

    Parameters
    ----------
    dcm_path : Path
        Path to T1 IR parameter map P3 DICOM (rCBF).

    Returns
    -------
    perfusion_map : ndarray (height, width), float32
    """
    try:
        import pydicom
    except ImportError:
        raise ImportError("pydicom not installed. Install with: pip install pydicom")

    ds = pydicom.dcmread(str(dcm_path))
    pixel_data = ds.pixel_array.astype(np.float32)

    # Apply RescaleSlope/Intercept
    # Check per-frame first (Enhanced DICOM)
    rescaled = False
    if hasattr(ds, 'PerFrameFunctionalGroupsSequence'):
        for i, frame_seq in enumerate(ds.PerFrameFunctionalGroupsSequence):
            if hasattr(frame_seq, 'PixelValueTransformationSequence'):
                pvt = frame_seq.PixelValueTransformationSequence[0]
                slope = float(getattr(pvt, 'RescaleSlope', 1.0))
                intercept = float(getattr(pvt, 'RescaleIntercept', 0.0))
                if slope != 1.0 or intercept != 0.0:
                    if pixel_data.ndim == 3 and i < pixel_data.shape[0]:
                        pixel_data[i] = pixel_data[i] * slope + intercept
                    elif pixel_data.ndim == 2:
                        pixel_data = pixel_data * slope + intercept
                    rescaled = True

    # Check shared functional groups
    if not rescaled and hasattr(ds, 'SharedFunctionalGroupsSequence'):
        shared = ds.SharedFunctionalGroupsSequence[0]
        if hasattr(shared, 'PixelValueTransformationSequence'):
            pvt = shared.PixelValueTransformationSequence[0]
            slope = float(getattr(pvt, 'RescaleSlope', 1.0))
            intercept = float(getattr(pvt, 'RescaleIntercept', 0.0))
            if slope != 1.0 or intercept != 0.0:
                pixel_data = pixel_data * slope + intercept
                rescaled = True

    # Check top-level RescaleSlope (standard DICOM)
    if not rescaled:
        slope = float(getattr(ds, 'RescaleSlope', 1.0))
        intercept = float(getattr(ds, 'RescaleIntercept', 0.0))
        if slope != 1.0 or intercept != 0.0:
            pixel_data = pixel_data * slope + intercept

    # For multi-frame, try to identify the rCBF frame
    if pixel_data.ndim == 3:
        n_frames = pixel_data.shape[0]
        if n_frames == 1:
            perfusion_map = pixel_data[0]
        else:
            # Try to find rCBF frame via FrameComments
            rcbf_idx = None
            if hasattr(ds, 'PerFrameFunctionalGroupsSequence'):
                for i, frame_seq in enumerate(ds.PerFrameFunctionalGroupsSequence):
                    comment = _get_frame_comment(frame_seq)
                    if comment and 'rcbf' in comment.lower():
                        rcbf_idx = i
                        break
                    if comment and 'cbf' in comment.lower():
                        rcbf_idx = i
                        break

            if rcbf_idx is not None:
                perfusion_map = pixel_data[rcbf_idx]
                logger.info(f"Using frame {rcbf_idx} as rCBF map")
            else:
                # Fallback: use first frame
                logger.warning(
                    f"Could not identify rCBF frame in {n_frames}-frame DICOM. "
                    f"Using first frame."
                )
                perfusion_map = pixel_data[0]
    else:
        perfusion_map = pixel_data

    return perfusion_map.astype(np.float32)


def _get_frame_comment(frame_seq) -> Optional[str]:
    """Extract frame comment from a per-frame functional group sequence item."""
    # FrameContentSequence (Bruker)
    if hasattr(frame_seq, 'FrameContentSequence'):
        fcs = frame_seq.FrameContentSequence[0]
        if hasattr(fcs, 'FrameComments'):
            return str(fcs.FrameComments)
        if hasattr(fcs, 'FrameLabel'):
            return str(fcs.FrameLabel)
    # Direct
    if hasattr(frame_seq, 'FrameComments'):
        return str(frame_seq.FrameComments)
    return None


def extract_reference_from_dicom(dcm_path: Path) -> np.ndarray:
    """
    Extract a reference image from a T2 anatomical or BOLD DICOM.

    Parameters
    ----------
    dcm_path : Path
        Path to T2 anatomical or first-echo BOLD DICOM.

    Returns
    -------
    reference : ndarray (height, width), float32
    """
    try:
        import pydicom
    except ImportError:
        raise ImportError("pydicom not installed. Install with: pip install pydicom")

    ds = pydicom.dcmread(str(dcm_path))
    pixel_data = ds.pixel_array.astype(np.float32)

    # Apply RescaleSlope if present
    slope = float(getattr(ds, 'RescaleSlope', 1.0))
    intercept = float(getattr(ds, 'RescaleIntercept', 0.0))
    if slope != 1.0 or intercept != 0.0:
        pixel_data = pixel_data * slope + intercept

    # Also check Enhanced DICOM shared groups
    if hasattr(ds, 'SharedFunctionalGroupsSequence'):
        shared = ds.SharedFunctionalGroupsSequence[0]
        if hasattr(shared, 'PixelValueTransformationSequence'):
            pvt = shared.PixelValueTransformationSequence[0]
            s = float(getattr(pvt, 'RescaleSlope', 1.0))
            i = float(getattr(pvt, 'RescaleIntercept', 0.0))
            if s != 1.0 or i != 0.0:
                pixel_data = pixel_data * s + i

    # For multi-frame DICOM, take first frame (best anatomical contrast)
    if pixel_data.ndim == 3:
        reference = pixel_data[0]
        logger.info(f"Using first frame of {pixel_data.shape[0]}-frame DICOM as reference")
    else:
        reference = pixel_data

    return reference.astype(np.float32)


# ============================================================================
# SAMPLE PREPARATION
# ============================================================================

def prepare_dicom_sample(
    sample_dir: Path,
    output_dir: Path,
    sample_name: Optional[str] = None,
    skip_perfusion: bool = False,
    no_reference: bool = False,
) -> Dict:
    """
    Prepare data from a single DICOM sample directory.

    Orchestrates: classify -> extract T2*/R2* per condition -> extract perfusion
    -> extract reference -> save .npy files + JSON config.

    Parameters
    ----------
    sample_dir : Path
        Directory containing DICOM files for one sample.
    output_dir : Path
        Output directory for .npy and JSON files.
    sample_name : str, optional
        Sample name for output files (default: directory name).
    skip_perfusion : bool
        Skip perfusion extraction.
    no_reference : bool
        Skip reference image extraction.

    Returns
    -------
    results : dict
        Processing results including file paths and metadata.
    """
    sample_dir = Path(sample_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if sample_name is None:
        sample_name = sample_dir.name

    print(f"\n{'=' * 70}")
    print(f"Processing DICOM sample: {sample_dir.name}")
    print(f"Sample name: {sample_name}")
    print(f"Output: {output_dir}")
    print(f"{'=' * 70}")

    results = {
        'sample_name': sample_name,
        'input_dir': str(sample_dir),
        'output_dir': str(output_dir),
        'files_created': [],
        'metadata': {
            'format': 'dicom',
            'extracted_at': datetime.now().isoformat(),
        },
    }

    # Step 1: Classify DICOM files
    print("\n1. Classifying DICOM files...")
    classified = classify_dicom_files(sample_dir)
    print_classification_summary(classified, sample_dir)

    results['metadata']['session'] = classified.get('session')
    results['metadata']['warnings'] = classified.get('warnings', [])

    # Step 2: Extract T2* for each condition
    print("\n2. Extracting T2* maps...")
    t2star_paths = {}
    r2star_paths = {}

    for condition in ['pre', '20%', 'post']:
        if condition not in classified['t2_param_maps']:
            print(f"  ! {condition}: no T2 parameter map found, skipping")
            continue

        dcm_path = classified['t2_param_maps'][condition]
        pipeline_condition = CONDITION_MAP[condition]

        print(f"  Extracting {condition} ({pipeline_condition})...")
        try:
            t2star_map, r2star_map = extract_t2star_from_dicom(dcm_path)

            # Save T2*
            t2_fname = output_dir / f"{sample_name}_{pipeline_condition}_t2star_bruker.npy"
            np.save(t2_fname, t2star_map)
            t2star_paths[pipeline_condition] = str(t2_fname.resolve())
            results['files_created'].append(str(t2_fname))

            # Save R2*
            r2_fname = output_dir / f"{sample_name}_{pipeline_condition}_r2star_bruker.npy"
            np.save(r2_fname, r2star_map)
            r2star_paths[pipeline_condition] = str(r2_fname.resolve())
            results['files_created'].append(str(r2_fname))

            # Stats
            valid = t2star_map > 0
            if np.any(valid):
                mean_t2 = float(np.mean(t2star_map[valid]))
                print(f"    T2* mean: {mean_t2:.1f} ms, shape: {t2star_map.shape}")

                if mean_t2 > 200 or mean_t2 < 1:
                    print(f"    WARNING: Suspicious T2* values!")
            else:
                print(f"    WARNING: No valid T2* pixels!")

        except Exception as e:
            print(f"    FAILED: {e}")
            import traceback
            traceback.print_exc()

    # Step 3: Extract perfusion
    perfusion_path = None
    if not skip_perfusion and classified['perfusion_param_p3']:
        print("\n3. Extracting perfusion (rCBF) map...")
        try:
            perfusion_map = extract_perfusion_from_dicom(
                classified['perfusion_param_p3']
            )
            perf_fname = output_dir / f"{sample_name}_perfusion.npy"
            np.save(perf_fname, perfusion_map)
            perfusion_path = str(perf_fname.resolve())
            results['files_created'].append(str(perf_fname))

            valid = perfusion_map > 0
            if np.any(valid):
                print(f"    Perfusion shape: {perfusion_map.shape}")
                print(f"    Perfusion mean: {float(np.mean(perfusion_map[valid])):.1f}")
            else:
                print(f"    WARNING: No valid perfusion pixels!")

        except Exception as e:
            print(f"    FAILED: {e}")
            import traceback
            traceback.print_exc()
    elif not skip_perfusion:
        print("\n3. Perfusion: no rCBF (P3) DICOM found, skipping")
    else:
        print("\n3. Perfusion: skipped (--skip-perfusion)")

    # Step 4: Extract reference
    reference_path = None
    if not no_reference:
        print("\n4. Extracting reference image...")
        ref_source = None
        if classified['t2_anatomical']:
            ref_source = classified['t2_anatomical']
            ref_type = "T2 anatomical"
        elif 'pre' in classified['bold']:
            ref_source = classified['bold']['pre']
            ref_type = "BOLD pre (first echo)"

        if ref_source:
            try:
                reference = extract_reference_from_dicom(ref_source)
                ref_fname = output_dir / f"{sample_name}_reference.npy"
                np.save(ref_fname, reference)
                reference_path = str(ref_fname.resolve())
                results['files_created'].append(str(ref_fname))
                print(f"    Source: {ref_type}")
                print(f"    Shape: {reference.shape}")
            except Exception as e:
                print(f"    FAILED: {e}")
        else:
            print(f"    No reference source found")
    else:
        print("\n4. Reference: skipped (--no-reference)")

    # Step 5: Generate JSON config
    print("\n5. Generating config JSON...")
    config = {
        'id': sample_name,
        'source_format': 'dicom',
        'extracted_at': datetime.now().isoformat(),
    }

    if t2star_paths:
        config['t2star_maps'] = t2star_paths
    if r2star_paths:
        config['r2star_maps'] = r2star_paths
    if perfusion_path:
        config['perfusion_map'] = perfusion_path
    if reference_path:
        config['reference'] = reference_path

    config_fname = output_dir / f"{sample_name}_config.json"
    with open(config_fname, 'w') as f:
        json.dump(config, f, indent=2)
    results['files_created'].append(str(config_fname))

    # Save detailed metadata
    metadata_fname = output_dir / f"{sample_name}_metadata.json"
    results['metadata']['classified_files'] = {
        'bold_conditions': list(classified['bold'].keys()),
        't2_param_conditions': list(classified['t2_param_maps'].keys()),
        'has_perfusion_raw': classified['perfusion_raw'] is not None,
        'has_perfusion_rcbf': classified['perfusion_param_p3'] is not None,
        'has_t2_anatomical': classified['t2_anatomical'] is not None,
        'n_unclassified': len(classified['unclassified']),
    }
    with open(metadata_fname, 'w') as f:
        json.dump(results['metadata'], f, indent=2)
    results['files_created'].append(str(metadata_fname))

    # Summary
    print(f"\n{'=' * 70}")
    print(f"Processing complete: {sample_name}")
    print(f"  Created {len(results['files_created'])} files in {output_dir}")
    print(f"  T2* conditions: {list(t2star_paths.keys())}")
    print(f"  Perfusion: {'yes' if perfusion_path else 'no'}")
    print(f"  Reference: {'yes' if reference_path else 'no'}")
    if classified['warnings']:
        print(f"  Warnings: {len(classified['warnings'])}")
        for w in classified['warnings']:
            print(f"    ! {w}")
    print(f"{'=' * 70}")

    return results


def prepare_batch(
    batch_dir: Path,
    output_dir: Path,
    skip_perfusion: bool = False,
    no_reference: bool = False,
) -> List[Dict]:
    """
    Batch process all sample subdirectories.

    Parameters
    ----------
    batch_dir : Path
        Parent directory containing one subdirectory per sample.
    output_dir : Path
        Output parent directory.
    skip_perfusion : bool
        Skip perfusion extraction for all samples.
    no_reference : bool
        Skip reference extraction for all samples.

    Returns
    -------
    all_results : list of dict
    """
    batch_dir = Path(batch_dir)
    output_dir = Path(output_dir)

    # Find subdirectories containing .dcm files
    sample_dirs = []
    for d in sorted(batch_dir.iterdir()):
        if d.is_dir() and list(d.glob('*.dcm')):
            sample_dirs.append(d)

    if not sample_dirs:
        print(f"No sample directories with .dcm files found in {batch_dir}")
        return []

    print(f"\n{'=' * 70}")
    print(f"BATCH PROCESSING: {len(sample_dirs)} samples")
    print(f"{'=' * 70}")
    for d in sample_dirs:
        n_files = len(list(d.glob('*.dcm')))
        print(f"  {d.name}: {n_files} DICOM files")

    all_results = []
    for i, sample_dir in enumerate(sample_dirs, 1):
        print(f"\n[{i}/{len(sample_dirs)}] {sample_dir.name}")
        try:
            sample_output = output_dir / sample_dir.name
            results = prepare_dicom_sample(
                sample_dir=sample_dir,
                output_dir=sample_output,
                skip_perfusion=skip_perfusion,
                no_reference=no_reference,
            )
            all_results.append(results)
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Batch summary
    print(f"\n{'=' * 70}")
    print(f"BATCH COMPLETE: {len(all_results)}/{len(sample_dirs)} successful")
    print(f"{'=' * 70}")

    for r in all_results:
        warnings = r.get('metadata', {}).get('warnings', [])
        status = "OK" if not warnings else f"{len(warnings)} warnings"
        print(f"  {r['sample_name']}: {len(r['files_created'])} files ({status})")

    return all_results


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Prepare DICOM data for BoldPy MLCO analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Classify files (dry run)
  python prepare_dicom.py --input-dir data/174005/ --list-only

  # Single sample
  python prepare_dicom.py --input-dir data/174005/ --output-dir prepared/174005/

  # Single sample with custom name
  python prepare_dicom.py --input-dir data/174005/ --output-dir prepared/174005/ \\
      --sample-name 174005_captopril

  # Batch process all samples
  python prepare_dicom.py --batch-dir data/ --output-dir prepared/

  # Skip perfusion
  python prepare_dicom.py --input-dir data/174005/ --output-dir prepared/174005/ \\
      --skip-perfusion
        """
    )

    # Input (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--input-dir', type=Path,
        help='Input directory containing DICOM files for one sample'
    )
    input_group.add_argument(
        '--batch-dir', type=Path,
        help='Parent directory with one subdirectory per sample (batch mode)'
    )

    # Output
    parser.add_argument(
        '--output-dir', type=Path, default=Path('prepared_data'),
        help='Output directory (default: prepared_data/)'
    )
    parser.add_argument(
        '--sample-name', type=str,
        help='Sample name for output files (default: directory name)'
    )

    # Options
    parser.add_argument(
        '--list-only', action='store_true',
        help='Classify files and print summary without extracting data'
    )
    parser.add_argument(
        '--skip-perfusion', action='store_true',
        help='Skip perfusion extraction'
    )
    parser.add_argument(
        '--no-reference', action='store_true',
        help='Skip reference image extraction'
    )

    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s: %(message)s'
    )

    # List-only mode
    if args.list_only:
        if args.input_dir:
            dirs = [args.input_dir]
        else:
            dirs = sorted(
                d for d in args.batch_dir.iterdir()
                if d.is_dir() and list(d.glob('*.dcm'))
            )

        for d in dirs:
            print(f"\n{'=' * 70}")
            print(f"Sample: {d.name}")
            print(f"{'=' * 70}")
            try:
                classified = classify_dicom_files(d)
                print_classification_summary(classified, d)
            except Exception as e:
                print(f"  ERROR: {e}")
        return

    # Single sample
    if args.input_dir:
        if not LOADER_AVAILABLE:
            print("ERROR: DICOM loader not available. Install pydicom.")
            sys.exit(1)

        prepare_dicom_sample(
            sample_dir=args.input_dir,
            output_dir=args.output_dir,
            sample_name=args.sample_name,
            skip_perfusion=args.skip_perfusion,
            no_reference=args.no_reference,
        )

    # Batch mode
    else:
        if not LOADER_AVAILABLE:
            print("ERROR: DICOM loader not available. Install pydicom.")
            sys.exit(1)

        prepare_batch(
            batch_dir=args.batch_dir,
            output_dir=args.output_dir,
            skip_perfusion=args.skip_perfusion,
            no_reference=args.no_reference,
        )


if __name__ == '__main__':
    main()
