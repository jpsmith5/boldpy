"""
Bruker DICOM Parametric Map Loader
===================================

Load processed parameter maps exported from Bruker ParaVision:
- T2* maps
- R2* maps  
- T1 maps
- T2 maps
- Fit quality metrics (standard deviations)

Your DICOM files:
    T2star_map_MGE_E11_P1_EnIm1.dcm  (fitted T2* map)
    Parameter_maps_T2_relaxation_*.dcm (fitted T2 maps)
    etc.

Enhanced DICOM format contains multiple frames:
    Frame 1: signal intensity
    Frame 2: std dev of signal intensity  
    Frame 3: relaxation time (T2*, T1, etc.)
    Frame 4: std dev of relaxation time
    Frame 5: std dev of fit
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import logging

logger = logging.getLogger(__name__)


def load_bruker_parametric_dicom(
    dicom_path: Union[str, Path],
    extract_uncertainty: bool = True
) -> Dict[str, np.ndarray]:
    """
    Load Bruker parametric map DICOM (Enhanced multi-frame format)
    
    Parameters
    ----------
    dicom_path : str or Path
        Path to DICOM file containing parametric map
    extract_uncertainty : bool
        If True, extract standard deviation frames
        
    Returns
    -------
    data : dict
        Dictionary containing:
        - 'map': Main parameter map (T2*, R2*, etc.)
        - 'map_std': Standard deviation of parameter (if available)
        - 'signal': Signal intensity
        - 'signal_std': Std dev of signal (if available)
        - 'fit_std': Std dev of fit (if available)
        - 'metadata': DICOM metadata
        
    Examples
    --------
    >>> data = load_bruker_parametric_dicom('T2star_map_MGE_E11_P1_EnIm1.dcm')
    >>> t2star_map = data['map']
    >>> t2star_uncertainty = data['map_std']
    >>> print(f"T2* range: {t2star_map.min():.1f} - {t2star_map.max():.1f} ms")
    """
    try:
        import pydicom
    except ImportError:
        raise ImportError(
            "pydicom not installed.\n"
            "Install with: pip install pydicom"
        )
    
    dicom_path = Path(dicom_path)
    
    if not dicom_path.exists():
        raise FileNotFoundError(f"DICOM file not found: {dicom_path}")
    
    logger.info(f"Loading Bruker parametric DICOM: {dicom_path.name}")
    
    # Load DICOM
    ds = pydicom.dcmread(str(dicom_path))
    
    # Check if Enhanced DICOM
    is_enhanced = hasattr(ds, 'NumberOfFrames') and ds.NumberOfFrames > 1
    
    if not is_enhanced:
        logger.warning(
            "Not an Enhanced DICOM (single frame). "
            "Loading as single parameter map."
        )
        pixel_arr = ds.pixel_array.astype(np.float32)
        # Apply RescaleSlope/Intercept for single-frame DICOMs
        slope = float(getattr(ds, 'RescaleSlope', 1.0))
        intercept = float(getattr(ds, 'RescaleIntercept', 0.0))
        if slope != 1.0 or intercept != 0.0:
            pixel_arr = pixel_arr * slope + intercept
            logger.info(f"Applied RescaleSlope={slope}, RescaleIntercept={intercept}")
        data = {
            'map': pixel_arr,
            'metadata': _extract_dicom_metadata(ds),
        }
        return data
    
    # Enhanced DICOM - multiple frames
    n_frames = int(ds.NumberOfFrames)
    logger.info(f"Enhanced DICOM with {n_frames} frames")
    
    # Get pixel array (shape: (n_frames, height, width))
    pixel_data = ds.pixel_array.astype(np.float32)

    # Apply per-frame RescaleSlope/Intercept (Bruker stores raw integers)
    if hasattr(ds, 'PerFrameFunctionalGroupsSequence'):
        for i, frame_seq in enumerate(ds.PerFrameFunctionalGroupsSequence):
            if hasattr(frame_seq, 'PixelValueTransformationSequence'):
                pvt = frame_seq.PixelValueTransformationSequence[0]
                slope = float(getattr(pvt, 'RescaleSlope', 1.0))
                intercept = float(getattr(pvt, 'RescaleIntercept', 0.0))
                if slope != 1.0 or intercept != 0.0:
                    if pixel_data.ndim == 3 and i < pixel_data.shape[0]:
                        pixel_data[i] = pixel_data[i] * slope + intercept
                        logger.debug(
                            f"Frame {i}: applied RescaleSlope={slope}, "
                            f"RescaleIntercept={intercept}"
                        )

    # Check SharedFunctionalGroupsSequence for shared rescaling
    if hasattr(ds, 'SharedFunctionalGroupsSequence'):
        shared = ds.SharedFunctionalGroupsSequence[0]
        if hasattr(shared, 'PixelValueTransformationSequence'):
            pvt = shared.PixelValueTransformationSequence[0]
            slope = float(getattr(pvt, 'RescaleSlope', 1.0))
            intercept = float(getattr(pvt, 'RescaleIntercept', 0.0))
            if slope != 1.0 or intercept != 0.0:
                pixel_data = pixel_data * slope + intercept
                logger.info(
                    f"Applied shared RescaleSlope={slope}, "
                    f"RescaleIntercept={intercept}"
                )

    if pixel_data.ndim == 2:
        # Single frame despite NumberOfFrames tag
        data = {
            'map': pixel_data,
            'metadata': _extract_dicom_metadata(ds),
        }
        return data
    
    # Parse frames based on FrameComments
    frame_map = _parse_frame_structure(ds)
    
    # Extract data
    data = {}
    
    # Main parameter map (T2*, R2*, T1, etc.)
    if 'relaxation_time' in frame_map:
        idx = frame_map['relaxation_time']
        data['map'] = pixel_data[idx].copy()
        logger.info(f"  Frame {idx}: Relaxation time map")
    elif 'map' in frame_map:
        idx = frame_map['map']
        data['map'] = pixel_data[idx].copy()
        logger.info(f"  Frame {idx}: Parameter map")
    else:
        # Fallback - use first frame
        logger.warning("Could not identify parameter map frame, using first frame")
        data['map'] = pixel_data[0].copy()
    
    if extract_uncertainty:
        # Standard deviation of map
        if 'relaxation_time_std' in frame_map:
            idx = frame_map['relaxation_time_std']
            data['map_std'] = pixel_data[idx].copy()
            logger.info(f"  Frame {idx}: Relaxation time std dev")
        
        # Signal intensity
        if 'signal' in frame_map:
            idx = frame_map['signal']
            data['signal'] = pixel_data[idx].copy()
            logger.info(f"  Frame {idx}: Signal intensity")
        
        # Signal std dev
        if 'signal_std' in frame_map:
            idx = frame_map['signal_std']
            data['signal_std'] = pixel_data[idx].copy()
            logger.info(f"  Frame {idx}: Signal std dev")
        
        # Fit std dev
        if 'fit_std' in frame_map:
            idx = frame_map['fit_std']
            data['fit_std'] = pixel_data[idx].copy()
            logger.info(f"  Frame {idx}: Fit std dev")
    
    # Extract metadata
    data['metadata'] = _extract_dicom_metadata(ds)
    data['frame_structure'] = frame_map
    
    # Log summary
    logger.info(f"Loaded parametric map: {data['map'].shape}")
    if 'map' in data:
        valid_pixels = np.isfinite(data['map'])
        if np.any(valid_pixels):
            logger.info(
                f"  Value range: {data['map'][valid_pixels].min():.2f} - "
                f"{data['map'][valid_pixels].max():.2f}"
            )
    
    return data


def _parse_frame_structure(ds) -> Dict[str, int]:
    """
    Parse Enhanced DICOM frame structure from FrameComments
    
    Returns mapping of frame type to frame index
    """
    frame_map = {}
    
    if not hasattr(ds, 'PerFrameFunctionalGroupsSequence'):
        logger.warning("No PerFrameFunctionalGroupsSequence found")
        return frame_map
    
    for frame_idx, frame_seq in enumerate(ds.PerFrameFunctionalGroupsSequence):
        # Try to get frame comments
        comment = None

        # Method 0: FrameContentSequence (Bruker Enhanced DICOM)
        if hasattr(frame_seq, 'FrameContentSequence'):
            fcs = frame_seq.FrameContentSequence[0]
            if hasattr(fcs, 'FrameComments'):
                comment = str(fcs.FrameComments).lower()
            elif hasattr(fcs, 'FrameLabel'):
                comment = str(fcs.FrameLabel).lower()

        # Method 1: Direct FrameComments
        if comment is None and hasattr(frame_seq, 'FrameComments'):
            comment = str(frame_seq.FrameComments).lower()

        # Method 2: PlaneOrientationSequence
        elif comment is None and hasattr(frame_seq, 'PlaneOrientationSequence'):
            if len(frame_seq.PlaneOrientationSequence) > 0:
                plane = frame_seq.PlaneOrientationSequence[0]
                if hasattr(plane, 'FrameComments'):
                    comment = str(plane.FrameComments).lower()
        
        # Method 3: Check MRImageFrameTypeSequence
        if comment is None and hasattr(frame_seq, 'MRImageFrameTypeSequence'):
            if len(frame_seq.MRImageFrameTypeSequence) > 0:
                frame_type = frame_seq.MRImageFrameTypeSequence[0]
                if hasattr(frame_type, 'FrameType'):
                    comment = str(frame_type.FrameType).lower()
        
        if comment:
            logger.debug(f"Frame {frame_idx}: {comment}")

            # Helper: detect std/sigma markers in comment
            # Bruker uses both "std dev" and "σ" (sigma) in frame comments
            is_std = ('std' in comment or '\u03c3' in comment or 'sigma' in comment)

            # Map comment to data type
            if 'signal intensity' in comment and not is_std:
                frame_map['signal'] = frame_idx
            elif ('signal' in comment) and is_std:
                frame_map['signal_std'] = frame_idx
            elif 't2 relaxation time' in comment or 't2*' in comment or 't2star' in comment:
                if is_std:
                    frame_map['relaxation_time_std'] = frame_idx
                else:
                    frame_map['relaxation_time'] = frame_idx
            elif 't1 relaxation time' in comment:
                if is_std:
                    frame_map['relaxation_time_std'] = frame_idx
                else:
                    frame_map['relaxation_time'] = frame_idx
            elif 'relaxation' in comment:
                if is_std:
                    frame_map['relaxation_time_std'] = frame_idx
                else:
                    frame_map['relaxation_time'] = frame_idx
            elif 'std dev of the fit' in comment or 'fit_std' in comment or 'fit chi' in comment or 'chi' in comment:
                frame_map['fit_std'] = frame_idx
            elif 'fit valid' in comment:
                frame_map['fit_valid'] = frame_idx
            elif 'parameter' in comment or 'map' in comment:
                frame_map['map'] = frame_idx
    
    logger.info(f"Identified {len(frame_map)} frame types: {list(frame_map.keys())}")
    
    return frame_map


def _extract_dicom_metadata(ds) -> Dict:
    """Extract useful DICOM metadata"""
    metadata = {}
    
    # Basic info
    for attr in ['PatientID', 'StudyDescription', 'SeriesDescription',
                 'SeriesNumber', 'AcquisitionNumber', 'InstanceNumber']:
        if hasattr(ds, attr):
            metadata[attr] = str(getattr(ds, attr))
    
    # Imaging parameters
    for attr in ['EchoTime', 'RepetitionTime', 'FlipAngle', 
                 'MagneticFieldStrength', 'ImagingFrequency']:
        if hasattr(ds, attr):
            try:
                metadata[attr] = float(getattr(ds, attr))
            except (ValueError, TypeError):
                pass
    
    # Geometry
    if hasattr(ds, 'PixelSpacing'):
        try:
            metadata['PixelSpacing'] = [float(x) for x in ds.PixelSpacing]
        except:
            pass
    
    if hasattr(ds, 'SliceThickness'):
        try:
            metadata['SliceThickness'] = float(ds.SliceThickness)
        except:
            pass
    
    # Image dimensions
    metadata['Rows'] = int(ds.Rows)
    metadata['Columns'] = int(ds.Columns)
    
    if hasattr(ds, 'NumberOfFrames'):
        metadata['NumberOfFrames'] = int(ds.NumberOfFrames)
    
    return metadata


def load_bruker_parametric_series(
    dicom_dir: Union[str, Path],
    pattern: str = "*T2star*.dcm",
    extract_uncertainty: bool = True
) -> List[Dict[str, np.ndarray]]:
    """
    Load multiple parametric DICOMs from directory
    
    Parameters
    ----------
    dicom_dir : str or Path
        Directory containing DICOM files
    pattern : str
        Glob pattern to match files (default: "*T2star*.dcm")
    extract_uncertainty : bool
        Extract uncertainty metrics
        
    Returns
    -------
    data_list : list of dict
        List of data dictionaries from each DICOM
        
    Examples
    --------
    >>> # Load all T2* maps in directory
    >>> maps = load_bruker_parametric_series(
    ...     'path/to/dicoms',
    ...     pattern='*T2star*.dcm'
    ... )
    >>> print(f"Loaded {len(maps)} T2* maps")
    """
    dicom_dir = Path(dicom_dir)
    
    if not dicom_dir.exists():
        raise FileNotFoundError(f"Directory not found: {dicom_dir}")
    
    # Find matching files
    dicom_files = sorted(dicom_dir.glob(pattern))
    
    if len(dicom_files) == 0:
        raise ValueError(f"No DICOM files matching '{pattern}' in {dicom_dir}")
    
    logger.info(f"Found {len(dicom_files)} DICOM files matching '{pattern}'")
    
    # Load all
    data_list = []
    for dcm_file in dicom_files:
        try:
            data = load_bruker_parametric_dicom(dcm_file, extract_uncertainty)
            data['filename'] = dcm_file.name
            data_list.append(data)
            logger.info(f"  ✓ {dcm_file.name}")
        except Exception as e:
            logger.error(f"  ✗ Failed to load {dcm_file.name}: {e}")
            continue
    
    logger.info(f"Successfully loaded {len(data_list)}/{len(dicom_files)} files")
    
    return data_list


def main():
    """CLI for testing DICOM parametric map loader"""
    import argparse
    import matplotlib.pyplot as plt
    
    parser = argparse.ArgumentParser(
        description='Load Bruker parametric DICOM maps'
    )
    parser.add_argument(
        'dicom_path',
        help='DICOM file or directory'
    )
    parser.add_argument(
        '--pattern',
        default='*.dcm',
        help='File pattern if loading directory'
    )
    parser.add_argument(
        '--no-uncertainty',
        action='store_true',
        help='Skip uncertainty extraction'
    )
    parser.add_argument('--save', help='Save output directory')
    parser.add_argument('--plot', action='store_true', help='Show plot')
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s: %(message)s'
    )
    
    path = Path(args.dicom_path)
    
    if path.is_file():
        # Single file
        data = load_bruker_parametric_dicom(
            path,
            extract_uncertainty=not args.no_uncertainty
        )
        data_list = [data]
    else:
        # Directory
        data_list = load_bruker_parametric_series(
            path,
            pattern=args.pattern,
            extract_uncertainty=not args.no_uncertainty
        )
    
    print("\n" + "="*70)
    print(f"LOADED {len(data_list)} PARAMETRIC MAP(S)")
    print("="*70)
    
    for i, data in enumerate(data_list):
        print(f"\nMap {i+1}:")
        print(f"  Shape: {data['map'].shape}")
        valid = np.isfinite(data['map'])
        if np.any(valid):
            print(f"  Range: {data['map'][valid].min():.2f} - {data['map'][valid].max():.2f}")
        if 'map_std' in data:
            print(f"  Has uncertainty: ✓")
        print(f"  Available data: {list(data.keys())}")
    
    if args.save:
        output = Path(args.save)
        output.mkdir(parents=True, exist_ok=True)
        
        for i, data in enumerate(data_list):
            prefix = f"map{i+1:02d}" if len(data_list) > 1 else "map"
            
            np.save(output / f'{prefix}_parameter.npy', data['map'])
            
            if 'map_std' in data:
                np.save(output / f'{prefix}_std.npy', data['map_std'])
            
            import json
            with open(output / f'{prefix}_metadata.json', 'w') as f:
                json.dump(data['metadata'], f, indent=2)
        
        print(f"\n✓ Saved to {output}/")
    
    if args.plot and len(data_list) > 0:
        data = data_list[0]
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        im = axes[0].imshow(data['map'], cmap='hot')
        axes[0].set_title('Parameter Map')
        plt.colorbar(im, ax=axes[0])
        
        if 'map_std' in data:
            im = axes[1].imshow(data['map_std'], cmap='viridis')
            axes[1].set_title('Uncertainty (Std Dev)')
            plt.colorbar(im, ax=axes[1])
        else:
            axes[1].axis('off')
            axes[1].text(0.5, 0.5, 'No uncertainty data', 
                        ha='center', va='center')
        
        plt.tight_layout()
        plt.show()
    
    return 0


if __name__ == '__main__':
    exit(main())
