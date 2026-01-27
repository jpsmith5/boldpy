#!/usr/bin/env python3
"""
ROI Format Utilities
===================

Convert between different ROI mask formats:
- .npy (NumPy binary format) - Fast loading in Python
- .json (JSON format) - Compatible with ImageJ/QuPath

Both formats store the same mask data plus metadata.
"""

import numpy as np
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional


def save_roi_npy(mask: np.ndarray, 
                 output_path: Path,
                 metadata: Optional[Dict] = None) -> None:
    """
    Save ROI mask in NumPy .npy format
    
    Parameters:
    -----------
    mask : ndarray
        Integer mask with layer labels
    output_path : Path
        Output file path (.npy)
    metadata : dict, optional
        Additional metadata to store
    """
    output_path = Path(output_path)
    
    # Ensure .npy extension
    if output_path.suffix != '.npy':
        output_path = output_path.with_suffix('.npy')
    
    # Save mask
    np.save(output_path, mask)
    
    # Save metadata if provided
    if metadata is not None:
        metadata_path = output_path.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    print(f"✓ Saved ROI mask: {output_path}")
    if metadata:
        print(f"✓ Saved metadata: {metadata_path}")


def load_roi_npy(input_path: Path) -> tuple:
    """
    Load ROI mask from NumPy .npy format
    
    Parameters:
    -----------
    input_path : Path
        Input file path (.npy)
    
    Returns:
    --------
    mask : ndarray
        Integer mask with layer labels
    metadata : dict or None
        Metadata if available
    """
    input_path = Path(input_path)
    
    # Load mask
    mask = np.load(input_path)
    
    # Try to load metadata
    metadata_path = input_path.with_suffix('.json')
    metadata = None
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    
    return mask, metadata


def roi_npy_to_json(input_path: Path, 
                    output_path: Optional[Path] = None,
                    include_coordinates: bool = False) -> None:
    """
    Convert ROI from .npy to .json format
    
    Parameters:
    -----------
    input_path : Path
        Input .npy file
    output_path : Path, optional
        Output .json file (default: same name, .json extension)
    include_coordinates : bool
        If True, include pixel coordinates for each layer (large files!)
    """
    input_path = Path(input_path)
    
    if output_path is None:
        output_path = input_path.with_suffix('.json')
    else:
        output_path = Path(output_path)
    
    # Load mask and existing metadata
    mask, existing_metadata = load_roi_npy(input_path)
    
    # Create JSON structure
    json_data = {
        'format': 'MLCO_ROI',
        'version': '2.1.0',
        'created': datetime.now().isoformat(),
        'source_file': str(input_path),
        'shape': list(mask.shape),
        'dtype': str(mask.dtype),
        'n_layers': int(mask.max()),
        'roi_type': 'mlco_mask'
    }
    
    # Add existing metadata if available
    if existing_metadata:
        json_data['metadata'] = existing_metadata
    
    # Add layer statistics
    unique_layers = np.unique(mask)
    unique_layers = unique_layers[unique_layers > 0]  # Exclude background
    
    json_data['layers'] = []
    for layer_id in unique_layers:
        layer_mask = (mask == layer_id)
        layer_info = {
            'layer_id': int(layer_id),
            'n_pixels': int(layer_mask.sum()),
            'bbox': {
                'min_row': int(np.where(layer_mask)[0].min()),
                'max_row': int(np.where(layer_mask)[0].max()),
                'min_col': int(np.where(layer_mask)[1].min()),
                'max_col': int(np.where(layer_mask)[1].max())
            }
        }
        
        # Optionally include pixel coordinates (can make files very large)
        if include_coordinates:
            rows, cols = np.where(layer_mask)
            layer_info['coordinates'] = {
                'rows': rows.tolist(),
                'cols': cols.tolist()
            }
        
        json_data['layers'].append(layer_info)
    
    # Save JSON
    with open(output_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    
    print(f"✓ Converted {input_path} → {output_path}")
    if include_coordinates:
        print(f"  (includes pixel coordinates - large file)")


def roi_json_to_npy(input_path: Path,
                    output_path: Optional[Path] = None) -> None:
    """
    Convert ROI from .json to .npy format
    
    Parameters:
    -----------
    input_path : Path
        Input .json file
    output_path : Path, optional
        Output .npy file (default: same name, .npy extension)
    """
    input_path = Path(input_path)
    
    if output_path is None:
        output_path = input_path.with_suffix('.npy')
    else:
        output_path = Path(output_path)
    
    # Load JSON
    with open(input_path, 'r') as f:
        json_data = json.load(f)
    
    # Check format
    if json_data.get('format') != 'MLCO_ROI':
        print(f"⚠️  Warning: Not a standard MLCO ROI format")
    
    # Create mask array
    shape = tuple(json_data['shape'])
    mask = np.zeros(shape, dtype=np.int32)
    
    # Fill in layers
    if 'coordinates' in json_data['layers'][0]:
        # Reconstruct from coordinates (if available)
        for layer in json_data['layers']:
            layer_id = layer['layer_id']
            rows = np.array(layer['coordinates']['rows'])
            cols = np.array(layer['coordinates']['cols'])
            mask[rows, cols] = layer_id
    else:
        print(f"⚠️  JSON does not contain pixel coordinates")
        print(f"Cannot reconstruct mask from summary statistics only")
        return
    
    # Save mask and metadata
    metadata = json_data.get('metadata', {})
    save_roi_npy(mask, output_path, metadata)
    
    print(f"✓ Converted {input_path} → {output_path}")


def convert_batch(input_dir: Path, 
                  from_format: str = 'npy',
                  to_format: str = 'json',
                  pattern: str = '*_mlco_mask_*.npy') -> None:
    """
    Batch convert ROI files
    
    Parameters:
    -----------
    input_dir : Path
        Input directory
    from_format : str
        Source format ('npy' or 'json')
    to_format : str
        Target format ('npy' or 'json')
    pattern : str
        File pattern to match
    """
    input_dir = Path(input_dir)
    
    if from_format == 'npy' and to_format == 'json':
        converter = roi_npy_to_json
    elif from_format == 'json' and to_format == 'npy':
        converter = roi_json_to_npy
    else:
        raise ValueError(f"Unsupported conversion: {from_format} → {to_format}")
    
    # Find files
    files = list(input_dir.glob(pattern))
    
    if not files:
        print(f"No files found matching pattern: {pattern}")
        return
    
    print(f"Found {len(files)} files to convert")
    
    # Convert each file
    for file_path in files:
        try:
            converter(file_path)
        except Exception as e:
            print(f"✗ Failed to convert {file_path}: {e}")


def main():
    """Command-line interface for ROI format conversion"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Convert between ROI mask formats (.npy ↔ .json)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert single file (npy → json)
  python roi_format_utils.py --input M1_mlco_mask_24layers.npy --to-json
  
  # Convert single file (json → npy)
  python roi_format_utils.py --input M1_mlco_mask_24layers.json --to-npy
  
  # Batch convert all .npy files in directory
  python roi_format_utils.py --batch-dir masks/ --from npy --to json
        """
    )
    
    # Single file conversion
    parser.add_argument('--input', type=Path, help='Input file')
    parser.add_argument('--output', type=Path, help='Output file (optional)')
    parser.add_argument('--to-json', action='store_true', help='Convert to JSON')
    parser.add_argument('--to-npy', action='store_true', help='Convert to NPY')
    parser.add_argument('--include-coords', action='store_true', 
                       help='Include pixel coordinates in JSON (large files!)')
    
    # Batch conversion
    parser.add_argument('--batch-dir', type=Path, help='Batch convert directory')
    parser.add_argument('--from', dest='from_format', choices=['npy', 'json'], 
                       default='npy', help='Source format for batch')
    parser.add_argument('--to', dest='to_format', choices=['npy', 'json'],
                       default='json', help='Target format for batch')
    parser.add_argument('--pattern', default='*_mlco_mask_*.npy',
                       help='File pattern for batch conversion')
    
    args = parser.parse_args()
    
    # Batch conversion
    if args.batch_dir:
        convert_batch(args.batch_dir, args.from_format, args.to_format, args.pattern)
        return
    
    # Single file conversion
    if not args.input:
        parser.print_help()
        return
    
    if args.to_json:
        roi_npy_to_json(args.input, args.output, args.include_coords)
    elif args.to_npy:
        roi_json_to_npy(args.input, args.output)
    else:
        print("Specify --to-json or --to-npy")
        parser.print_help()


if __name__ == '__main__':
    main()
