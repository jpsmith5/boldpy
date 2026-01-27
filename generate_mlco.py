#!/usr/bin/env python3
"""
Enhanced MLCO Layer Generation with Multi-Region Support
=========================================================

Generate concentric layers for:
- Single-region organs
- Bilateral organs (left/right split)
- Multi-region organs (cortex, medulla, papilla, etc.)
- Bilateral + multi-region combinations

Multi-region encoding: region_id * 1000 + layer_num
Example: Region 2, Layer 5 → 2005
"""

import numpy as np
from scipy import ndimage
from scipy.spatial import distance
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, Dict, List, Optional


def detect_mask_type(mask: np.ndarray) -> Dict[str, any]:
    """
    Detect if mask is binary, bilateral, or multi-region
    
    Parameters:
    -----------
    mask : ndarray
        Input mask to analyze
        
    Returns:
    --------
    info : dict
        Dictionary with mask type information:
        - 'type': 'binary', 'bilateral', or 'multi-region'
        - 'n_regions': Number of distinct regions
        - 'region_ids': List of region IDs
        - 'is_multi_region': Boolean
    """
    unique_vals = np.unique(mask[mask > 0])
    n_regions = len(unique_vals)
    
    # Check if values are sequential starting from 1
    is_multi_label = n_regions > 1 and np.array_equal(unique_vals, np.arange(1, n_regions + 1))
    
    if n_regions == 0:
        raise ValueError("Mask is empty (no regions found)")
    
    elif n_regions == 1:
        # Single region - could be unilateral or bilateral
        # Check if it has 2 disconnected components
        labeled, n_components = ndimage.label(mask > 0)
        
        if n_components >= 2:
            mask_type = 'bilateral'
        else:
            mask_type = 'binary'
            
        info = {
            'type': mask_type,
            'n_regions': 1,
            'n_components': n_components,
            'region_ids': [1],
            'is_multi_region': False,
            'is_bilateral': n_components >= 2
        }
    
    elif is_multi_label:
        # Multi-region mask with sequential integer labels
        mask_type = 'multi-region'
        
        info = {
            'type': mask_type,
            'n_regions': n_regions,
            'region_ids': list(unique_vals),
            'is_multi_region': True,
            'is_bilateral': False  # Will check per-region if needed
        }
        
    else:
        raise ValueError(f"Unexpected mask format. Unique values: {unique_vals}. "
                        f"Expected: binary (0,1) or multi-label (0,1,2,3...)")
    
    return info


def split_mask(mask: np.ndarray,
               component_names: Tuple[str, str] = ('left', 'right'),
               apply_mri_flip: bool = True,
               min_size: int = 100) -> Dict[str, np.ndarray]:
    """
    Split bilateral mask into two components
    
    For MRI data (default), applies left-right flip convention where
    anatomical left appears on image right side, and vice versa.
    
    Parameters:
    -----------
    mask : ndarray
        Bilateral mask with two separate regions
    component_names : tuple
        Names for the two components (default: ('left', 'right'))
    apply_mri_flip : bool
        Apply MRI left-right flip convention (default: True)
    min_size : int
        Minimum size for valid component in pixels
        
    Returns:
    --------
    components : dict
        Dictionary mapping component names to masks
    """
    # Label connected components
    labeled, n_components = ndimage.label(mask)
    
    if n_components == 0:
        raise ValueError("No components found in mask")
    
    # Get properties of each component
    components_list = []
    for i in range(1, n_components + 1):
        comp_mask = labeled == i
        size = np.sum(comp_mask)
        
        if size >= min_size:
            y_coords, x_coords = np.where(comp_mask)
            center_x = np.mean(x_coords)
            center_y = np.mean(y_coords)
            
            components_list.append({
                'label': i,
                'mask': comp_mask,
                'size': size,
                'center_x': center_x,
                'center_y': center_y
            })
    
    if len(components_list) < 2:
        raise ValueError(f"Found {len(components_list)} components (need 2). "
                        f"Try reducing --min-component-size")
    
    if len(components_list) > 2:
        print(f"  ⚠️  Warning: Found {len(components_list)} components, using 2 largest")
    
    # Sort by size and take two largest
    components_list = sorted(components_list, key=lambda x: x['size'], reverse=True)[:2]
    
    # Sort by x-coordinate (left to right in image)
    components_list = sorted(components_list, key=lambda x: x['center_x'])
    
    image_left_mask = components_list[0]['mask']
    image_right_mask = components_list[1]['mask']
    
    if apply_mri_flip:
        result = {
            component_names[0]: image_right_mask,
            component_names[1]: image_left_mask
        }
        print(f"  {component_names[0]}: {np.sum(result[component_names[0]]):,} pixels (anatomical {component_names[0]} = image right)")
        print(f"  {component_names[1]}: {np.sum(result[component_names[1]]):,} pixels (anatomical {component_names[1]} = image left)")
    else:
        result = {
            component_names[0]: image_left_mask,
            component_names[1]: image_right_mask
        }
        print(f"  {component_names[0]}: {np.sum(result[component_names[0]]):,} pixels (image left)")
        print(f"  {component_names[1]}: {np.sum(result[component_names[1]]):,} pixels (image right)")
    
    return result


def generate_mlco_layers(kidney_mask: np.ndarray,
                        n_layers: int = 12,
                        method: str = 'distance') -> np.ndarray:
    """
    Generate concentric layers from outer (cortex) to inner (medulla)
    
    Parameters:
    -----------
    kidney_mask : ndarray
        Boolean kidney ROI mask
    n_layers : int
        Number of layers to generate (default: 12)
    method : str
        'distance' = distance transform based (recommended)
        'erosion' = successive erosion based
        
    Returns:
    --------
    layer_mask : ndarray
        Integer mask where:
        0 = background
        1 = outermost layer (cortex)
        n_layers = innermost layer (medulla)
    """
    layer_mask = np.zeros_like(kidney_mask, dtype=np.int32)
    
    if method == 'distance':
        distance_from_edge = ndimage.distance_transform_edt(kidney_mask)
        
        max_dist = distance_from_edge[kidney_mask].max()
        if max_dist > 0:
            normalized_dist = distance_from_edge / max_dist
        else:
            normalized_dist = distance_from_edge
        
        for layer_idx in range(1, n_layers + 1):
            depth_min = (layer_idx - 1) / n_layers
            depth_max = layer_idx / n_layers
            
            in_range = (normalized_dist >= depth_min) & (normalized_dist < depth_max) & kidney_mask
            layer_mask[in_range] = layer_idx
        
        # Handle remaining pixels
        remaining = kidney_mask & (layer_mask == 0)
        if np.any(remaining):
            layer_mask[remaining] = n_layers
    
    elif method == 'erosion':
        remaining_mask = kidney_mask.copy()
        
        for layer_idx in range(1, n_layers + 1):
            if not np.any(remaining_mask):
                break
            
            boundary = remaining_mask & ~ndimage.binary_erosion(remaining_mask)
            layer_mask[boundary] = layer_idx
            remaining_mask = remaining_mask & ~boundary
        
        if np.any(remaining_mask):
            layer_mask[remaining_mask] = n_layers
    
    return layer_mask


def generate_bilateral_mlco_layers(bilateral_mask: np.ndarray,
                                   left_mask: np.ndarray,
                                   right_mask: np.ndarray,
                                   n_layers: int = 12) -> np.ndarray:
    """
    Generate MLCO layers for bilateral organ mask
    
    Layer numbering:
    - Layers 1-n_layers: Right component
    - Layers (n_layers+1)-(2*n_layers): Left component
    """
    print(f"\nGenerating bilateral MLCO layers...")
    print(f"  Layers per component: {n_layers}")
    
    mlco_mask = np.zeros_like(bilateral_mask, dtype=np.int32)
    
    # Generate layers for right component (1 to n_layers)
    print(f"\n  Right component:")
    right_layers = generate_mlco_layers(right_mask, n_layers)
    mlco_mask[right_mask] = right_layers[right_mask]
    
    # Generate layers for left component (n_layers+1 to 2*n_layers)
    print(f"\n  Left component:")
    left_layers = generate_mlco_layers(left_mask, n_layers)
    mlco_mask[left_mask] = left_layers[left_mask] + n_layers
    
    return mlco_mask


def generate_multiregion_mlco_layers(mask: np.ndarray,
                                      region_ids: List[int],
                                      layers_per_region: List[int],
                                      method: str = 'distance') -> np.ndarray:
    """
    Generate MLCO layers for multi-region mask
    
    Encoding: region_id * 1000 + layer_num
    Example:
      Region 1, Layer 1-8: 1001-1008
      Region 2, Layer 1-8: 2001-2008
      Region 3, Layer 1-6: 3001-3006
    
    Parameters:
    -----------
    mask : ndarray
        Multi-label mask where each pixel value = region ID
    region_ids : list of int
        List of region IDs to process
    layers_per_region : list of int
        Number of layers for each region
    method : str
        Layer generation method
        
    Returns:
    --------
    mlco_mask : ndarray
        Multi-region MLCO mask with encoded layer values
    """
    print(f"\n{'='*70}")
    print("MULTI-REGION MLCO GENERATION")
    print(f"{'='*70}")
    print(f"\nRegions to process: {len(region_ids)}")
    
    if len(layers_per_region) != len(region_ids):
        raise ValueError(f"layers_per_region length ({len(layers_per_region)}) "
                        f"must match number of regions ({len(region_ids)})")
    
    mlco_mask = np.zeros_like(mask, dtype=np.int32)
    
    for region_id, n_layers in zip(region_ids, layers_per_region):
        print(f"\n{'─'*70}")
        print(f"Region {region_id}: Generating {n_layers} layers")
        print(f"{'─'*70}")
        
        # Extract region mask
        region_mask = mask == region_id
        n_pixels = np.sum(region_mask)
        
        if n_pixels == 0:
            print(f"  ⚠️  Warning: Region {region_id} is empty, skipping")
            continue
        
        print(f"  Pixels in region: {n_pixels:,}")
        
        # Generate layers for this region
        region_layers = generate_mlco_layers(region_mask, n_layers, method)
        
        # Encode as: region_id * 1000 + layer_num
        encoded_layers = np.zeros_like(region_layers)
        for layer_num in range(1, n_layers + 1):
            layer_pixels = region_layers == layer_num
            encoded_value = region_id * 1000 + layer_num
            encoded_layers[layer_pixels] = encoded_value
        
        # Add to output mask
        mlco_mask[region_mask] = encoded_layers[region_mask]
        
        # Print layer distribution
        print(f"\n  Layer distribution:")
        for layer_num in range(1, n_layers + 1):
            layer_pixels = region_layers == layer_num
            n_layer_pixels = np.sum(layer_pixels)
            pct = 100 * n_layer_pixels / n_pixels if n_pixels > 0 else 0
            encoded_value = region_id * 1000 + layer_num
            print(f"    Layer {layer_num:2d} ({encoded_value:4d}): {n_layer_pixels:5d} pixels ({pct:5.1f}%)")
    
    print(f"\n{'='*70}")
    print("✓ MULTI-REGION MLCO COMPLETE")
    print(f"{'='*70}")
    
    return mlco_mask


def decode_multiregion_mlco(encoded_value: int) -> Tuple[int, int]:
    """
    Decode multi-region MLCO value
    
    Parameters:
    -----------
    encoded_value : int
        Encoded value (region_id * 1000 + layer_num)
        
    Returns:
    --------
    region_id : int
    layer_num : int
    """
    region_id = encoded_value // 1000
    layer_num = encoded_value % 1000
    return region_id, layer_num


def visualize_multiregion_mlco(anatomical: np.ndarray,
                                mlco_mask: np.ndarray,
                                region_ids: List[int],
                                layers_per_region: List[int],
                                output_file: Path,
                                title: str = "Multi-Region MLCO Layers"):
    """Visualize multi-region MLCO layers"""
    
    n_regions = len(region_ids)
    
    fig, axes = plt.subplots(1, n_regions + 1, figsize=(5*(n_regions+1), 5))
    
    if n_regions == 1:
        axes = [axes]
    
    # Overall view
    axes[0].imshow(anatomical, cmap='gray')
    
    # Create composite overlay
    composite = np.zeros_like(mlco_mask, dtype=float)
    for region_id in region_ids:
        region_pixels = (mlco_mask >= region_id * 1000) & (mlco_mask < (region_id + 1) * 1000)
        if np.any(region_pixels):
            # Decode layers
            for encoded_val in np.unique(mlco_mask[region_pixels]):
                if encoded_val > 0:
                    _, layer_num = decode_multiregion_mlco(encoded_val)
                    composite[mlco_mask == encoded_val] = layer_num
    
    axes[0].imshow(np.ma.masked_where(composite == 0, composite),
                   cmap='jet', alpha=0.5, vmin=1, vmax=max(layers_per_region))
    axes[0].set_title('All Regions')
    axes[0].axis('off')
    
    # Individual regions
    for idx, (region_id, n_layers) in enumerate(zip(region_ids, layers_per_region)):
        ax = axes[idx + 1]
        ax.imshow(anatomical, cmap='gray')
        
        # Extract this region's layers
        region_pixels = (mlco_mask >= region_id * 1000) & (mlco_mask < (region_id + 1) * 1000)
        
        if np.any(region_pixels):
            # Decode to layer numbers
            region_layers = np.zeros_like(mlco_mask)
            for encoded_val in np.unique(mlco_mask[region_pixels]):
                if encoded_val > 0:
                    _, layer_num = decode_multiregion_mlco(encoded_val)
                    region_layers[mlco_mask == encoded_val] = layer_num
            
            ax.imshow(np.ma.masked_where(region_layers == 0, region_layers),
                     cmap='jet', alpha=0.5, vmin=1, vmax=n_layers)
        
        ax.set_title(f'Region {region_id}')
        ax.axis('off')
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved multi-region visualization: {output_file}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Enhanced MLCO layer generation with multi-region support',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:

  # Single-region, unilateral
  python generate_mlco.py --mask kidney.npy --anatomical ref.npy \\
      -o output/ -l sample --n-layers 12
  
  # Bilateral (auto-split)
  python generate_mlco.py --mask bilateral.npy --anatomical ref.npy \\
      -o output/ -l sample --split --n-layers 12
  
  # Multi-region (from multi-label mask)
  python generate_mlco.py --mask regions.npy --anatomical ref.npy \\
      -o output/ -l sample --multi-region --layers-per-region 8 8 6
        """
    )
    
    # Required arguments
    parser.add_argument('--mask', type=Path, required=True,
                       help='Organ mask (.npy file) - can be binary, bilateral, or multi-label')
    parser.add_argument('--anatomical', type=Path, required=True,
                       help='Anatomical reference image (.npy)')
    parser.add_argument('--output-dir', '-o', type=Path, required=True,
                       help='Output directory')
    parser.add_argument('--label', '-l', required=True,
                       help='Label for output files')
    
    # Splitting options (for bilateral)
    parser.add_argument('--split', action='store_true',
                       help='Split binary mask into bilateral components')
    parser.add_argument('--component-names', nargs=2, default=['left', 'right'],
                       help='Names for bilateral components')
    parser.add_argument('--no-mri-flip', action='store_true',
                       help='Disable MRI left-right flip')
    parser.add_argument('--min-component-size', type=int, default=100,
                       help='Minimum component size in pixels')
    
    # Multi-region options
    parser.add_argument('--multi-region', action='store_true',
                       help='Process as multi-region mask (auto-detected from multi-label masks)')
    parser.add_argument('--layers-per-region', nargs='+', type=int,
                       help='Number of layers for each region (required for --multi-region)')
    
    # Layer generation options
    parser.add_argument('--n-layers', type=int, default=12,
                       help='Number of layers (single/bilateral mode)')
    parser.add_argument('--method', choices=['distance', 'erosion'], default='distance',
                       help='Layer generation method')
    
    args = parser.parse_args()
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("ENHANCED MLCO LAYER GENERATION")
    print("="*70)
    
    # Load data
    print("\nLoading data...")
    mask = np.load(args.mask)
    anatomical = np.load(args.anatomical)
    
    print(f"  Mask shape: {mask.shape}")
    print(f"  Anatomical shape: {anatomical.shape}")
    
    # Detect mask type
    print("\nDetecting mask type...")
    mask_info = detect_mask_type(mask)
    print(f"  Type: {mask_info['type']}")
    print(f"  Regions: {mask_info['n_regions']}")
    print(f"  Region IDs: {mask_info['region_ids']}")
    
    # Process based on type and arguments
    if args.multi_region or mask_info['is_multi_region']:
        # MULTI-REGION MODE
        if not args.layers_per_region:
            raise ValueError("--layers-per-region required for multi-region masks")
        
        if len(args.layers_per_region) != mask_info['n_regions']:
            raise ValueError(f"Expected {mask_info['n_regions']} values for --layers-per-region, "
                           f"got {len(args.layers_per_region)}")
        
        mlco_mask = generate_multiregion_mlco_layers(
            mask,
            mask_info['region_ids'],
            args.layers_per_region,
            args.method
        )
        
        # Save
        output_file = args.output_dir / f'{args.label}_mlco_multiregion.npy'
        np.save(output_file, mlco_mask)
        print(f"\n✓ Saved multi-region MLCO mask: {output_file}")
        
        # Visualize
        viz_file = args.output_dir / f'{args.label}_mlco_multiregion_viz.png'
        visualize_multiregion_mlco(
            anatomical, mlco_mask,
            mask_info['region_ids'],
            args.layers_per_region,
            viz_file,
            f'{args.label} - Multi-Region MLCO'
        )
        
        print(f"\n✓ Multi-region MLCO generation complete!")
        print(f"  Total layers: {sum(args.layers_per_region)}")
        for region_id, n_layers in zip(mask_info['region_ids'], args.layers_per_region):
            start = region_id * 1000 + 1
            end = region_id * 1000 + n_layers
            print(f"  Region {region_id}: {start}-{end} ({n_layers} layers)")
    
    elif args.split or mask_info['is_bilateral']:
        # BILATERAL MODE
        print(f"\nSplitting bilateral mask...")
        components = split_mask(
            mask,
            component_names=tuple(args.component_names),
            apply_mri_flip=not args.no_mri_flip,
            min_size=args.min_component_size
        )
        
        mlco_mask = generate_bilateral_mlco_layers(
            mask,
            components[args.component_names[0]],
            components[args.component_names[1]],
            args.n_layers
        )
        
        # Save
        output_file = args.output_dir / f'{args.label}_mlco_bilateral.npy'
        np.save(output_file, mlco_mask)
        print(f"\n✓ Saved bilateral MLCO mask: {output_file}")
        
        # Visualize
        viz_file = args.output_dir / f'{args.label}_mlco_bilateral_viz.png'
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.imshow(anatomical, cmap='gray')
        ax.imshow(np.ma.masked_where(mlco_mask == 0, mlco_mask),
                 cmap='jet', alpha=0.5, vmin=1, vmax=2*args.n_layers)
        ax.set_title(f'{args.label} - Bilateral MLCO')
        ax.axis('off')
        plt.colorbar(ax.images[1], ax=ax, label='Layer')
        plt.savefig(viz_file, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved visualization: {viz_file}")
        
        print(f"\n✓ Bilateral MLCO generation complete!")
        print(f"  {args.component_names[0]}: Layers 1-{args.n_layers}")
        print(f"  {args.component_names[1]}: Layers {args.n_layers+1}-{2*args.n_layers}")
    
    else:
        # SINGLE-REGION MODE
        print(f"\nGenerating single-region MLCO...")
        mlco_mask = generate_mlco_layers(mask, args.n_layers, args.method)
        
        # Save
        output_file = args.output_dir / f'{args.label}_mlco_single.npy'
        np.save(output_file, mlco_mask)
        print(f"\n✓ Saved single-region MLCO mask: {output_file}")
        
        # Visualize
        viz_file = args.output_dir / f'{args.label}_mlco_single_viz.png'
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.imshow(anatomical, cmap='gray')
        ax.imshow(np.ma.masked_where(mlco_mask == 0, mlco_mask),
                 cmap='jet', alpha=0.5, vmin=1, vmax=args.n_layers)
        ax.set_title(f'{args.label} - Single-Region MLCO')
        ax.axis('off')
        plt.colorbar(ax.images[1], ax=ax, label='Layer')
        plt.savefig(viz_file, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved visualization: {viz_file}")
        
        print(f"\n✓ Single-region MLCO generation complete!")
        print(f"  Layers: 1-{args.n_layers}")


if __name__ == '__main__':
    main()
