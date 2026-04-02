#!/usr/bin/env python3
"""
Perfusion Data Loader
=====================

Load and process perfusion maps from Bruker ASL data.
"""

import numpy as np
import zipfile
from pathlib import Path
from typing import Optional


def load_bruker_perfusion(pvdatasets_path: Path) -> Optional[np.ndarray]:
    """
    Load perfusion map from Bruker pdata/2, frame 5
    
    Bruker ASL processing stores perfusion in pdata/2:
    - Frame 1: M0 (reference)
    - Frame 5: CBF/Perfusion map (relative %)
    
    Parameters:
    -----------
    pvdatasets_path : Path
        Path to perfusion .PvDatasets file
        
    Returns:
    --------
    perfusion_map : ndarray or None
        Perfusion map in relative % units
    """
    print(f"\nLoading perfusion from: {pvdatasets_path.name}")
    
    try:
        with zipfile.ZipFile(pvdatasets_path, 'r') as zf:
            # Find pdata/2/2dseq
            seq_files = [f for f in zf.namelist() if f.endswith('/pdata/2/2dseq')]
            
            if not seq_files:
                print("  ⚠️  No pdata/2 found (not a processed perfusion scan)")
                return None
            
            # Read 2dseq
            with zf.open(seq_files[0]) as f:
                data_raw = f.read()
            
            # Get dimensions from visu_pars
            visu_files = [f for f in zf.namelist() if f.endswith('/pdata/2/visu_pars')]
            with zf.open(visu_files[0]) as f:
                visu_content = f.read().decode('utf-8', errors='ignore')
            
            frame_count = None
            core_size = []
            
            for line in visu_content.split('\n'):
                if '##$VisuCoreFrameCount=' in line:
                    frame_count = int(line.split('=')[1])
                elif '##$VisuCoreSize=' in line and not core_size:
                    idx = visu_content.index(line)
                    size_line = visu_content[idx:].split('\n')[1]
                    core_size = [int(x) for x in size_line.split()]
        
        if frame_count < 5:
            print(f"  ⚠️  Only {frame_count} frames found (need frame 5 for perfusion)")
            return None
        
        # Parse data (32-bit float)
        data = np.frombuffer(data_raw, dtype='<f4')
        height, width = core_size
        data = data.reshape((frame_count, height, width))
        
        # Frame 5 (index 4) is perfusion
        perfusion_map = data[4]
        
        # Filter out unrealistic values
        # Perfusion should be positive and reasonable (0-200% relative)
        valid = (perfusion_map > 0) & (perfusion_map < 200)
        
        print(f"  ✓ Loaded perfusion map: {perfusion_map.shape}")
        print(f"  Valid pixels: {np.sum(valid):,} / {perfusion_map.size:,}")
        
        if np.any(valid):
            print(f"  Perfusion range: {perfusion_map[valid].min():.2f} - {perfusion_map[valid].max():.2f}")
            print(f"  Mean perfusion: {perfusion_map[valid].mean():.2f}")
        
        return perfusion_map
        
    except Exception as e:
        print(f"  ⚠️  Error loading perfusion: {e}")
        return None


def resample_to_bold_resolution(perfusion_map: np.ndarray,
                                target_shape: tuple) -> np.ndarray:
    """
    Resample perfusion map to match BOLD resolution
    
    Parameters:
    -----------
    perfusion_map : ndarray
        Perfusion map (may be different resolution)
    target_shape : tuple
        Target shape (BOLD map shape)
        
    Returns:
    --------
    resampled : ndarray
        Perfusion map at BOLD resolution
    """
    from scipy.ndimage import zoom
    
    if perfusion_map.shape == target_shape:
        print(f"  Perfusion already at BOLD resolution")
        return perfusion_map
    
    zoom_factors = (target_shape[0] / perfusion_map.shape[0],
                   target_shape[1] / perfusion_map.shape[1])
    
    print(f"  Resampling perfusion: {perfusion_map.shape} → {target_shape}")
    print(f"  Zoom factors: {zoom_factors}")
    
    # Use bilinear interpolation (order=1)
    resampled = zoom(perfusion_map, zoom_factors, order=1)
    
    return resampled


def analyze_perfusion_tlco_layers(perfusion_map: np.ndarray,
                                  tlco_mask: np.ndarray,
                                  kidney_start_layer: int,
                                  n_layers: int,
                                  kidney_label: str = "kidney") -> dict:
    """
    Analyze perfusion across TLCO layers
    
    Parameters:
    -----------
    perfusion_map : ndarray
        Perfusion map (relative %)
    tlco_mask : ndarray
        TLCO layer mask
    kidney_start_layer : int
        First layer number for this kidney
    n_layers : int
        Number of layers
    kidney_label : str
        Kidney identifier
        
    Returns:
    --------
    results : dict
        Layer-by-layer perfusion statistics
    """
    print(f"\n{'─'*70}")
    print(f"{kidney_label.upper()} - PERFUSION")
    print(f"{'─'*70}")
    
    layers = []
    
    for layer_offset in range(n_layers):
        layer_num = kidney_start_layer + layer_offset
        layer_idx = layer_offset + 1
        
        # Get pixels in this layer
        layer_pixels = (tlco_mask == layer_num) & (perfusion_map > 0) & ~np.isnan(perfusion_map)
        
        n_pixels = np.sum(layer_pixels)
        
        if n_pixels == 0:
            print(f"  Layer {layer_idx:2d}: No valid perfusion data")
            continue
        
        # Extract values
        perf_values = perfusion_map[layer_pixels]
        
        # Calculate statistics
        layer_stats = {
            'layer': layer_idx,
            'layer_number': layer_num,
            'n_pixels': int(n_pixels),
            'perfusion': {
                'mean': float(np.mean(perf_values)),
                'median': float(np.median(perf_values)),
                'std': float(np.std(perf_values)),
                'min': float(np.min(perf_values)),
                'max': float(np.max(perf_values))
            }
        }
        
        layers.append(layer_stats)
        
        depth_pct = (layer_idx - 1) / (n_layers - 1) * 100 if n_layers > 1 else 0
        print(f"  Layer {layer_idx:2d} ({depth_pct:5.1f}%): "
              f"{n_pixels:5d} pix | "
              f"Perf: {layer_stats['perfusion']['median']:6.2f}±{layer_stats['perfusion']['std']:5.2f}")
    
    # Calculate gradient
    if len(layers) >= 2:
        cortex_perf = layers[0]['perfusion']['median']
        medulla_perf = layers[-1]['perfusion']['median']
        gradient = medulla_perf - cortex_perf
        
        print(f"\n  PERFUSION GRADIENT:")
        print(f"    Cortex (Layer 1):   {cortex_perf:.2f}")
        print(f"    Medulla (Layer {n_layers}): {medulla_perf:.2f}")
        print(f"    Gradient:            {gradient:+.2f}")
        
        # Normally cortex has higher perfusion than medulla
        if gradient > 0:
            print(f"    ⚠️  ABNORMAL: Medulla > Cortex (unusual)")
        else:
            print(f"    ✓ Normal: Cortex > Medulla")
        
        gradient_info = {
            'cortex': cortex_perf,
            'medulla': medulla_perf,
            'gradient': gradient,
            'abnormal': gradient > 0
        }
    else:
        gradient_info = None
    
    return {
        'kidney': kidney_label,
        'n_layers': n_layers,
        'layers': layers,
        'gradient': gradient_info
    }
