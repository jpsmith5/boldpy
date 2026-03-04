#!/usr/bin/env python3
"""
Enhanced MLCO Analysis - Multi-Region Support
==============================================

Analyzes T2*, R2*, and perfusion across MLCO layers with:
- Single-region analysis (24-layer bilateral)
- Multi-region analysis (independent regions with per-region zones)
- Tissue viability classification
- Regional statistics

Supports both traditional bilateral MLCO and new multi-region MLCO formats.
"""

import numpy as np
from typing import Dict, Optional, List, Tuple
from pathlib import Path
import sys

# Import tissue zone module (access definitions dynamically)
try:
    import tissue_zones
    from tissue_zones import (
        classify_tissue_viability, calculate_tissue_quality,
        get_zone_name, get_zone_layers, interpret_tissue_state,
        calculate_effect_size, interpret_effect_size,
        # Multi-region support
        is_multiregion_config, decode_multiregion_value,
        get_all_region_ids, get_region_name, get_region_zones,
        extract_region_from_mlco
    )
except ImportError:
    print("WARNING: tissue_zones module not found. Zone analysis will be limited.")
    tissue_zones = None


# ============================================================================
# MULTI-REGION DETECTION
# ============================================================================

def detect_multiregion_mlco(mlco_mask: np.ndarray) -> Tuple[bool, Optional[List[int]]]:
    """
    Detect if MLCO mask is multi-region format
    
    Multi-region format uses encoding: region_id * 1000 + layer_num
    Examples: 1001-1008 (region 1), 2001-2008 (region 2)
    
    Parameters:
    -----------
    mlco_mask : ndarray
        MLCO layer mask
        
    Returns:
    --------
    is_multiregion : bool
        True if multi-region format detected
    region_ids : list of int or None
        List of detected region IDs if multi-region, None otherwise
    """
    unique_values = np.unique(mlco_mask[mlco_mask > 0])
    
    if len(unique_values) == 0:
        return False, None
    
    # Check if any values >= 1000 (multi-region encoding threshold)
    if np.any(unique_values >= 1000):
        # Extract region IDs
        region_ids = sorted(list(set([v // 1000 for v in unique_values if v >= 1000])))
        return True, region_ids
    else:
        # Standard single/bilateral format (values < 1000)
        return False, None


def _analyze_mlco_single_organ(
    t2_map: np.ndarray,
    r2_map: np.ndarray,
    mlco_mask: np.ndarray,
    organ_start_layer: int,
    n_layers: int = 24,
    tissue_mask: Optional[np.ndarray] = None,
    perfusion_map: Optional[np.ndarray] = None,
    organ_label: str = "kidney",
    zone_config: Optional[Dict] = None
) -> Dict:
    """
    Enhanced MLCO analysis with 24 layers and tissue quality assessment
    
    Parameters:
    -----------
    t2_map : ndarray
        T2* map (ms)
    r2_map : ndarray
        R2* map (Hz)
    mlco_mask : ndarray
        MLCO layer mask
    organ_start_layer : int
        First layer number for this kidney
    n_layers : int
        Number of layers (default 24)
    tissue_mask : ndarray, optional
        Tissue validity mask
    perfusion_map : ndarray, optional
        Perfusion map (ml/100g/min)
    organ_label : str
        Kidney identifier
        
    Returns:
    --------
    results : dict
        Enhanced layer-by-layer statistics with tissue quality
    """
    print(f"\n{'─'*70}")
    print(f"{organ_label.upper()} - Layers {organ_start_layer} to {organ_start_layer+n_layers-1}")
    print(f"{'─'*70}")
    
    layers = []
    
    # Determine if this is cortex or medulla for each layer
    def get_region_type(layer_idx):
        """Determine if layer is in cortex or medulla"""
        if layer_idx <= 10:
            return 'cortex'
        elif layer_idx >= 14:
            return 'medulla'
        else:
            return 'cmj'  # Corticomedullary junction
    
    for layer_offset in range(n_layers):
        layer_num = organ_start_layer + layer_offset
        layer_idx = layer_offset + 1  # 1-indexed for display
        
        # Get pixels in this layer
        layer_pixels = (mlco_mask == layer_num)
        
        if tissue_mask is not None:
            layer_pixels = layer_pixels & tissue_mask
        
        layer_pixels = layer_pixels & ~np.isnan(t2_map) & ~np.isnan(r2_map)
        
        n_pixels = np.sum(layer_pixels)
        
        if n_pixels == 0:
            print(f"  Layer {layer_idx:2d}: No valid pixels")
            continue
        
        # Extract values
        t2_values = t2_map[layer_pixels]
        r2_values = r2_map[layer_pixels]
        
        # Calculate statistics
        layer_stats = {
            'layer': layer_idx,
            'layer_number': layer_num,
            'zone': get_zone_name(layer_idx, zone_config=zone_config),
            'n_pixels': int(n_pixels),
            't2star': {
                'mean': float(np.mean(t2_values)),
                'median': float(np.median(t2_values)),
                'std': float(np.std(t2_values)),
                'min': float(np.min(t2_values)),
                'max': float(np.max(t2_values)),
                'q25': float(np.percentile(t2_values, 25)),
                'q75': float(np.percentile(t2_values, 75))
            },
            'r2star': {
                'mean': float(np.mean(r2_values)),
                'median': float(np.median(r2_values)),
                'std': float(np.std(r2_values)),
                'min': float(np.min(r2_values)),
                'max': float(np.max(r2_values))
            }
        }
        
        # Add perfusion if available
        if perfusion_map is not None:
            perf_values = perfusion_map[layer_pixels]
            perf_values = perf_values[~np.isnan(perf_values)]
            if len(perf_values) > 0:
                layer_stats['perfusion'] = {
                    'mean': float(np.mean(perf_values)),
                    'median': float(np.median(perf_values)),
                    'std': float(np.std(perf_values)),
                    'min': float(np.min(perf_values)),
                    'max': float(np.max(perf_values))
                }
        
        # Tissue quality assessment
        region_type = get_region_type(layer_idx)
        tissue_quality = calculate_tissue_quality(
            t2_map, layer_pixels,
            perfusion_map if perfusion_map is not None else None,
            region=region_type
        )
        layer_stats['tissue_quality'] = tissue_quality
        
        layers.append(layer_stats)
        
        # Print compact summary with tissue quality
        depth_pct = (layer_idx - 1) / (n_layers - 1) * 100 if n_layers > 1 else 0
        viable_pct = tissue_quality['viable_pct']
        
        quality_flag = "✓" if viable_pct > 90 else "⚠️" if viable_pct > 70 else "🔴"
        
        print(f"  Layer {layer_idx:2d} ({depth_pct:5.1f}%) {quality_flag}: "
              f"{n_pixels:5d} pix | "
              f"T2*: {layer_stats['t2star']['median']:6.2f}±{layer_stats['t2star']['std']:5.2f} ms | "
              f"R2*: {layer_stats['r2star']['median']:5.2f}±{layer_stats['r2star']['std']:4.2f} Hz", end="")
        
        if 'perfusion' in layer_stats:
            print(f" | Perf: {layer_stats['perfusion']['median']:5.0f}±{layer_stats['perfusion']['std']:4.0f}", end="")
        
        print(f" | Viable: {viable_pct:4.0f}%")
    
    # Calculate gradient
    if len(layers) >= 2:
        t2_cortex = layers[0]['t2star']['median']
        t2_medulla = layers[-1]['t2star']['median']
        t2_gradient = t2_medulla - t2_cortex
        
        r2_cortex = layers[0]['r2star']['median']
        r2_medulla = layers[-1]['r2star']['median']
        r2_gradient = r2_medulla - r2_cortex
        
        print(f"\n  GRADIENT ANALYSIS:")
        print(f"    T2* Cortex (Layer 1):      {t2_cortex:.2f} ms")
        print(f"    T2* Medulla (Layer {n_layers}):    {t2_medulla:.2f} ms")
        print(f"    T2* Gradient:               {t2_gradient:+.2f} ms")
        
        if t2_gradient > 10:
            print(f"    ⚠️  ABNORMAL: Medulla T2* >> Cortex (suggests fluid/edema/necrosis)")
        elif t2_gradient > 0:
            print(f"    ⚠️  ABNORMAL: Medulla T2* > Cortex (mild)")
        else:
            print(f"    ✓ Normal: Cortex T2* > Medulla")
        
        gradient_info = {
            't2star': {
                'cortex': t2_cortex,
                'medulla': t2_medulla,
                'gradient': t2_gradient,
                'abnormal': t2_gradient > 0
            },
            'r2star': {
                'cortex': r2_cortex,
                'medulla': r2_medulla,
                'gradient': r2_gradient
            }
        }
        
        # Add perfusion gradient if available
        if 'perfusion' in layers[0] and 'perfusion' in layers[-1]:
            perf_cortex = layers[0]['perfusion']['median']
            perf_medulla = layers[-1]['perfusion']['median']
            perf_gradient = perf_medulla - perf_cortex
            
            gradient_info['perfusion'] = {
                'cortex': perf_cortex,
                'medulla': perf_medulla,
                'gradient': perf_gradient
            }
            
            print(f"\n    Perfusion Cortex:           {perf_cortex:.0f} ml/100g/min")
            print(f"    Perfusion Medulla:          {perf_medulla:.0f} ml/100g/min")
            print(f"    Perfusion Gradient:         {perf_gradient:+.0f} ml/100g/min")
    else:
        gradient_info = None
    
    return {
        'kidney': organ_label,
        'n_layers': n_layers,
        'layers': layers,
        'gradient': gradient_info
    }


def calculate_zone_statistics(layers: List[Dict], zone_name: str) -> Dict:
    """
    Calculate aggregate statistics for a specific zone
    
    Parameters:
    -----------
    layers : list
        List of layer statistics
    zone_name : str
        Zone name (e.g., 'outer_cortex', 'inner_medulla')
        
    Returns:
    --------
    zone_stats : dict
        Aggregate statistics for the zone
    """
    zone_layers = get_zone_layers(zone_name)
    
    # Filter layers in this zone
    zone_data = [l for l in layers if l['layer'] in zone_layers]
    
    if not zone_data:
        return None
    
    # Aggregate T2* values
    t2_values = [l['t2star']['median'] for l in zone_data]
    r2_values = [l['r2star']['median'] for l in zone_data]
    
    stats = {
        'zone': zone_name,
        'n_layers': len(zone_data),
        't2star': {
            'mean': float(np.mean(t2_values)),
            'median': float(np.median(t2_values)),
            'std': float(np.std(t2_values))
        },
        'r2star': {
            'mean': float(np.mean(r2_values)),
            'median': float(np.median(r2_values)),
            'std': float(np.std(r2_values))
        }
    }
    
    # Add perfusion if available
    if 'perfusion' in zone_data[0]:
        perf_values = [l['perfusion']['median'] for l in zone_data if 'perfusion' in l]
        if perf_values:
            stats['perfusion'] = {
                'mean': float(np.mean(perf_values)),
                'median': float(np.median(perf_values)),
                'std': float(np.std(perf_values))
            }
    
    # Aggregate tissue quality
    viable_pcts = [l['tissue_quality']['viable_pct'] for l in zone_data]
    necrosis_pcts = [l['tissue_quality']['likely_necrosis_pct'] for l in zone_data]
    edema_pcts = [l['tissue_quality']['suspect_edema_pct'] for l in zone_data]
    
    stats['tissue_quality'] = {
        'viable_pct': float(np.mean(viable_pcts)),
        'suspect_edema_pct': float(np.mean(edema_pcts)),
        'likely_necrosis_pct': float(np.mean(necrosis_pcts))
    }
    
    # Generate interpretation
    region_type = 'cortex' if 'cortex' in zone_name else 'medulla'
    stats['interpretation'] = interpret_tissue_state(
        stats['t2star']['mean'],
        stats.get('perfusion', {}).get('mean'),
        stats['tissue_quality'],
        zone_name
    )
    
    return stats


def _analyze_mlco_bilateral(
    t2_map: np.ndarray,
    r2_map: np.ndarray,
    mlco_mask: np.ndarray,
    n_layers_per_organ: int = 24,
    tissue_mask: Optional[np.ndarray] = None,
    perfusion_map: Optional[np.ndarray] = None,
    scan_label: str = "scan",
    zone_config: Optional[Dict] = None
) -> Dict:
    """
    Enhanced bilateral MLCO analysis with 5-zone breakdown
    
    Parameters:
    -----------
    t2_map : ndarray
        T2* map
    r2_map : ndarray
        R2* map
    mlco_mask : ndarray
        MLCO layer mask (right: 1-24, left: 25-48)
    n_layers_per_organ : int
        Number of layers per kidney (24)
    tissue_mask : ndarray, optional
        Tissue validity mask
    perfusion_map : ndarray, optional
        Perfusion map
    scan_label : str
        Scan identifier
        
    Returns:
    --------
    results : dict
        Complete bilateral analysis with 5-zone statistics
    """
    print(f"\n{'='*70}")
    print(f"ENHANCED MLCO ANALYSIS (24 LAYERS): {scan_label}")
    print(f"{'='*70}")
    
    # Analyze right kidney
    right_results = _analyze_mlco_single_organ(
        t2_map, r2_map, mlco_mask,
        organ_start_layer=1,
        n_layers=n_layers_per_organ,
        tissue_mask=tissue_mask,
        perfusion_map=perfusion_map,
        organ_label="Right Kidney (Anatomical)",
        zone_config=zone_config
    )

    # Analyze left kidney
    left_results = _analyze_mlco_single_organ(
        t2_map, r2_map, mlco_mask,
        organ_start_layer=n_layers_per_organ + 1,
        n_layers=n_layers_per_organ,
        tissue_mask=tissue_mask,
        perfusion_map=perfusion_map,
        organ_label="Left Kidney (Anatomical)",
        zone_config=zone_config
    )
    
    # Calculate zone statistics for each kidney
    n_zones = len(tissue_zones.ZONE_DEFINITIONS)
    print(f"\n{'─'*70}")
    print(f"{n_zones}-ZONE REGIONAL ANALYSIS")
    print(f"{'─'*70}")

    right_zones = {}
    left_zones = {}

    for zone_name in tissue_zones.ZONE_DEFINITIONS.keys():
        right_zone = calculate_zone_statistics(right_results['layers'], zone_name)
        left_zone = calculate_zone_statistics(left_results['layers'], zone_name)
        
        if right_zone:
            right_zones[zone_name] = right_zone
            print(f"\nRight {zone_name:20s}: T2* = {right_zone['t2star']['mean']:5.1f} ms, "
                  f"Viable = {right_zone['tissue_quality']['viable_pct']:4.0f}%")
            print(f"  {right_zone['interpretation']}")
        
        if left_zone:
            left_zones[zone_name] = left_zone
            print(f"Left  {zone_name:20s}: T2* = {left_zone['t2star']['mean']:5.1f} ms, "
                  f"Viable = {left_zone['tissue_quality']['viable_pct']:4.0f}%")
            print(f"  {left_zone['interpretation']}")
    
    # Store zone results
    right_results['zones'] = right_zones
    left_results['zones'] = left_zones
    
    # Bilateral combined analysis (average of both kidneys)
    print(f"\n{'─'*70}")
    print("BILATERAL (COMBINED)")
    print(f"{'─'*70}")
    
    bilateral_layers = []
    
    for layer_idx in range(1, n_layers_per_organ + 1):
        # Combine same layer from both kidneys
        right_layer_num = layer_idx
        left_layer_num = n_layers_per_organ + layer_idx
        
        layer_pixels = ((mlco_mask == right_layer_num) | (mlco_mask == left_layer_num))
        
        if tissue_mask is not None:
            layer_pixels = layer_pixels & tissue_mask
        
        layer_pixels = layer_pixels & ~np.isnan(t2_map) & ~np.isnan(r2_map)
        
        n_pixels = np.sum(layer_pixels)
        
        if n_pixels == 0:
            continue
        
        t2_values = t2_map[layer_pixels]
        r2_values = r2_map[layer_pixels]
        
        layer_stats = {
            'layer': layer_idx,
            'zone': get_zone_name(layer_idx, zone_config=zone_config),
            'n_pixels': int(n_pixels),
            't2star': {
                'mean': float(np.mean(t2_values)),
                'median': float(np.median(t2_values)),
                'std': float(np.std(t2_values))
            },
            'r2star': {
                'mean': float(np.mean(r2_values)),
                'median': float(np.median(r2_values)),
                'std': float(np.std(r2_values))
            }
        }
        
        # Add perfusion
        if perfusion_map is not None:
            perf_values = perfusion_map[layer_pixels]
            perf_values = perf_values[~np.isnan(perf_values)]
            if len(perf_values) > 0:
                layer_stats['perfusion'] = {
                    'mean': float(np.mean(perf_values)),
                    'median': float(np.median(perf_values)),
                    'std': float(np.std(perf_values))
                }
        
        # Tissue quality
        region_type = 'cortex' if layer_idx <= 10 else 'medulla'
        tissue_quality = calculate_tissue_quality(
            t2_map, layer_pixels,
            perfusion_map if perfusion_map is not None else None,
            region=region_type
        )
        layer_stats['tissue_quality'] = tissue_quality
        
        bilateral_layers.append(layer_stats)
        
        depth_pct = (layer_idx - 1) / (n_layers_per_organ - 1) * 100
        print(f"  Layer {layer_idx:2d} ({depth_pct:5.1f}%): "
              f"{n_pixels:5d} pix | "
              f"T2*: {layer_stats['t2star']['median']:6.2f}±{layer_stats['t2star']['std']:5.2f} ms")
    
    # Calculate bilateral gradient
    if len(bilateral_layers) >= 2:
        t2_gradient = bilateral_layers[-1]['t2star']['median'] - bilateral_layers[0]['t2star']['median']
        r2_gradient = bilateral_layers[-1]['r2star']['median'] - bilateral_layers[0]['r2star']['median']
        
        print(f"\n  BILATERAL GRADIENT:")
        print(f"    T2* Gradient: {t2_gradient:+.2f} ms")
        print(f"    R2* Gradient: {r2_gradient:+.2f} Hz")
        
        if t2_gradient > 10:
            print(f"    ⚠️  SEVERELY ABNORMAL gradient (likely tissue necrosis)")
        elif t2_gradient > 0:
            print(f"    ⚠️  ABNORMAL gradient (medulla > cortex)")
        
        bilateral_gradient = {
            't2star': {
                'cortex': bilateral_layers[0]['t2star']['median'],
                'medulla': bilateral_layers[-1]['t2star']['median'],
                'gradient': t2_gradient,
                'abnormal': t2_gradient > 0
            },
            'r2star': {
                'cortex': bilateral_layers[0]['r2star']['median'],
                'medulla': bilateral_layers[-1]['r2star']['median'],
                'gradient': r2_gradient
            }
        }
        
        if 'perfusion' in bilateral_layers[0] and 'perfusion' in bilateral_layers[-1]:
            perf_gradient = bilateral_layers[-1]['perfusion']['median'] - bilateral_layers[0]['perfusion']['median']
            bilateral_gradient['perfusion'] = {
                'cortex': bilateral_layers[0]['perfusion']['median'],
                'medulla': bilateral_layers[-1]['perfusion']['median'],
                'gradient': perf_gradient
            }
    else:
        bilateral_gradient = None
    
    # Calculate bilateral zone statistics
    bilateral_zones = {}
    for zone_name in tissue_zones.ZONE_DEFINITIONS.keys():
        bilateral_zone = calculate_zone_statistics(bilateral_layers, zone_name)
        if bilateral_zone:
            bilateral_zones[zone_name] = bilateral_zone
    
    return {
        'scan_label': scan_label,
        'n_layers_per_organ': n_layers_per_organ,
        'right_kidney': right_results,
        'left_kidney': left_results,
        'bilateral': {
            'layers': bilateral_layers,
            'gradient': bilateral_gradient,
            'zones': bilateral_zones
        }
    }


if __name__ == "__main__":
    print("Enhanced MLCO Analysis Module")
    print("=" * 70)
    print("Features:")
    print("  ✓ 24 layers per kidney (2x resolution)")
    print("  ✓ 5-zone regional analysis")
    print("  ✓ Tissue viability classification")
    print("  ✓ Perfusion integration")
    print("  ✓ Separate left/right + bilateral combined")


# ==============================================================================
# MULTI-REGION ANALYSIS
# ==============================================================================

def _analyze_mlco_multiregion(
    t2_map: np.ndarray,
    r2_map: np.ndarray,
    mlco_mask: np.ndarray,
    region_ids: List[int],
    zone_config: Dict,
    tissue_mask: Optional[np.ndarray] = None,
    perfusion_map: Optional[np.ndarray] = None,
    scan_label: str = "scan"
) -> Dict:
    """
    Analyze multi-region MLCO mask with per-region zones
    
    Parameters:
    -----------
    t2_map : ndarray
        T2* map (ms)
    r2_map : ndarray
        R2* map (Hz)
    mlco_mask : ndarray
        Multi-region MLCO mask with encoded values
    region_ids : list of int
        List of region IDs to analyze
    zone_config : dict
        Multi-region zone configuration
    tissue_mask : ndarray, optional
        Tissue validity mask
    perfusion_map : ndarray, optional
        Perfusion map (ml/100g/min)
    scan_label : str
        Scan identifier
        
    Returns:
    --------
    results : dict
        Hierarchical results: {
            'scan_label': str,
            'mode': 'multi_region',
            'regions': {
                region_name: {
                    'region_id': int,
                    'layers': [...],
                    'zones': {...},
                    'summary': {...}
                }
            }
        }
    """
    print(f"\n{'='*70}")
    print(f"MULTI-REGION MLCO ANALYSIS")
    print(f"{'='*70}")
    print(f"Scan: {scan_label}")
    print(f"Regions: {len(region_ids)}")
    
    results = {
        'scan_label': scan_label,
        'mode': 'multi_region',
        'n_regions': len(region_ids),
        'regions': {}
    }
    
    # Analyze each region independently
    for region_id in region_ids:
        region_name = get_region_name(region_id, zone_config)
        region_config = zone_config['regions'][region_name]
        n_layers = region_config['n_layers']
        
        print(f"\n{'─'*70}")
        print(f"REGION {region_id}: {region_name.upper()}")
        print(f"{'─'*70}")
        print(f"Layers: {n_layers}")
        
        # Extract region mask
        region_mask = extract_region_from_mlco(mlco_mask, region_id)
        n_region_pixels = np.sum(region_mask)
        print(f"Pixels in region: {n_region_pixels:,}")
        
        # Analyze each layer in this region
        region_layers = []
        
        for layer_num in range(1, n_layers + 1):
            # Encoded value for this layer
            encoded_value = region_id * 1000 + layer_num
            
            # Get pixels in this layer
            layer_pixels = (mlco_mask == encoded_value)
            
            if tissue_mask is not None:
                layer_pixels = layer_pixels & tissue_mask
            
            layer_pixels = layer_pixels & ~np.isnan(t2_map) & ~np.isnan(r2_map)
            
            n_pixels = np.sum(layer_pixels)
            
            if n_pixels == 0:
                print(f"  Layer {layer_num:2d}: No valid pixels")
                continue
            
            # Extract values
            t2_values = t2_map[layer_pixels]
            r2_values = r2_map[layer_pixels]
            
            layer_stats = {
                'layer': layer_num,
                'encoded_value': int(encoded_value),
                'n_pixels': int(n_pixels),
                't2star': {
                    'mean': float(np.mean(t2_values)),
                    'median': float(np.median(t2_values)),
                    'std': float(np.std(t2_values)),
                    'min': float(np.min(t2_values)),
                    'max': float(np.max(t2_values))
                },
                'r2star': {
                    'mean': float(np.mean(r2_values)),
                    'median': float(np.median(r2_values)),
                    'std': float(np.std(r2_values)),
                    'min': float(np.min(r2_values)),
                    'max': float(np.max(r2_values))
                }
            }
            
            # Add perfusion if available
            if perfusion_map is not None:
                perf_values = perfusion_map[layer_pixels]
                perf_values = perf_values[~np.isnan(perf_values)]
                if len(perf_values) > 0:
                    layer_stats['perfusion'] = {
                        'mean': float(np.mean(perf_values)),
                        'median': float(np.median(perf_values)),
                        'std': float(np.std(perf_values))
                    }
            
            # Tissue quality (map custom region names to standard names)
            # Extract anatomical region type from region name
            # e.g., 'right_cortex' -> 'cortex', 'left_medulla' -> 'medulla'
            region_type = 'cortex'  # Default
            region_name_lower = region_name.lower()
            if 'cortex' in region_name_lower:
                region_type = 'cortex'
            elif 'medulla' in region_name_lower or 'papilla' in region_name_lower:
                region_type = 'medulla'
            
            try:
                tissue_quality = calculate_tissue_quality(
                    t2_map, layer_pixels,
                    perfusion_map if perfusion_map is not None else None,
                    region=region_type  # Use mapped region type
                )
                layer_stats['tissue_quality'] = tissue_quality
            except (KeyError, ValueError) as e:
                # Skip tissue quality if thresholds not available for this region
                # This is optional analysis anyway
                pass
            
            region_layers.append(layer_stats)
            
            # Print layer summary
            depth_pct = (layer_num - 0.5) / n_layers * 100
            print(f"  Layer {layer_num:2d} ({depth_pct:5.1f}%): "
                  f"{n_pixels:5d} pix | "
                  f"T2*: {layer_stats['t2star']['median']:6.2f}±{layer_stats['t2star']['std']:5.2f} ms")
        
        # Calculate region gradient
        if len(region_layers) >= 2:
            t2_gradient = region_layers[-1]['t2star']['median'] - region_layers[0]['t2star']['median']
            r2_gradient = region_layers[-1]['r2star']['median'] - region_layers[0]['r2star']['median']
            
            print(f"\n  REGION GRADIENT:")
            print(f"    T2* Gradient: {t2_gradient:+.2f} ms (outer → inner)")
            print(f"    R2* Gradient: {r2_gradient:+.2f} Hz (outer → inner)")
            
            region_gradient = {
                't2star': {
                    'outer': region_layers[0]['t2star']['median'],
                    'inner': region_layers[-1]['t2star']['median'],
                    'gradient': t2_gradient
                },
                'r2star': {
                    'outer': region_layers[0]['r2star']['median'],
                    'inner': region_layers[-1]['r2star']['median'],
                    'gradient': r2_gradient
                }
            }
            
            if 'perfusion' in region_layers[0] and 'perfusion' in region_layers[-1]:
                perf_gradient = region_layers[-1]['perfusion']['median'] - region_layers[0]['perfusion']['median']
                region_gradient['perfusion'] = {
                    'outer': region_layers[0]['perfusion']['median'],
                    'inner': region_layers[-1]['perfusion']['median'],
                    'gradient': perf_gradient
                }
        else:
            region_gradient = None
        
        # Calculate zone statistics for this region
        region_zones = {}
        region_zone_defs = get_region_zones(region_id, zone_config)
        
        for zone_name, zone_info in region_zone_defs.items():
            zone_encoded_layers = zone_info['layers']
            
            # Find layers in this zone
            zone_layers = [l for l in region_layers 
                          if l['encoded_value'] in zone_encoded_layers]
            
            if not zone_layers:
                continue
            
            # Calculate zone statistics
            zone_t2_values = []
            zone_r2_values = []
            zone_perf_values = []
            zone_pixels = 0
            
            for layer in zone_layers:
                # Get all pixels for this layer
                layer_pixels = (mlco_mask == layer['encoded_value'])
                if tissue_mask is not None:
                    layer_pixels = layer_pixels & tissue_mask
                layer_pixels = layer_pixels & ~np.isnan(t2_map) & ~np.isnan(r2_map)
                
                zone_t2_values.extend(t2_map[layer_pixels].tolist())
                zone_r2_values.extend(r2_map[layer_pixels].tolist())
                zone_pixels += np.sum(layer_pixels)
                
                if perfusion_map is not None:
                    perf = perfusion_map[layer_pixels]
                    perf = perf[~np.isnan(perf)]
                    zone_perf_values.extend(perf.tolist())
            
            if zone_pixels > 0:
                zone_stats = {
                    'n_pixels': int(zone_pixels),
                    'n_layers': len(zone_layers),
                    't2star': {
                        'mean': float(np.mean(zone_t2_values)),
                        'median': float(np.median(zone_t2_values)),
                        'std': float(np.std(zone_t2_values))
                    },
                    'r2star': {
                        'mean': float(np.mean(zone_r2_values)),
                        'median': float(np.median(zone_r2_values)),
                        'std': float(np.std(zone_r2_values))
                    }
                }
                
                if zone_perf_values:
                    zone_stats['perfusion'] = {
                        'mean': float(np.mean(zone_perf_values)),
                        'median': float(np.median(zone_perf_values)),
                        'std': float(np.std(zone_perf_values))
                    }
                
                region_zones[zone_name] = zone_stats
                
                print(f"\n  Zone: {zone_name}")
                print(f"    Layers: {len(zone_layers)}, Pixels: {zone_pixels:,}")
                print(f"    T2*: {zone_stats['t2star']['median']:.2f}±{zone_stats['t2star']['std']:.2f} ms")
        
        # Store region results
        results['regions'][region_name] = {
            'region_id': int(region_id),
            'n_layers': n_layers,
            'n_pixels': int(n_region_pixels),
            'layers': region_layers,
            'zones': region_zones,
            'gradient': region_gradient
        }
    
    print(f"\n{'='*70}")
    print("✓ MULTI-REGION ANALYSIS COMPLETE")
    print(f"{'='*70}")
    
    return results


# ==============================================================================
# PUBLIC API - Consolidated Function
# ==============================================================================

def analyze_mlco(
    t2_map: np.ndarray,
    r2_map: np.ndarray,
    mlco_mask: np.ndarray,
    n_layers_per_organ: int = 24,
    tissue_mask: Optional[np.ndarray] = None,
    perfusion_map: Optional[np.ndarray] = None,
    scan_label: str = "scan",
    analysis_mode: str = 'bilateral',
    side: Optional[str] = None,
    zone_config: Optional[Dict] = None
) -> Dict:
    """
    Analyze Multi-Layer Concentric Object (MLCO)
    
    Flexible function that supports:
    - Bilateral analysis (left + right organs)
    - Single organ analysis  
    - Multi-region analysis (NEW! - independent regions with per-region zones)
    
    Automatically detects MLCO format and routes to appropriate analysis.
    
    Parameters:
    -----------
    t2_map : ndarray
        T2* map in milliseconds
    r2_map : ndarray
        R2* map in Hz
    mlco_mask : ndarray
        MLCO layer mask with integer labels
        - Standard bilateral: layers 1-N (organ 1), (N+1)-2N (organ 2)
        - Standard single: layers 1-N
        - Multi-region: encoded as region_id * 1000 + layer_num
          (e.g., 1001-1008, 2001-2008, 3001-3006)
    n_layers_per_organ : int, default=24
        Number of layers per organ (for bilateral/single modes)
        Ignored in multi-region mode (uses zone_config)
    tissue_mask : ndarray, optional
        Boolean mask of valid tissue pixels
    perfusion_map : ndarray, optional
        Perfusion map in ml/100g/min
    scan_label : str
        Scan identifier for output
    analysis_mode : str, default='bilateral'
        Analysis mode (ignored if multi-region detected):
        - 'bilateral': Analyze both organs together
        - 'single': Analyze one organ only
    side : str, optional
        Required if analysis_mode='single'
        - 'right': Analyze right organ (layers 1-N)
        - 'left': Analyze left organ (layers N+1 to 2N)
    zone_config : dict, optional
        Zone configuration dictionary (required for multi-region analysis)
        If None, uses default single-region zones
        
    Returns:
    --------
    results : dict
        Analysis results with structure depending on detected mode:
        - bilateral mode: {'bilateral': {...}, 'right_organ': {...}, 'left_organ': {...}}
        - single mode: {'layers': [...], 'zones': {...}}
        - multi-region mode: {'mode': 'multi_region', 'regions': {...}}
        
    Examples:
    ---------
    # Bilateral analysis (both kidneys)
    >>> results = analyze_mlco(
    ...     t2_map, r2_map, mlco_mask,
    ...     n_layers_per_organ=24,
    ...     analysis_mode='bilateral'
    ... )
    
    # Multi-region analysis (cortex, medulla, papilla)
    >>> from tissue_zones import load_zone_config
    >>> config = load_zone_config('configs/zones/kidney_multiregion_8_8_6.yaml')
    >>> results = analyze_mlco(
    ...     t2_map, r2_map, mlco_mask,
    ...     zone_config=config
    ... )
    
    Notes:
    ------
    - Multi-region format auto-detected (values >= 1000)
    - Multi-region requires zone_config parameter
    - Layer count customizable (not hardcoded to 24)
    - Zone definitions customizable via tissue_zones.py
    """
    # Validate parameters
    if analysis_mode not in ['bilateral', 'single']:
        raise ValueError(
            f"analysis_mode must be 'bilateral' or 'single', got '{analysis_mode}'"
        )
    
    if analysis_mode == 'single' and side not in ['right', 'left']:
        raise ValueError(
            "side must be 'right' or 'left' when analysis_mode='single'"
        )
    
    # AUTO-DETECT MULTI-REGION MLCO
    is_multiregion, region_ids = detect_multiregion_mlco(mlco_mask)
    
    if is_multiregion:
        # MULTI-REGION MODE
        print(f"\n{'='*70}")
        print("MLCO FORMAT: MULTI-REGION DETECTED")
        print(f"{'='*70}")
        print(f"Detected {len(region_ids)} regions: {region_ids}")
        print(f"Encoding: region_id * 1000 + layer_num")
        
        # Require zone_config for multi-region
        if zone_config is None:
            raise ValueError(
                "Multi-region MLCO detected but zone_config not provided. "
                "Please load zone config: "
                "zone_config = load_zone_config('configs/zones/kidney_multiregion_8_8_6.yaml')"
            )
        
        # Verify zone_config is multi-region format
        if not is_multiregion_config(zone_config):
            raise ValueError(
                "Multi-region MLCO detected but zone_config is single-region format. "
                "Please use a multi-region zone config (mode: multi_region)"
            )
        
        # Route to multi-region analysis
        return _analyze_mlco_multiregion(
            t2_map=t2_map,
            r2_map=r2_map,
            mlco_mask=mlco_mask,
            region_ids=region_ids,
            zone_config=zone_config,
            tissue_mask=tissue_mask,
            perfusion_map=perfusion_map,
            scan_label=scan_label
        )
    
    # STANDARD BILATERAL/SINGLE MODE (backward compatible)
    print(f"\n{'='*70}")
    print("MLCO FORMAT: STANDARD (BILATERAL/SINGLE)")
    print(f"{'='*70}")
    
    # Route to appropriate internal function
    if analysis_mode == 'bilateral':
        return _analyze_mlco_bilateral(
            t2_map=t2_map,
            r2_map=r2_map,
            mlco_mask=mlco_mask,
            n_layers_per_organ=n_layers_per_organ,  # Legacy parameter name
            tissue_mask=tissue_mask,
            perfusion_map=perfusion_map,
            scan_label=scan_label,
            zone_config=zone_config
        )

    else:  # analysis_mode == 'single'
        # Determine starting layer based on side
        if side == 'right':
            start_layer = 1
            organ_label = "Right Organ"
        else:  # side == 'left'
            start_layer = n_layers_per_organ + 1
            organ_label = "Left Organ"

        return _analyze_mlco_single_organ(
            t2_map=t2_map,
            r2_map=r2_map,
            mlco_mask=mlco_mask,
            organ_start_layer=start_layer,  # Legacy parameter name
            n_layers=n_layers_per_organ,
            tissue_mask=tissue_mask,
            perfusion_map=perfusion_map,
            organ_label=organ_label,
            zone_config=zone_config
        )


# ============================================================================
# PHASE 5.5: ENHANCED MULTI-REGION ANALYSIS FUNCTIONS
# ============================================================================
# Functions for bilateral averaging, whole kidney metrics, and oxygen response
# Added: 2026-01-21
# ============================================================================

def average_bilateral_regions(
    results: dict,
    region_pairs: list = None
) -> dict:
    """
    Average left and right regions as technical replicates
    
    Parameters:
    -----------
    results : dict
        Analysis results from analyze_mlco() in multi-region mode
        Contains results['regions'] with individual regions
    region_pairs : list of tuples, optional
        Pairs of (left, right) region names to average
        Default: [('left_cortex', 'right_cortex'), 
                  ('left_medulla', 'right_medulla'),
                  ('left_papilla', 'right_papilla')]
    
    Returns:
    --------
    averaged : dict
        {
            'cortex': {layers, zones, gradient, n_pixels, asymmetry_index},
            'medulla': {...},
            'papilla': {...}
        }
    
    Notes:
    ------
    For each layer:
    - mean_value = (left_mean + right_mean) / 2
    - std_pooled = sqrt((std_left² + std_right²) / 2)
    - n_pixels = n_left + n_right
    
    Asymmetry index = abs(left - right) / mean(left, right)
    Flag if > 0.20 (20% difference)
    """
    # Auto-detect region pairs if not provided
    if region_pairs is None:
        region_pairs = _detect_bilateral_pairs(results)
    
    if not region_pairs:
        print("  Note: No bilateral pairs detected, returning original results")
        return results
    
    averaged = {}
    
    for left_name, right_name in region_pairs:
        # Determine averaged region name (e.g., 'left_cortex' + 'right_cortex' → 'cortex')
        avg_name = _get_averaged_region_name(left_name, right_name)
        
        # Get individual region data
        left_data = results['regions'].get(left_name)
        right_data = results['regions'].get(right_name)
        
        if left_data is None or right_data is None:
            print(f"  Warning: Missing region for pair ({left_name}, {right_name}), skipping")
            continue
        
        # Average the regions
        averaged_region = _average_two_regions(left_data, right_data, left_name, right_name)
        averaged[avg_name] = averaged_region
    
    return averaged


def _detect_bilateral_pairs(results: dict) -> list:
    """Auto-detect left/right region pairs"""
    regions = list(results['regions'].keys())
    pairs = []
    
    for region in regions:
        if region.startswith('left_'):
            tissue = region.replace('left_', '')
            right_name = f'right_{tissue}'
            if right_name in regions:
                pairs.append((region, right_name))
    
    return pairs


def _get_averaged_region_name(left_name: str, right_name: str) -> str:
    """Get averaged region name from left/right pair"""
    name = left_name.replace('left_', '').replace('right_', '')
    return name


def _average_two_regions(left_data: dict, right_data: dict, left_name: str, right_name: str) -> dict:
    """Average two regions (left + right)"""
    averaged = {
        'source_regions': [left_name, right_name],
        'n_layers': left_data['n_layers'],
        'layers': [],
        'zones': {},
        'gradient': {},
        'n_pixels': left_data['n_pixels'] + right_data['n_pixels']
    }
    
    # Average layers
    left_layers = left_data['layers']
    right_layers = right_data['layers']
    n_layers = min(len(left_layers), len(right_layers))
    asymmetry_values = []
    
    for i in range(n_layers):
        left_layer = left_layers[i]
        right_layer = right_layers[i]
        avg_layer = _average_two_layers(left_layer, right_layer)
        averaged['layers'].append(avg_layer)
        
        # Track asymmetry for T2*
        if 't2star' in left_layer and 't2star' in right_layer:
            left_t2 = left_layer['t2star']['median']
            right_t2 = right_layer['t2star']['median']
            mean_t2 = (left_t2 + right_t2) / 2
            if mean_t2 > 0:
                asym = abs(left_t2 - right_t2) / mean_t2
                asymmetry_values.append(asym)
    
    # Average zones
    if 'zones' in left_data and 'zones' in right_data:
        for zone_name in left_data['zones'].keys():
            if zone_name in right_data['zones']:
                left_zone = left_data['zones'][zone_name]
                right_zone = right_data['zones'][zone_name]
                averaged['zones'][zone_name] = _average_two_zones(left_zone, right_zone)
    
    # Average gradients
    if 'gradient' in left_data and 'gradient' in right_data:
        averaged['gradient'] = _average_two_gradients(left_data['gradient'], right_data['gradient'])
    
    # Calculate asymmetry metrics
    averaged['asymmetry'] = {
        'mean_asymmetry_index': float(np.mean(asymmetry_values)) if asymmetry_values else 0.0,
        'max_asymmetry_index': float(np.max(asymmetry_values)) if asymmetry_values else 0.0,
        'asymmetry_flag': float(np.mean(asymmetry_values)) > 0.20 if asymmetry_values else False,
        'interpretation': 'high_asymmetry' if (asymmetry_values and np.mean(asymmetry_values) > 0.20) 
                         else 'normal_asymmetry'
    }
    
    return averaged


def _average_two_layers(left_layer: dict, right_layer: dict) -> dict:
    """Average two corresponding layers"""
    avg_layer = {
        'layer': left_layer['layer'],
        'n_pixels': left_layer['n_pixels'] + right_layer['n_pixels']
    }
    
    metrics = ['t2star', 'r2star', 'perfusion']
    for metric in metrics:
        if metric in left_layer and metric in right_layer:
            left_stats = left_layer[metric]
            right_stats = right_layer[metric]
            
            avg_layer[metric] = {
                'mean': (left_stats['mean'] + right_stats['mean']) / 2,
                'median': (left_stats['median'] + right_stats['median']) / 2,
                'std': np.sqrt((left_stats['std']**2 + right_stats['std']**2) / 2),
                'min': min(left_stats.get('min', float('inf')), right_stats.get('min', float('inf'))),
                'max': max(left_stats.get('max', 0), right_stats.get('max', 0))
            }
            
            if 'q1' in left_stats and 'q1' in right_stats:
                avg_layer[metric]['q1'] = (left_stats['q1'] + right_stats['q1']) / 2
                avg_layer[metric]['q3'] = (left_stats['q3'] + right_stats['q3']) / 2
    
    return avg_layer


def _average_two_zones(left_zone: dict, right_zone: dict) -> dict:
    """Average two corresponding zones"""
    avg_zone = {
        'n_pixels': left_zone['n_pixels'] + right_zone['n_pixels'],
        'layers': left_zone.get('layers', []) + right_zone.get('layers', [])
    }
    
    metrics = ['t2star', 'r2star', 'perfusion']
    for metric in metrics:
        if metric in left_zone and metric in right_zone:
            left_stats = left_zone[metric]
            right_stats = right_zone[metric]
            
            avg_zone[metric] = {
                'mean': (left_stats['mean'] + right_stats['mean']) / 2,
                'median': (left_stats['median'] + right_stats['median']) / 2,
                'std': np.sqrt((left_stats['std']**2 + right_stats['std']**2) / 2)
            }
    
    return avg_zone


def _average_two_gradients(left_grad: dict, right_grad: dict) -> dict:
    """Average two gradient dicts"""
    avg_grad = {}
    
    metrics = ['t2star', 'r2star', 'perfusion']
    for metric in metrics:
        if metric in left_grad and metric in right_grad:
            left_g = left_grad[metric]
            right_g = right_grad[metric]
            
            avg_grad[metric] = {
                'outer': (left_g['outer'] + right_g['outer']) / 2,
                'inner': (left_g['inner'] + right_g['inner']) / 2,
                'gradient': (left_g['gradient'] + right_g['gradient']) / 2
            }
    
    return avg_grad


def calculate_whole_kidney_average(results: dict, use_bilateral_averaged: bool = True) -> dict:
    """
    Calculate single whole-kidney value averaging across all regions
    
    Parameters:
    -----------
    results : dict
        Either raw multi-region results or bilateral-averaged results
    use_bilateral_averaged : bool
        If True, expects results from average_bilateral_regions()
        If False, uses raw regions directly
    
    Returns:
    --------
    whole_kidney : dict
        Single averaged value across all tissue types
    """
    whole_kidney = {'n_pixels': 0, 'n_regions': 0}
    
    # Get regions to average
    if use_bilateral_averaged:
        regions_dict = results
    else:
        regions_dict = results.get('regions', {})
    
    # Collect all layer data across all regions
    all_t2star = []
    all_r2star = []
    all_perfusion = []
    total_pixels = 0
    
    for region_name, region_data in regions_dict.items():
        if 'layers' not in region_data:
            continue
            
        whole_kidney['n_regions'] += 1
        
        for layer in region_data['layers']:
            n_pix = layer.get('n_pixels', 0)
            total_pixels += n_pix
            
            if 't2star' in layer:
                all_t2star.extend([layer['t2star']['median']] * n_pix)
            if 'r2star' in layer:
                all_r2star.extend([layer['r2star']['median']] * n_pix)
            if 'perfusion' in layer:
                all_perfusion.extend([layer['perfusion']['median']] * n_pix)
    
    whole_kidney['n_pixels'] = total_pixels
    
    # Calculate statistics
    if all_t2star:
        whole_kidney['t2star'] = {
            'mean': float(np.mean(all_t2star)),
            'median': float(np.median(all_t2star)),
            'std': float(np.std(all_t2star))
        }
    
    if all_r2star:
        whole_kidney['r2star'] = {
            'mean': float(np.mean(all_r2star)),
            'median': float(np.median(all_r2star)),
            'std': float(np.std(all_r2star))
        }
    
    if all_perfusion:
        whole_kidney['perfusion'] = {
            'mean': float(np.mean(all_perfusion)),
            'median': float(np.median(all_perfusion)),
            'std': float(np.std(all_perfusion))
        }
    
    return whole_kidney


def calculate_oxygen_response_multiregion(condition_results: dict, regions: list = None) -> dict:
    """
    Calculate oxygen responsiveness from temporal sequence: O₁ → Air → O₂
    
    Parameters:
    -----------
    condition_results : dict
        Results from all conditions: {'oxygen_1': {...}, 'air': {...}, 'oxygen_2': {...}}
    regions : list of str, optional
        Region names to analyze. If None, uses all available regions.
    
    Returns:
    --------
    oxygen_response : dict
        Detailed oxygen response analysis per region
    """
    # Check that we have all required conditions
    required_conditions = ['oxygen_1', 'air', 'oxygen_2']
    for cond in required_conditions:
        if cond not in condition_results:
            raise ValueError(f"Missing required condition: {cond}")
    
    # Get results
    o1_results = condition_results['oxygen_1']
    air_results = condition_results['air']
    o2_results = condition_results['oxygen_2']
    
    # Determine if raw or averaged
    if 'regions' in o1_results:
        o1_regions = o1_results['regions']
        air_regions = air_results['regions']
        o2_regions = o2_results['regions']
    else:
        o1_regions = o1_results
        air_regions = air_results
        o2_regions = o2_results
    
    # Auto-detect regions if not provided
    if regions is None:
        regions = list(air_regions.keys())
    
    oxygen_response = {'regions': {}}
    response_magnitudes = {}
    
    for region_name in regions:
        if region_name not in air_regions:
            continue
        
        region_response = {
            'initial_response': {},
            'recovery_response': {},
            'reproducibility': {}
        }
        
        # Calculate responses
        if region_name in o1_regions:
            region_response['initial_response'] = _calculate_delta_response(
                o1_regions[region_name], air_regions[region_name], 'oxygen_1 - air'
            )
        
        if region_name in o2_regions:
            region_response['recovery_response'] = _calculate_delta_response(
                o2_regions[region_name], air_regions[region_name], 'oxygen_2 - air'
            )
        
        if region_name in o1_regions and region_name in o2_regions:
            region_response['reproducibility'] = _calculate_delta_response(
                o2_regions[region_name], o1_regions[region_name], 'oxygen_2 - oxygen_1'
            )
        
        oxygen_response['regions'][region_name] = region_response
        
        # Track response magnitude
        if 'initial_response' in region_response and 't2star_delta' in region_response['initial_response']:
            response_magnitudes[region_name] = region_response['initial_response']['t2star_delta']['mean']
    
    # Calculate summary
    oxygen_response['summary'] = _calculate_response_summary(oxygen_response['regions'], response_magnitudes)
    
    return oxygen_response


def _calculate_delta_response(oxygen_data: dict, baseline_data: dict, comparison_name: str) -> dict:
    """Calculate delta between two conditions"""
    delta = {'comparison': comparison_name}
    
    # Layer deltas
    if 'layers' in oxygen_data and 'layers' in baseline_data:
        delta['layer_deltas'] = _calculate_layer_deltas(oxygen_data['layers'], baseline_data['layers'])
    
    # Zone deltas
    if 'zones' in oxygen_data and 'zones' in baseline_data:
        delta['zone_deltas'] = _calculate_zone_deltas(oxygen_data['zones'], baseline_data['zones'])
    
    # Gradient deltas
    if 'gradient' in oxygen_data and 'gradient' in baseline_data:
        delta['gradient_delta'] = _calculate_gradient_delta(oxygen_data['gradient'], baseline_data['gradient'])
    
    # Summary deltas
    for metric in ['t2star', 'r2star', 'perfusion']:
        delta[f'{metric}_delta'] = _calculate_metric_summary_delta(oxygen_data, baseline_data, metric)
    
    return delta


def _calculate_layer_deltas(oxygen_layers: list, baseline_layers: list) -> list:
    """Calculate per-layer deltas"""
    layer_deltas = []
    n_layers = min(len(oxygen_layers), len(baseline_layers))
    
    for i in range(n_layers):
        o_layer = oxygen_layers[i]
        b_layer = baseline_layers[i]
        layer_delta = {'layer': o_layer['layer']}
        
        for metric in ['t2star', 'r2star', 'perfusion']:
            if metric in o_layer and metric in b_layer:
                delta_val = o_layer[metric]['median'] - b_layer[metric]['median']
                pct_change = (delta_val / b_layer[metric]['median'] * 100) if b_layer[metric]['median'] != 0 else 0
                
                layer_delta[metric] = {
                    'delta': float(delta_val),
                    'percent_change': float(pct_change),
                    'oxygen_value': o_layer[metric]['median'],
                    'baseline_value': b_layer[metric]['median']
                }
        
        layer_deltas.append(layer_delta)
    
    return layer_deltas


def _calculate_zone_deltas(oxygen_zones: dict, baseline_zones: dict) -> dict:
    """Calculate per-zone deltas"""
    zone_deltas = {}
    
    for zone_name in oxygen_zones.keys():
        if zone_name not in baseline_zones:
            continue
        
        zone_deltas[zone_name] = {}
        for metric in ['t2star', 'r2star', 'perfusion']:
            if metric in oxygen_zones[zone_name] and metric in baseline_zones[zone_name]:
                o_val = oxygen_zones[zone_name][metric]['median']
                b_val = baseline_zones[zone_name][metric]['median']
                delta_val = o_val - b_val
                pct_change = (delta_val / b_val * 100) if b_val != 0 else 0
                
                zone_deltas[zone_name][metric] = {
                    'delta': float(delta_val),
                    'percent_change': float(pct_change)
                }
    
    return zone_deltas


def _calculate_gradient_delta(oxygen_gradient: dict, baseline_gradient: dict) -> dict:
    """Calculate gradient changes"""
    gradient_delta = {}
    
    for metric in ['t2star', 'r2star', 'perfusion']:
        if metric in oxygen_gradient and metric in baseline_gradient:
            o_grad = oxygen_gradient[metric]['gradient']
            b_grad = baseline_gradient[metric]['gradient']
            
            gradient_delta[metric] = {
                'gradient_delta': float(o_grad - b_grad),
                'oxygen_gradient': float(o_grad),
                'baseline_gradient': float(b_grad),
                'interpretation': 'gradient_increased' if (o_grad - b_grad) > 0 else 'gradient_decreased'
            }
    
    return gradient_delta


def _calculate_metric_summary_delta(oxygen_data: dict, baseline_data: dict, metric: str) -> dict:
    """Calculate overall summary delta"""
    if 'layers' not in oxygen_data or 'layers' not in baseline_data:
        return {}
    
    o_values = [layer[metric]['median'] for layer in oxygen_data['layers'] if metric in layer]
    b_values = [layer[metric]['median'] for layer in baseline_data['layers'] if metric in layer]
    
    if not o_values or not b_values:
        return {}
    
    o_mean = np.mean(o_values)
    b_mean = np.mean(b_values)
    delta = o_mean - b_mean
    pct_change = (delta / b_mean * 100) if b_mean != 0 else 0
    
    return {
        'mean': float(delta),
        'percent_change': float(pct_change),
        'oxygen_mean': float(o_mean),
        'baseline_mean': float(b_mean)
    }


def _calculate_response_summary(regions_response: dict, response_magnitudes: dict) -> dict:
    """Calculate summary statistics across all regions"""
    if not response_magnitudes:
        return {}
    
    best_responder = max(response_magnitudes.items(), key=lambda x: x[1])
    poorest_responder = min(response_magnitudes.items(), key=lambda x: x[1])
    
    reprod_scores = []
    for region_name, region_resp in regions_response.items():
        if 'reproducibility' in region_resp and 't2star_delta' in region_resp['reproducibility']:
            reprod_scores.append(abs(region_resp['reproducibility']['t2star_delta']['mean']))
    
    return {
        'best_responder': best_responder[0],
        'best_response_magnitude': float(best_responder[1]),
        'poorest_responder': poorest_responder[0],
        'poorest_response_magnitude': float(poorest_responder[1]),
        'mean_response': float(np.mean(list(response_magnitudes.values()))),
        'reproducibility_score': float(np.mean(reprod_scores)) if reprod_scores else 0.0,
        'interpretation': 'good_reproducibility' if (reprod_scores and np.mean(reprod_scores) < 1.0) 
                         else 'variable_reproducibility'
    }
