#!/usr/bin/env python3
"""
Complete Enhanced MLCO Analysis with Perfusion Integration
============================================================

Integrates:
- 24-layer MLCO analysis
- Tissue quality assessment
- Perfusion integration (3 modalities: T2*, R2*, Perfusion)
- Cortex-only statistics
- Tissue heterogeneity metrics
- Oxygen responsiveness analysis
- WT vs KO comparison

Usage:
    python analyze_tlco_enhanced.py \\
        --config analysis_config.json \\
        --output-dir results/
        
Or specify everything on command line (see examples below)
"""

import argparse
import json
import numpy as np
import sys
from pathlib import Path
from datetime import datetime

# Import enhanced modules
try:
    from mlco_analysis import analyze_mlco
    from boldpy_plots import (
        plot_mlco_profile,
        plot_perfusion_profile,
        plot_t2star_perfusion_scatter,
        plot_triple_overlay,
        plot_mlco_comparison
    )
    # Multi-region plotting (Phase 5)
    try:
        from boldpy_plots import generate_all_multiregion_plots
        MULTIREGION_PLOTTING_AVAILABLE = True
    except ImportError:
        MULTIREGION_PLOTTING_AVAILABLE = False
        print("  Note: Multi-region plotting not available (boldpy_plots.py not found)")
    
    # Phase 2 multiregion functions (enhanced with bilateral averaging)
    try:
        from mlco_analysis import average_bilateral_regions
        from boldpy_plots import plot_group_comparison_overview
        PHASE2_PLOTTING_AVAILABLE = True
    except ImportError:
        PHASE2_PLOTTING_AVAILABLE = False
        print("  Note: Phase 2 plotting not available")
    
    from tissue_zones import (
        load_zone_config,
        load_threshold_config,
        update_configs,
        calculate_effect_size
    )
    ENHANCED_MODULES = True
except ImportError as e:
    print("="*70)
    print("ERROR: Required modules not found!")
    print("="*70)
    print(f"Import error: {e}")
    print("\nRequired files (must be in same directory as this script):")
    print("  - mlco_analysis.py (or mlco_analysis_FIXED2.py)")
    print("  - boldpy_plots.py (unified plotting module)")
    print("  - tissue_zones.py (or tissue_zones_FIXED3.py)")
    print(f"\nScript location: {Path(__file__).parent}")
    print("\nPlease ensure all required files are in the same directory.")
    print("="*70)
    sys.exit(1)

# Add src directory to path for boldpy imports
script_dir = Path(__file__).parent
src_dir = script_dir / 'src'
if src_dir.exists():
    sys.path.insert(0, str(src_dir))

# Import BoldPy modules - only if needed for other functionality
# Note: Since we load pre-computed .npy files, we don't need these imports anymore
# Keeping this section in case we add features that need BoldPy in the future


# ==============================================================================
# DATA LOADING FUNCTIONS
# ==============================================================================


def load_data(config: dict) -> dict:
    """
    Load pre-computed T2*/R2* maps and perfusion from config
    
    Parameters:
    -----------
    config : dict
        Configuration with paths to pre-computed maps:
        {
          "id": "sample_id",
          "t2star_maps": {"condition1": "path.npy", ...},
          "r2star_maps": {"condition1": "path.npy", ...},
          "perfusion_map": "path.npy",  // optional
          "mlco_mask": "path.npy"
        }
    
    Returns:
    --------
    data : dict
        Loaded scan data with T2*, R2*, and perfusion maps
    """
    print(f"\n{'='*70}")
    print(f"Loading data for: {config['id']}")
    print(f"{'='*70}")
    
    # Validate required fields
    required = ['id', 't2star_maps', 'r2star_maps', 'mlco_mask']
    missing = [f for f in required if f not in config]
    if missing:
        raise ValueError(f"Config missing required fields: {missing}")
    
    # Check that t2star_maps and r2star_maps have same conditions
    t2_conditions = set(config['t2star_maps'].keys())
    r2_conditions = set(config['r2star_maps'].keys())
    if t2_conditions != r2_conditions:
        raise ValueError(
            f"t2star_maps and r2star_maps must have same conditions.\n"
            f"  t2star_maps: {sorted(t2_conditions)}\n"
            f"  r2star_maps: {sorted(r2_conditions)}"
        )
    
    data = {
        'id': config['id'],
        'scans': {},
        'mlco_mask': None,
        'perfusion_map': None
    }
    
    # Load T2* and R2* maps for each condition
    print(f"\nLoading maps for conditions: {sorted(t2_conditions)}")
    for condition in sorted(t2_conditions):
        print(f"\n  Loading {condition}:")
        
        # Load T2* map
        t2_path = Path(config['t2star_maps'][condition])
        if not t2_path.exists():
            raise FileNotFoundError(f"T2* map not found: {t2_path}")
        
        t2_map = np.load(t2_path)
        print(f"    ✓ T2* map: {t2_path.name}")
        print(f"      Shape: {t2_map.shape}")
        print(f"      Range: {t2_map[t2_map>0].min():.1f} - {t2_map[t2_map>0].max():.1f} ms")
        
        # Load R2* map
        r2_path = Path(config['r2star_maps'][condition])
        if not r2_path.exists():
            raise FileNotFoundError(f"R2* map not found: {r2_path}")
        
        r2_map = np.load(r2_path)
        print(f"    ✓ R2* map: {r2_path.name}")
        
        # Validate shapes match
        if t2_map.shape != r2_map.shape:
            raise ValueError(
                f"T2* and R2* maps have different shapes for {condition}:\n"
                f"  T2*: {t2_map.shape}\n"
                f"  R2*: {r2_map.shape}"
            )
        
        # Store
        data['scans'][condition] = {
            't2_map': t2_map,
            'r2_map': r2_map
        }
    
    # Load MLCO mask
    print(f"\n  Loading MLCO mask:")
    mlco_path = Path(config['mlco_mask'])
    if not mlco_path.exists():
        raise FileNotFoundError(f"MLCO mask not found: {mlco_path}")
    
    data['mlco_mask'] = np.load(mlco_path)
    unique_layers = np.unique(data['mlco_mask'])
    unique_layers = unique_layers[unique_layers > 0]
    print(f"    ✓ {mlco_path.name}")
    print(f"      {len(unique_layers)} layers loaded")
    
    # Load perfusion (optional)
    if 'perfusion_map' in config:
        print(f"\n  Loading perfusion map:")
        perf_path = Path(config['perfusion_map'])
        
        if not perf_path.exists():
            print(f"    ⚠️  Perfusion map specified but not found: {perf_path}")
        else:
            perfusion_map = np.load(perf_path)
            print(f"    ✓ {perf_path.name}")
            print(f"      Shape: {perfusion_map.shape}")
            
            # Check if resampling needed
            first_scan = list(data['scans'].values())[0]
            t2_shape = first_scan['t2_map'].shape
            
            if perfusion_map.shape != t2_shape:
                print(f"    ⚠️  Perfusion shape {perfusion_map.shape} doesn't match T2* {t2_shape}")
                print(f"       Resampling perfusion to match T2* resolution...")
                from scipy import ndimage
                zoom_factors = (t2_shape[0] / perfusion_map.shape[0],
                              t2_shape[1] / perfusion_map.shape[1])
                # Use order=1 (bilinear) for smoother resampling of perfusion data
                perfusion_map = ndimage.zoom(perfusion_map, zoom_factors, order=1)
                print(f"    ✓ Resampled to {perfusion_map.shape}")
                print(f"    ℹ️  Note: For best results, run prepare_data.py with --resample-to")
            
            data['perfusion_map'] = perfusion_map
    else:
        print(f"\n  ℹ️  No perfusion map specified (optional)")
    
    print(f"\n{'='*70}")
    print(f"✓ Data loading complete for {config['id']}")
    print(f"{'='*70}")
    
    return data



def calculate_tissue_heterogeneity(layers_data: list) -> dict:
    """
    Calculate tissue heterogeneity metrics (CV of T2* per layer)
    
    Parameters:
    -----------
    layers_data : list
        Layer statistics
    
    Returns:
    --------
    heterogeneity : dict
        Heterogeneity metrics
    """
    cv_values = []
    
    for layer in layers_data:
        mean_t2 = layer['t2star']['mean']
        std_t2 = layer['t2star']['std']
        
        if mean_t2 > 0:
            cv = (std_t2 / mean_t2) * 100
            cv_values.append({
                'layer': layer['layer'],
                'cv': cv,
                'interpretation': 'homogeneous' if cv < 25 else 'heterogeneous' if cv < 50 else 'severe_heterogeneity'
            })
    
    # Calculate zonal averages
    zones = {
        'cortex': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'medulla': [14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
    }
    
    zone_cv = {}
    for zone_name, zone_layers in zones.items():
        zone_cvs = [cv['cv'] for cv in cv_values if cv['layer'] in zone_layers]
        if zone_cvs:
            zone_cv[zone_name] = {
                'mean': float(np.mean(zone_cvs)),
                'max': float(np.max(zone_cvs)),
                'interpretation': 'homogeneous' if np.mean(zone_cvs) < 25 else 
                                 'moderately_heterogeneous' if np.mean(zone_cvs) < 50 else 
                                 'severely_heterogeneous'
            }
    
    return {
        'layer_cv': cv_values,
        'zone_cv': zone_cv
    }


def calculate_oxygen_responsiveness(conditions_data: dict) -> dict:
    """
    Calculate oxygen responsiveness index
    
    Parameters:
    -----------
    conditions_data : dict
        Data for air, oxygen_1, oxygen_2
    
    Returns:
    --------
    responsiveness : dict
        Oxygen response metrics
    """
    if 'air' not in conditions_data or 'oxygen_1' not in conditions_data:
        return {}
    
    air_layers = conditions_data['air']['bilateral']['layers']
    oxy1_layers = conditions_data['oxygen_1']['bilateral']['layers']
    
    responses = []
    
    for air_layer, oxy_layer in zip(air_layers, oxy1_layers):
        air_t2 = air_layer['t2star']['median']
        oxy_t2 = oxy_layer['t2star']['median']
        
        delta_t2 = oxy_t2 - air_t2
        pct_change = (delta_t2 / air_t2) * 100 if air_t2 > 0 else 0
        
        responses.append({
            'layer': air_layer['layer'],
            'air_t2star': air_t2,
            'oxygen_t2star': oxy_t2,
            'delta_t2star': delta_t2,
            'percent_change': pct_change,
            'responsive': abs(delta_t2) > 1  # >1ms change considered responsive
        })
    
    # Zonal responsiveness
    zones = {
        'cortex': range(1, 11),
        'medulla': range(14, 25)
    }
    
    zone_response = {}
    for zone_name, zone_range in zones.items():
        zone_responses = [r for r in responses if r['layer'] in zone_range]
        if zone_responses:
            zone_response[zone_name] = {
                'mean_delta': float(np.mean([r['delta_t2star'] for r in zone_responses])),
                'mean_percent': float(np.mean([r['percent_change'] for r in zone_responses])),
                'responsive_layers': sum([r['responsive'] for r in zone_responses]),
                'total_layers': len(zone_responses),
                'interpretation': 'viable_responsive' if np.mean([r['delta_t2star'] for r in zone_responses]) > 1 else
                                 'minimal_response' if np.mean([r['delta_t2star'] for r in zone_responses]) > -1 else
                                 'non_responsive'
            }
    
    return {
        'layer_response': responses,
        'zone_response': zone_response
    }


def extract_cortex_only_statistics(results: dict) -> dict:
    """
    Extract cortex-only (layers 1-10) statistics
    
    Parameters:
    -----------
    results : dict
        Full MLCO results
    
    Returns:
    --------
    cortex_stats : dict
        Cortex-only statistics
    """
    cortex_layers = range(1, 11)
    
    cortex_stats = {}
    
    for condition, condition_data in results.items():
        bilateral = condition_data['bilateral']
        
        # Filter cortex layers
        cortex_layer_data = [l for l in bilateral['layers'] if l['layer'] in cortex_layers]
        
        if not cortex_layer_data:
            continue
        
        # Calculate statistics
        t2_values = [l['t2star']['median'] for l in cortex_layer_data]
        r2_values = [l['r2star']['median'] for l in cortex_layer_data]
        
        stats = {
            'n_layers': len(cortex_layer_data),
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
                'std': float(np.std(r2_values))
            }
        }
        
        # Add perfusion if available
        if 'perfusion' in cortex_layer_data[0]:
            perf_values = [l['perfusion']['median'] for l in cortex_layer_data if 'perfusion' in l]
            if perf_values:
                stats['perfusion'] = {
                    'mean': float(np.mean(perf_values)),
                    'median': float(np.median(perf_values)),
                    'std': float(np.std(perf_values))
                }
        
        # Tissue quality
        viable_pcts = [l['tissue_quality']['viable_pct'] for l in cortex_layer_data]
        stats['tissue_quality'] = {
            'mean_viable_pct': float(np.mean(viable_pcts)),
            'interpretation': 'healthy' if np.mean(viable_pcts) > 90 else
                            'mild_damage' if np.mean(viable_pcts) > 75 else
                            'moderate_damage'
        }
        
        cortex_stats[condition] = stats
    
    return cortex_stats


def analyze_sample(data: dict, n_layers: int = 24, output_dir: Path = None):
    """
    Complete analysis for one sample
    
    Parameters:
    -----------
    data : dict
        Loaded sample data
    n_layers : int
        Number of layers per organ
    output_dir : Path
        Output directory
    
    Returns:
    --------
    results : dict
        Complete analysis results
    """
    sample_id = data['id']
    mlco_mask = data['mlco_mask']
    
    print(f"\n{'='*70}")
    print(f"Analyzing: {sample_id}")
    print(f"{'='*70}")
    
    # Analyze each condition
    condition_results = {}
    
    for condition, scan_data in data['scans'].items():
        print(f"\nAnalyzing condition: {condition}")
        
        # Extract maps (loaded data has 't2_map' and 'r2_map' keys)
        t2_map = scan_data['t2_map']
        r2_map = scan_data['r2_map']
        
        # Get perfusion map if available
        perfusion_map = data.get('perfusion_map', None)
        
        # Load zone config if not already loaded
        # This is needed for multi-region analysis
        zone_config = None
        try:
            from tissue_zones import load_zone_config, ZONE_CONFIG
            zone_config = ZONE_CONFIG  # Use currently loaded config
        except:
            pass
        
        # Run enhanced MLCO analysis (auto-detects format)
        results = analyze_mlco(
            t2_map=t2_map,
            r2_map=r2_map,
            mlco_mask=mlco_mask,
            n_layers_per_organ=n_layers,
            perfusion_map=perfusion_map,
            scan_label=f"{sample_id}_{condition}",
            zone_config=zone_config  # NEW: Pass zone_config for multi-region support
        )
        
        # Calculate additional metrics
        print(f"\nCalculating enhanced metrics...")
        
        # Tissue heterogeneity (only for standard format)
        if results.get('mode') != 'multi_region':
            heterogeneity = calculate_tissue_heterogeneity(results['bilateral']['layers'])
            results['heterogeneity'] = heterogeneity
        else:
            print("  Skipping heterogeneity (multi-region mode)")
        
        condition_results[condition] = results
    
    # Calculate oxygen responsiveness (compare conditions)
    # Only for standard format - multi-region needs different implementation
    oxygen_response = None
    cortex_stats = None
    
    # Check if any condition is multi-region
    is_multi_region = any(
        cond_result.get('mode') == 'multi_region' 
        for cond_result in condition_results.values()
    )
    
    if not is_multi_region:
        print(f"\nCalculating oxygen responsiveness...")
        oxygen_response = calculate_oxygen_responsiveness(condition_results)
        
        # Extract cortex-only statistics
        print(f"\nExtracting cortex-only statistics...")
        cortex_stats = extract_cortex_only_statistics(condition_results)
    else:
        print(f"\nSkipping oxygen responsiveness (multi-region mode)")
        print(f"Skipping cortex-only statistics (multi-region mode)")
    
    # Compile complete results
    complete_results = {
        'sample_id': sample_id,
        'n_layers_per_organ': n_layers,
        'analysis_date': datetime.now().isoformat(),
        'conditions': condition_results
    }
    
    # Add optional metrics if calculated
    if oxygen_response is not None:
        complete_results['oxygen_responsiveness'] = oxygen_response
    if cortex_stats is not None:
        complete_results['cortex_only_statistics'] = cortex_stats
    
    # Generate plots if output directory specified
    if output_dir and not is_multi_region:
        print(f"\n{'='*70}")
        print(f"GENERATING PLOTS")
        print(f"{'='*70}")
        
        # Get list of conditions
        scan_conditions = list(condition_results.keys())
        
        # 1. Main MLCO profile plot (all conditions, includes perfusion)
        plot_path = output_dir / f"{sample_id}_tlco_profiles"
        plot_mlco_profile(
            results=condition_results,
            scan_conditions=scan_conditions,
            output_path=plot_path,
            animal_label=sample_id
        )
        print(f"  ✓ Saved profile plots: {plot_path.name}.*")
        
        # Check if any condition has perfusion data
        has_perfusion = any(
            condition_results[cond].get('bilateral', {}).get('layers', [{}])[0].get('perfusion') is not None
            for cond in scan_conditions
        )
        
        if has_perfusion:
            # 2. Dedicated perfusion profile plot
            perf_plot_path = output_dir / f"{sample_id}_perfusion_profile"
            plot_perfusion_profile(
                results=condition_results,
                scan_conditions=scan_conditions,
                output_path=perf_plot_path,
                animal_label=sample_id
            )
            print(f"  ✓ Saved perfusion plot: {perf_plot_path.name}.*")
            
            # 3. T2* vs Perfusion scatter plot (for each condition)
            for condition in scan_conditions:
                scatter_path = output_dir / f"{sample_id}_{condition}_t2star_perfusion_scatter"
                plot_t2star_perfusion_scatter(
                    results=condition_results[condition],  # Single condition results
                    output_path=scatter_path,
                    animal_label=f"{sample_id}_{condition}"
                )
            print(f"  ✓ Saved scatter plots for {len(scan_conditions)} conditions")
            
            # 4. Triple overlay (for each condition)
            for condition in scan_conditions:
                overlay_path = output_dir / f"{sample_id}_{condition}_triple_overlay"
                plot_triple_overlay(
                    results=condition_results,  # Full results dict
                    condition=condition,  # Specify which condition
                    output_path=overlay_path,
                    animal_label=sample_id
                )
            print(f"  ✓ Saved triple overlay plots for {len(scan_conditions)} conditions")
    
    elif output_dir and is_multi_region:
        print(f"\n{'='*70}")
        print(f"GENERATING MULTI-REGION PLOTS (Phase 5)")
        print(f"{'='*70}")
        
        if not MULTIREGION_PLOTTING_AVAILABLE:
            print("  ✗ Multi-region plotting module not available")
            print("  Please ensure boldpy_plots.py is in the same directory")
        else:
            # Generate plots for each condition
            for condition in condition_results.keys():
                cond_result = condition_results[condition]
                
                if cond_result.get('mode') != 'multi_region':
                    continue
                
                print(f"\nGenerating plots for condition: {condition}")
                
                # Create condition-specific output directory
                condition_dir = output_dir / condition
                
                # Generate all multi-region plots (Phase 1 style)
                scan_label = f"{sample_id}_{condition}"
                generate_all_multiregion_plots(
                    results=cond_result,
                    output_dir=condition_dir,
                    scan_label=scan_label
                )
                
                # Generate continuous whole-kidney plot (NEW!)
                try:
                    from boldpy_plots import plot_whole_kidney_continuous
                    from mlco_analysis import average_bilateral_regions
                    
                    print(f"  Generating continuous profile...")
                    
                    # Bilateral averaging for continuous plot
                    averaged = average_bilateral_regions(cond_result)
                    
                    # Debug: Check structure and identify missing layers
                    print(f"    Regions: {list(averaged.keys())}")
                    total_expected = 0
                    total_actual = 0
                    for region in averaged.keys():
                        n_layers = averaged[region]['n_layers']
                        actual_layers = averaged[region]['layers']
                        actual = len(actual_layers)
                        
                        # Check which layers exist
                        existing_layers = [layer['layer_number'] for layer in actual_layers]
                        missing_layers = [i for i in range(1, n_layers + 1) if i not in existing_layers]
                        
                        status = "✓" if actual == n_layers else f"⚠ missing {len(missing_layers)} layer(s)"
                        print(f"    {region}: expected={n_layers}, actual={actual} {status}")
                        if missing_layers:
                            print(f"      Missing layers: {missing_layers} (will be shown as gaps)")
                        
                        total_expected += n_layers
                        total_actual += actual
                    
                    print(f"    Total: {total_actual}/{total_expected} layers have data")
                    
                    continuous_path = condition_dir / f"{sample_id}_{condition}_continuous"
                    plot_whole_kidney_continuous(
                        averaged_results=averaged,
                        condition=condition,
                        output_path=continuous_path,
                        sample_id=sample_id,
                        include_perfusion=True,
                        t2_threshold=40.0,
                        perfusion_threshold=200.0
                    )
                    print(f"  ✓ Saved continuous profile: {continuous_path.name}.png/svg")
                    
                except Exception as e:
                    print(f"  ⚠ Could not generate continuous plot: {e}")
                    import traceback
                    print("  Full error:")
                    traceback.print_exc()
            
            print(f"\n✓ All multi-region plots generated")
    
    # Save JSON
    if output_dir:
        json_path = output_dir / f"{sample_id}_complete_analysis.json"
        with open(json_path, 'w') as f:
            json.dump(complete_results, f, indent=2)
        print(f"\n✓ Saved results: {json_path.name}")
    
    return complete_results


def compare_groups(group1_results: dict, group2_results: dict, output_dir: Path):
    """
    Compare group 1 vs group 2 with enhanced statistics
    
    Parameters:
    -----------
    group1_results : dict
        Group 1 analysis results (e.g., WT, control)
    group2_results : dict
        Group 2 analysis results (e.g., KO, treatment)
    output_dir : Path
        Output directory
    """
    print(f"\n{'='*70}")
    print(f"Comparing {group1_results['sample_id']} vs {group2_results['sample_id']}")
    print(f"{'='*70}")
    
    # Check if both groups have cortex_only_statistics (not available in multi-region mode)
    has_cortex_stats = (
        'cortex_only_statistics' in group1_results and 
        'cortex_only_statistics' in group2_results and
        group1_results['cortex_only_statistics'] is not None and
        group2_results['cortex_only_statistics'] is not None
    )
    
    comparison = {
        'group1_id': group1_results['sample_id'],
        'group2_id': group2_results['sample_id'],
        'cortex_comparison': {},
        'full_organ_comparison': {},
        'tissue_quality_comparison': {}
    }
    
    # Compare cortex-only statistics (if available - not in multi-region mode)
    if has_cortex_stats:
        for condition in group1_results['cortex_only_statistics'].keys():
            if condition in group2_results['cortex_only_statistics']:
                g1_cortex = group1_results['cortex_only_statistics'][condition]
                g2_cortex = group2_results['cortex_only_statistics'][condition]
                
                # T2* comparison
                g1_t2 = g1_cortex['t2star']['mean']
                g2_t2 = g2_cortex['t2star']['mean']
                delta_t2 = g2_t2 - g1_t2
                pct_change = (delta_t2 / g1_t2) * 100
                
                effect_size = calculate_effect_size(
                    g1_t2, g1_cortex['t2star']['std'],
                    g2_t2, g2_cortex['t2star']['std']
                )
                
                comparison['cortex_comparison'][condition] = {
                    't2star': {
                        'group1_mean': g1_t2,
                        'group2_mean': g2_t2,
                        'delta': delta_t2,
                        'percent_change': pct_change,
                        'effect_size': effect_size,
                        'interpretation': 'large_effect' if abs(effect_size) > 0.8 else
                                        'medium_effect' if abs(effect_size) > 0.5 else
                                        'small_effect'
                    }
                }
                
                # Perfusion comparison if available
                if 'perfusion' in g1_cortex and 'perfusion' in g2_cortex:
                    g1_perf = g1_cortex['perfusion']['mean']
                    g2_perf = g2_cortex['perfusion']['mean']
                    delta_perf = g2_perf - g1_perf
                    pct_change_perf = (delta_perf / g1_perf) * 100
                    
                    comparison['cortex_comparison'][condition]['perfusion'] = {
                        'group1_mean': g1_perf,
                        'group2_mean': g2_perf,
                        'delta': delta_perf,
                        'percent_change': pct_change_perf,
                        'interpretation': 'severely_reduced' if pct_change_perf < -25 else
                                        'moderately_reduced' if pct_change_perf < -10 else
                                        'mildly_reduced' if pct_change_perf < 0 else
                                        'normal_or_elevated'
                    }
                
                print(f"\n{condition.upper()} - Cortex Comparison:")
                print(f"  T2*:       Group1={g1_t2:.1f}ms, Group2={g2_t2:.1f}ms, Δ={delta_t2:+.1f}ms ({pct_change:+.0f}%)")
                print(f"  Effect size: {effect_size:.2f}")
                if 'perfusion' in comparison['cortex_comparison'][condition]:
                    perf_comp = comparison['cortex_comparison'][condition]['perfusion']
                    print(f"  Perfusion: Group1={perf_comp['group1_mean']:.0f}, Group2={perf_comp['group2_mean']:.0f}, "
                          f"Δ={perf_comp['delta']:+.0f} ({perf_comp['percent_change']:+.0f}%)")
    else:
        print("\nCortex-only comparison skipped (multi-region mode or statistics not available)")
    
    # Check if multi-region mode
    is_multi_region = any(
        cond_result.get('mode') == 'multi_region'
        for cond_result in group1_results.get('conditions', {}).values()
    )
    
    # Generate comparison plots
    if not is_multi_region:
        # Standard bilateral comparison plots
        print(f"\n{'='*70}")
        print(f"GENERATING COMPARISON PLOTS")
        print(f"{'='*70}")
        
        # Plot comparison for each condition
        for condition in group1_results['conditions'].keys():
            if condition in group2_results['conditions']:
                print(f"\nPlotting {condition} comparison...")
                
                # Extract single condition data for each group
                g1_condition_data = group1_results['conditions'][condition]
                g2_condition_data = group2_results['conditions'][condition]
                
                # Group 1 vs Group 2 comparison plot
                comparison_plot_path = output_dir / f"group1_vs_group2_{condition}_comparison"
                plot_mlco_comparison(
                    wt_results=g1_condition_data,  # plot function still uses wt/ko parameter names
                    ko_results=g2_condition_data,
                    condition=condition,
                    output_path=comparison_plot_path
                )
                print(f"  ✓ Saved comparison plot: {comparison_plot_path.name}.*")
    else:
        # Multi-region comparison plots
        print(f"\n{'='*70}")
        print(f"MULTI-REGION COMPARISON PLOTS")
        print(f"{'='*70}")
        
        if not PHASE2_PLOTTING_AVAILABLE:
            print("  Phase 2 plotting functions not available.")
            print("  JSON comparison data has been saved.")
        else:
            print("\nPreparing multi-region comparison data...")
            
            # Bilaterally average ALL conditions for both groups
            g1_averaged_all = {}
            g2_averaged_all = {}
            
            for condition in group1_results['conditions'].keys():
                if condition in group2_results['conditions']:
                    print(f"  Averaging {condition} for both groups...")
                    
                    g1_raw = group1_results['conditions'][condition]
                    g2_raw = group2_results['conditions'][condition]
                    
                    g1_averaged_all[condition] = average_bilateral_regions(g1_raw)
                    g2_averaged_all[condition] = average_bilateral_regions(g2_raw)
            
            # Generate single comprehensive comparison plot (3×3 grid)
            comparison_plot_path = output_dir / "group1_vs_group2_multiregion_comparison"
            
            print(f"\nGenerating comprehensive multi-region comparison...")
            print(f"  Layout: 3 regions (rows) × {len(g1_averaged_all)} conditions (columns)")
            
            try:
                plot_group_comparison_overview(
                    group1_averaged=g1_averaged_all,
                    group2_averaged=g2_averaged_all,
                    group1_name='Group 1 (WT)',
                    group2_name='Group 2 (KO)',
                    output_path=comparison_plot_path
                )
                print(f"  ✓ Saved: {comparison_plot_path.name}.png/svg")
                
                print(f"\n{'='*70}")
                print(f"✓ MULTI-REGION COMPARISON COMPLETE!")
                print(f"{'='*70}")
                print(f"\nGenerated plot shows:")
                print(f"  • 3×{len(g1_averaged_all)} grid (cortex/medulla/papilla × conditions)")
                print(f"  • Group 1 (solid lines, circles) vs Group 2 (dashed lines, squares)")
                print(f"  • T2* profiles with zone shading")
                print(f"  • All conditions: {', '.join(g1_averaged_all.keys())}")
                
            except Exception as e:
                print(f"  ✗ Error generating comparison plot: {e}")
                import traceback
                traceback.print_exc()
            
            # Generate continuous whole-kidney comparisons (NEW!)
            print(f"\n{'='*70}")
            print(f"CONTINUOUS WHOLE-KIDNEY COMPARISONS")
            print(f"{'='*70}")
            
            try:
                from boldpy_plots import plot_whole_kidney_comparison
                
                for condition in g1_averaged_all.keys():
                    print(f"\n  Generating continuous comparison for {condition}...")
                    
                    continuous_path = output_dir / f"group1_vs_group2_{condition}_continuous"
                    
                    plot_whole_kidney_comparison(
                        group1_averaged=g1_averaged_all[condition],
                        group2_averaged=g2_averaged_all[condition],
                        condition=condition,
                        output_path=continuous_path,
                        group1_name='Group 1 (WT)',
                        group2_name='Group 2 (KO)',
                        include_perfusion=True,
                        t2_threshold=40.0,
                        perfusion_threshold=200.0
                    )
                    print(f"    ✓ Saved: {continuous_path.name}.png/svg")
                
                print(f"\n✓ Continuous whole-kidney comparisons complete!")
                print(f"\n  Generated {len(g1_averaged_all)} continuous comparison plots:")
                print(f"    • Shows entire kidney: cortex → medulla → papilla")
                print(f"    • Includes T2* comparison panel")
                print(f"    • Includes perfusion comparison panel")
                print(f"    • Tissue viability thresholds marked")
                
            except Exception as e:
                print(f"  ✗ Error generating continuous plots: {e}")
                import traceback
                traceback.print_exc()
    
    # Save comparison
    json_path = output_dir / "group1_vs_group2_comparison.json"
    with open(json_path, 'w') as f:
        json.dump(comparison, f, indent=2)
    print(f"\n✓ Saved comparison: {json_path.name}")
    
    return comparison


def main():
    parser = argparse.ArgumentParser(
        description='Enhanced MLCO Analysis with Perfusion Integration',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example Usage:

  # Analyze single sample
  python boldpy_analyze.py \\
      --config m1_config.json \\
      --n-layers 24 \\
      --output-dir results/M1/
  
  # Compare two groups
  python boldpy_analyze.py \\
      --group1-config wt_config.json \\
      --group2-config ko_config.json \\
      --compare \\
      --n-layers 24 \\
      --output-dir results/comparison/

Config JSON format:
{
  "id": "M1_WT",
  "t2star_maps": {
    "air": "path/M1_prepared/M1_air_t2star_custom.npy",
    "oxygen_1": "path/M1_prepared/M1_oxygen1_t2star_custom.npy",
    "oxygen_2": "path/M1_prepared/M1_oxygen2_t2star_custom.npy"
  },
  "r2star_maps": {
    "air": "path/M1_prepared/M1_air_r2star_custom.npy",
    "oxygen_1": "path/M1_prepared/M1_oxygen1_r2star_custom.npy",
    "oxygen_2": "path/M1_prepared/M1_oxygen2_r2star_custom.npy"
  },
  "perfusion_map": "path/M1_prepared/M1_perfusion.npy",
  "mlco_mask": "path/M1_mlco/M1_mlco_layers.npy"
}

Note: Use prepare_data.py to generate t2star_maps, r2star_maps, and perfusion_map
      from PvDatasets files before running analysis.
        """
    )
    
    # Config file mode
    parser.add_argument('--config', type=Path, help='Sample configuration JSON')
    
    # Comparison mode
    parser.add_argument('--group1-config', type=Path, help='Group 1 config JSON (e.g., WT, control)')
    parser.add_argument('--group2-config', type=Path, help='Group 2 config JSON (e.g., KO, treatment)')
    parser.add_argument('--compare', action='store_true', help='Compare group 1 vs group 2')
    
    # Common parameters
    parser.add_argument('--n-layers', type=int, default=24, help='Layers per organ (default: 24)')
    parser.add_argument('--zone-config', type=Path, help='Zone configuration YAML (default: configs/zones/kidney_24layer.yaml)')
    parser.add_argument('--threshold-config', type=Path, help='Threshold configuration YAML (default: configs/thresholds/kidney_mouse_default.yaml)')
    parser.add_argument('--output-dir', type=Path, required=True, help='Output directory')
    
    args = parser.parse_args()
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("ENHANCED MLCO ANALYSIS")
    print("="*70)
    print(f"Layers per organ: {args.n_layers}")
    print(f"Output directory: {args.output_dir}")
    
    # Load custom configs if provided
    if args.zone_config or args.threshold_config:
        print(f"\nLoading custom configurations:")
        if args.zone_config:
            print(f"  Zone config: {args.zone_config}")
        if args.threshold_config:
            print(f"  Threshold config: {args.threshold_config}")
        update_configs(
            zone_config_path=args.zone_config,
            threshold_config_path=args.threshold_config
        )
        print("  ✓ Configs updated")
    else:
        print("\nUsing default configurations:")
        print("  Zone config: configs/zones/kidney_24layer.yaml")
        print("  Threshold config: configs/thresholds/kidney_mouse_default.yaml")
    
    print("="*70)
    
    # Determine mode
    if args.compare and args.group1_config and args.group2_config:
        # Comparison mode
        print("\nMode: Group 1 vs Group 2 Comparison")
        
        # Load configs
        with open(args.group1_config) as f:
            group1_config = json.load(f)
        with open(args.group2_config) as f:
            group2_config = json.load(f)
        
        # Load and analyze Group 1
        group1_data = load_data(group1_config)
        group1_results = analyze_sample(group1_data, args.n_layers, args.output_dir)
        
        # Load and analyze Group 2
        group2_data = load_data(group2_config)
        group2_results = analyze_sample(group2_data, args.n_layers, args.output_dir)
        
        # Compare
        comparison = compare_groups(group1_results, group2_results, args.output_dir)
        
        # Check if multi-region mode (already handled inside compare_groups)
        is_multi_region = any(
            cond_result.get('mode') == 'multi_region'
            for cond_result in group1_results.get('conditions', {}).values()
        )
        
        # Generate additional comparison plots (only for standard bilateral mode)
        # Note: compare_groups() already handles plotting, but this section is kept for legacy support
        if not is_multi_region:
            print("\nGenerating comparison plots...")
            for condition in group1_results['conditions'].keys():
                if condition in group2_results['conditions']:
                    output_path = args.output_dir / f"group1_vs_group2_{condition}_comparison.png"
                    plot_mlco_comparison(
                        group1_results['conditions'][condition],
                        group2_results['conditions'][condition],
                        condition,
                        output_path
                    )
    
    elif args.config:
        # Config file mode
        print("\nMode: Single Sample Analysis (Config File)")
        
        with open(args.config) as f:
            config = json.load(f)
        
        data = load_data(config)
        results = analyze_sample(data, args.n_layers, args.output_dir)
        
        # Check if multi-region mode
        is_multi_region = any(
            cond_result.get('mode') == 'multi_region'
            for cond_result in results.get('conditions', {}).values()
        )
        
        # Generate plots (only for standard bilateral mode)
        if not is_multi_region:
            print("\nGenerating plots...")
            scan_conditions = list(results['conditions'].keys())
            
            output_path = args.output_dir / f"{config['id']}_tlco_profiles.png"
            plot_mlco_profile(
                results['conditions'],
                scan_conditions,
                output_path,
                config['id']
            )
        else:
            print("\nPlots skipped (multi-region mode - handled in analyze_sample)")
    
    else:
        print("\nERROR: Invalid arguments.")
        print("  Use --config for single sample analysis")
        print("  Or use --group1-config and --group2-config with --compare for group comparison")
        parser.print_help()
        sys.exit(1)
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE!")
    print("="*70)
    print(f"\nResults saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
