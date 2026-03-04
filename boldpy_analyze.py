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
    
    # Zonal responsiveness — use dynamic zone config from tissue_zones
    try:
        from tissue_zones import ZONE_CONFIG, AGGREGATE_ZONES
        if AGGREGATE_ZONES:
            # Use aggregate zones (total_cortex, total_medulla) if available
            zones = {name: layers for name, layers in AGGREGATE_ZONES.items()
                     if name != 'all_zones'}
        else:
            # Fall back to primary zones from config
            zones = {name: info['layers'] for name, info in ZONE_CONFIG['zones'].items()}
    except ImportError:
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


def extract_superficial_zone_statistics(results: dict) -> dict:
    """
    Extract statistics for the most superficial zone.

    Resolves the superficial layers from aggregate zones using positional
    logic: 'total_superficial' (clustered configs) or 'total_cortex'
    (hardcoded anatomical configs), falling back to the first zone in
    ZONE_CONFIG.

    Parameters:
    -----------
    results : dict
        Full MLCO results

    Returns:
    --------
    superficial_stats : dict
        Superficial-zone statistics
    """
    # Determine superficial layers dynamically from zone config
    try:
        from tissue_zones import AGGREGATE_ZONES, ZONE_CONFIG
        if 'total_superficial' in AGGREGATE_ZONES:
            superficial_layers = AGGREGATE_ZONES['total_superficial']
        elif 'total_cortex' in AGGREGATE_ZONES:
            superficial_layers = AGGREGATE_ZONES['total_cortex']
        else:
            # Use the first zone (shallowest)
            first_zone = next(iter(ZONE_CONFIG['zones'].values()))
            superficial_layers = first_zone['layers']
    except (ImportError, StopIteration):
        superficial_layers = range(1, 11)
    
    superficial_stats = {}

    for condition, condition_data in results.items():
        bilateral = condition_data['bilateral']

        # Filter to superficial zone layers
        zone_layer_data = [l for l in bilateral['layers'] if l['layer'] in superficial_layers]

        if not zone_layer_data:
            continue

        # Calculate statistics
        t2_values = [l['t2star']['median'] for l in zone_layer_data]
        r2_values = [l['r2star']['median'] for l in zone_layer_data]

        stats = {
            'n_layers': len(zone_layer_data),
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
        if 'perfusion' in zone_layer_data[0]:
            perf_values = [l['perfusion']['median'] for l in zone_layer_data if 'perfusion' in l]
            if perf_values:
                stats['perfusion'] = {
                    'mean': float(np.mean(perf_values)),
                    'median': float(np.median(perf_values)),
                    'std': float(np.std(perf_values))
                }

        # Tissue quality
        viable_pcts = [l['tissue_quality']['viable_pct'] for l in zone_layer_data]
        stats['tissue_quality'] = {
            'mean_viable_pct': float(np.mean(viable_pcts)),
            'interpretation': 'healthy' if np.mean(viable_pcts) > 90 else
                            'mild_damage' if np.mean(viable_pcts) > 75 else
                            'moderate_damage'
        }

        superficial_stats[condition] = stats

    return superficial_stats


def analyze_sample(data: dict, n_layers: int = 24, output_dir: Path = None,
                   cluster_args: dict = None, zone_config_override: dict = None):
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
    cluster_args : dict, optional
        If provided, enables k-means clustering for zone boundaries.
        Keys: n_clusters (int), method (str), condition (str or None),
              save_config_path (Path or None).
    zone_config_override : dict, optional
        If provided, apply these zone boundaries directly (Workflow A:
        shared reference). Skips clustering. Must be a valid zone config
        dict (same format as load_zone_config() output).

    Returns:
    --------
    results : dict
        Complete analysis results (includes 'zone_config' key)
    """
    sample_id = data['id']
    mlco_mask = data['mlco_mask']

    print(f"\n{'='*70}")
    print(f"Analyzing: {sample_id}")
    print(f"{'='*70}")

    # --- Workflow A: Apply shared reference zone boundaries ---
    if zone_config_override is not None:
        from tissue_zones import update_configs_from_dict
        print(f"\n  Applying shared reference zone config (Workflow A)")
        update_configs_from_dict(zone_config_override)
        for zname, zinfo in zone_config_override['zones'].items():
            print(f"    {zname}: layers {zinfo['layers']}")
        print(f"  Zone config applied for downstream analysis.")

    # --- Workflow B / default: K-means clustering for data-driven zone boundaries ---
    elif cluster_args is not None:
        try:
            from cluster_zones import cluster_and_build_zones, compare_zone_configs, plot_clustering_diagnostics
            from tissue_zones import update_configs_from_dict, ZONE_CONFIG

            # Pick condition to cluster on
            condition_name = cluster_args.get('condition')
            if condition_name is None:
                condition_name = next(iter(data['scans']))

            if condition_name not in data['scans']:
                print(f"  WARNING: Cluster condition '{condition_name}' not found. "
                      f"Using '{next(iter(data['scans']))}'.")
                condition_name = next(iter(data['scans']))

            scan_data = data['scans'][condition_name]
            print(f"\n  Running k-means clustering on condition: {condition_name}")
            print(f"  Clusters: {cluster_args.get('n_clusters', 3)}, "
                  f"Method: {cluster_args.get('method', 'kmeans')}")

            zone_config, diagnostics = cluster_and_build_zones(
                t2_map=scan_data['t2_map'],
                r2_map=scan_data['r2_map'],
                mlco_mask=mlco_mask,
                n_layers=n_layers,
                n_clusters=cluster_args.get('n_clusters', 3),
                method=cluster_args.get('method', 'kmeans'),
                perfusion_map=data.get('perfusion_map'),
            )

            sil = zone_config['metadata'].get('silhouette_score', 0)
            print(f"  Silhouette score: {sil:.3f}")

            # Show zone assignments
            for zname, zlayers in zone_config['zones'].items():
                print(f"    {zname}: layers {zlayers['layers']}")

            # Compare with reference config
            ref_config = ZONE_CONFIG
            comparison = compare_zone_configs(zone_config, ref_config)
            print(f"\n  Zone comparison vs reference:")
            for zname, info in comparison.items():
                print(f"    {zname}: Jaccard={info['jaccard']:.2f}, "
                      f"shift={info.get('boundary_shift', {})}")

            # Inject into module globals so all downstream code uses new zones
            update_configs_from_dict(zone_config)
            print(f"  Zone config updated for downstream analysis.")

            # Save diagnostic plot
            if output_dir is not None:
                diag_path = output_dir / f"{sample_id}_cluster_diagnostics.png"
                cluster_info = diagnostics['cluster_info']
                plot_clustering_diagnostics(
                    layer_features=diagnostics['layer_features'],
                    labels=cluster_info['labels'],
                    assignments=diagnostics['tissue_assignments'],
                    centroids=cluster_info['centroids'],
                    feature_names=cluster_info['feature_names'],
                    reference_config=ref_config,
                    output_path=diag_path,
                )

            # Save cluster config as YAML if requested
            save_path = cluster_args.get('save_config_path')
            if save_path is not None:
                import yaml
                save_path = Path(save_path)
                with open(save_path, 'w') as f:
                    yaml.dump(zone_config, f, default_flow_style=False, sort_keys=False)
                print(f"  Saved cluster config: {save_path}")

        except ImportError as e:
            print(f"  WARNING: Clustering unavailable ({e}). Using default zone config.")
        except Exception as e:
            print(f"  WARNING: Clustering failed ({e}). Using default zone config.")

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
    superficial_stats = None
    
    # Check if any condition is multi-region
    is_multi_region = any(
        cond_result.get('mode') == 'multi_region' 
        for cond_result in condition_results.values()
    )
    
    if not is_multi_region:
        print(f"\nCalculating oxygen responsiveness...")
        oxygen_response = calculate_oxygen_responsiveness(condition_results)
        
        # Extract superficial zone statistics
        print(f"\nExtracting superficial zone statistics...")
        superficial_stats = extract_superficial_zone_statistics(condition_results)
    else:
        print(f"\nSkipping oxygen responsiveness (multi-region mode)")
        print(f"Skipping superficial zone statistics (multi-region mode)")
    
    # Compile complete results
    complete_results = {
        'sample_id': sample_id,
        'n_layers_per_organ': n_layers,
        'analysis_date': datetime.now().isoformat(),
        'conditions': condition_results
    }

    # Store the active zone config so downstream comparison can detect boundaries
    try:
        from tissue_zones import ZONE_CONFIG
        # Deep copy the config to avoid mutation — convert ranges to lists for JSON
        import copy
        stored_zc = copy.deepcopy(ZONE_CONFIG)
        # Ensure layers are plain lists (not range objects) for JSON serialization
        if 'zones' in stored_zc:
            for zinfo in stored_zc['zones'].values():
                if 'layers' in zinfo and not isinstance(zinfo['layers'], list):
                    zinfo['layers'] = list(zinfo['layers'])
        if 'aggregate_zones' in stored_zc:
            for ainfo in stored_zc['aggregate_zones'].values():
                if 'layers' in ainfo and not isinstance(ainfo['layers'], list):
                    ainfo['layers'] = list(ainfo['layers'])
        complete_results['zone_config'] = stored_zc
    except ImportError:
        pass

    # Add optional metrics if calculated
    if oxygen_response is not None:
        complete_results['oxygen_responsiveness'] = oxygen_response
    if superficial_stats is not None:
        complete_results['superficial_zone_statistics'] = superficial_stats
    
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
                        # Use get() with index fallback for compatibility with averaged data
                        existing_layers = [layer.get('layer_number', idx+1) for idx, layer in enumerate(actual_layers)]
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


def _zone_boundaries_match(zc1: dict, zc2: dict) -> bool:
    """
    Check whether two zone configs have identical zone-to-layer mappings.

    Returns True if both are None (no clustering), or both have the same
    zone names with the same layer sets. Returns False otherwise.
    """
    if zc1 is None and zc2 is None:
        return True
    if zc1 is None or zc2 is None:
        return False

    zones1 = zc1.get('zones', {})
    zones2 = zc2.get('zones', {})

    if set(zones1.keys()) != set(zones2.keys()):
        return False

    for zname in zones1:
        layers1 = set(zones1[zname].get('layers', []))
        layers2 = set(zones2[zname].get('layers', []))
        if layers1 != layers2:
            return False

    return True


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

    # --- Zone boundary compatibility detection ---
    g1_zc = group1_results.get('zone_config')
    g2_zc = group2_results.get('zone_config')
    boundaries_match = _zone_boundaries_match(g1_zc, g2_zc)
    boundary_comparison = None

    if g1_zc is not None and g2_zc is not None:
        if boundaries_match:
            print("\nZone boundaries: MATCH (Workflow A — shared reference)")
            print("  Zone-level statistics are directly comparable.")
        else:
            print("\nZone boundaries: DIFFER (Workflow B — per-sample clustering)")
            print("  Zone-level stats are NOT directly comparable.")
            print("  Computing boundary comparison instead...")
            try:
                from cluster_zones import compare_zone_configs
                boundary_comparison = compare_zone_configs(g1_zc, g2_zc)
                print(f"\n  {'Zone':<20s} {'Jaccard':>8s} {'Lower shift':>12s} {'Upper shift':>12s}")
                print(f"  {'-'*52}")
                for zname, info in boundary_comparison.items():
                    jac = f"{info['jaccard']:.2f}"
                    lower = info.get('boundary_shift', {}).get('lower', '-')
                    upper = info.get('boundary_shift', {}).get('upper', '-')
                    lower_str = f"{lower:+d}" if isinstance(lower, int) else str(lower)
                    upper_str = f"{upper:+d}" if isinstance(upper, int) else str(upper)
                    print(f"  {zname:<20s} {jac:>8s} {lower_str:>12s} {upper_str:>12s}")
            except ImportError:
                print("  WARNING: cluster_zones not available for boundary comparison")

    # Check if both groups have superficial zone statistics (not available in multi-region mode)
    # Support both new ('superficial_zone_statistics') and legacy ('cortex_only_statistics') keys
    sup_key_1 = 'superficial_zone_statistics' if 'superficial_zone_statistics' in group1_results else 'cortex_only_statistics'
    sup_key_2 = 'superficial_zone_statistics' if 'superficial_zone_statistics' in group2_results else 'cortex_only_statistics'
    has_superficial_stats = (
        sup_key_1 in group1_results and
        sup_key_2 in group2_results and
        group1_results[sup_key_1] is not None and
        group2_results[sup_key_2] is not None
    )

    comparison = {
        'group1_id': group1_results['sample_id'],
        'group2_id': group2_results['sample_id'],
        'boundaries_match': boundaries_match,
        'superficial_zone_comparison': {},
        'full_organ_comparison': {},
        'tissue_quality_comparison': {}
    }
    if boundary_comparison is not None:
        comparison['boundary_comparison'] = boundary_comparison

    # Compare superficial zone statistics (if available - not in multi-region mode)
    if has_superficial_stats:
        for condition in group1_results[sup_key_1].keys():
            if condition in group2_results[sup_key_2]:
                g1_cortex = group1_results[sup_key_1][condition]
                g2_cortex = group2_results[sup_key_2][condition]
                
                # T2* comparison
                g1_t2 = g1_cortex['t2star']['mean']
                g2_t2 = g2_cortex['t2star']['mean']
                delta_t2 = g2_t2 - g1_t2
                pct_change = (delta_t2 / g1_t2) * 100
                
                effect_size = calculate_effect_size(
                    g1_t2, g1_cortex['t2star']['std'],
                    g2_t2, g2_cortex['t2star']['std']
                )
                
                comparison['superficial_zone_comparison'][condition] = {
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
                    
                    comparison['superficial_zone_comparison'][condition]['perfusion'] = {
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
                if 'perfusion' in comparison['superficial_zone_comparison'][condition]:
                    perf_comp = comparison['superficial_zone_comparison'][condition]['perfusion']
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
                    output_path=comparison_plot_path,
                    wt_zone_config=g1_zc,
                    ko_zone_config=g2_zc
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

    # K-means clustering for data-driven zone boundaries
    cluster_group = parser.add_argument_group(
        'clustering', 'Data-driven zone boundaries via k-means clustering'
    )
    cluster_group.add_argument(
        '--cluster-zones', action='store_true',
        help='Enable k-means clustering to determine zone boundaries from the data'
    )
    cluster_group.add_argument(
        '--n-clusters', type=int, default=3,
        help='Number of tissue clusters (default: 3 = cortex/medulla/papilla; 5 = 5-zone)'
    )
    cluster_group.add_argument(
        '--cluster-method', choices=['kmeans', 'gmm'], default='kmeans',
        help='Clustering method (default: kmeans)'
    )
    cluster_group.add_argument(
        '--cluster-condition', type=str, default=None,
        help='Which condition to cluster on (default: first condition in config)'
    )
    cluster_group.add_argument(
        '--save-cluster-config', type=Path, default=None,
        help='Save clustered zone config as YAML for reuse with --zone-config'
    )
    cluster_group.add_argument(
        '--cluster-reference', type=Path, default=None,
        help='Load a saved clustered YAML and apply to ALL samples (Workflow A: '
             'shared reference boundaries). Mutually exclusive with --cluster-zones.'
    )

    args = parser.parse_args()

    # Validate mutual exclusion: --cluster-zones and --cluster-reference
    if args.cluster_zones and args.cluster_reference:
        parser.error("--cluster-zones and --cluster-reference are mutually exclusive.\n"
                     "  Use --cluster-zones for per-sample clustering (Workflow B)\n"
                     "  Use --cluster-reference for shared reference boundaries (Workflow A)")

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
    
    # Build cluster_args if clustering enabled
    cluster_args = None
    zone_config_override = None

    if args.cluster_reference:
        # Workflow A: Load shared reference zone config
        print(f"\nWorkflow A: Shared reference zone boundaries")
        print(f"  Loading: {args.cluster_reference}")
        zone_config_override = load_zone_config(args.cluster_reference)
        n_zones = len(zone_config_override.get('zones', {}))
        print(f"  Loaded {n_zones}-zone config")
        for zname, zinfo in zone_config_override['zones'].items():
            print(f"    {zname}: layers {zinfo['layers']}")
    elif args.cluster_zones:
        print(f"\nK-means clustering enabled (per-sample):")
        print(f"  Clusters: {args.n_clusters}")
        print(f"  Method: {args.cluster_method}")
        if args.cluster_condition:
            print(f"  Condition: {args.cluster_condition}")
        cluster_args = {
            'n_clusters': args.n_clusters,
            'method': args.cluster_method,
            'condition': args.cluster_condition,
            'save_config_path': args.save_cluster_config,
        }

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
        group1_results = analyze_sample(group1_data, args.n_layers, args.output_dir,
                                        cluster_args=cluster_args,
                                        zone_config_override=zone_config_override)

        # Load and analyze Group 2
        group2_data = load_data(group2_config)
        group2_results = analyze_sample(group2_data, args.n_layers, args.output_dir,
                                        cluster_args=cluster_args,
                                        zone_config_override=zone_config_override)
        
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
                        output_path,
                        wt_zone_config=group1_results.get('zone_config'),
                        ko_zone_config=group2_results.get('zone_config')
                    )
        
        # Generate oxygen challenge plots if we have the three conditions
        print("\n" + "="*70)
        print("CHECKING FOR OXYGEN CHALLENGE DATA")
        print("="*70)
        
        # Helper function to transform data for plotting
        def transform_for_oxygen_plotting(condition_results):
            """Transform analyze_sample results into format expected by plotting functions
            
            Uses zone config to determine region order and layer counts for sequential numbering
            """
            transformed = {'regions': {}}
            
            # Check if this is multi-region data
            if 'mode' in condition_results and condition_results['mode'] == 'multi_region':
                
                # Import zone config to get region structure
                try:
                    from tissue_zones import ZONE_CONFIG
                    zone_config = ZONE_CONFIG
                except ImportError:
                    zone_config = None
                
                # Raw multi-region format (regions: right_cortex, left_cortex, etc.)
                if 'regions' in condition_results:
                    raw_regions = condition_results['regions']
                    
                    # Group regions by base name (cortex, medulla, papilla)
                    region_groups = {}
                    for raw_region_name, raw_region_data in raw_regions.items():
                        # Remove right_/left_ prefix to get base region name
                        base_region = raw_region_name.replace('right_', '').replace('left_', '')
                        
                        if base_region not in region_groups:
                            region_groups[base_region] = []
                        
                        region_groups[base_region].append(raw_region_data)
                    
                    # Determine region order and layer counts from zone config if available
                    if zone_config and 'regions' in zone_config:
                        # Extract unique base regions and their layer counts from zone config
                        zone_regions = {}
                        for region_name, region_data in zone_config['regions'].items():
                            base_name = region_name.replace('right_', '').replace('left_', '')
                            if base_name not in zone_regions and 'n_layers' in region_data:
                                zone_regions[base_name] = region_data['n_layers']
                        
                        # Create ordered list based on first appearance in zone config
                        region_order = list(zone_regions.keys())
                        print(f"\n  Using zone config region order: {region_order}")
                        print(f"  Layer counts per region: {zone_regions}")
                    else:
                        # Fallback: use standard kidney order
                        region_order = ['cortex', 'medulla', 'papilla']
                        print(f"\n  Using default region order: {region_order}")
                    
                    # Process regions in determined order with sequential layer numbering
                    global_layer_number = 1  # Sequential layer numbering across all regions
                    
                    for base_region in region_order:
                        if base_region not in region_groups:
                            continue
                            
                        region_list = region_groups[base_region]
                        
                        # Collect all layers by original layer_number within this region
                        layers_by_number = {}
                        
                        for region_data in region_list:
                            if isinstance(region_data, dict) and 'layers' in region_data:
                                for layer in region_data['layers']:
                                    layer_num = layer.get('layer_number', layer.get('layer', None))
                                    
                                    if layer_num is not None:
                                        if layer_num not in layers_by_number:
                                            layers_by_number[layer_num] = []
                                        
                                        layers_by_number[layer_num].append({
                                            't2star_mean': layer.get('t2star', {}).get('mean', layer.get('t2star_mean', np.nan)),
                                            'r2star_mean': layer.get('r2star', {}).get('mean', layer.get('r2star_mean', np.nan)),
                                            'perfusion_mean': layer.get('perfusion', {}).get('mean', layer.get('perfusion_mean', np.nan))
                                        })
                        
                        # Average bilateral layers and assign sequential global layer numbers
                        averaged_layers = []
                        start_layer = global_layer_number  # Track where this region starts
                        
                        for layer_num in sorted(layers_by_number.keys()):
                            layer_group = layers_by_number[layer_num]
                            
                            # Average across bilateral measurements
                            t2star_vals = [l['t2star_mean'] for l in layer_group if not np.isnan(l['t2star_mean'])]
                            r2star_vals = [l['r2star_mean'] for l in layer_group if not np.isnan(l['r2star_mean'])]
                            perfusion_vals = [l['perfusion_mean'] for l in layer_group if not np.isnan(l['perfusion_mean'])]
                            
                            averaged_layer = {
                                'layer_number': global_layer_number,  # Use sequential numbering
                                't2star_mean': np.mean(t2star_vals) if t2star_vals else np.nan,
                                'r2star_mean': np.mean(r2star_vals) if r2star_vals else np.nan,
                                'perfusion_mean': np.mean(perfusion_vals) if perfusion_vals else np.nan
                            }
                            averaged_layers.append(averaged_layer)
                            global_layer_number += 1  # Increment for next layer
                        
                        end_layer = global_layer_number - 1  # Track where this region ends
                        print(f"    {base_region}: {len(averaged_layers)} layers (global #{start_layer}-{end_layer})")
                        
                        transformed['regions'][base_region] = {'layers': averaged_layers}
                
                # Bilateral average format (if it exists - some workflows use this)
                elif 'bilateral_average' in condition_results:
                    bilateral = condition_results['bilateral_average']
                    
                    for region_name, region_data in bilateral.items():
                        simple_name = region_name.replace('right_', '').replace('left_', '')
                        
                        if simple_name not in transformed['regions']:
                            transformed['regions'][simple_name] = {'layers': []}
                        
                        if isinstance(region_data, dict) and 'layers' in region_data:
                            for layer in region_data['layers']:
                                transformed_layer = {
                                    'layer_number': layer.get('layer_number', layer.get('layer', len(transformed['regions'][simple_name]['layers']) + 1)),
                                    't2star_mean': layer.get('t2star', {}).get('mean', layer.get('t2star_mean', np.nan)),
                                    'r2star_mean': layer.get('r2star', {}).get('mean', layer.get('r2star_mean', np.nan)),
                                    'perfusion_mean': layer.get('perfusion', {}).get('mean', layer.get('perfusion_mean', np.nan))
                                }
                                transformed['regions'][simple_name]['layers'].append(transformed_layer)
            
            return transformed
        
        # Detect oxygen challenge conditions
        conditions = list(group1_results['conditions'].keys())
        print(f"Conditions found: {conditions}")
        
        # Map condition names (handle variations like oxygen1, oxygen_1, etc.)
        condition_map = {}
        for cond in conditions:
            cond_lower = cond.lower()
            if 'oxygen1' in cond_lower or 'oxygen_1' in cond_lower:
                condition_map['oxygen1'] = cond
            elif 'air' in cond_lower:
                condition_map['air'] = cond
            elif 'oxygen2' in cond_lower or 'oxygen_2' in cond_lower:
                condition_map['oxygen2'] = cond
        
        # Check if we have all three conditions
        has_oxygen_challenge = all(k in condition_map for k in ['oxygen1', 'air', 'oxygen2'])
        
        if has_oxygen_challenge:
            print("\n✓ Oxygen challenge conditions detected!")
            print(f"  • Oxygen 1: {condition_map['oxygen1']}")
            print(f"  • Air: {condition_map['air']}")
            print(f"  • Oxygen 2: {condition_map['oxygen2']}")
            print("\nGenerating comprehensive oxygen challenge plots...")
            
            try:
                from boldpy_plots import plot_comprehensive_oxygen_analysis
                
                # Organize data by condition
                # Extract group names from config files
                group1_name = group1_config.get('id', 'Group 1')
                group2_name = group2_config.get('id', 'Group 2')
                
                # Prepare data structure for plotting with transformation
                # Format: {condition: {group_name: data}}
                print("\nTransforming data for plotting...")
                oxygen1_data = {
                    group1_name: transform_for_oxygen_plotting(group1_results['conditions'][condition_map['oxygen1']]),
                    group2_name: transform_for_oxygen_plotting(group2_results['conditions'][condition_map['oxygen1']])
                }
                
                air_data = {
                    group1_name: transform_for_oxygen_plotting(group1_results['conditions'][condition_map['air']]),
                    group2_name: transform_for_oxygen_plotting(group2_results['conditions'][condition_map['air']])
                }
                
                oxygen2_data = {
                    group1_name: transform_for_oxygen_plotting(group1_results['conditions'][condition_map['oxygen2']]),
                    group2_name: transform_for_oxygen_plotting(group2_results['conditions'][condition_map['oxygen2']])
                }
                
                # Verify data was extracted
                for group_name in [group1_name, group2_name]:
                    for cond_name, cond_data in [('oxygen1', oxygen1_data), ('air', air_data), ('oxygen2', oxygen2_data)]:
                        n_regions = len(cond_data[group_name].get('regions', {}))
                        print(f"  {group_name} {cond_name}: {n_regions} regions", end='')
                        if n_regions > 0:
                            region_names = list(cond_data[group_name]['regions'].keys())
                            print(f" ({', '.join(region_names)})")
                            # Check layer counts
                            for rname in region_names:
                                n_layers = len(cond_data[group_name]['regions'][rname].get('layers', []))
                                print(f"    {rname}: {n_layers} layers")
                        else:
                            print()
                            print(f"    ⚠ No regions found for {group_name} {cond_name}!")
                
                # Create output directory
                oxygen_output_dir = args.output_dir / 'oxygen_challenge_analysis'
                oxygen_output_dir.mkdir(exist_ok=True)
                
                # Generate comprehensive plots
                plot_comprehensive_oxygen_analysis(
                    oxygen1_data=oxygen1_data,
                    air_data=air_data,
                    oxygen2_data=oxygen2_data,
                    output_dir=oxygen_output_dir,
                    group_names=[group1_name, group2_name],
                    sample_prefix='oxygen_analysis'
                )
                
                print("\n" + "="*70)
                print("✓ OXYGEN CHALLENGE PLOTS GENERATED!")
                print("="*70)
                print(f"Output directory: {oxygen_output_dir}")
                print("\nGenerated plots:")
                print("  1. Multi-parameter continuous (T2*, R2*, Perfusion) - 3 plots")
                print("  2. Oxygen response profiles (ΔT2*, ΔR2*, ΔPerfusion) - 2 plots")
                print("  3. Regional response bars (cortex/medulla/papilla) - 2 plots")
                print("  4. Whole vs regional comparison - 1 plot")
                print(f"\nTotal: 8 plots × 3 formats = {len(list(oxygen_output_dir.glob('*')))} files")
                
            except ImportError as e:
                print(f"\n⚠ Could not import oxygen challenge plotting functions: {e}")
                print("  Make sure boldpy_plots.py has the oxygen challenge functions")
            except Exception as e:
                print(f"\n✗ Error generating oxygen challenge plots: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("\n⚠ Oxygen challenge conditions not complete")
            missing = []
            if 'oxygen1' not in condition_map:
                missing.append('oxygen1/oxygen_1')
            if 'air' not in condition_map:
                missing.append('air')
            if 'oxygen2' not in condition_map:
                missing.append('oxygen2/oxygen_2')
            print(f"  Missing: {', '.join(missing)}")
            print("  Skipping oxygen challenge analysis")
            print("  (Need all three conditions for oxygen challenge plots)")

    
    elif args.config:
        # Config file mode
        print("\nMode: Single Sample Analysis (Config File)")
        
        with open(args.config) as f:
            config = json.load(f)
        
        data = load_data(config)
        results = analyze_sample(data, args.n_layers, args.output_dir,
                                cluster_args=cluster_args,
                                zone_config_override=zone_config_override)
        
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
