#!/usr/bin/env python3
"""
Phase 5: Multi-Region MLCO Plotting Functions
==============================================

Plotting functions specifically designed for multi-region MLCO analysis.
Handles hierarchical structure: region → zone → layer

Usage:
    from boldpy_plots_multiregion import (
        plot_multiregion_profile,
        plot_region_comparison,
        plot_zone_heatmap,
        plot_multiregion_overview
    )
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.gridspec import GridSpec
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Set style
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['font.size'] = 10

# ============================================================================
# COLOR SCHEMES
# ============================================================================

# Colorblind-friendly palette for regions (up to 8 regions)
REGION_COLORS = {
    1: '#E69F00',  # Orange
    2: '#56B4E9',  # Sky Blue
    3: '#009E73',  # Bluish Green
    4: '#F0E442',  # Yellow
    5: '#0072B2',  # Blue
    6: '#D55E00',  # Vermillion
    7: '#CC79A7',  # Reddish Purple
    8: '#999999'   # Gray
}

# Region name to color mapping (case-insensitive)
REGION_NAME_COLORS = {
    'cortex': '#E69F00',        # Orange
    'right_cortex': '#E69F00',
    'left_cortex': '#F0E442',   # Yellow
    'medulla': '#56B4E9',       # Sky Blue
    'right_medulla': '#56B4E9',
    'left_medulla': '#0072B2',  # Blue
    'papilla': '#009E73',       # Bluish Green
    'right_papilla': '#009E73',
    'left_papilla': '#CC79A7'   # Reddish Purple
}

# Zone colors (light shading)
ZONE_COLORS = {
    'outer': '#FFE5CC',
    'inner': '#FFD9B3'
}


def get_region_color(region_id: int = None, region_name: str = None) -> str:
    """
    Get color for a region by ID or name
    
    Parameters:
    -----------
    region_id : int, optional
        Region ID (1, 2, 3, ...)
    region_name : str, optional
        Region name (e.g., 'cortex', 'right_medulla')
        
    Returns:
    --------
    color : str
        Hex color code
    """
    if region_name is not None:
        region_name_lower = region_name.lower()
        return REGION_NAME_COLORS.get(region_name_lower, REGION_COLORS.get(1))
    elif region_id is not None:
        return REGION_COLORS.get(region_id, '#999999')
    else:
        return '#999999'


def save_figure_multiple_formats(fig, base_path: Path, dpi: int = 300):
    """Save figure in PNG, SVG, and PDF formats"""
    base_path = Path(base_path)
    base_no_ext = base_path.with_suffix('')
    
    formats = []
    
    # PNG
    png_path = base_no_ext.with_suffix('.png')
    fig.savefig(png_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    formats.append(f"PNG: {png_path.name}")
    
    # SVG
    svg_path = base_no_ext.with_suffix('.svg')
    fig.savefig(svg_path, format='svg', bbox_inches='tight', facecolor='white')
    formats.append(f"SVG: {svg_path.name}")
    
    # PDF
    pdf_path = base_no_ext.with_suffix('.pdf')
    fig.savefig(pdf_path, format='pdf', bbox_inches='tight', facecolor='white')
    formats.append(f"PDF: {pdf_path.name}")
    
    return formats


# ============================================================================
# SINGLE REGION PROFILE PLOT
# ============================================================================

def plot_multiregion_profile(
    results: Dict,
    region_name: str,
    output_path: Path,
    metric: str = 't2star',
    show_zones: bool = True
):
    """
    Plot T2*/R2*/Perfusion profile for a single region
    
    Parameters:
    -----------
    results : dict
        Multi-region analysis results
    region_name : str
        Name of region to plot (e.g., 'right_cortex')
    output_path : Path
        Output file path (without extension)
    metric : str
        Metric to plot: 't2star', 'r2star', or 'perfusion'
    show_zones : bool
        Whether to show zone boundaries
    """
    region_data = results['regions'][region_name]
    layers = region_data['layers']
    zones = region_data.get('zones', {})
    
    # Extract data
    layer_nums = [l['layer'] for l in layers]
    values = [l[metric]['median'] for l in layers if metric in l]
    stds = [l[metric]['std'] for l in layers if metric in l]
    
    if not values:
        print(f"  ✗ No {metric} data for {region_name}")
        return
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Zone shading
    if show_zones and zones:
        y_min, y_max = min(values) - max(stds), max(values) + max(stds)
        for zone_name, zone_data in zones.items():
            zone_layers = [l['layer'] for l in layers 
                          if l['encoded_value'] in zone_data.get('layers', [])]
            if zone_layers:
                x_start = min(zone_layers) - 0.5
                x_end = max(zone_layers) + 0.5
                ax.axvspan(x_start, x_end, alpha=0.15, 
                          color=ZONE_COLORS.get(zone_name, '#EEEEEE'),
                          label=f'{zone_name.capitalize()} zone')
    
    # Plot profile with error bars
    color = get_region_color(region_name=region_name)
    ax.errorbar(layer_nums, values, yerr=stds, 
                marker='o', markersize=6, linewidth=2,
                capsize=4, capthick=1.5,
                color=color, label=region_name.replace('_', ' ').title())
    
    # Labels
    metric_labels = {
        't2star': 'T2* (ms)',
        'r2star': 'R2* (Hz)',
        'perfusion': 'Perfusion (ml/100g/min)'
    }
    
    ax.set_xlabel('Layer (Outer → Inner)', fontsize=12, fontweight='bold')
    ax.set_ylabel(metric_labels.get(metric, metric.upper()), fontsize=12, fontweight='bold')
    ax.set_title(f'{region_name.replace("_", " ").title()} - {metric.upper()} Profile',
                 fontsize=14, fontweight='bold', pad=15)
    
    # Grid and legend
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best', framealpha=0.9)
    
    # Annotations
    if region_data.get('gradient'):
        grad = region_data['gradient'].get(metric, {}).get('gradient')
        if grad is not None:
            ax.text(0.02, 0.98, f'Gradient: {grad:+.2f}',
                   transform=ax.transAxes, va='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                   fontsize=10)
    
    plt.tight_layout()
    
    # Save
    formats = save_figure_multiple_formats(fig, output_path)
    print(f"  ✓ Saved {region_name} profile: {', '.join(formats)}")
    plt.close()


# ============================================================================
# MULTI-REGION COMPARISON PLOT
# ============================================================================

def plot_region_comparison(
    results: Dict,
    region_names: List[str],
    output_path: Path,
    metric: str = 't2star'
):
    """
    Compare multiple regions on the same plot
    
    Parameters:
    -----------
    results : dict
        Multi-region analysis results
    region_names : list of str
        Names of regions to compare
    output_path : Path
        Output file path
    metric : str
        Metric to plot
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    
    for region_name in region_names:
        if region_name not in results['regions']:
            continue
            
        region_data = results['regions'][region_name]
        layers = region_data['layers']
        
        # Extract data
        layer_nums = [l['layer'] for l in layers]
        values = [l[metric]['median'] for l in layers if metric in l]
        
        if not values:
            continue
        
        # Plot
        color = get_region_color(region_name=region_name)
        label = region_name.replace('_', ' ').title()
        ax.plot(layer_nums, values, marker='o', linewidth=2,
               markersize=5, color=color, label=label, alpha=0.8)
    
    # Labels
    metric_labels = {
        't2star': 'T2* (ms)',
        'r2star': 'R2* (Hz)',
        'perfusion': 'Perfusion (ml/100g/min)'
    }
    
    ax.set_xlabel('Layer (Outer → Inner)', fontsize=12, fontweight='bold')
    ax.set_ylabel(metric_labels.get(metric, metric.upper()), fontsize=12, fontweight='bold')
    ax.set_title(f'Region Comparison - {metric.upper()}',
                 fontsize=14, fontweight='bold', pad=15)
    
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best', framealpha=0.9, ncol=2)
    
    plt.tight_layout()
    
    formats = save_figure_multiple_formats(fig, output_path)
    print(f"  ✓ Saved region comparison: {', '.join(formats)}")
    plt.close()


# ============================================================================
# ZONE HEATMAP
# ============================================================================

def plot_zone_heatmap(
    results: Dict,
    output_path: Path,
    metric: str = 't2star'
):
    """
    Create heatmap of zone statistics across all regions
    
    Parameters:
    -----------
    results : dict
        Multi-region analysis results
    output_path : Path
        Output file path
    metric : str
        Metric to display
    """
    # Collect zone data
    data = {}
    region_names = []
    zone_names = set()
    
    for region_name, region_data in results['regions'].items():
        region_names.append(region_name)
        zones = region_data.get('zones', {})
        data[region_name] = {}
        
        for zone_name, zone_stats in zones.items():
            zone_names.add(zone_name)
            if metric in zone_stats:
                data[region_name][zone_name] = zone_stats[metric]['median']
    
    if not data:
        print(f"  ✗ No zone data available")
        return
    
    zone_names = sorted(list(zone_names))
    
    # Build matrix
    matrix = []
    for region_name in region_names:
        row = [data[region_name].get(zone, np.nan) for zone in zone_names]
        matrix.append(row)
    
    matrix = np.array(matrix)
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(8, len(region_names) * 0.6 + 2))
    
    im = ax.imshow(matrix, aspect='auto', cmap='RdYlBu_r', interpolation='nearest')
    
    # Labels
    ax.set_xticks(range(len(zone_names)))
    ax.set_xticklabels([z.capitalize() for z in zone_names], fontsize=11)
    ax.set_yticks(range(len(region_names)))
    ax.set_yticklabels([r.replace('_', ' ').title() for r in region_names], fontsize=11)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, pad=0.02)
    metric_labels = {
        't2star': 'T2* (ms)',
        'r2star': 'R2* (Hz)',
        'perfusion': 'Perfusion (ml/100g/min)'
    }
    cbar.set_label(metric_labels.get(metric, metric.upper()), fontsize=11)
    
    # Annotate cells with values
    for i in range(len(region_names)):
        for j in range(len(zone_names)):
            if not np.isnan(matrix[i, j]):
                text = ax.text(j, i, f'{matrix[i, j]:.1f}',
                             ha="center", va="center", color="black", fontsize=9)
    
    ax.set_title(f'Zone Statistics Heatmap - {metric.upper()}',
                 fontsize=13, fontweight='bold', pad=15)
    
    plt.tight_layout()
    
    formats = save_figure_multiple_formats(fig, output_path)
    print(f"  ✓ Saved zone heatmap: {', '.join(formats)}")
    plt.close()


# ============================================================================
# MULTI-REGION OVERVIEW (Multi-panel)
# ============================================================================

def plot_multiregion_overview(
    results: Dict,
    output_path: Path,
    scan_label: str = "scan"
):
    """
    Create multi-panel overview of all regions
    
    Shows T2* profiles for all regions in a grid layout
    
    Parameters:
    -----------
    results : dict
        Multi-region analysis results
    output_path : Path
        Output file path
    scan_label : str
        Scan label for title
    """
    regions = results['regions']
    n_regions = len(regions)
    
    # Determine grid layout
    if n_regions <= 3:
        nrows, ncols = 1, n_regions
        figsize = (5 * n_regions, 5)
    elif n_regions <= 6:
        nrows, ncols = 2, 3
        figsize = (15, 10)
    else:
        nrows = int(np.ceil(n_regions / 3))
        ncols = 3
        figsize = (15, 5 * nrows)
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    if n_regions == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
    
    for idx, (region_name, region_data) in enumerate(regions.items()):
        if idx >= len(axes):
            break
            
        ax = axes[idx]
        layers = region_data['layers']
        
        # Extract T2* data
        layer_nums = [l['layer'] for l in layers]
        t2_values = [l['t2star']['median'] for l in layers if 't2star' in l]
        t2_stds = [l['t2star']['std'] for l in layers if 't2star' in l]
        
        if not t2_values:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                   transform=ax.transAxes, fontsize=12)
            ax.set_title(region_name.replace('_', ' ').title(), fontweight='bold')
            continue
        
        # Plot
        color = get_region_color(region_name=region_name)
        ax.errorbar(layer_nums, t2_values, yerr=t2_stds,
                   marker='o', markersize=4, linewidth=1.5,
                   capsize=3, color=color, alpha=0.8)
        
        # Zone shading
        zones = region_data.get('zones', {})
        if zones:
            y_min, y_max = ax.get_ylim()
            for zone_name, zone_data in zones.items():
                zone_layers = [l['layer'] for l in layers 
                              if l['encoded_value'] in zone_data.get('layers', [])]
                if zone_layers:
                    x_start = min(zone_layers) - 0.5
                    x_end = max(zone_layers) + 0.5
                    ax.axvspan(x_start, x_end, alpha=0.1,
                             color=ZONE_COLORS.get(zone_name, '#EEEEEE'))
        
        # Labels
        ax.set_xlabel('Layer', fontsize=10)
        ax.set_ylabel('T2* (ms)', fontsize=10)
        ax.set_title(region_name.replace('_', ' ').title(), fontweight='bold', fontsize=11)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Gradient annotation
        if region_data.get('gradient'):
            grad = region_data['gradient'].get('t2star', {}).get('gradient')
            if grad is not None:
                ax.text(0.98, 0.02, f'Δ={grad:+.1f}',
                       transform=ax.transAxes, ha='right', va='bottom',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.7),
                       fontsize=8)
    
    # Hide empty subplots
    for idx in range(n_regions, len(axes)):
        axes[idx].axis('off')
    
    fig.suptitle(f'Multi-Region MLCO Overview - {scan_label}',
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    formats = save_figure_multiple_formats(fig, output_path)
    print(f"  ✓ Saved multi-region overview: {', '.join(formats)}")
    plt.close()


# ============================================================================
# LATERAL COMPARISON (Left vs Right)
# ============================================================================

def plot_lateral_comparison(
    results: Dict,
    output_path: Path,
    anatomical_region: str = 'cortex',
    metric: str = 't2star'
):
    """
    Compare left vs right for a specific anatomical region
    
    Parameters:
    -----------
    results : dict
        Multi-region analysis results
    output_path : Path
        Output file path
    anatomical_region : str
        Anatomical region name (e.g., 'cortex', 'medulla')
    metric : str
        Metric to plot
    """
    # Find left and right regions
    right_name = f'right_{anatomical_region}'
    left_name = f'left_{anatomical_region}'
    
    regions_to_plot = []
    if right_name in results['regions']:
        regions_to_plot.append(right_name)
    if left_name in results['regions']:
        regions_to_plot.append(left_name)
    
    if len(regions_to_plot) < 2:
        print(f"  ✗ Cannot compare - need both left and right {anatomical_region}")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for region_name in regions_to_plot:
        region_data = results['regions'][region_name]
        layers = region_data['layers']
        
        # Extract data
        layer_nums = [l['layer'] for l in layers]
        values = [l[metric]['median'] for l in layers if metric in l]
        stds = [l[metric]['std'] for l in layers if metric in l]
        
        if not values:
            continue
        
        # Plot
        color = get_region_color(region_name=region_name)
        side = 'Right' if 'right' in region_name else 'Left'
        ax.errorbar(layer_nums, values, yerr=stds,
                   marker='o', markersize=6, linewidth=2,
                   capsize=4, color=color, label=f'{side} {anatomical_region.title()}',
                   alpha=0.8)
    
    # Labels
    metric_labels = {
        't2star': 'T2* (ms)',
        'r2star': 'R2* (Hz)',
        'perfusion': 'Perfusion (ml/100g/min)'
    }
    
    ax.set_xlabel('Layer (Outer → Inner)', fontsize=12, fontweight='bold')
    ax.set_ylabel(metric_labels.get(metric, metric.upper()), fontsize=12, fontweight='bold')
    ax.set_title(f'Left vs Right {anatomical_region.title()} - {metric.upper()}',
                 fontsize=14, fontweight='bold', pad=15)
    
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best', framealpha=0.9, fontsize=11)
    
    plt.tight_layout()
    
    formats = save_figure_multiple_formats(fig, output_path)
    print(f"  ✓ Saved lateral comparison: {', '.join(formats)}")
    plt.close()


# ============================================================================
# CONVENIENCE FUNCTION
# ============================================================================

def generate_all_multiregion_plots(
    results: Dict,
    output_dir: Path,
    scan_label: str = "scan"
):
    """
    Generate complete set of multi-region plots
    
    Parameters:
    -----------
    results : dict
        Multi-region analysis results (single condition)
    output_dir : Path
        Output directory for plots
    scan_label : str
        Scan label for filenames
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nGenerating multi-region plots for {scan_label}...")
    
    # 1. Multi-region overview
    overview_path = output_dir / f"{scan_label}_overview"
    plot_multiregion_overview(results, overview_path, scan_label)
    
    # 2. Individual region profiles
    for region_name in results['regions'].keys():
        region_path = output_dir / f"{scan_label}_{region_name}_profile"
        plot_multiregion_profile(results, region_name, region_path)
    
    # 3. Zone heatmap
    heatmap_path = output_dir / f"{scan_label}_zone_heatmap"
    plot_zone_heatmap(results, heatmap_path)
    
    # 4. Region comparison (all regions)
    comparison_path = output_dir / f"{scan_label}_all_regions_comparison"
    plot_region_comparison(results, list(results['regions'].keys()), comparison_path)
    
    # 5. Lateral comparisons (if applicable)
    for anatomical_region in ['cortex', 'medulla', 'papilla']:
        right_name = f'right_{anatomical_region}'
        left_name = f'left_{anatomical_region}'
        if right_name in results['regions'] and left_name in results['regions']:
            lateral_path = output_dir / f"{scan_label}_{anatomical_region}_left_vs_right"
            plot_lateral_comparison(results, lateral_path, anatomical_region)
    
    print(f"✓ All multi-region plots generated")


if __name__ == '__main__':
    print("Phase 5: Multi-Region Plotting Module")
    print("Import this module and use the plotting functions")
