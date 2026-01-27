#!/usr/bin/env python3
"""
BoldPy Unified Plotting Functions
==================================

Comprehensive plotting suite for BOLD MRI analysis supporting:
1. Whole-kidney analysis (single 24-layer MLCO with 5 zones)
2. Phase 1 Multi-region analysis (raw region data before averaging)
3. Phase 2 Multi-region analysis (bilateral averaging, perfusion, oxygen response)

WHOLE-KIDNEY FUNCTIONS (for single MLCO):
==========================================
    plot_perfusion_profile()        - Perfusion layer-by-layer profile
    plot_t2star_perfusion_scatter() - T2* vs perfusion scatter plot
    plot_triple_overlay()           - T2* + R2* + Perfusion integrated plot
    plot_mlco_profile()             - Single MLCO layer profile with zones
    plot_mlco_comparison()          - WT vs KO whole-kidney comparison

PHASE 1 MULTI-REGION FUNCTIONS (for backward compatibility):
=============================================================
    generate_all_multiregion_plots()      - Wrapper function for boldpy_analyze.py
    plot_multiregion_profile_phase1()     - Single region profile
    plot_multiregion_overview_phase1()    - All regions overview
    plot_zone_heatmap()                   - Zone comparison bars
    plot_region_comparison()              - Compare specific regions
    plot_lateral_comparison()             - Left vs Right comparison

PHASE 2 MULTI-REGION FUNCTIONS (Enhanced, recommended):
========================================================
    plot_multicondition_profile()      - O₁→Air→O₂ progression (4 panels with perfusion)
    plot_regions_overview()            - All regions side-by-side for one condition
    plot_oxygen_response_summary()     - Delta response magnitudes (bar charts)
    plot_group_comparison_overview()   - M1 vs M2 comparison (3×3 grid)
    plot_asymmetry_check()             - QC asymmetry check (left vs right)

Author: BoldPy Development Team
Version: 3.0 (Complete Unified)
Date: January 2026
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pathlib import Path
from typing import Dict, List, Optional

# Set style to avoid font warnings
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']

# Import zone definitions
try:
    from tissue_zones import ZONE_DEFINITIONS, calculate_effect_size, interpret_effect_size
    ZONES_AVAILABLE = True
except ImportError:
    print("WARNING: tissue_zones module not available")
    ZONES_AVAILABLE = False

# Color schemes
CONDITION_COLORS = {
    'oxygen1': '#4A90E2',    # Blue
    'oxygen_1': '#4A90E2',
    'air': '#E27A3F',        # Orange
    'oxygen2': '#50C878',    # Green
    'oxygen_2': '#50C878'
}

ZONE_COLORS = {
    'outer_cortex': '#E8F4F8',
    'inner_cortex': '#C5E3ED',
    'cmj': '#FFE5CC',
    'outer_medulla': '#FFD9B3',
    'inner_medulla': '#FFC999'
}


def save_figure_multiple_formats(fig, base_path: Path, dpi: int = 300):
    """Save figure in PNG, SVG, and PDF formats"""
    base_path = Path(base_path)
    base_no_ext = base_path.with_suffix('')
    
    # PNG
    png_path = base_no_ext.with_suffix('.png')
    fig.savefig(png_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    print(f"  ✓ Saved PNG: {png_path.name}")
    
    # SVG
    svg_path = base_no_ext.with_suffix('.svg')
    fig.savefig(svg_path, format='svg', bbox_inches='tight', facecolor='white')
    print(f"  ✓ Saved SVG: {svg_path.name}")
    
    # PDF
    pdf_path = base_no_ext.with_suffix('.pdf')
    fig.savefig(pdf_path, format='pdf', bbox_inches='tight', facecolor='white')
    print(f"  ✓ Saved PDF: {pdf_path.name}")


def add_zone_shading(ax, n_layers: int = 24, alpha: float = 0.15):
    """Add background shading for 5 zones"""
    if not ZONES_AVAILABLE:
        return
    
    y_min, y_max = ax.get_ylim()
    
    for zone_name, zone_info in ZONE_DEFINITIONS.items():
        layers = zone_info['layers']
        x_start = min(layers) - 0.5
        x_end = max(layers) + 0.5
        
        ax.add_patch(Rectangle(
            (x_start, y_min), x_end - x_start, y_max - y_min,
            facecolor=ZONE_COLORS.get(zone_name, '#EEEEEE'),
            edgecolor='none',
            alpha=alpha,
            zorder=0
        ))


def plot_perfusion_profile(results: Dict,
                                   scan_conditions: List[str],
                                   output_path: Path,
                                   animal_label: str = "animal"):
    """
    Create perfusion MLCO profile plots
    
    Parameters:
    -----------
    results : dict
        Dictionary of MLCO results per condition
    scan_conditions : list
        List of condition labels
    output_path : Path
        Output file path
    animal_label : str
        Animal identifier
    """
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3)
    
    colors = ['steelblue', 'coral', 'forestgreen', 'purple']
    
    kidneys_to_plot = [
        ('bilateral', 'Bilateral Kidneys', 0),
        ('right_kidney', 'Right Kidney', 1),
        ('left_kidney', 'Left Kidney', 2)
    ]
    
    # Row 1: Perfusion profiles
    for kidney_key, kidney_title, col_idx in kidneys_to_plot:
        ax = fig.add_subplot(gs[0, col_idx])
        
        # Add zone shading
        add_zone_shading(ax)
        
        for cond_idx, condition in enumerate(scan_conditions):
            if condition not in results:
                continue
            
            # Extract layer data
            if kidney_key == 'bilateral':
                layers_data = results[condition]['bilateral']['layers']
            else:
                layers_data = results[condition][kidney_key]['layers']
            
            if not layers_data or 'perfusion' not in layers_data[0]:
                continue
            
            layers = [l['layer'] for l in layers_data if 'perfusion' in l]
            perf_means = [l['perfusion']['median'] for l in layers_data if 'perfusion' in l]
            perf_stds = [l['perfusion']['std'] for l in layers_data if 'perfusion' in l]
            
            # Plot with error bars
            ax.errorbar(layers, perf_means, yerr=perf_stds,
                       marker='o', linestyle='-', linewidth=2, markersize=6,
                       capsize=4, color=colors[cond_idx % len(colors)],
                       label=condition.replace('_', ' ').title(),
                       alpha=0.8, zorder=10)
        
        ax.set_xlabel('Layer (1=Cortex → 24=Medulla)', fontweight='bold', fontsize=10)
        ax.set_ylabel('Perfusion (ml/100g/min)', fontweight='bold', fontsize=10)
        ax.set_title(f'{kidney_title}\nPerfusion Profile', fontweight='bold', fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, zorder=1)
        ax.set_xticks(range(1, 25, 4))
    
    # Row 2: Tissue quality summary for each kidney
    for kidney_key, kidney_title, col_idx in kidneys_to_plot:
        ax = fig.add_subplot(gs[1, col_idx])
        ax.axis('off')
        
        summary_lines = [f"{kidney_title.upper()} - TISSUE QUALITY", "="*40, ""]
        
        for condition in scan_conditions:
            if condition not in results:
                continue
            
            if kidney_key == 'bilateral':
                zones = results[condition]['bilateral'].get('zones', {})
            else:
                zones = results[condition][kidney_key].get('zones', {})
            
            if zones:
                summary_lines.append(f"{condition.upper()}:")
                for zone_name in ['outer_cortex', 'inner_cortex', 'cmj', 'outer_medulla', 'inner_medulla']:
                    if zone_name in zones:
                        z = zones[zone_name]
                        viable = z['tissue_quality']['viable_pct']
                        summary_lines.append(f"  {zone_name:20s}: {viable:4.0f}% viable")
                summary_lines.append("")
        
        summary_text = "\n".join(summary_lines)
        ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
               fontsize=8, family='monospace', verticalalignment='top')
    
    plt.suptitle(f'{animal_label} - Perfusion MLCO Analysis',
                fontsize=14, fontweight='bold')
    
    save_figure_multiple_formats(fig, output_path)
    plt.close()


def plot_t2star_perfusion_scatter(results: Dict,
                                    output_path: Path,
                                    animal_label: str = "animal"):
    """
    Create T2* vs Perfusion scatter plots
    
    Shows relationship between T2* and perfusion with points colored by layer depth
    
    Parameters:
    -----------
    results : dict
        MLCO results for one condition
    output_path : Path
        Output file path
    animal_label : str
        Animal identifier
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    kidneys_to_plot = [
        ('bilateral', 'Bilateral', 0),
        ('right_kidney', 'Right', 1),
        ('left_kidney', 'Left', 2)
    ]
    
    for kidney_key, kidney_title, idx in kidneys_to_plot:
        ax = axes[idx]
        
        # Extract data
        if kidney_key == 'bilateral':
            layers_data = results['bilateral']['layers']
        else:
            layers_data = results[kidney_key]['layers']
        
        # Get T2* and perfusion values
        t2_values = []
        perf_values = []
        layer_nums = []
        viable_fracs = []
        
        for layer in layers_data:
            if 'perfusion' in layer:
                t2_values.append(layer['t2star']['median'])
                perf_values.append(layer['perfusion']['median'])
                layer_nums.append(layer['layer'])
                viable_fracs.append(layer['tissue_quality']['viable_pct'] / 100)
        
        if not t2_values:
            ax.text(0.5, 0.5, 'No perfusion data', 
                   ha='center', va='center', transform=ax.transAxes)
            continue
        
        # Create scatter with color gradient by layer depth
        scatter = ax.scatter(perf_values, t2_values, 
                           c=layer_nums, cmap='viridis', 
                           s=[v*200 for v in viable_fracs],  # Size by viability
                           alpha=0.7, edgecolors='black', linewidth=0.5)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Layer (Cortex→Medulla)', fontsize=9)
        
        # Add quadrant lines
        ax.axhline(y=40, color='red', linestyle='--', alpha=0.5, linewidth=1)
        ax.axvline(x=150, color='red', linestyle='--', alpha=0.5, linewidth=1)
        
        # Label quadrants
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        
        ax.text(xlim[1]*0.75, ylim[1]*0.9, 'Normal\nOxygenated',
               ha='center', va='center', fontsize=8, style='italic', alpha=0.6)
        ax.text(xlim[1]*0.25, ylim[1]*0.9, 'Ischemic\nHypoxia',
               ha='center', va='center', fontsize=8, style='italic', alpha=0.6)
        ax.text(xlim[1]*0.25, ylim[0] + (ylim[1]-ylim[0])*0.1, 'Necrosis/\nEdema',
               ha='center', va='center', fontsize=8, style='italic', alpha=0.6,
               color='red')
        
        ax.set_xlabel('Perfusion (ml/100g/min)', fontweight='bold', fontsize=10)
        ax.set_ylabel('T2* (ms)', fontweight='bold', fontsize=10)
        ax.set_title(f'{kidney_title} Kidney', fontweight='bold', fontsize=11)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'{animal_label} - T2* vs Perfusion Analysis\n(Point size = tissue viability)',
                fontsize=13, fontweight='bold')
    plt.tight_layout()
    
    save_figure_multiple_formats(fig, output_path)
    plt.close()


def plot_triple_overlay(results: Dict,
                               condition: str,
                               output_path: Path,
                               animal_label: str = "animal"):
    """
    Create triple overlay plot: T2* + R2* + Perfusion on same graph
    
    Parameters:
    -----------
    results : dict
        MLCO results
    condition : str
        Condition label
    output_path : Path
        Output file path
    animal_label : str
        Animal identifier
    """
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    kidneys_to_plot = [
        ('bilateral', 'Bilateral', 0),
        ('right_kidney', 'Right', 1),
        ('left_kidney', 'Left', 2)
    ]
    
    for kidney_key, kidney_title, idx in kidneys_to_plot:
        ax = axes[idx]
        ax2 = ax.twinx()  # Second y-axis for R2*
        ax3 = ax.twinx()  # Third y-axis for perfusion
        ax3.spines['right'].set_position(('outward', 60))
        
        # Add zone shading
        add_zone_shading(ax)
        
        # Extract data
        if kidney_key == 'bilateral':
            layers_data = results[condition]['bilateral']['layers']
        else:
            # Check if this kidney data exists (might not for single-organ analysis)
            if kidney_key not in results[condition]:
                # Skip this subplot if data doesn't exist
                ax.text(0.5, 0.5, f'No {kidney_title} kidney data\n(single organ analysis)',
                       ha='center', va='center', transform=ax.transAxes, fontsize=12)
                ax.set_xlabel('Layer')
                ax.set_ylabel('T2* (ms)')
                ax2.set_ylabel('R2* (Hz)')
                continue
            layers_data = results[condition][kidney_key]['layers']
        
        # Check if layers_data is empty
        if not layers_data:
            ax.text(0.5, 0.5, f'No {kidney_title} data available',
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_xlabel('Layer')
            ax.set_ylabel('T2* (ms)')
            ax2.set_ylabel('R2* (Hz)')
            continue
        
        layers = [l['layer'] for l in layers_data]
        t2_values = [l['t2star']['median'] for l in layers_data]
        r2_values = [l['r2star']['median'] for l in layers_data]
        
        # Plot T2* (primary y-axis)
        line1 = ax.plot(layers, t2_values, 'o-', color='#4A90E2', 
                       linewidth=2.5, markersize=7, label='T2*', zorder=10)
        ax.set_ylabel('T2* (ms)', color='#4A90E2', fontweight='bold', fontsize=11)
        ax.tick_params(axis='y', labelcolor='#4A90E2')
        
        # Plot R2* (secondary y-axis)
        line2 = ax2.plot(layers, r2_values, 's-', color='#E27A3F',
                        linewidth=2.5, markersize=7, label='R2*', zorder=10)
        ax2.set_ylabel('R2* (Hz)', color='#E27A3F', fontweight='bold', fontsize=11)
        ax2.tick_params(axis='y', labelcolor='#E27A3F')
        
        # Plot Perfusion (tertiary y-axis) if available
        if layers_data and 'perfusion' in layers_data[0]:
            perf_values = [l['perfusion']['median'] for l in layers_data if 'perfusion' in l]
            perf_layers = [l['layer'] for l in layers_data if 'perfusion' in l]
            
            line3 = ax3.plot(perf_layers, perf_values, '^-', color='#50C878',
                           linewidth=2.5, markersize=7, label='Perfusion', zorder=10)
            ax3.set_ylabel('Perfusion (ml/100g/min)', color='#50C878', 
                          fontweight='bold', fontsize=11)
            ax3.tick_params(axis='y', labelcolor='#50C878')
        
        ax.set_xlabel('Layer (1=Cortex → 24=Medulla)', fontweight='bold', fontsize=11)
        ax.set_title(f'{kidney_title} Kidney - {condition.replace("_", " ").title()}',
                    fontweight='bold', fontsize=12)
        ax.grid(True, alpha=0.3, zorder=1)
        ax.set_xticks(range(1, 25, 3))
    
    plt.suptitle(f'{animal_label} - Integrated Analysis: T2* + R2* + Perfusion',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    save_figure_multiple_formats(fig, output_path)
    plt.close()


def plot_mlco_profile(results: Dict,
                                           scan_conditions: List[str],
                                           output_path: Path,
                                           animal_label: str = "animal"):
    """
    Create comprehensive MLCO profile plots for single animal
    
    Updated for 24 layers with zone shading
    """
    # Reorder conditions
    ordered_conditions = []
    condition_map = {c.lower(): c for c in scan_conditions}
    
    for preferred in ['oxygen_1', 'oxygen1', 'air', 'oxygen_2', 'oxygen2']:
        if preferred in condition_map:
            ordered_conditions.append(condition_map[preferred])
            del condition_map[preferred]
    
    ordered_conditions.extend(condition_map.values())
    
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 4, hspace=0.35, wspace=0.3)
    
    colors = ['steelblue', 'coral', 'forestgreen', 'purple']
    
    kidneys_to_plot = [
        ('bilateral', 'Bilateral Kidneys', 0),
        ('right_kidney', 'Right Kidney', 1),
        ('left_kidney', 'Left Kidney', 2)
    ]
    
    # Row 1: T2* profiles
    for kidney_key, kidney_title, col_idx in kidneys_to_plot:
        ax = fig.add_subplot(gs[0, col_idx])
        add_zone_shading(ax)
        
        for cond_idx, condition in enumerate(ordered_conditions):
            if condition not in results:
                continue
            
            if kidney_key == 'bilateral':
                layers_data = results[condition]['bilateral']['layers']
            else:
                layers_data = results[condition][kidney_key]['layers']
            
            if not layers_data:
                continue
            
            layers = [l['layer'] for l in layers_data]
            t2_means = [l['t2star']['median'] for l in layers_data]
            t2_stds = [l['t2star']['std'] for l in layers_data]
            
            ax.errorbar(layers, t2_means, yerr=t2_stds,
                       marker='o', linestyle='-', linewidth=2, markersize=6,
                       capsize=4, color=colors[cond_idx % len(colors)],
                       label=condition.replace('_', ' ').title(),
                       alpha=0.8, zorder=10)
        
        ax.set_xlabel('Layer (1=Cortex → 24=Medulla)', fontweight='bold', fontsize=10)
        ax.set_ylabel('T2* (ms)', fontweight='bold', fontsize=10)
        ax.set_title(f'{kidney_title}\nT2* Profile', fontweight='bold', fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, zorder=1)
        ax.set_xticks(range(1, 25, 4))
    
    # Row 2: R2* profiles
    for kidney_key, kidney_title, col_idx in kidneys_to_plot:
        ax = fig.add_subplot(gs[1, col_idx])
        add_zone_shading(ax)
        
        for cond_idx, condition in enumerate(ordered_conditions):
            if condition not in results:
                continue
            
            if kidney_key == 'bilateral':
                layers_data = results[condition]['bilateral']['layers']
            else:
                layers_data = results[condition][kidney_key]['layers']
            
            if not layers_data:
                continue
            
            layers = [l['layer'] for l in layers_data]
            r2_means = [l['r2star']['median'] for l in layers_data]
            r2_stds = [l['r2star']['std'] for l in layers_data]
            
            ax.errorbar(layers, r2_means, yerr=r2_stds,
                       marker='s', linestyle='-', linewidth=2, markersize=6,
                       capsize=4, color=colors[cond_idx % len(colors)],
                       label=condition.replace('_', ' ').title(),
                       alpha=0.8, zorder=10)
        
        ax.set_xlabel('Layer (1=Cortex → 24=Medulla)', fontweight='bold', fontsize=10)
        ax.set_ylabel('R2* (Hz)', fontweight='bold', fontsize=10)
        ax.set_title(f'{kidney_title}\nR2* Profile', fontweight='bold', fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, zorder=1)
        ax.set_xticks(range(1, 25, 4))
    
    # Row 3: Gradient summary
    ax = fig.add_subplot(gs[2, :2])
    
    gradients_t2 = []
    gradient_labels = []
    
    for condition in ordered_conditions:
        if condition not in results:
            continue
        
        bilateral_gradient = results[condition]['bilateral'].get('gradient')
        if bilateral_gradient and bilateral_gradient['t2star']:
            gradients_t2.append(bilateral_gradient['t2star']['gradient'])
            gradient_labels.append(condition.replace('_', ' ').title())
    
    if gradients_t2:
        x = np.arange(len(gradients_t2))
        bars = ax.bar(x, gradients_t2, 
                     color=['steelblue' if g < 0 else 'coral' for g in gradients_t2])
        ax.set_ylabel('T2* Gradient (Medulla - Cortex, ms)', fontweight='bold', fontsize=10)
        ax.set_title('Cortex-Medulla Gradient\n(Negative = Normal, Positive = Abnormal)',
                    fontweight='bold', fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(gradient_labels)
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax.grid(True, alpha=0.3, axis='y')
        
        for bar, val in zip(bars, gradients_t2):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:+.1f}',
                   ha='center', va='bottom' if height > 0 else 'top',
                   fontweight='bold', fontsize=9)
    
    # Row 3: Summary statistics table
    ax = fig.add_subplot(gs[2, 2:4])
    ax.axis('off')
    
    summary_lines = [f"{animal_label.upper()} - ANALYSIS SUMMARY", "="*45, ""]
    
    for condition in ordered_conditions:
        if condition not in results:
            continue
        
        summary_lines.append(f"{condition.upper()}:")
        
        bilateral = results[condition]['bilateral']
        if bilateral['layers']:
            cortex_t2 = bilateral['layers'][0]['t2star']['median']
            medulla_t2 = bilateral['layers'][-1]['t2star']['median']
            
            summary_lines.append(f"  Cortex T2*:  {cortex_t2:.1f} ms")
            summary_lines.append(f"  Medulla T2*: {medulla_t2:.1f} ms")
            
            if bilateral.get('gradient') and bilateral['gradient']['t2star']:
                grad = bilateral['gradient']['t2star']['gradient']
                summary_lines.append(f"  Gradient:    {grad:+.1f} ms")
                
                if bilateral['gradient']['t2star']['abnormal']:
                    summary_lines.append(f"  WARNING: ABNORMAL gradient")
                else:
                    summary_lines.append(f"  OK: Normal gradient")
        
        summary_lines.append("")
    
    summary_text = "\n".join(summary_lines)
    ax.text(0.1, 0.5, summary_text, transform=ax.transAxes,
           fontsize=9, family='monospace', verticalalignment='center')
    
    plt.suptitle(f'{animal_label} - MLCO Analysis (24 Layers)',
                fontsize=15, fontweight='bold')
    
    save_figure_multiple_formats(fig, output_path)
    plt.close()


def plot_mlco_comparison(wt_results: Dict,
                                ko_results: Dict,
                                condition: str,
                                output_path: Path):
    """
    Compare WT vs KO with 5-zone statistics
    
    Enhanced version with effect sizes and zone-by-zone comparison
    """
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.3)
    
    # Row 1: T2* and R2* bilateral comparison
    ax1 = fig.add_subplot(gs[0, 0])
    add_zone_shading(ax1)
    
    wt_layers = wt_results['bilateral']['layers']
    ko_layers = ko_results['bilateral']['layers']
    
    if wt_layers:
        layers = [l['layer'] for l in wt_layers]
        t2_means = [l['t2star']['median'] for l in wt_layers]
        t2_stds = [l['t2star']['std'] for l in wt_layers]
        
        ax1.errorbar(layers, t2_means, yerr=t2_stds,
                   marker='o', linestyle='-', linewidth=2, markersize=6,
                   capsize=4, color='steelblue', label='WT',
                   alpha=0.8, zorder=10)
    
    if ko_layers:
        layers = [l['layer'] for l in ko_layers]
        t2_means = [l['t2star']['median'] for l in ko_layers]
        t2_stds = [l['t2star']['std'] for l in ko_layers]
        
        ax1.errorbar(layers, t2_means, yerr=t2_stds,
                   marker='o', linestyle='-', linewidth=2, markersize=6,
                   capsize=4, color='coral', label='KO',
                   alpha=0.8, zorder=10)
    
    ax1.set_xlabel('Layer (1=Cortex → 24=Medulla)', fontweight='bold', fontsize=10)
    ax1.set_ylabel('T2* (ms)', fontweight='bold', fontsize=10)
    ax1.set_title(f'Bilateral T2* - {condition.title()}', fontweight='bold', fontsize=11)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3, zorder=1)
    ax1.set_xticks(range(1, 25, 4))
    
    # R2* comparison
    ax2 = fig.add_subplot(gs[0, 1])
    add_zone_shading(ax2)
    
    if wt_layers:
        layers = [l['layer'] for l in wt_layers]
        r2_means = [l['r2star']['median'] for l in wt_layers]
        r2_stds = [l['r2star']['std'] for l in wt_layers]
        
        ax2.errorbar(layers, r2_means, yerr=r2_stds,
                   marker='s', linestyle='-', linewidth=2, markersize=6,
                   capsize=4, color='steelblue', label='WT',
                   alpha=0.8, zorder=10)
    
    if ko_layers:
        layers = [l['layer'] for l in ko_layers]
        r2_means = [l['r2star']['median'] for l in ko_layers]
        r2_stds = [l['r2star']['std'] for l in ko_layers]
        
        ax2.errorbar(layers, r2_means, yerr=r2_stds,
                   marker='s', linestyle='-', linewidth=2, markersize=6,
                   capsize=4, color='coral', label='KO',
                   alpha=0.8, zorder=10)
    
    ax2.set_xlabel('Layer (1=Cortex → 24=Medulla)', fontweight='bold', fontsize=10)
    ax2.set_ylabel('R2* (Hz)', fontweight='bold', fontsize=10)
    ax2.set_title(f'Bilateral R2* - {condition.title()}', fontweight='bold', fontsize=11)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, zorder=1)
    ax2.set_xticks(range(1, 25, 4))
    
    # Gradient comparison
    ax3 = fig.add_subplot(gs[0, 2])
    
    gradients = []
    labels = []
    colors_list = []
    
    wt_grad = wt_results['bilateral'].get('gradient')
    if wt_grad and wt_grad['t2star']:
        gradients.append(wt_grad['t2star']['gradient'])
        labels.append('WT')
        colors_list.append('steelblue')
    
    ko_grad = ko_results['bilateral'].get('gradient')
    if ko_grad and ko_grad['t2star']:
        gradients.append(ko_grad['t2star']['gradient'])
        labels.append('KO')
        colors_list.append('coral')
    
    if gradients:
        x = np.arange(len(gradients))
        bars = ax3.bar(x, gradients, color=colors_list)
        ax3.set_ylabel('T2* Gradient (ms)', fontweight='bold', fontsize=10)
        ax3.set_title('Cortex-Medulla Gradient', fontweight='bold', fontsize=11)
        ax3.set_xticks(x)
        ax3.set_xticklabels(labels)
        ax3.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax3.grid(True, alpha=0.3, axis='y')
        
        for bar, val in zip(bars, gradients):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:+.1f}',
                   ha='center', va='bottom' if height > 0 else 'top',
                   fontweight='bold', fontsize=10)
    
    # Row 2: Layer-by-layer difference
    ax4 = fig.add_subplot(gs[1, :])
    add_zone_shading(ax4)
    
    if wt_layers and ko_layers and len(wt_layers) == len(ko_layers):
        layers = [l['layer'] for l in wt_layers]
        
        t2_diff = []
        for wt_l, ko_l in zip(wt_layers, ko_layers):
            diff = ko_l['t2star']['median'] - wt_l['t2star']['median']
            t2_diff.append(diff)
        
        ax4.plot(layers, t2_diff, marker='o', linestyle='-', linewidth=3,
               markersize=8, color='purple', label='T2* Difference (KO - WT)',
               zorder=10)
        ax4.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax4.set_xlabel('Layer (1=Cortex → 24=Medulla)', fontweight='bold', fontsize=11)
        ax4.set_ylabel('Difference (ms)', fontweight='bold', fontsize=11)
        ax4.set_title(f'Layer-by-Layer T2* Difference (KO - WT) - {condition.title()}',
                    fontweight='bold', fontsize=12)
        ax4.legend(fontsize=10)
        ax4.grid(True, alpha=0.3, zorder=1)
        ax4.set_xticks(range(1, 25, 2))
        
        # Highlight significant differences
        max_diff = max(abs(min(t2_diff)), abs(max(t2_diff)))
        if max_diff > 5:
            ax4.fill_between(layers, 0, t2_diff, where=np.array(t2_diff) > 5,
                           alpha=0.2, color='red', label='KO >> WT (>5ms)', zorder=5)
            ax4.fill_between(layers, t2_diff, 0, where=np.array(t2_diff) < -5,
                           alpha=0.2, color='blue', label='WT >> KO (>5ms)', zorder=5)
    
    # Row 3: 5-Zone statistical comparison
    ax5 = fig.add_subplot(gs[2, :])
    ax5.axis('off')
    
    summary_lines = ["5-ZONE REGIONAL COMPARISON (WT vs KO)", "="*70, ""]
    
    if ZONES_AVAILABLE:
        wt_zones = wt_results['bilateral'].get('zones', {})
        ko_zones = ko_results['bilateral'].get('zones', {})
        
        for zone_name in ['outer_cortex', 'inner_cortex', 'cmj', 'outer_medulla', 'inner_medulla']:
            if zone_name in wt_zones and zone_name in ko_zones:
                wt_z = wt_zones[zone_name]
                ko_z = ko_zones[zone_name]
                
                wt_t2 = wt_z['t2star']['mean']
                ko_t2 = ko_z['t2star']['mean']
                delta = ko_t2 - wt_t2
                pct_change = (delta / wt_t2) * 100
                
                # Calculate effect size
                effect_size = calculate_effect_size(
                    wt_t2, wt_z['t2star']['std'],
                    ko_t2, ko_z['t2star']['std']
                )
                
                summary_lines.append(f"{zone_name.upper().replace('_', ' '):25s}:")
                summary_lines.append(f"  WT:  T2* = {wt_t2:5.1f} ± {wt_z['t2star']['std']:4.1f} ms, "
                                   f"Viable = {wt_z['tissue_quality']['viable_pct']:4.0f}%")
                summary_lines.append(f"  KO:  T2* = {ko_t2:5.1f} ± {ko_z['t2star']['std']:4.1f} ms, "
                                   f"Viable = {ko_z['tissue_quality']['viable_pct']:4.0f}%")
                summary_lines.append(f"  Δ = {delta:+.1f} ms ({pct_change:+.0f}%), "
                                   f"Effect size: {effect_size:.2f} ({interpret_effect_size(effect_size)})")
                summary_lines.append("")
    
    summary_text = "\n".join(summary_lines)
    ax5.text(0.05, 0.95, summary_text, transform=ax5.transAxes,
           fontsize=9, family='monospace', verticalalignment='top')
    
    plt.suptitle(f'WT vs KO Comparison - {condition.title()} (24 Layers)',
                fontsize=15, fontweight='bold')
    
    save_figure_multiple_formats(fig, output_path)
    plt.close()


# ==============================================================================
# MULTI-REGION PLOTTING FUNCTIONS (PHASE 5)
# ==============================================================================

    print("  ✓ Multi-region plotting (Phase 5)")


# ============================================================================
# PHASE 1 MULTI-REGION FUNCTIONS (for backward compatibility)
# ============================================================================
#
# These are the original Phase 1 multiregion functions that boldpy_analyze.py
# depends on. They work with raw multi-region MLCO output (before bilateral
# averaging).
#
# Note: Phase 2 functions (above) are more advanced and include bilateral
# averaging, perfusion support, and oxygen response analysis.
# ============================================================================

def _add_zone_shading_phase1(ax, zones: Dict, n_layers: int):
    """Add colored shading for zones (Phase 1 version)"""
    colors = ['#E8F4F8', '#FFE5CC']  # Light blue, light orange
    
    for idx, (zone_name, zone_data) in enumerate(zones.items()):
        if 'layers' not in zone_data:
            continue
        
        # Get layer numbers from encoded values
        encoded_layers = zone_data['layers']
        if not encoded_layers:
            continue
        
        # Decode to get actual layer numbers
        layer_nums = [v % 1000 for v in encoded_layers]
        if not layer_nums:
            continue
        
        min_layer = min(layer_nums)
        max_layer = max(layer_nums)
        
        ax.axvspan(min_layer - 0.5, max_layer + 0.5, 
                  alpha=0.2, color=colors[idx % 2], zorder=0)
        
        # Add zone label
        mid_layer = (min_layer + max_layer) / 2
        y_pos = ax.get_ylim()[1] * 0.95
        ax.text(mid_layer, y_pos, zone_name.upper(),
               ha='center', va='top', fontsize=8, 
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))


def plot_multiregion_profile_phase1(
    region_data: Dict,
    region_name: str,
    condition: str = 'air',
    output_path: Optional[Path] = None,
    show_perfusion: bool = True
) -> None:
    """
    Plot MLCO profile for a single region (Phase 1 version)
    
    Parameters:
    -----------
    region_data : dict
        Region data from multi-region analysis
        Contains: layers, zones, gradient
    region_name : str
        Region name (e.g., 'right_cortex')
    condition : str
        Condition name (e.g., 'air', 'oxygen')
    output_path : Path, optional
        Output file path
    show_perfusion : bool
        Include perfusion subplot if available
    """
    layers = region_data['layers']
    n_layers = len(layers)
    
    if n_layers == 0:
        print(f"Warning: No layers in {region_name}")
        return
    
    # Check if perfusion available
    has_perfusion = 'perfusion' in layers[0] and show_perfusion
    
    # Setup figure
    if has_perfusion:
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 9))
    else:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
        ax3 = None
    
    fig.suptitle(f"{region_name.replace('_', ' ').title()} - {condition.upper()}", 
                 fontsize=14, fontweight='bold')
    
    # Extract data
    layer_nums = [l['layer'] for l in layers]
    t2_values = [l['t2star']['median'] for l in layers]
    t2_stds = [l['t2star']['std'] for l in layers]
    r2_values = [l['r2star']['median'] for l in layers]
    r2_stds = [l['r2star']['std'] for l in layers]
    
    # Plot T2*
    ax1.errorbar(layer_nums, t2_values, yerr=t2_stds, 
                 marker='o', capsize=3, linewidth=2, markersize=6,
                 color='#E27A3F', label='T2*')
    ax1.set_ylabel('T2* (ms)', fontsize=11, fontweight='bold')
    ax1.set_title('T2* Profile', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Add zone shading
    if 'zones' in region_data:
        _add_zone_shading_phase1(ax1, region_data['zones'], n_layers)
    
    # Plot R2*
    ax2.errorbar(layer_nums, r2_values, yerr=r2_stds,
                 marker='s', capsize=3, linewidth=2, markersize=6,
                 color='#4A90E2', label='R2*')
    ax2.set_ylabel('R2* (Hz)', fontsize=11, fontweight='bold')
    ax2.set_xlabel('Layer Number', fontsize=11, fontweight='bold')
    ax2.set_title('R2* Profile', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    if 'zones' in region_data:
        _add_zone_shading_phase1(ax2, region_data['zones'], n_layers)
    
    # Plot perfusion if available
    if has_perfusion and ax3 is not None:
        perf_values = [l['perfusion']['median'] for l in layers if 'perfusion' in l]
        perf_stds = [l['perfusion']['std'] for l in layers if 'perfusion' in l]
        perf_layers = [l['layer'] for l in layers if 'perfusion' in l]
        
        if perf_values:
            ax3.errorbar(perf_layers, perf_values, yerr=perf_stds,
                        marker='^', capsize=3, linewidth=2, markersize=6,
                        color='#50C878', label='Perfusion')
            ax3.set_ylabel('Perfusion\n(ml/100g/min)', fontsize=11, fontweight='bold')
            ax3.set_xlabel('Layer Number', fontsize=11, fontweight='bold')
            ax3.set_title('Perfusion Profile', fontsize=10)
            ax3.grid(True, alpha=0.3)
            ax3.legend()
            
            if 'zones' in region_data:
                _add_zone_shading_phase1(ax3, region_data['zones'], n_layers)
    
    plt.tight_layout()
    
    # Save
    if output_path:
        for fmt in ['png', 'svg']:
            save_path = output_path.parent / f"{output_path.stem}.{fmt}"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {output_path.stem}")
    
    plt.close()


def plot_multiregion_overview_phase1(
    results: Dict,
    output_path: Optional[Path] = None,
    scan_label: str = "scan"
) -> None:
    """
    Create multi-panel overview of all regions (Phase 1 version)
    
    Parameters:
    -----------
    results : dict
        Multi-region analysis results
    output_path : Path, optional
        Output file path
    scan_label : str
        Label for the scan
    """
    if results.get('mode') != 'multi_region':
        print("Warning: Not multi-region results")
        return
    
    regions = results['regions']
    n_regions = len(regions)
    
    # Setup figure (2 columns, n_regions/2 rows)
    n_rows = (n_regions + 1) // 2
    fig, axes = plt.subplots(n_rows, 2, figsize=(14, 4*n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle(f"Multi-Region Overview - {scan_label.upper()}", 
                 fontsize=16, fontweight='bold')
    
    # Plot each region
    for idx, (region_name, region_data) in enumerate(regions.items()):
        row = idx // 2
        col = idx % 2
        ax = axes[row, col]
        
        layers = region_data['layers']
        if not layers:
            ax.text(0.5, 0.5, f'No data\n{region_name}', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.axis('off')
            continue
        
        # Extract T2* data
        layer_nums = [l['layer'] for l in layers]
        t2_values = [l['t2star']['median'] for l in layers]
        
        # Plot
        ax.plot(layer_nums, t2_values, marker='o', linewidth=2, 
               markersize=6, color='#E27A3F')
        ax.set_title(region_name.replace('_', ' ').title(), fontsize=11, fontweight='bold')
        ax.set_xlabel('Layer', fontsize=9)
        ax.set_ylabel('T2* (ms)', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Add zone shading
        if 'zones' in region_data:
            _add_zone_shading_phase1(ax, region_data['zones'], len(layers))
        
        # Add gradient info if available
        if 'gradient' in region_data and region_data['gradient']:
            grad = region_data['gradient']['t2star']['gradient']
            ax.text(0.02, 0.98, f'Gradient: {grad:+.1f} ms',
                   transform=ax.transAxes, va='top', fontsize=8,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Hide unused subplots
    for idx in range(n_regions, n_rows * 2):
        row = idx // 2
        col = idx % 2
        axes[row, col].axis('off')
    
    plt.tight_layout()
    
    # Save
    if output_path:
        for fmt in ['png', 'svg']:
            save_path = output_path.parent / f"{output_path.stem}.{fmt}"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved overview: {output_path.stem}")
    
    plt.close()


def plot_zone_heatmap(
    results: Dict,
    output_path: Optional[Path] = None
) -> None:
    """
    Compare zones across all regions as heatmap
    
    Parameters:
    -----------
    results : dict
        Multi-region analysis results
    output_path : Path, optional
        Output file path
    """
    if results.get('mode') != 'multi_region':
        print("Warning: Not multi-region results")
        return
    
    regions = results['regions']
    
    # Collect zone data
    region_names = []
    outer_t2 = []
    inner_t2 = []
    
    for region_name, region_data in regions.items():
        if 'zones' not in region_data:
            continue
        
        zones = region_data['zones']
        if 'outer' in zones and 'inner' in zones:
            region_names.append(region_name.replace('_', '\n'))
            outer_t2.append(zones['outer']['t2star']['median'])
            inner_t2.append(zones['inner']['t2star']['median'])
    
    if not region_names:
        print("Warning: No zone data to plot")
        return
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(region_names))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, outer_t2, width, label='Outer Zone',
                   color='#4A90E2', alpha=0.8)
    bars2 = ax.bar(x + width/2, inner_t2, width, label='Inner Zone',
                   color='#E27A3F', alpha=0.8)
    
    ax.set_xlabel('Region', fontsize=12, fontweight='bold')
    ax.set_ylabel('T2* (ms)', fontsize=12, fontweight='bold')
    ax.set_title('Zone Comparison Across Regions', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(region_names, fontsize=9)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}',
                   ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    # Save
    if output_path:
        for fmt in ['png', 'svg']:
            save_path = output_path.parent / f"{output_path.stem}.{fmt}"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved zone heatmap: {output_path.stem}")
    
    plt.close()


def plot_region_comparison(
    results: Dict,
    region_names: List[str],
    output_path: Optional[Path] = None
) -> None:
    """
    Compare specified regions
    
    Parameters:
    -----------
    results : dict
        Multi-region analysis results
    region_names : list
        List of region names to compare
    output_path : Path, optional
        Output file path
    """
    if results.get('mode') != 'multi_region':
        print("Warning: Not multi-region results")
        return
    
    regions = results['regions']
    n_regions = len(region_names)
    
    # Setup figure
    fig, axes = plt.subplots(1, n_regions, figsize=(4*n_regions, 5))
    if n_regions == 1:
        axes = [axes]
    
    fig.suptitle('Region Comparison', fontsize=14, fontweight='bold')
    
    # Plot each region
    for idx, region_name in enumerate(region_names):
        ax = axes[idx]
        
        if region_name not in regions:
            ax.text(0.5, 0.5, f'No data\n{region_name}',
                   ha='center', va='center', transform=ax.transAxes)
            ax.axis('off')
            continue
        
        region_data = regions[region_name]
        layers = region_data['layers']
        
        if not layers:
            ax.text(0.5, 0.5, f'No data\n{region_name}',
                   ha='center', va='center', transform=ax.transAxes)
            ax.axis('off')
            continue
        
        # Extract T2* data
        layer_nums = [l['layer'] for l in layers]
        t2_values = [l['t2star']['median'] for l in layers]
        
        # Plot
        ax.plot(layer_nums, t2_values, marker='o', linewidth=2, color='#E27A3F')
        ax.set_title(region_name.replace('_', ' ').title(), fontsize=11, fontweight='bold')
        ax.set_xlabel('Layer', fontsize=9)
        if idx == 0:
            ax.set_ylabel('T2* (ms)', fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    if output_path:
        for fmt in ['png', 'svg']:
            save_path = output_path.parent / f"{output_path.stem}.{fmt}"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved region comparison: {output_path.stem}")
    
    plt.close()


def plot_lateral_comparison(
    results: Dict,
    output_path: Optional[Path],
    anatomical_region: str = 'cortex'
) -> None:
    """
    Compare left vs right for a specific anatomical region
    
    Parameters:
    -----------
    results : dict
        Multi-region analysis results
    output_path : Path
        Output file path
    anatomical_region : str
        'cortex', 'medulla', or 'papilla'
    """
    if results.get('mode') != 'multi_region':
        print("Warning: Not multi-region results")
        return
    
    regions = results['regions']
    right_name = f'right_{anatomical_region}'
    left_name = f'left_{anatomical_region}'
    
    if right_name not in regions or left_name not in regions:
        print(f"Warning: Missing {right_name} or {left_name}")
        return
    
    right_data = regions[right_name]
    left_data = regions[left_name]
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Right kidney
    right_layers = [l['layer'] for l in right_data['layers']]
    right_t2 = [l['t2star']['median'] for l in right_data['layers']]
    ax.plot(right_layers, right_t2, marker='o', linewidth=2, 
           label='Right', color='#4A90E2')
    
    # Left kidney
    left_layers = [l['layer'] for l in left_data['layers']]
    left_t2 = [l['t2star']['median'] for l in left_data['layers']]
    ax.plot(left_layers, left_t2, marker='s', linewidth=2,
           label='Left', color='#E27A3F')
    
    ax.set_xlabel('Layer', fontsize=11, fontweight='bold')
    ax.set_ylabel('T2* (ms)', fontsize=11, fontweight='bold')
    ax.set_title(f'{anatomical_region.title()} - Left vs Right Comparison',
                fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    if output_path:
        for fmt in ['png', 'svg']:
            save_path = output_path.parent / f"{output_path.stem}.{fmt}"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved lateral comparison: {output_path.stem}")
    
    plt.close()


def generate_all_multiregion_plots(
    results: Dict,
    output_dir: Path,
    scan_label: str = "scan"
):
    """
    Generate complete set of multi-region plots (Phase 1)
    
    This is the wrapper function that boldpy_analyze.py calls.
    
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
    plot_multiregion_overview_phase1(results, overview_path, scan_label)
    
    # 2. Individual region profiles
    for region_name, region_data in results['regions'].items():
        region_path = output_dir / f"{scan_label}_{region_name}_profile"
        plot_multiregion_profile_phase1(region_data, region_name, 
                                       scan_label.split('_')[-1], region_path)
    
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


plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['font.size'] = 10
plt.rcParams['figure.dpi'] = 300

# ============================================================================
# COLOR SCHEMES
# ============================================================================

# Condition colors (in experimental order)
CONDITION_COLORS = {
    'oxygen_1': '#ff7f0e',   # Orange (first exposure)
    'air': '#1f77b4',        # Blue (baseline)
    'oxygen_2': '#ae5100'    # Darker orange (second exposure, 10% darker than oxygen_1)
}

# Region colors
REGION_COLORS = {
    'cortex': '#fe4a49',     # Red
    'medulla': '#fed766',    # Yellow
    'papilla': '#009fb7'     # Blue
}

# Asymmetry colors
ASYMMETRY_COLORS = {
    'normal': '#2ca02c',     # Green
    'flagged': '#d62728'     # Red
}

# Zone shading alpha
ZONE_ALPHA = 0.15

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _save_figure_formats(fig, base_path: Path, dpi: int = 300):
    """
    Save figure in multiple formats (PNG + SVG)
    
    Parameters:
    -----------
    fig : matplotlib.figure.Figure
        Figure to save
    base_path : Path
        Base path without extension
    dpi : int
        DPI for PNG output
    """
    base_path = Path(base_path)
    
    # Save PNG
    png_path = base_path.with_suffix('.png')
    fig.savefig(png_path, dpi=dpi, bbox_inches='tight')
    print(f"  ✓ Saved: {png_path.name}")
    
    # Save SVG
    svg_path = base_path.with_suffix('.svg')
    fig.savefig(svg_path, format='svg', bbox_inches='tight')
    print(f"  ✓ Saved: {svg_path.name}")


def _add_zone_shading_to_axis(ax, zones: Dict, n_layers: int, alpha: float = ZONE_ALPHA):
    """
    Add zone shading to axis based on 50/50 split
    
    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        Axis to add shading to
    zones : dict
        Zone definitions (used to verify zones exist)
    n_layers : int
        Total number of layers
    alpha : float
        Transparency for shading
    """
    if 'outer' in zones and 'inner' in zones:
        # Use 50/50 split - outer is first half, inner is second half
        split_point = n_layers // 2
        
        # Outer zone shading (first half)
        outer_start = 1
        outer_end = split_point
        ax.axvspan(outer_start - 0.5, outer_end + 0.5, 
                   color='lightblue', alpha=alpha, label='Outer zone')
        
        # Inner zone shading (second half)
        inner_start = split_point + 1
        inner_end = n_layers
        ax.axvspan(inner_start - 0.5, inner_end + 0.5,
                   color='lightcoral', alpha=alpha, label='Inner zone')


def _setup_plot_style():
    """Setup consistent plot styling"""
    plt.style.use('default')
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans'],
        'font.size': 10,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linestyle': '--',
        'axes.axisbelow': True
    })


# ============================================================================
# FUNCTION 1: MULTI-CONDITION PROFILE
# ============================================================================

def plot_multicondition_profile(
    averaged_results: Dict,
    region_name: str,
    output_path: Path,
    sample_id: str = 'sample'
) -> None:
    """
    Plot O₁→Air→O₂ temporal progression for a single region
    
    Creates 4-panel figure showing T2*, R2*, Perfusion, and gradient comparison
    across all three conditions.
    
    Parameters:
    -----------
    averaged_results : dict
        Dictionary with keys ['air', 'oxygen_1', 'oxygen_2'], each containing
        region data from average_bilateral_regions()
    region_name : str
        Region to plot ('cortex', 'medulla', or 'papilla')
    output_path : Path
        Base path for saving (without extension)
    sample_id : str, optional
        Sample identifier for title
        
    Output:
    -------
    Saves PNG and SVG files:
    - {output_path}.png
    - {output_path}.svg
    
    Example:
    --------
    >>> plot_multicondition_profile(
    ...     averaged_results=averaged,
    ...     region_name='cortex',
    ...     output_path='M1_cortex_multicondition',
    ...     sample_id='M1'
    ... )
    """
    _setup_plot_style()
    
    # Validate inputs (order matches experimental sequence)
    required_conditions = ['oxygen_1', 'air', 'oxygen_2']
    for cond in required_conditions:
        if cond not in averaged_results:
            raise ValueError(f"Missing condition: {cond}")
        if region_name not in averaged_results[cond]:
            raise ValueError(f"Region '{region_name}' not found in {cond} results")
    
    # Extract data for this region across all conditions
    region_data = {
        cond: averaged_results[cond][region_name]
        for cond in required_conditions
    }
    
    # Check if perfusion data is available
    has_perfusion = any('perfusion' in region_data[cond]['layers'][0] 
                       for cond in required_conditions 
                       if len(region_data[cond]['layers']) > 0)
    
    # Create figure with 3 or 4 panels depending on perfusion availability
    n_panels = 4 if has_perfusion else 3
    fig, axes = plt.subplots(1, n_panels, figsize=(5*n_panels, 5))
    fig.suptitle(f'{sample_id} - {region_name.capitalize()} - Multi-Condition Profile',
                 fontsize=14, fontweight='bold')
    
    # Get number of layers
    n_layers = region_data['air']['n_layers']
    layer_positions = np.arange(1, n_layers + 1)
    
    # Panel 1: T2* Profile
    ax_t2 = axes[0]
    for cond in required_conditions:
        data = region_data[cond]
        
        # Extract T2* values per layer
        t2_values = []
        t2_errors = []
        for layer in data['layers']:
            if 't2star' in layer:
                t2_values.append(layer['t2star']['median'])
                t2_errors.append(layer['t2star']['std'])
            else:
                t2_values.append(np.nan)
                t2_errors.append(np.nan)
        
        # Plot with error bars
        ax_t2.errorbar(layer_positions[:len(t2_values)], t2_values, yerr=t2_errors,
                       marker='o', linewidth=2, markersize=6,
                       color=CONDITION_COLORS[cond],
                       label=cond.replace('_', ' ').title(),
                       capsize=3, alpha=0.8)
    
    ax_t2.set_xlabel('Layer (Outer → Inner)', fontweight='bold')
    ax_t2.set_ylabel('T2* (ms)', fontweight='bold')
    ax_t2.set_title('T2* Profile', fontweight='bold')
    ax_t2.legend(loc='best', frameon=True, fancybox=True)
    ax_t2.grid(True, alpha=0.3)
    
    # Force integer x-axis ticks
    from matplotlib.ticker import MaxNLocator
    ax_t2.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    # Add zone shading
    if 'zones' in region_data['air']:
        _add_zone_shading_to_axis(ax_t2, region_data['air']['zones'], n_layers)
    
    # Panel 2: R2* Profile
    ax_r2 = axes[1]
    for cond in required_conditions:
        data = region_data[cond]
        
        # Extract R2* values per layer
        r2_values = []
        r2_errors = []
        for layer in data['layers']:
            if 'r2star' in layer:
                r2_values.append(layer['r2star']['median'])
                r2_errors.append(layer['r2star']['std'])
            else:
                r2_values.append(np.nan)
                r2_errors.append(np.nan)
        
        # Plot with error bars
        ax_r2.errorbar(layer_positions[:len(r2_values)], r2_values, yerr=r2_errors,
                       marker='s', linewidth=2, markersize=6,
                       color=CONDITION_COLORS[cond],
                       label=cond.replace('_', ' ').title(),
                       capsize=3, alpha=0.8)
    
    ax_r2.set_xlabel('Layer (Outer → Inner)', fontweight='bold')
    ax_r2.set_ylabel('R2* (Hz)', fontweight='bold')
    ax_r2.set_title('R2* Profile', fontweight='bold')
    ax_r2.legend(loc='best', frameon=True, fancybox=True)
    ax_r2.grid(True, alpha=0.3)
    
    # Force integer x-axis ticks
    ax_r2.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    # Add zone shading
    if 'zones' in region_data['air']:
        _add_zone_shading_to_axis(ax_r2, region_data['air']['zones'], n_layers)
    
    # Panel 3: Perfusion Profile (if available)
    if has_perfusion:
        ax_perf = axes[2]
        for cond in required_conditions:
            data = region_data[cond]
            
            # Extract Perfusion values per layer
            perf_values = []
            perf_errors = []
            for layer in data['layers']:
                if 'perfusion' in layer:
                    perf_values.append(layer['perfusion']['median'])
                    perf_errors.append(layer['perfusion']['std'])
                else:
                    perf_values.append(np.nan)
                    perf_errors.append(np.nan)
            
            # Plot with error bars
            ax_perf.errorbar(layer_positions[:len(perf_values)], perf_values, yerr=perf_errors,
                           marker='^', linewidth=2, markersize=6,
                           color=CONDITION_COLORS[cond],
                           label=cond.replace('_', ' ').title(),
                           capsize=3, alpha=0.8)
        
        ax_perf.set_xlabel('Layer (Outer → Inner)', fontweight='bold')
        ax_perf.set_ylabel('Perfusion (ml/100g/min)', fontweight='bold')
        ax_perf.set_title('Perfusion Profile', fontweight='bold')
        ax_perf.legend(loc='best', frameon=True, fancybox=True)
        ax_perf.grid(True, alpha=0.3)
        
        # Force integer x-axis ticks
        ax_perf.xaxis.set_major_locator(MaxNLocator(integer=True))
        
        # Add zone shading
        if 'zones' in region_data['air']:
            _add_zone_shading_to_axis(ax_perf, region_data['air']['zones'], n_layers)
    
    # Panel 4 (or 3): Gradient Comparison (Bar chart)
    ax_grad = axes[-1]
    gradient_values = []
    gradient_labels = []
    
    for cond in required_conditions:
        data = region_data[cond]
        if 'gradient' in data and 't2star' in data['gradient']:
            gradient_values.append(data['gradient']['t2star']['gradient'])
            gradient_labels.append(cond.replace('_', ' ').title())
    
    if gradient_values:
        bars = ax_grad.bar(gradient_labels, gradient_values,
                          color=[CONDITION_COLORS[cond] for cond in required_conditions],
                          alpha=0.8, edgecolor='black', linewidth=1.5)
        
        # Add value labels on bars
        for bar, val in zip(bars, gradient_values):
            height = bar.get_height()
            ax_grad.text(bar.get_x() + bar.get_width()/2., height,
                        f'{val:.2f}',
                        ha='center', va='bottom' if height >= 0 else 'top',
                        fontweight='bold', fontsize=10)
    
    ax_grad.set_ylabel('T2* Gradient (ms)', fontweight='bold')
    ax_grad.set_title('Gradient Comparison\n(Outer → Inner)', fontweight='bold')
    ax_grad.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax_grad.grid(True, alpha=0.3, axis='y')
    
    # Adjust layout and save
    plt.tight_layout()
    _save_figure_formats(fig, output_path)
    plt.close(fig)


# ============================================================================
# FUNCTION 2: REGIONS OVERVIEW
# ============================================================================

def plot_regions_overview(
    averaged_results: Dict,
    condition_name: str,
    output_path: Path,
    sample_id: str = 'sample'
) -> None:
    """
    Plot all regions side-by-side for a single condition
    
    Creates 3-panel figure showing cortex, medulla, and papilla profiles
    for one condition (e.g., air, oxygen_1, or oxygen_2).
    
    Parameters:
    -----------
    averaged_results : dict
        Results for one condition from average_bilateral_regions()
        Should contain keys: 'cortex', 'medulla', 'papilla'
    condition_name : str
        Condition being plotted (e.g., 'air', 'oxygen_1', 'oxygen_2')
    output_path : Path
        Base path for saving (without extension)
    sample_id : str, optional
        Sample identifier for title
        
    Output:
    -------
    Saves PNG and SVG files:
    - {output_path}.png
    - {output_path}.svg
    
    Example:
    --------
    >>> plot_regions_overview(
    ...     averaged_results=averaged['air'],
    ...     condition_name='air',
    ...     output_path='M1_air_all_regions',
    ...     sample_id='M1'
    ... )
    """
    _setup_plot_style()
    
    # Expected regions in order
    regions = ['cortex', 'medulla', 'papilla']
    
    # Validate inputs
    for region in regions:
        if region not in averaged_results:
            raise ValueError(f"Region '{region}' not found in results")
    
    # Create figure with 3 panels
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    
    # Format condition name for title
    condition_title = condition_name.replace('_', ' ').title()
    fig.suptitle(f'{sample_id} - {condition_title} - All Regions',
                 fontsize=14, fontweight='bold')
    
    # Plot each region
    for idx, region in enumerate(regions):
        ax = axes[idx]
        data = averaged_results[region]
        
        # Get number of layers and positions
        n_layers = data['n_layers']
        layer_positions = np.arange(1, n_layers + 1)
        
        # Extract T2* values per layer
        t2_values = []
        t2_errors = []
        for layer in data['layers']:
            if 't2star' in layer:
                t2_values.append(layer['t2star']['median'])
                t2_errors.append(layer['t2star']['std'])
            else:
                t2_values.append(np.nan)
                t2_errors.append(np.nan)
        
        # Plot T2* profile with error bars
        ax.errorbar(layer_positions[:len(t2_values)], t2_values, yerr=t2_errors,
                   marker='o', linewidth=2.5, markersize=7,
                   color=REGION_COLORS[region],
                   label=f'{region.capitalize()}',
                   capsize=4, alpha=0.9)
        
        # Add zone shading
        if 'zones' in data:
            _add_zone_shading_to_axis(ax, data['zones'], n_layers)
        
        # Add gradient annotation
        if 'gradient' in data and 't2star' in data['gradient']:
            gradient = data['gradient']['t2star']['gradient']
            
            # Determine direction from gradient value
            if gradient > 0:
                direction = 'increasing'
            elif gradient < 0:
                direction = 'decreasing'
            else:
                direction = 'flat'
            
            # Position annotation in upper right
            ax.text(0.98, 0.98, f'Gradient: {gradient:.2f} ms\n({direction})',
                   transform=ax.transAxes,
                   fontsize=9, fontweight='bold',
                   verticalalignment='top',
                   horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Formatting
        ax.set_xlabel('Layer (Outer → Inner)', fontweight='bold')
        ax.set_title(f'{region.capitalize()}', fontweight='bold', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left', frameon=True, fancybox=True)
        
        # Force integer x-axis ticks
        ax.set_xticks(np.arange(1, n_layers + 1))
        ax.set_xlim(0.5, n_layers + 0.5)
    
    # Only set ylabel on leftmost panel
    axes[0].set_ylabel('T2* (ms)', fontweight='bold')
    
    # Adjust layout and save
    plt.tight_layout()
    _save_figure_formats(fig, output_path)
    plt.close(fig)


# ============================================================================
# FUNCTION 3: OXYGEN RESPONSE SUMMARY
# ============================================================================

def plot_oxygen_response_summary(
    oxygen_response: Dict,
    output_path: Path,
    sample_id: str = 'sample'
) -> None:
    """
    Plot oxygen response magnitudes as bar charts
    
    Creates 3-panel figure showing:
    - Initial response (Oxygen 1 - Air)
    - Recovery response (Oxygen 2 - Air)
    - Reproducibility (Oxygen 2 - Oxygen 1)
    
    Parameters:
    -----------
    oxygen_response : dict
        Output from calculate_oxygen_response_multiregion()
        Should contain 'regions' with response data for each region
    output_path : Path
        Base path for saving (without extension)
    sample_id : str, optional
        Sample identifier for title
        
    Output:
    -------
    Saves PNG and SVG files:
    - {output_path}.png
    - {output_path}.svg
    
    Example:
    --------
    >>> plot_oxygen_response_summary(
    ...     oxygen_response=oxygen_response,
    ...     output_path='M1_oxygen_response',
    ...     sample_id='M1'
    ... )
    """
    _setup_plot_style()
    
    # Validate input
    if 'regions' not in oxygen_response:
        raise ValueError("oxygen_response must contain 'regions' key")
    
    # Expected regions in order
    regions = ['cortex', 'medulla', 'papilla']
    
    # Create figure with 3 panels
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f'{sample_id} - Oxygen Response Summary',
                 fontsize=14, fontweight='bold')
    
    # Response types to plot
    response_types = [
        ('initial_response', 'Initial Response\n(Oxygen 1 - Air)', axes[0]),
        ('recovery_response', 'Recovery Response\n(Oxygen 2 - Air)', axes[1]),
        ('reproducibility', 'Reproducibility\n(Oxygen 2 - Oxygen 1)', axes[2])
    ]
    
    for response_key, title, ax in response_types:
        # Collect data for each region
        region_names = []
        delta_values = []
        delta_errors = []
        colors = []
        
        for region in regions:
            if region in oxygen_response['regions']:
                region_data = oxygen_response['regions'][region]
                
                if response_key in region_data:
                    # Get T2* delta
                    if 't2star_delta' in region_data[response_key]:
                        delta_info = region_data[response_key]['t2star_delta']
                        
                        region_names.append(region.capitalize())
                        delta_values.append(delta_info['mean'])
                        
                        # Use std if available, otherwise 0
                        if 'std' in delta_info:
                            delta_errors.append(delta_info['std'])
                        else:
                            delta_errors.append(0)
                        
                        colors.append(REGION_COLORS[region])
        
        # Plot bars
        if delta_values:
            x_pos = np.arange(len(region_names))
            bars = ax.bar(x_pos, delta_values, yerr=delta_errors,
                          color=colors, alpha=0.8, 
                          edgecolor='black', linewidth=1.5,
                          capsize=5, error_kw={'linewidth': 2})
            
            # Add value labels on bars
            for i, (bar, val) in enumerate(zip(bars, delta_values)):
                height = bar.get_height()
                # Position label above or below bar depending on sign
                y_pos = height + (delta_errors[i] if height >= 0 else -delta_errors[i])
                va = 'bottom' if height >= 0 else 'top'
                
                ax.text(bar.get_x() + bar.get_width()/2., y_pos,
                       f'{val:.2f}',
                       ha='center', va=va,
                       fontweight='bold', fontsize=10)
            
            # Formatting
            ax.set_xticks(x_pos)
            ax.set_xticklabels(region_names, fontweight='bold')
            ax.set_ylabel('ΔT2* (ms)', fontweight='bold')
            ax.set_title(title, fontweight='bold', fontsize=11)
            ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
            ax.grid(True, alpha=0.3, axis='y')
    
    # Adjust layout and save
    plt.tight_layout()
    _save_figure_formats(fig, output_path)
    plt.close(fig)


# ============================================================================
# FUNCTION 4: GROUP COMPARISON OVERVIEW
# ============================================================================

def plot_group_comparison_overview(
    group1_averaged: Dict,
    group2_averaged: Dict,
    group1_name: str,
    group2_name: str,
    output_path: Path
) -> None:
    """
    Plot multi-panel group comparison (e.g., M1 vs M2)
    
    Creates 3×3 grid:
    - Rows: cortex, medulla, papilla
    - Columns: oxygen_1, air, oxygen_2 (experimental order)
    - Each panel: overlaid profiles for both groups
    
    Parameters:
    -----------
    group1_averaged : dict
        Averaged results for group 1 (e.g., M1)
        Should contain keys ['oxygen_1', 'air', 'oxygen_2']
    group2_averaged : dict
        Averaged results for group 2 (e.g., M2)
        Should contain keys ['oxygen_1', 'air', 'oxygen_2']
    group1_name : str
        Label for group 1 (e.g., 'WT', 'M1')
    group2_name : str
        Label for group 2 (e.g., 'KO', 'M2')
    output_path : Path
        Base path for saving (without extension)
        
    Output:
    -------
    Saves PNG and SVG files:
    - {output_path}.png
    - {output_path}.svg
    
    Example:
    --------
    >>> plot_group_comparison_overview(
    ...     group1_averaged=M1_averaged,
    ...     group2_averaged=M2_averaged,
    ...     group1_name='WT',
    ...     group2_name='KO',
    ...     output_path='WT_vs_KO_comparison'
    ... )
    """
    _setup_plot_style()
    
    # Expected regions and conditions (in experimental order)
    regions = ['cortex', 'medulla', 'papilla']
    conditions = ['oxygen_1', 'air', 'oxygen_2']
    
    # Validate inputs
    for cond in conditions:
        if cond not in group1_averaged or cond not in group2_averaged:
            raise ValueError(f"Missing condition: {cond}")
        for region in regions:
            if region not in group1_averaged[cond]:
                raise ValueError(f"Region '{region}' not found in group1 {cond}")
            if region not in group2_averaged[cond]:
                raise ValueError(f"Region '{region}' not found in group2 {cond}")
    
    # Create 3×3 grid
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    fig.suptitle(f'Group Comparison: {group1_name} vs {group2_name}',
                 fontsize=16, fontweight='bold')
    
    # Plot each combination
    for row_idx, region in enumerate(regions):
        for col_idx, condition in enumerate(conditions):
            ax = axes[row_idx, col_idx]
            
            # Get data for both groups
            g1_data = group1_averaged[condition][region]
            g2_data = group2_averaged[condition][region]
            
            # Get number of layers (use max of both)
            n_layers = max(g1_data['n_layers'], g2_data['n_layers'])
            layer_positions = np.arange(1, n_layers + 1)
            
            # Plot Group 1
            g1_t2_values = []
            g1_t2_errors = []
            for layer in g1_data['layers']:
                if 't2star' in layer:
                    g1_t2_values.append(layer['t2star']['median'])
                    g1_t2_errors.append(layer['t2star']['std'])
                else:
                    g1_t2_values.append(np.nan)
                    g1_t2_errors.append(np.nan)
            
            ax.errorbar(layer_positions[:len(g1_t2_values)], g1_t2_values, 
                       yerr=g1_t2_errors,
                       marker='o', linewidth=2.5, markersize=6,
                       color=CONDITION_COLORS[condition],
                       label=group1_name,
                       capsize=3, alpha=0.9, linestyle='-')
            
            # Plot Group 2
            g2_t2_values = []
            g2_t2_errors = []
            for layer in g2_data['layers']:
                if 't2star' in layer:
                    g2_t2_values.append(layer['t2star']['median'])
                    g2_t2_errors.append(layer['t2star']['std'])
                else:
                    g2_t2_values.append(np.nan)
                    g2_t2_errors.append(np.nan)
            
            ax.errorbar(layer_positions[:len(g2_t2_values)], g2_t2_values,
                       yerr=g2_t2_errors,
                       marker='s', linewidth=2.5, markersize=6,
                       color=CONDITION_COLORS[condition],
                       label=group2_name,
                       capsize=3, alpha=0.6, linestyle='--')
            
            # Add zone shading
            if 'zones' in g1_data:
                _add_zone_shading_to_axis(ax, g1_data['zones'], n_layers)
            
            # Formatting
            ax.set_xlabel('Layer', fontweight='bold' if row_idx == 2 else None)
            ax.grid(True, alpha=0.3)
            
            # Force integer x-axis ticks
            ax.set_xticks(np.arange(1, n_layers + 1))
            ax.set_xlim(0.5, n_layers + 0.5)
            
            # Add title to top row
            if row_idx == 0:
                cond_title = condition.replace('_', ' ').title()
                ax.set_title(cond_title, fontweight='bold', fontsize=12)
            
            # Add y-label to leftmost column
            if col_idx == 0:
                ax.set_ylabel(f'{region.capitalize()}\nT2* (ms)', 
                            fontweight='bold', fontsize=11)
            
            # Add legend to top-right panel
            if row_idx == 0 and col_idx == 2:
                ax.legend(loc='best', frameon=True, fancybox=True, fontsize=10)
    
    # Adjust layout and save
    plt.tight_layout()
    _save_figure_formats(fig, output_path)
    plt.close(fig)


# ============================================================================
# FUNCTION 5: ASYMMETRY CHECK (QC)
# ============================================================================

def plot_asymmetry_check(
    averaged_results: Dict,
    condition_name: str,
    output_path: Path,
    sample_id: str = 'sample',
    threshold: float = 0.20
) -> None:
    """
    Plot asymmetry QC showing left vs right kidney differences
    
    Creates bar chart showing asymmetry index for each region.
    Flags regions exceeding threshold (default 20%) in red.
    
    Parameters:
    -----------
    averaged_results : dict
        Results for one condition from average_bilateral_regions()
        Should contain 'asymmetry' data for each region
    condition_name : str
        Condition being checked (e.g., 'air', 'oxygen_1', 'oxygen_2')
    output_path : Path
        Base path for saving (without extension)
    sample_id : str, optional
        Sample identifier for title
    threshold : float, optional
        Asymmetry threshold for flagging (default 0.20 = 20%)
        
    Output:
    -------
    Saves PNG and SVG files:
    - {output_path}.png
    - {output_path}.svg
    
    Example:
    --------
    >>> plot_asymmetry_check(
    ...     averaged_results=averaged['air'],
    ...     condition_name='air',
    ...     output_path='M1_air_asymmetry_QC',
    ...     sample_id='M1'
    ... )
    """
    _setup_plot_style()
    
    # Expected regions in order
    regions = ['cortex', 'medulla', 'papilla']
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Format condition name for title
    condition_title = condition_name.replace('_', ' ').title()
    fig.suptitle(f'{sample_id} - {condition_title} - Asymmetry QC Check',
                 fontsize=14, fontweight='bold')
    
    # Collect asymmetry data
    region_names = []
    asymmetry_values = []
    colors = []
    flags = []
    source_info = []
    
    for region in regions:
        if region in averaged_results:
            region_data = averaged_results[region]
            
            if 'asymmetry' in region_data:
                asym_data = region_data['asymmetry']
                
                # Get mean asymmetry index
                if 'mean_asymmetry_index' in asym_data:
                    asym_value = asym_data['mean_asymmetry_index']
                    
                    region_names.append(region.capitalize())
                    asymmetry_values.append(asym_value * 100)  # Convert to percentage
                    
                    # Color based on threshold
                    if asym_value > threshold:
                        colors.append(ASYMMETRY_COLORS['flagged'])
                        flags.append('FLAGGED')
                    else:
                        colors.append(ASYMMETRY_COLORS['normal'])
                        flags.append('Normal')
                    
                    # Get source region info
                    if 'source_regions' in region_data:
                        sources = region_data['source_regions']
                        source_info.append(f"{sources[0]} + {sources[1]}")
                    else:
                        source_info.append('')
    
    # Plot bars
    if asymmetry_values:
        x_pos = np.arange(len(region_names))
        bars = ax.bar(x_pos, asymmetry_values,
                     color=colors, alpha=0.8,
                     edgecolor='black', linewidth=2)
        
        # Add threshold line
        ax.axhline(y=threshold * 100, color='red', linestyle='--', 
                   linewidth=2, label=f'Threshold ({threshold*100:.0f}%)',
                   alpha=0.7)
        
        # Add value labels on bars
        for i, (bar, val, flag) in enumerate(zip(bars, asymmetry_values, flags)):
            height = bar.get_height()
            
            # Label with value
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{val:.1f}%',
                   ha='center', va='bottom',
                   fontweight='bold', fontsize=11)
            
            # Add flag status below bar
            ax.text(bar.get_x() + bar.get_width()/2., -2,
                   flag,
                   ha='center', va='top',
                   fontsize=9, fontstyle='italic',
                   color='red' if flag == 'FLAGGED' else 'green')
            
            # Add source info at bottom
            if source_info[i]:
                ax.text(bar.get_x() + bar.get_width()/2., -5,
                       source_info[i],
                       ha='center', va='top',
                       fontsize=8, color='gray')
        
        # Formatting
        ax.set_xticks(x_pos)
        ax.set_xticklabels(region_names, fontweight='bold', fontsize=12)
        ax.set_ylabel('Asymmetry Index (%)', fontweight='bold', fontsize=12)
        ax.set_title('Left/Right Kidney Asymmetry', fontweight='bold', fontsize=11, pad=10)
        ax.set_ylim(-8, max(asymmetry_values) * 1.2)
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend(loc='upper right', frameon=True, fancybox=True)
        
        # Add interpretation text
        n_flagged = sum(1 for f in flags if f == 'FLAGGED')
        if n_flagged > 0:
            status_text = f'⚠️ {n_flagged} region(s) flagged for high asymmetry'
            status_color = 'red'
        else:
            status_text = '✓ All regions within normal asymmetry range'
            status_color = 'green'
        
        ax.text(0.02, 0.98, status_text,
               transform=ax.transAxes,
               fontsize=11, fontweight='bold',
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', 
                        edgecolor=status_color, linewidth=2, alpha=0.9))
    
    # Adjust layout and save
    plt.tight_layout()
    _save_figure_formats(fig, output_path)
    plt.close(fig)


# ============================================================================
# FUNCTION 5: ASYMMETRY CHECK (QC)
# ============================================================================

def plot_asymmetry_check(
    averaged_results: Dict,
    condition_name: str,
    output_path: Path,
    sample_id: str = 'sample'
) -> None:
    """
    Plot asymmetry QC check for left vs right kidneys
    
    Creates bar chart showing asymmetry index for each region.
    Flags regions with >20% left-right difference in red.
    
    Parameters:
    -----------
    averaged_results : dict
        Results from average_bilateral_regions() for one condition
        Should contain 'asymmetry' data for each region
    condition_name : str
        Condition being checked (e.g., 'air', 'oxygen_1', 'oxygen_2')
    output_path : Path
        Base path for saving (without extension)
    sample_id : str, optional
        Sample identifier for title
        
    Output:
    -------
    Saves PNG and SVG files:
    - {output_path}.png
    - {output_path}.svg
    
    Example:
    --------
    >>> plot_asymmetry_check(
    ...     averaged_results=averaged['air'],
    ...     condition_name='air',
    ...     output_path='M1_air_asymmetry_QC',
    ...     sample_id='M1'
    ... )
    """
    _setup_plot_style()
    
    # Expected regions in order
    regions = ['cortex', 'medulla', 'papilla']
    
    # Validate inputs
    for region in regions:
        if region not in averaged_results:
            raise ValueError(f"Region '{region}' not found in results")
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Format condition name for title
    condition_title = condition_name.replace('_', ' ').title()
    fig.suptitle(f'{sample_id} - {condition_title} - Asymmetry QC Check',
                 fontsize=14, fontweight='bold')
    
    # Collect asymmetry data
    region_names = []
    asymmetry_values = []
    colors = []
    source_regions_list = []
    
    for region in regions:
        region_data = averaged_results[region]
        
        if 'asymmetry' in region_data:
            asym = region_data['asymmetry']
            
            # Get mean asymmetry index (as percentage)
            asym_index = asym['mean_asymmetry_index'] * 100  # Convert to percentage
            
            region_names.append(region.capitalize())
            asymmetry_values.append(asym_index)
            
            # Determine color based on threshold (20%)
            is_flagged = asym.get('asymmetry_flag', False)
            colors.append(ASYMMETRY_COLORS['flagged'] if is_flagged else ASYMMETRY_COLORS['normal'])
            
            # Get source regions for annotation
            if 'source_regions' in region_data:
                source_regions_list.append(region_data['source_regions'])
            else:
                source_regions_list.append([])
    
    # Plot bars
    if asymmetry_values:
        x_pos = np.arange(len(region_names))
        bars = ax.bar(x_pos, asymmetry_values,
                      color=colors, alpha=0.8,
                      edgecolor='black', linewidth=1.5)
        
        # Add value labels on bars
        for bar, val in zip(bars, asymmetry_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.1f}%',
                   ha='center', va='bottom',
                   fontweight='bold', fontsize=11)
        
        # Add threshold line at 20%
        ax.axhline(y=20, color='red', linestyle='--', linewidth=2,
                  label='20% Threshold', alpha=0.7)
        
        # Formatting
        ax.set_xticks(x_pos)
        ax.set_xticklabels(region_names, fontweight='bold', fontsize=12)
        ax.set_ylabel('Asymmetry Index (%)', fontweight='bold', fontsize=12)
        ax.set_title('Left vs Right Kidney Comparison', fontweight='bold', fontsize=12, pad=10)
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend(loc='upper right', frameon=True, fancybox=True, fontsize=10)
        
        # Set y-axis to start at 0
        ax.set_ylim(bottom=0)
        
        # Add interpretation text
        n_flagged = sum([1 for c in colors if c == ASYMMETRY_COLORS['flagged']])
        if n_flagged > 0:
            interpretation = f"⚠️  {n_flagged} region(s) flagged with >20% asymmetry"
            text_color = 'red'
        else:
            interpretation = "✓ All regions show normal asymmetry (<20%)"
            text_color = 'green'
        
        ax.text(0.02, 0.98, interpretation,
               transform=ax.transAxes,
               fontsize=11, fontweight='bold',
               verticalalignment='top',
               color=text_color,
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Adjust layout and save
    plt.tight_layout()
    _save_figure_formats(fig, output_path)
    plt.close(fig)


# ============================================================================
# TESTING
# ============================================================================


# ============================================================================
# PHASE 2 CONTINUOUS PROFILES (Whole-kidney cortex→medulla→papilla)
# ============================================================================
#
# These functions show the entire kidney as a continuous spatial profile,
# respecting the sequential layer numbering from outer cortex to inner papilla.
# Includes tissue viability thresholds and perfusion integration.
#
# Functions:
#   - plot_whole_kidney_continuous()    - Single sample continuous profile
#   - plot_whole_kidney_comparison()    - Group comparison with perfusion
# ============================================================================

# Configurable tissue viability thresholds
DEFAULT_THRESHOLDS = {
    't2star_fluid': 40.0,           # T2* > 40 ms = likely fluid/degraded tissue
    't2star_ischemic': 8.0,          # T2* < 8 ms = severe ischemia
    'perfusion_cortex_min': 250.0,   # Cortex normal > 250 ml/100g/min
    'perfusion_medulla_min': 100.0,  # Medulla normal > 100 ml/100g/min
    'perfusion_severe': 50.0,        # < 50 anywhere is severe compromise
    'gradient_abnormal': 40.0        # |gradient| > 40 ms is abnormal
}


def plot_whole_kidney_continuous(
    averaged_results: Dict,
    condition: str,
    output_path: Path,
    sample_id: str = 'sample',
    include_perfusion: bool = True,
    t2_threshold: float = 40.0,
    perfusion_threshold: float = 200.0,
    region_order: List[str] = ['cortex', 'medulla', 'papilla']
) -> None:
    """
    Plot continuous profile across entire kidney (cortex→medulla→papilla)
    
    Creates visualization showing all regions in spatial/anatomical order as
    a single continuous profile, respecting sequential layer numbering.
    
    Parameters:
    -----------
    averaged_results : dict
        Results from average_bilateral_regions() for ONE condition
        Should contain keys for each region in region_order
    condition : str
        Condition name (e.g., 'air', 'oxygen_1')
    output_path : Path
        Base path for saving (without extension)
    sample_id : str
        Sample identifier for title
    include_perfusion : bool
        Include perfusion panel if data available
    t2_threshold : float
        T2* threshold (ms) - above = likely fluid
    perfusion_threshold : float
        Perfusion threshold (ml/100g/min) - below = compromised
    region_order : list
        Order of regions from outer to inner
        
    Output:
    -------
    Saves PNG and SVG showing:
    - T2* profile with threshold line
    - R2* profile
    - Perfusion profile (if available)
    - Region boundaries marked
    - Whole-kidney gradient calculated
    
    Example:
    --------
    >>> plot_whole_kidney_continuous(
    ...     averaged_results=averaged['air'],
    ...     condition='air',
    ...     output_path='M1_air_continuous',
    ...     sample_id='M1'
    ... )
    """
    # Check all regions present
    for region in region_order:
        if region not in averaged_results:
            raise ValueError(f"Missing region: {region}")
    
    # Build continuous arrays in anatomical order
    continuous_t2, continuous_t2_err = [], []
    continuous_r2, continuous_r2_err = [], []
    continuous_perf, continuous_perf_err = [], []
    region_boundaries = []
    cumulative_layers = 0
    has_perfusion = False
    
    for region in region_order:
        region_data = averaged_results[region]
        n_layers = region_data['n_layers']
        
        # Mark boundary
        region_boundaries.append({
            'name': region,
            'start': cumulative_layers,
            'end': cumulative_layers + n_layers,
            'color': REGION_COLORS[region]
        })
        
        # Create a mapping of layer numbers to data
        # Note: averaged layers may not have 'layer_number' field
        # so we infer from position (index + 1)
        layer_dict = {}
        for idx, layer in enumerate(region_data['layers'], start=1):
            # Try to get layer_number if it exists, otherwise use position
            layer_num = layer.get('layer_number', idx)
            layer_dict[layer_num] = layer
        
        # Fill ALL expected layers (use NaN for missing layers)
        for layer_num in range(1, n_layers + 1):
            if layer_num in layer_dict:
                # Layer exists - use actual data
                layer = layer_dict[layer_num]
                continuous_t2.append(layer['t2star']['median'])
                continuous_t2_err.append(layer['t2star']['std'])
                continuous_r2.append(layer['r2star']['median'])
                continuous_r2_err.append(layer['r2star']['std'])
                
                if 'perfusion' in layer:
                    continuous_perf.append(layer['perfusion']['median'])
                    continuous_perf_err.append(layer['perfusion']['std'])
                    has_perfusion = True
                else:
                    continuous_perf.append(np.nan)
                    continuous_perf_err.append(np.nan)
            else:
                # Layer missing (0 pixels) - use NaN
                continuous_t2.append(np.nan)
                continuous_t2_err.append(np.nan)
                continuous_r2.append(np.nan)
                continuous_r2_err.append(np.nan)
                continuous_perf.append(np.nan)
                continuous_perf_err.append(np.nan)
        
        cumulative_layers += n_layers
    
    # Verify array lengths (should now always match)
    actual_length = len(continuous_t2)
    if actual_length != cumulative_layers:
        raise ValueError(
            f"Array length mismatch after padding: expected {cumulative_layers} layers, "
            f"got {actual_length} data points. This should not happen!"
        )
    
    # Setup figure
    layer_positions = np.arange(1, cumulative_layers + 1)
    n_panels = 3 if (has_perfusion and include_perfusion) else 2
    
    fig, axes = plt.subplots(n_panels, 1, figsize=(14, 4*n_panels))
    if n_panels == 1:
        axes = [axes]
    
    fig.suptitle(f'{sample_id} - Whole Kidney Continuous Profile - {condition.upper()}',
                 fontsize=14, fontweight='bold')
    
    # Panel 1: T2*
    ax_t2 = axes[0]
    ax_t2.errorbar(layer_positions, continuous_t2, yerr=continuous_t2_err,
                   marker='o', linewidth=2, markersize=5,
                   color=CONDITION_COLORS.get(condition, '#1f77b4'),
                   capsize=3, alpha=0.8, label='T2*')
    
    # Threshold line
    ax_t2.axhline(y=t2_threshold, color='red', linestyle='--', linewidth=2,
                 label=f'Fluid threshold ({t2_threshold} ms)', alpha=0.7)
    
    ax_t2.set_ylabel('T2* (ms)', fontweight='bold', fontsize=11)
    ax_t2.set_title('T2* Profile (Cortex → Medulla → Papilla)', 
                   fontweight='bold', fontsize=12)
    ax_t2.legend(loc='best', frameon=True, fancybox=True)
    ax_t2.grid(True, alpha=0.3)
    
    # Force integer x-axis ticks
    from matplotlib.ticker import MaxNLocator
    ax_t2.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    # Region shading and labels
    for boundary in region_boundaries:
        ax_t2.axvspan(boundary['start'] + 0.5, boundary['end'] + 0.5,
                     alpha=0.1, color=boundary['color'], zorder=0)
        mid = (boundary['start'] + boundary['end']) / 2 + 0.5
        y_pos = ax_t2.get_ylim()[1] * 0.95
        ax_t2.text(mid, y_pos, boundary['name'].upper(),
                  ha='center', va='top', fontsize=10, fontweight='bold',
                  bbox=dict(boxstyle='round', facecolor='white',
                           edgecolor=boundary['color'], alpha=0.8))
    
    # Panel 2: R2*
    ax_r2 = axes[1]
    ax_r2.errorbar(layer_positions, continuous_r2, yerr=continuous_r2_err,
                   marker='s', linewidth=2, markersize=5,
                   color=CONDITION_COLORS.get(condition, '#1f77b4'),
                   capsize=3, alpha=0.8, label='R2*')
    
    ax_r2.set_ylabel('R2* (Hz)', fontweight='bold', fontsize=11)
    ax_r2.set_title('R2* Profile (Cortex → Medulla → Papilla)',
                   fontweight='bold', fontsize=12)
    ax_r2.legend(loc='best', frameon=True, fancybox=True)
    ax_r2.grid(True, alpha=0.3)
    
    # Force integer x-axis ticks
    ax_r2.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    for boundary in region_boundaries:
        ax_r2.axvspan(boundary['start'] + 0.5, boundary['end'] + 0.5,
                     alpha=0.1, color=boundary['color'], zorder=0)
    
    if not has_perfusion or not include_perfusion:
        ax_r2.set_xlabel('Layer Position (Outer → Inner)', fontweight='bold', fontsize=11)
    
    # Panel 3: Perfusion (if available)
    if has_perfusion and include_perfusion:
        ax_perf = axes[2]
        
        # Convert to numpy arrays first for proper indexing
        continuous_perf_arr = np.array(continuous_perf)
        continuous_perf_err_arr = np.array(continuous_perf_err)
        
        # Filter out NaN values
        valid_idx = ~np.isnan(continuous_perf_arr)
        
        if np.sum(valid_idx) > 0:  # Only plot if we have valid data
            valid_pos = layer_positions[valid_idx]
            valid_perf = continuous_perf_arr[valid_idx]
            valid_perf_err = continuous_perf_err_arr[valid_idx]
            
            # Verify array sizes match
            if len(valid_pos) == len(valid_perf) == len(valid_perf_err):
                ax_perf.errorbar(valid_pos, valid_perf, yerr=valid_perf_err,
                                marker='^', linewidth=2, markersize=5,
                                color='#50C878', capsize=3, alpha=0.8, label='Perfusion')
            else:
                print(f"  ⚠ Warning: Perfusion array size mismatch: pos={len(valid_pos)}, "
                      f"perf={len(valid_perf)}, err={len(valid_perf_err)}")
        
        ax_perf.axhline(y=perfusion_threshold, color='red', linestyle='--',
                       linewidth=2, alpha=0.7,
                       label=f'Compromise threshold ({perfusion_threshold} ml/100g/min)')
        
        ax_perf.set_ylabel('Perfusion (ml/100g/min)', fontweight='bold', fontsize=11)
        ax_perf.set_xlabel('Layer Position (Outer → Inner)', fontweight='bold', fontsize=11)
        ax_perf.set_title('Perfusion Profile (Cortex → Medulla → Papilla)',
                         fontweight='bold', fontsize=12)
        ax_perf.legend(loc='best', frameon=True, fancybox=True)
        ax_perf.grid(True, alpha=0.3)
        
        # Force integer x-axis ticks
        ax_perf.xaxis.set_major_locator(MaxNLocator(integer=True))
        
        for boundary in region_boundaries:
            ax_perf.axvspan(boundary['start'] + 0.5, boundary['end'] + 0.5,
                           alpha=0.1, color=boundary['color'], zorder=0)
    
    # Calculate whole-kidney gradient
    cortex_end = region_boundaries[0]['end']
    papilla_start = region_boundaries[-1]['start']
    
    cortex_t2_mean = np.nanmean(continuous_t2[:cortex_end])
    papilla_t2_mean = np.nanmean(continuous_t2[papilla_start:])
    gradient = papilla_t2_mean - cortex_t2_mean
    
    gradient_text = f"Whole-Kidney T2* Gradient: {gradient:+.1f} ms\n"
    gradient_text += f"(Cortex: {cortex_t2_mean:.1f} ms → Papilla: {papilla_t2_mean:.1f} ms)"
    
    fig.text(0.5, 0.02, gradient_text, ha='center', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout(rect=[0, 0.04, 1, 0.98])
    _save_figure_formats(fig, output_path)
    plt.close(fig)


def plot_whole_kidney_comparison(
    group1_averaged: Dict,
    group2_averaged: Dict,
    condition: str,
    output_path: Path,
    group1_name: str = 'Group 1',
    group2_name: str = 'Group 2',
    include_perfusion: bool = True,
    t2_threshold: float = 40.0,
    perfusion_threshold: float = 200.0,
    region_order: List[str] = ['cortex', 'medulla', 'papilla']
) -> None:
    """
    Plot whole-kidney comparison between two groups for ONE condition
    
    Shows continuous profiles with Group 1 vs Group 2 overlaid, including
    perfusion comparison.
    
    Parameters:
    -----------
    group1_averaged : dict
        Averaged results for group 1 (one condition)
    group2_averaged : dict
        Averaged results for group 2 (one condition)
    condition : str
        Condition name
    output_path : Path
        Base path for saving
    group1_name : str
        Label for group 1
    group2_name : str
        Label for group 2
    include_perfusion : bool
        Include perfusion panel
    t2_threshold : float
        T2* threshold (ms)
    perfusion_threshold : float
        Perfusion threshold (ml/100g/min)
    region_order : list
        Region order from outer to inner
        
    Output:
    -------
    Saves PNG and SVG with 2 panels:
    - T2* comparison (overlaid)
    - Perfusion comparison (overlaid, if available)
    
    Example:
    --------
    >>> plot_whole_kidney_comparison(
    ...     group1_averaged=M1_averaged['air'],
    ...     group2_averaged=M2_averaged['air'],
    ...     condition='air',
    ...     output_path='WT_vs_KO_air_continuous',
    ...     group1_name='WT',
    ...     group2_name='KO'
    ... )
    """
    # Build continuous arrays for both groups
    def build_continuous(data, regions):
        t2_vals, t2_errs = [], []
        r2_vals, r2_errs = [], []
        perf_vals, perf_errs = [], []
        boundaries = []
        pos = 0
        has_perf = False
        
        for region in regions:
            if region not in data:
                continue
            
            n_layers = data[region]['n_layers']
            
            boundaries.append({
                'name': region,
                'start': pos,
                'end': pos + n_layers,
                'color': REGION_COLORS[region]
            })
            
            # Create mapping of layer numbers to data
            # Note: averaged layers may not have 'layer_number' field
            # so we infer from position (index + 1)
            layer_dict = {}
            for idx, layer in enumerate(data[region]['layers'], start=1):
                # Try to get layer_number if it exists, otherwise use position
                layer_num = layer.get('layer_number', idx)
                layer_dict[layer_num] = layer
            
            # Fill ALL expected layers (use NaN for missing)
            for layer_num in range(1, n_layers + 1):
                if layer_num in layer_dict:
                    # Layer exists
                    layer = layer_dict[layer_num]
                    t2_vals.append(layer['t2star']['median'])
                    t2_errs.append(layer['t2star']['std'])
                    r2_vals.append(layer['r2star']['median'])
                    r2_errs.append(layer['r2star']['std'])
                    
                    if 'perfusion' in layer:
                        perf_vals.append(layer['perfusion']['median'])
                        perf_errs.append(layer['perfusion']['std'])
                        has_perf = True
                    else:
                        perf_vals.append(np.nan)
                        perf_errs.append(np.nan)
                else:
                    # Layer missing (0 pixels) - use NaN
                    t2_vals.append(np.nan)
                    t2_errs.append(np.nan)
                    r2_vals.append(np.nan)
                    r2_errs.append(np.nan)
                    perf_vals.append(np.nan)
                    perf_errs.append(np.nan)
            
            pos += n_layers
        
        return {
            't2': (t2_vals, t2_errs),
            'r2': (r2_vals, r2_errs),
            'perf': (perf_vals, perf_errs),
            'boundaries': boundaries,
            'has_perfusion': has_perf
        }
    
    g1 = build_continuous(group1_averaged, region_order)
    g2 = build_continuous(group2_averaged, region_order)
    
    # Setup figure
    n_panels = 2 if (g1['has_perfusion'] and include_perfusion) else 1
    fig, axes = plt.subplots(n_panels, 1, figsize=(14, 5*n_panels))
    if n_panels == 1:
        axes = [axes]
    
    fig.suptitle(f'Whole Kidney Comparison - {condition.upper()}\n{group1_name} vs {group2_name}',
                 fontsize=14, fontweight='bold')
    
    x_pos = np.arange(1, len(g1['t2'][0]) + 1)
    
    # Panel 1: T2* Comparison
    ax_t2 = axes[0]
    
    ax_t2.errorbar(x_pos, g1['t2'][0], yerr=g1['t2'][1],
                   marker='o', linewidth=2, markersize=5, linestyle='-',
                   color='#1f77b4', capsize=3, alpha=0.8,
                   label=f'{group1_name}')
    
    ax_t2.errorbar(x_pos, g2['t2'][0], yerr=g2['t2'][1],
                   marker='s', linewidth=2, markersize=5, linestyle='--',
                   color='#ff7f0e', capsize=3, alpha=0.8,
                   label=f'{group2_name}')
    
    ax_t2.axhline(y=t2_threshold, color='red', linestyle=':', linewidth=2,
                 label=f'Fluid threshold ({t2_threshold} ms)', alpha=0.7)
    
    ax_t2.set_ylabel('T2* (ms)', fontweight='bold', fontsize=11)
    ax_t2.set_title('T2* Profile Comparison (Cortex → Medulla → Papilla)',
                   fontweight='bold', fontsize=12)
    ax_t2.legend(loc='best', frameon=True, fancybox=True, fontsize=9)
    ax_t2.grid(True, alpha=0.3)
    
    # Force integer x-axis ticks
    from matplotlib.ticker import MaxNLocator
    ax_t2.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    # Region shading and labels
    for boundary in g1['boundaries']:
        ax_t2.axvspan(boundary['start'] + 0.5, boundary['end'] + 0.5,
                     alpha=0.1, color=boundary['color'], zorder=0)
        mid = (boundary['start'] + boundary['end']) / 2 + 0.5
        y_pos = ax_t2.get_ylim()[1] * 0.95
        ax_t2.text(mid, y_pos, boundary['name'].upper(),
                  ha='center', va='top', fontsize=10, fontweight='bold',
                  bbox=dict(boxstyle='round', facecolor='white',
                           edgecolor=boundary['color'], alpha=0.8))
    
    if not (g1['has_perfusion'] and include_perfusion):
        ax_t2.set_xlabel('Layer Position (Outer → Inner)', fontweight='bold', fontsize=11)
    
    # Panel 2: Perfusion Comparison (if available)
    if g1['has_perfusion'] and include_perfusion:
        ax_perf = axes[1]
        
        # Convert to numpy arrays for proper indexing
        g1_perf_arr = np.array(g1['perf'][0])
        g1_perf_err_arr = np.array(g1['perf'][1])
        g2_perf_arr = np.array(g2['perf'][0])
        g2_perf_err_arr = np.array(g2['perf'][1])
        
        # Find valid (non-NaN) indices
        g1_valid = ~np.isnan(g1_perf_arr)
        g2_valid = ~np.isnan(g2_perf_arr)
        
        # Plot group 1 if has valid data
        if np.sum(g1_valid) > 0:
            g1_pos = x_pos[g1_valid]
            g1_vals = g1_perf_arr[g1_valid]
            g1_errs = g1_perf_err_arr[g1_valid]
            
            if len(g1_pos) == len(g1_vals) == len(g1_errs):
                ax_perf.errorbar(g1_pos, g1_vals, yerr=g1_errs,
                                marker='^', linewidth=2, markersize=5, linestyle='-',
                                color='#1f77b4', capsize=3, alpha=0.8,
                                label=f'{group1_name}')
        
        # Plot group 2 if has valid data
        if np.sum(g2_valid) > 0:
            g2_pos = x_pos[g2_valid]
            g2_vals = g2_perf_arr[g2_valid]
            g2_errs = g2_perf_err_arr[g2_valid]
            
            if len(g2_pos) == len(g2_vals) == len(g2_errs):
                ax_perf.errorbar(g2_pos, g2_vals, yerr=g2_errs,
                                marker='v', linewidth=2, markersize=5, linestyle='--',
                                color='#ff7f0e', capsize=3, alpha=0.8,
                                label=f'{group2_name}')
        
        ax_perf.axhline(y=perfusion_threshold, color='red', linestyle=':',
                       linewidth=2, alpha=0.7,
                       label=f'Compromise threshold ({perfusion_threshold} ml/100g/min)')
        
        ax_perf.set_ylabel('Perfusion (ml/100g/min)', fontweight='bold', fontsize=11)
        ax_perf.set_xlabel('Layer Position (Outer → Inner)', fontweight='bold', fontsize=11)
        ax_perf.set_title('Perfusion Profile Comparison (Cortex → Medulla → Papilla)',
                         fontweight='bold', fontsize=12)
        ax_perf.legend(loc='best', frameon=True, fancybox=True, fontsize=9)
        ax_perf.grid(True, alpha=0.3)
        
        # Force integer x-axis ticks
        ax_perf.xaxis.set_major_locator(MaxNLocator(integer=True))
        
        for boundary in g1['boundaries']:
            ax_perf.axvspan(boundary['start'] + 0.5, boundary['end'] + 0.5,
                           alpha=0.1, color=boundary['color'], zorder=0)
    
    plt.tight_layout()
    _save_figure_formats(fig, output_path)
    plt.close(fig)


