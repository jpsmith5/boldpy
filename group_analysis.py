#!/usr/bin/env python3
"""
Group-Level MLCO Profile Analysis for BOLD MRI
===============================================

Project-level complement to boldpy_analyze.py. Reads per-sample analysis
JSON outputs and generates cross-group comparison plots using per-layer
bilateral MLCO profiles. Supports n > 1 per group — computes mean ± SEM.
Zones are shaded using the standard 5-zone kidney config.

Usage:
    cd boldpy
    python group_analysis.py --pep code/analysis/captopril/project_config.yaml

project_config.yaml format (PEP):
    See pipeline/examples/project_config.yaml for a full template.
    Key sections used by this script:
      sample_table   — CSV with columns: sample_name, group, ...
      output_dir     — where group-level figures are saved
      analysis_dir   — where per-sample _complete_analysis.json live
      hematology_csv — optional path to hematology CSV
      group_styles   — per-group colors / line styles / labels (keyed by group id)

Outputs saved to: {output_dir}/
"""

import argparse
import csv
import json
import peppy
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
from scipy.stats import mannwhitneyu, linregress
from pathlib import Path
import warnings

# ── Configuration ─────────────────────────────────────────────────────────────

BASE = Path(__file__).resolve().parents[3]          # BOLD_MRI root
ANALYSIS_DIR = BASE / 'processed' / 'analysis'
OUTPUT_DIR   = BASE / 'processed' / 'analysis' / 'group_comparison'
HEMATOLOGY_CSV = None   # Set via --config hematology_csv key

# Populated at runtime via --pep project_config.yaml
GROUPS       = {}
GROUP_STYLES = {}


def load_pep_project(pep_path):
    """Load GROUPS, GROUP_STYLES, OUTPUT_DIR, ANALYSIS_DIR, and optional
    HEMATOLOGY_CSV from a PEP project_config.yaml."""
    global GROUPS, GROUP_STYLES, OUTPUT_DIR, ANALYSIS_DIR, HEMATOLOGY_CSV
    pep_path = Path(pep_path).resolve()
    project  = peppy.Project(str(pep_path))
    samples  = project.sample_table
    styles   = project.config.get('group_styles', {})

    # Build GROUPS and GROUP_STYLES keyed by display label (preserves existing API)
    GROUPS       = {}
    GROUP_STYLES = {}
    for grp_id, df in samples.groupby('group', sort=False):
        style   = dict(styles.get(grp_id, {}))
        display = style.pop('label', grp_id)
        GROUPS[display]       = list(df['sample_name'])
        GROUP_STYLES[display] = style

    # Project-level path overrides
    cfg = project.config
    base = pep_path.parent

    def _resolve(key, default):
        val = cfg.get(key)
        if not val:
            return default
        p = Path(val)
        return p if p.is_absolute() else base / p

    OUTPUT_DIR   = _resolve('group_output_dir', BASE / 'processed' / 'analysis' / 'group_comparison')
    ANALYSIS_DIR = _resolve('analysis_dir',     BASE / 'processed' / 'analysis')

    hema = cfg.get('hematology_csv')
    if hema:
        HEMATOLOGY_CSV = Path(hema) if Path(hema).is_absolute() else base / hema

ZONE_COLORS = {
    'outer_cortex':  '#E8F4F8',
    'inner_cortex':  '#C5E3ED',
    'cmj':           '#FFE5CC',
    'outer_medulla': '#FFD9B3',
    'inner_medulla': '#FFC999',
}
ZONE_ORDER = ['outer_cortex', 'inner_cortex', 'cmj', 'outer_medulla', 'inner_medulla']
ZONE_LABELS = {
    'outer_cortex':  'Outer\nCortex',
    'inner_cortex':  'Inner\nCortex',
    'cmj':           'CMJ',
    'outer_medulla': 'Outer\nMedulla',
    'inner_medulla': 'Inner\nMedulla',
}

CONDITION_LABELS = {
    'oxygen_1': 'Pre-O₂ (100%)',
    'air':      'Air (21%)',
    'oxygen_2': 'Post-O₂ (100%)',
}
CONDITION_ORDER = ['oxygen_1', 'air', 'oxygen_2']

METRIC_LABELS = {
    't2star':   'T₂* (ms)',
    'r2star':   'R₂* (s⁻¹)',
    'perfusion': 'Perfusion (a.u.)',
}

# ── Data loading ───────────────────────────────────────────────────────────────

def load_result(sample_id: str) -> dict:
    path = ANALYSIS_DIR / sample_id / f'{sample_id}_complete_analysis.json'
    with open(path) as f:
        return json.load(f)


def get_bilateral_layers(result: dict, condition: str) -> dict:
    """Return {layer_num: layer_dict} for the bilateral average."""
    layers = result['conditions'][condition]['bilateral']['layers']
    return {l['layer']: l for l in layers}


def load_group(sample_ids: list) -> list:
    """Load results for a group of samples."""
    results = []
    for sid in sample_ids:
        try:
            results.append(load_result(sid))
            print(f"  Loaded {sid}")
        except FileNotFoundError:
            print(f"  WARNING: {sid} not found, skipping")
    return results


# ── Profile extraction ─────────────────────────────────────────────────────────

def extract_profile(results: list, condition: str, metric: str, stat: str = 'mean') -> dict:
    """
    For each layer, collect per-sample values and compute group stats.

    Returns:
        dict with keys: layers, group_mean, group_sem, group_std, n, per_sample
    """
    # Collect all bilateral layers across samples
    per_layer = {}  # layer_num -> list of values
    for r in results:
        if condition not in r.get('conditions', {}):
            continue
        ldict = get_bilateral_layers(r, condition)
        for layer_num, ldata in ldict.items():
            if metric in ldata and stat in ldata[metric]:
                val = ldata[metric][stat]
                if val is None or (isinstance(val, float) and np.isnan(val)):
                    continue
                # Reject physiologically implausible values (artifact / DICOM overflow)
                if metric == 't2star' and val > 200:
                    continue
                if metric == 'r2star' and val > 500:
                    continue
                per_layer.setdefault(layer_num, []).append(val)

    layers = sorted(per_layer.keys())
    group_mean = np.array([np.mean(per_layer[l]) for l in layers])
    group_std  = np.array([np.std(per_layer[l], ddof=1) if len(per_layer[l]) > 1 else 0.0
                           for l in layers])
    n_per      = np.array([len(per_layer[l]) for l in layers])
    group_sem  = np.where(n_per > 1, group_std / np.sqrt(n_per), 0.0)

    return {
        'layers':     np.array(layers),
        'group_mean': group_mean,
        'group_sem':  group_sem,
        'group_std':  group_std,
        'n':          n_per,
        'per_sample': per_layer,
    }


# ── Plotting helpers ───────────────────────────────────────────────────────────

def add_zone_shading(ax, zone_config: dict, alpha: float = 0.20):
    """Draw background zone shading on the given axis."""
    y0, y1 = ax.get_ylim()
    for zone_name in ZONE_ORDER:
        if zone_name not in zone_config.get('zones', {}):
            continue
        layers = zone_config['zones'][zone_name]['layers']
        if not layers:
            continue
        x0 = min(layers) - 0.5
        x1 = max(layers) + 0.5
        color = ZONE_COLORS.get(zone_name, '#EEEEEE')
        ax.add_patch(Rectangle(
            (x0, y0), x1 - x0, y1 - y0,
            facecolor=color, edgecolor='none', alpha=alpha, zorder=0
        ))
    ax.set_ylim(y0, y1)


def add_zone_labels(ax, zone_config: dict, y_pos: float = None, fontsize: float = 6.5):
    """Add zone name text labels along the x-axis inside the plot."""
    y0, y1 = ax.get_ylim()
    label_y = y0 + 0.03 * (y1 - y0) if y_pos is None else y_pos
    for zone_name in ZONE_ORDER:
        if zone_name not in zone_config.get('zones', {}):
            continue
        layers = zone_config['zones'][zone_name]['layers']
        if not layers:
            continue
        x_mid = (min(layers) + max(layers)) / 2.0
        ax.text(x_mid, label_y, ZONE_LABELS.get(zone_name, zone_name),
                ha='center', va='bottom', fontsize=fontsize,
                color='#555555', zorder=5, style='italic')


def plot_group_profile(ax, profile: dict, style: dict, label: str, n_samples: int,
                       show_sem: bool = True):
    """Plot a group mean (±SEM) line on ax."""
    x    = profile['layers']
    mean = profile['group_mean']
    sem  = profile['group_sem']

    n_label = f"{label} (n={n_samples})" if n_samples > 1 else label

    ax.plot(x, mean, color=style['color'], ls=style['ls'], lw=style['lw'],
            label=n_label, zorder=style['zorder'])
    if show_sem and n_samples > 1:
        ax.fill_between(x, mean - sem, mean + sem,
                        color=style['color'], alpha=0.20, zorder=style['zorder'] - 1)


def figure_perfusion_baseline(group_results: dict, zone_config: dict, output_dir: Path):
    """
    Single-panel group comparison of baseline perfusion (air condition only).
    Perfusion is a single measurement per sample, not condition-varying.
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.suptitle('MLCO Group Comparison — Perfusion (baseline)',
                 fontsize=12, fontweight='bold')

    for group_name, results in group_results.items():
        p = extract_profile(results, 'air', 'perfusion')
        if len(p['layers']) == 0:
            continue
        # Count only samples that actually contributed perfusion data
        n_perf = sum(
            1 for r in results
            if 'air' in r.get('conditions', {})
            and any(
                'perfusion' in ldata and ldata['perfusion'].get('mean') is not None
                for ldata in get_bilateral_layers(r, 'air').values()
            )
        )
        plot_group_profile(ax, p, GROUP_STYLES[group_name], group_name, n_perf)

    ax.autoscale(axis='y')
    ax.set_ylabel(METRIC_LABELS['perfusion'], fontsize=9)
    ax.set_title('Baseline (air)', fontsize=10)
    ax.set_xlabel('MLCO Layer (surface → center)', fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(axis='y', alpha=0.3, zorder=1)
    ax.set_xlim(0.5, 24.5)
    ax.legend(fontsize=8, loc='upper right')
    fig.canvas.draw()
    add_zone_shading(ax, zone_config)
    add_zone_labels(ax, zone_config)

    plt.tight_layout()
    out = output_dir / 'group_comparison_perfusion'
    for fmt in ('png', 'svg', 'pdf'):
        fig.savefig(out.with_suffix(f'.{fmt}'), dpi=300, bbox_inches='tight',
                    facecolor='white')
        print(f"  ✓ Saved: {out.with_suffix(f'.{fmt}').name}")
    plt.close(fig)


# ── Main figures ───────────────────────────────────────────────────────────────

def figure_metric_profiles(group_results: dict, zone_config: dict,
                           metric: str, output_dir: Path):
    """
    3-row × 1-col figure: one row per condition (oxygen_1, air, oxygen_2).
    Shows group mean ± SEM for the given metric vs MLCO layer.
    """
    conditions = [c for c in CONDITION_ORDER if c in
                  next(iter(group_results.values()))[0]['conditions']]

    fig, axes = plt.subplots(len(conditions), 1,
                             figsize=(8, 3 * len(conditions)),
                             sharex=True)
    if len(conditions) == 1:
        axes = [axes]

    fig.suptitle(f'MLCO Group Comparison — {METRIC_LABELS[metric]}',
                 fontsize=12, fontweight='bold', y=1.01)

    for row, condition in enumerate(conditions):
        ax = axes[row]
        profiles = {}
        for group_name, results in group_results.items():
            p = extract_profile(results, condition, metric)
            profiles[group_name] = p
            n = len(results)
            plot_group_profile(ax, p, GROUP_STYLES[group_name], group_name, n)

        # Axis formatting — do before zone shading so ylims are set
        ax.autoscale(axis='y')
        ax.set_ylabel(METRIC_LABELS[metric], fontsize=9)
        ax.set_title(CONDITION_LABELS.get(condition, condition), fontsize=10)
        ax.tick_params(labelsize=8)
        ax.grid(axis='y', alpha=0.3, zorder=1)
        ax.set_xlim(0.5, 24.5)

        # Force y-axis to render so get_ylim() works before zone shading
        fig.canvas.draw()
        add_zone_shading(ax, zone_config)
        add_zone_labels(ax, zone_config)

        if row == 0:
            ax.legend(fontsize=8, loc='upper right')

    axes[-1].set_xlabel('MLCO Layer (surface → center)', fontsize=9)

    plt.tight_layout()
    out = output_dir / f'group_comparison_{metric}'
    for fmt in ('png', 'svg', 'pdf'):
        fig.savefig(out.with_suffix(f'.{fmt}'), dpi=300, bbox_inches='tight',
                    facecolor='white')
        print(f"  ✓ Saved: {out.with_suffix(f'.{fmt}').name}")
    plt.close(fig)


def figure_oxygen_response(group_results: dict, zone_config: dict, output_dir: Path):
    """
    ΔT2* oxygen responsiveness: (T2*_oxygen_2 − T2*_air) vs layer.
    One panel: all groups overlaid.
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.suptitle('MLCO Group Comparison — ΔT₂* Oxygen Response\n'
                 '(Post-O₂ − Air, bilateral average)',
                 fontsize=12, fontweight='bold')

    for group_name, results in group_results.items():
        # Collect per-sample ΔT2* per layer
        per_layer_delta = {}
        for r in results:
            conds = r.get('conditions', {})
            if 'oxygen_2' not in conds or 'air' not in conds:
                continue
            o2_layers  = get_bilateral_layers(r, 'oxygen_2')
            air_layers = get_bilateral_layers(r, 'air')
            common = set(o2_layers.keys()) & set(air_layers.keys())
            for layer_num in common:
                t2o2  = o2_layers[layer_num]['t2star']['mean']
                t2air = air_layers[layer_num]['t2star']['mean']
                if t2o2 is None or t2air is None:
                    continue
                delta = t2o2 - t2air
                per_layer_delta.setdefault(layer_num, []).append(delta)

        if not per_layer_delta:
            continue

        layers = sorted(per_layer_delta.keys())
        means  = np.array([np.mean(per_layer_delta[l]) for l in layers])
        sems   = np.array([np.std(per_layer_delta[l], ddof=1) / np.sqrt(len(per_layer_delta[l]))
                           if len(per_layer_delta[l]) > 1 else 0.0
                           for l in layers])
        n = len(results)
        style = GROUP_STYLES[group_name]
        n_label = f"{group_name} (n={n})" if n > 1 else group_name
        ax.plot(layers, means, color=style['color'], ls=style['ls'],
                lw=style['lw'], label=n_label, zorder=style['zorder'])
        if n > 1:
            ax.fill_between(layers, means - sems, means + sems,
                            color=style['color'], alpha=0.20, zorder=style['zorder'] - 1)

    ax.axhline(0, color='gray', lw=0.8, ls=':', zorder=2)
    ax.set_xlabel('MLCO Layer (surface → center)', fontsize=9)
    ax.set_ylabel('ΔT₂* (ms)', fontsize=9)
    ax.set_xlim(0.5, 24.5)
    ax.tick_params(labelsize=8)
    ax.grid(axis='y', alpha=0.3, zorder=1)
    ax.legend(fontsize=8, loc='upper right')

    fig.canvas.draw()
    add_zone_shading(ax, zone_config)
    add_zone_labels(ax, zone_config)
    plt.tight_layout()

    out = output_dir / 'group_comparison_delta_t2star'
    for fmt in ('png', 'svg', 'pdf'):
        fig.savefig(out.with_suffix(f'.{fmt}'), dpi=300, bbox_inches='tight',
                    facecolor='white')
        print(f"  ✓ Saved: {out.with_suffix(f'.{fmt}').name}")
    plt.close(fig)


def figure_combined_conditions(group_results: dict, zone_config: dict,
                               metric: str, output_dir: Path):
    """
    Single figure showing all 3 conditions overlaid per group.
    Useful for seeing the within-group oxygen challenge response.
    """
    conditions = [c for c in CONDITION_ORDER if c in
                  next(iter(group_results.values()))[0]['conditions']]

    COND_COLORS = {
        'oxygen_1': '#4A90E2',
        'air':      '#E27A3F',
        'oxygen_2': '#50C878',
    }
    COND_LS = {
        'oxygen_1': '-',
        'air':      '--',
        'oxygen_2': ':',
    }

    n_groups = len(group_results)
    fig, axes = plt.subplots(1, n_groups,
                             figsize=(6 * n_groups, 4.5),
                             sharey=True)
    if n_groups == 1:
        axes = [axes]

    fig.suptitle(f'Oxygen Challenge — {METRIC_LABELS[metric]} by Group',
                 fontsize=12, fontweight='bold')

    for col, (group_name, results) in enumerate(group_results.items()):
        ax = axes[col]
        n = len(results)
        for condition in conditions:
            p = extract_profile(results, condition, metric)
            color = COND_COLORS.get(condition, 'gray')
            ls    = COND_LS.get(condition, '-')
            label = CONDITION_LABELS.get(condition, condition)
            ax.plot(p['layers'], p['group_mean'], color=color, ls=ls,
                    lw=1.8, label=label, zorder=3)
            if n > 1:
                ax.fill_between(p['layers'],
                                p['group_mean'] - p['group_sem'],
                                p['group_mean'] + p['group_sem'],
                                color=color, alpha=0.15, zorder=2)

        ax.set_title(f'{group_name} (n={n})', fontsize=10)
        ax.set_xlabel('MLCO Layer (surface → center)', fontsize=9)
        ax.set_xlim(0.5, 24.5)
        ax.tick_params(labelsize=8)
        ax.grid(axis='y', alpha=0.3, zorder=1)
        ax.legend(fontsize=7.5, loc='upper right')
        if col == 0:
            ax.set_ylabel(METRIC_LABELS[metric], fontsize=9)

    # Draw all group data first so the shared y-axis captures the full range
    # across both groups before zone shading locks in the limits.
    fig.canvas.draw()
    for ax in axes:
        add_zone_shading(ax, zone_config)
        add_zone_labels(ax, zone_config)

    plt.tight_layout()
    out = output_dir / f'within_group_{metric}_oxygen_challenge'
    for fmt in ('png', 'svg', 'pdf'):
        fig.savefig(out.with_suffix(f'.{fmt}'), dpi=300, bbox_inches='tight',
                    facecolor='white')
        print(f"  ✓ Saved: {out.with_suffix(f'.{fmt}').name}")
    plt.close(fig)


def figure_zone_barplot(group_results: dict, zone_config: dict,
                        condition: str, metric: str, output_dir: Path):
    """
    Bar chart: zone-averaged metric for each group.
    One bar per zone per group, error bars = SEM across samples.
    """
    zone_names = [z for z in ZONE_ORDER if z in zone_config.get('zones', {})]
    zone_layer_map = {z: zone_config['zones'][z]['layers'] for z in zone_names}

    fig, ax = plt.subplots(figsize=(8, 4.5))
    fig.suptitle(f'Zone-Averaged {METRIC_LABELS[metric]} — '
                 f'{CONDITION_LABELS.get(condition, condition)}',
                 fontsize=11, fontweight='bold')

    n_zones  = len(zone_names)
    n_groups = len(group_results)
    bar_width = 0.8 / n_groups
    x = np.arange(n_zones)

    for gidx, (group_name, results) in enumerate(group_results.items()):
        # For each sample, average across zone layers
        zone_vals = {z: [] for z in zone_names}
        for r in results:
            if condition not in r.get('conditions', {}):
                continue
            ldict = get_bilateral_layers(r, condition)
            for z in zone_names:
                vals = []
                for layer_num in zone_layer_map[z]:
                    if layer_num in ldict and metric in ldict[layer_num]:
                        v = ldict[layer_num][metric]['mean']
                        if v is not None and not np.isnan(v):
                            vals.append(v)
                if vals:
                    zone_vals[z].append(np.mean(vals))

        group_means = [np.mean(zone_vals[z]) if zone_vals[z] else np.nan
                       for z in zone_names]
        group_sems  = [(np.std(zone_vals[z], ddof=1) / np.sqrt(len(zone_vals[z]))
                        if len(zone_vals[z]) > 1 else 0.0)
                       for z in zone_names]

        offset = (gidx - (n_groups - 1) / 2) * bar_width
        style  = GROUP_STYLES[group_name]
        n      = len(results)
        bars   = ax.bar(x + offset, group_means, bar_width,
                        label=f'{group_name} (n={n})',
                        color=style['color'], alpha=0.75,
                        edgecolor=style['color'], linewidth=0.8)
        ax.errorbar(x + offset, group_means, yerr=group_sems,
                    fmt='none', ecolor='#333333', elinewidth=1.2, capsize=3)

    ax.set_xticks(x)
    ax.set_xticklabels([ZONE_LABELS[z].replace('\n', ' ') for z in zone_names], fontsize=9)
    ax.set_ylabel(METRIC_LABELS[metric], fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(axis='y', alpha=0.3)
    ax.legend(fontsize=8)
    plt.tight_layout()

    out = output_dir / f'zone_bar_{metric}_{condition}'
    for fmt in ('png', 'svg', 'pdf'):
        fig.savefig(out.with_suffix(f'.{fmt}'), dpi=300, bbox_inches='tight',
                    facecolor='white')
        print(f"  ✓ Saved: {out.with_suffix(f'.{fmt}').name}")
    plt.close(fig)


def save_group_stats_json(group_results: dict, zone_config: dict, output_dir: Path):
    """Save per-layer and per-zone group stats to JSON for downstream use."""
    stats = {}
    for group_name, results in group_results.items():
        stats[group_name] = {
            'n_samples': len(results),
            'sample_ids': [r['sample_id'] for r in results],
            'conditions': {},
        }
        for condition in CONDITION_ORDER:
            if condition not in results[0].get('conditions', {}):
                continue
            cond_stats = {}
            for metric in ['t2star', 'r2star', 'perfusion']:
                p = extract_profile(results, condition, metric)
                cond_stats[metric] = {
                    'layers': p['layers'].tolist(),
                    'group_mean': p['group_mean'].tolist(),
                    'group_sem': p['group_sem'].tolist(),
                    'group_std': p['group_std'].tolist(),
                    'n_per_layer': p['n'].tolist(),
                }
            stats[group_name]['conditions'][condition] = cond_stats

    out = output_dir / 'group_comparison_stats.json'
    with open(out, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"  ✓ Saved: {out.name}")


def load_hematology(csv_path: Path = None) -> dict:
    """
    Load per-sample hematology data from CSV.

    Returns {sample_id: {'RBC': float, 'HGB': float, 'HCT': float, 'group': str}}
    or an empty dict if the file is not found.
    """
    if csv_path is None:
        csv_path = HEMATOLOGY_CSV
    if not Path(csv_path).exists():
        print(f"  NOTE: Hematology CSV not found at {csv_path}, skipping correction.")
        return {}
    data = {}
    with open(csv_path, newline='', encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            sid = row['Mouse ID'].strip()
            data[sid] = {
                'RBC':   float(row['RBC']),
                'HGB':   float(row['HGB']),
                'HCT':   float(row['HCT']),
                'group': row['Group'].strip(),
            }
    print(f"  Loaded hematology for: {sorted(data.keys())}")
    return data


def _per_sample_delta_t2star(group_results: dict, oc_layers: set = None) -> dict:
    """
    Compute per-sample ΔT2* scalars (post-O₂ − air).

    Returns {group_name: {sample_id: {wk_abs, oc_abs, wk_pct, oc_pct}}}
    """
    if oc_layers is None:
        oc_layers = set(range(1, 6))   # outer cortex layers 1–5

    out = {}
    for group_name, results in group_results.items():
        out[group_name] = {}
        for r in results:
            sid   = r['sample_id']
            conds = r.get('conditions', {})
            if 'oxygen_2' not in conds or 'air' not in conds:
                continue
            o2_layers  = get_bilateral_layers(r, 'oxygen_2')
            air_layers = get_bilateral_layers(r, 'air')
            common = sorted(set(o2_layers.keys()) & set(air_layers.keys()))

            deltas_wk, deltas_oc = [], []
            baselines_wk, baselines_oc = [], []
            for ln in common:
                t2o2  = o2_layers[ln]['t2star'].get('mean')
                t2air = air_layers[ln]['t2star'].get('mean')
                if t2o2 is None or t2air is None:
                    continue
                if np.isnan(t2o2) or np.isnan(t2air) or t2air <= 0:
                    continue
                d = t2o2 - t2air
                deltas_wk.append(d)
                baselines_wk.append(t2air)
                if ln in oc_layers:
                    deltas_oc.append(d)
                    baselines_oc.append(t2air)

            if not deltas_wk:
                continue
            out[group_name][sid] = {
                'wk_abs': float(np.mean(deltas_wk)),
                'wk_pct': float(np.mean(deltas_wk) / np.mean(baselines_wk) * 100),
                'oc_abs': float(np.mean(deltas_oc)) if deltas_oc else np.nan,
                'oc_pct': (float(np.mean(deltas_oc) / np.mean(baselines_oc) * 100)
                           if deltas_oc else np.nan),
            }
    return out


def figure_oxygen_response_index(group_results: dict, output_dir: Path):
    """
    Per-sample oxygen response index: ΔT2* = T2*_post_O2 − T2*_air.

    Four metrics, each compared between groups via Mann-Whitney:
      1. Whole-kidney ΔT2* (ms)            — mean over all 24 layers
      2. Outer cortex ΔT2* (ms)            — mean over layers 1–5
      3. Whole-kidney % ΔT2*               — ΔT2*/T2*_air × 100 (fractional)
      4. Outer cortex % ΔT2*               — same, outer cortex only

    The fractional metrics control for captopril's higher baseline T2*, making
    them the fairest test of whether responsiveness itself is altered.
    """
    # ── Collect per-sample scalars ─────────────────────────────────────────────
    per_sample = _per_sample_delta_t2star(group_results)
    metric_keys = ['wk_abs', 'oc_abs', 'wk_pct', 'oc_pct']

    sample_vals = {}
    for group_name in group_results:
        sample_vals[group_name] = {k: [] for k in metric_keys}
        for sid_vals in per_sample[group_name].values():
            for k in metric_keys:
                sample_vals[group_name][k].append(sid_vals[k])

    # ── Print values ───────────────────────────────────────────────────────────
    print('\n── Oxygen Response Index ──────────────────────────────────')
    ctrl_key, capt_key = list(group_results.keys())
    for k, label in [('wk_abs', 'WK ΔT2* (ms)'), ('oc_abs', 'OC ΔT2* (ms)'),
                     ('wk_pct', 'WK %ΔT2*'),     ('oc_pct', 'OC %ΔT2*')]:
        cv = sample_vals[ctrl_key][k]
        av = sample_vals[capt_key][k]
        cv_c = [v for v in cv if not np.isnan(v)]
        av_c = [v for v in av if not np.isnan(v)]
        if len(cv_c) >= 2 and len(av_c) >= 2:
            _, p = mannwhitneyu(av_c, cv_c, alternative='two-sided')
            pstr = f'p={p:.3f}'
        else:
            pstr = 'n/a'
        print(f'  {label:18s}  Ctrl: {np.mean(cv_c):.2f}±{np.std(cv_c):.2f}'
              f'  Capt: {np.mean(av_c):.2f}±{np.std(av_c):.2f}  {pstr}')

    # ── Figure ─────────────────────────────────────────────────────────────────
    panel_specs = [
        ('wk_abs', 'Whole-kidney ΔT₂* (ms)',
         'Whole-kidney O₂ response\n(Post-O₂ − Air)', 'two-sided'),
        ('oc_abs', 'Outer cortex ΔT₂* (ms)',
         'Outer cortex O₂ response\n(layers 1–5)', 'two-sided'),
        ('wk_pct', 'Whole-kidney %ΔT₂*',
         'Fractional O₂ response\n(whole kidney)', 'two-sided'),
        ('oc_pct', 'Outer cortex %ΔT₂*',
         'Fractional O₂ response\n(outer cortex)', 'two-sided'),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(14, 4.5))
    fig.suptitle(
        'Per-Sample Oxygen Response Index  (ΔT₂* = Post-O₂ − Air)\n'
        'Absolute (ms) and fractional (%) metrics — Mann-Whitney, two-sided',
        fontsize=11, fontweight='bold'
    )

    rng = np.random.default_rng(42)
    group_list = [(ctrl_key, GROUP_STYLES[ctrl_key]),
                  (capt_key, GROUP_STYLES[capt_key])]

    for ax, (key, ylabel, title, alt) in zip(axes, panel_specs):
        for xi, (gname, style) in enumerate(group_list):
            vals = [v for v in sample_vals[gname][key] if not np.isnan(v)]
            if not vals:
                continue
            jitter = rng.uniform(-0.09, 0.09, len(vals))
            ax.scatter(np.full(len(vals), xi) + jitter, vals,
                       color=style['color'], s=55, zorder=4,
                       edgecolors='white', lw=0.6)
            ax.plot([xi - 0.18, xi + 0.18], [np.mean(vals)] * 2,
                    color=style['color'], lw=2.8, zorder=5)

        # significance bracket
        cv = [v for v in sample_vals[ctrl_key][key] if not np.isnan(v)]
        av = [v for v in sample_vals[capt_key][key] if not np.isnan(v)]
        if len(cv) >= 2 and len(av) >= 2:
            _, p = mannwhitneyu(av, cv, alternative=alt)
            pstr = f'p = {p:.3f}'
            sig  = '*' if p < 0.05 else ''
        else:
            pstr, sig = 'p = n/a', ''

        y_lo, y_hi = ax.get_ylim()
        margin = 0.22 * (y_hi - y_lo)
        ax.set_ylim(y_lo, y_hi + margin)
        y_lo, y_hi = ax.get_ylim()
        y_bar = y_hi - margin * 0.55
        ax.plot([0, 1], [y_bar, y_bar], color='#555', lw=1.0)
        ax.text(0.5, y_bar + 0.01 * (y_hi - y_lo), f'{pstr}{sig}',
                ha='center', va='bottom', fontsize=8, color='#333')

        n_ctrl = len(cv)
        n_capt = len(av)
        ax.set_xticks([0, 1])
        ax.set_xticklabels([f'Control\n(n={n_ctrl})', f'Captopril\n(n={n_capt})'],
                           fontsize=8)
        ax.axhline(0, color='#999', lw=0.8, ls=':', zorder=1)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=9.5, style='italic', color='#333')
        ax.tick_params(labelsize=8)
        ax.grid(axis='y', alpha=0.25, zorder=1)

    plt.tight_layout()
    out = output_dir / 'oxygen_response_index'
    for fmt in ('png', 'svg', 'pdf'):
        fig.savefig(out.with_suffix(f'.{fmt}'), dpi=300, bbox_inches='tight',
                    facecolor='white')
        print(f"  ✓ Saved: {out.with_suffix(f'.{fmt}').name}")
    plt.close(fig)


def figure_hematology_correction(group_results: dict, hematology: dict,
                                 output_dir: Path):
    """
    6-panel figure combining hematology values and Hct-corrected BOLD response.

    Top row:    HCT | HGB | RBC  per animal, by group
    Bottom row: Raw OC ΔT2* | Hct-corrected OC ΔT2* | HCT vs OC ΔT2* scatter

    Correction formula (linear approximation):
        ΔT2*_corrected = ΔT2*_raw × (HCT_ref / HCT_sample)

    Rationale: ΔR2* with oxygen breathing is proportional to Hct (Ogawa model).
    Lower Hct in captopril (ACE-inhibitor anemia) suppresses ΔT2* independently
    of physiology. Correcting up to a common reference Hct reveals the true
    vascular O₂ response. Because captopril animals have LOWER Hct, their
    corrected ΔT2* values are larger than raw — the anemia was attenuating
    the observed response.

    Samples without hematology data (M1_WT) are shown in raw panels only,
    marked with an open symbol, and excluded from corrected panels.
    """
    if not hematology:
        print("  Skipping hematology correction figure (no data).")
        return

    ctrl_key, capt_key = list(group_results.keys())

    # ── Reference HCT = mean of controls with known hematology ────────────────
    ctrl_sids = [r['sample_id'] for r in group_results[ctrl_key]]
    ctrl_hct_vals = [hematology[sid]['HCT']
                     for sid in ctrl_sids
                     if sid in hematology]
    if not ctrl_hct_vals:
        print("  No control hematology found; cannot compute reference HCT.")
        return
    hct_ref = float(np.mean(ctrl_hct_vals))
    print(f"  Reference HCT = {hct_ref:.2f}%  "
          f"(mean of controls: {[sid for sid in GROUPS[ctrl_key] if sid in hematology]})")

    # ── Per-sample ΔT2* ───────────────────────────────────────────────────────
    per_sample = _per_sample_delta_t2star(group_results)

    # Build combined table: sid → {group, hct, oc_abs_raw, oc_abs_corr}
    records = []
    for group_name, results in group_results.items():
        style = GROUP_STYLES[group_name]
        for r in results:
            sid = r['sample_id']
            if sid not in per_sample[group_name]:
                continue
            oc_raw = per_sample[group_name][sid]['oc_abs']
            hct    = hematology.get(sid, {}).get('HCT', np.nan)
            oc_cor = (oc_raw * hct_ref / hct
                      if not np.isnan(oc_raw) and not np.isnan(hct) and hct > 0
                      else np.nan)
            records.append({
                'sid':    sid,
                'group':  group_name,
                'color':  style['color'],
                'hct':    hct,
                'hgb':    hematology.get(sid, {}).get('HGB', np.nan),
                'rbc':    hematology.get(sid, {}).get('RBC', np.nan),
                'oc_raw': oc_raw,
                'oc_cor': oc_cor,
                'has_hemato': sid in hematology,
            })

    def group_records(gname):
        return [rc for rc in records if rc['group'] == gname]

    # ── Figure layout ─────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(13, 8.5))
    fig.suptitle(
        'Hematology Values and Hct-Corrected BOLD Oxygen Response\n'
        f'Reference HCT = {hct_ref:.1f}%  (control mean)  |  '
        'Correction: ΔT₂*_corr = ΔT₂*_raw × (HCT_ref / HCT_sample)',
        fontsize=10.5, color='#222'
    )

    rng = np.random.default_rng(42)

    # ── Helper: dot plot with group mean bar ──────────────────────────────────
    def dot_plot_hemato(ax, field, ylabel, title, show_stats=False,
                        show_only_with_hemato=False):
        """Dot plot for one hematology or BOLD metric."""
        group_vals = {}
        for xi, gname in enumerate([ctrl_key, capt_key]):
            recs = group_records(gname)
            if show_only_with_hemato:
                recs = [rc for rc in recs if rc['has_hemato']]
            vals = [rc[field] for rc in recs if not np.isnan(rc[field])]
            cols = [rc['color'] for rc in recs if not np.isnan(rc[field])]
            sids = [rc['sid']   for rc in recs if not np.isnan(rc[field])]
            has_h = [rc['has_hemato'] for rc in recs if not np.isnan(rc[field])]
            group_vals[gname] = vals

            jitter = rng.uniform(-0.09, 0.09, len(vals))
            for j, (v, c, s, h) in enumerate(zip(vals, cols, sids, has_h)):
                marker = 'o' if h else 'D'   # diamond = no hematology data
                facecolor = c if h else 'none'
                ax.scatter(xi + jitter[j], v, color=c, s=55, zorder=4,
                           marker=marker, facecolors=facecolor,
                           edgecolors=c, lw=1.2)
            if vals:
                ax.plot([xi - 0.18, xi + 0.18], [np.mean(vals)] * 2,
                        color=GROUP_STYLES[gname]['color'], lw=2.8, zorder=5)

        ax.set_xticks([0, 1])
        ctrl_n = len([rc for rc in group_records(ctrl_key)
                      if (not show_only_with_hemato or rc['has_hemato'])
                      and not np.isnan(rc[field])])
        capt_n = len([rc for rc in group_records(capt_key)
                      if (not show_only_with_hemato or rc['has_hemato'])
                      and not np.isnan(rc[field])])
        ax.set_xticklabels([f'Control\n(n={ctrl_n})', f'Captopril\n(n={capt_n})'],
                           fontsize=8)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=9.5, style='italic', color='#333')
        ax.tick_params(labelsize=8)
        ax.grid(axis='y', alpha=0.25)
        ax.axhline(0, color='#999', lw=0.8, ls=':', zorder=1)

        if show_stats:
            cv = [v for v in group_vals.get(ctrl_key, []) if not np.isnan(v)]
            av = [v for v in group_vals.get(capt_key, []) if not np.isnan(v)]
            if len(cv) >= 2 and len(av) >= 2:
                _, p = mannwhitneyu(av, cv, alternative='two-sided')
                pstr = f'p = {p:.3f}'
                sig  = '*' if p < 0.05 else ''
            elif len(cv) >= 1 and len(av) >= 1:
                pstr, sig = '(n too small for MW)', ''
            else:
                pstr, sig = '', ''

            if pstr:
                y_lo, y_hi = ax.get_ylim()
                margin = 0.20 * (y_hi - y_lo)
                ax.set_ylim(y_lo, y_hi + margin)
                y_lo, y_hi = ax.get_ylim()
                y_bar = y_hi - margin * 0.55
                ax.plot([0, 1], [y_bar, y_bar], color='#555', lw=1.0)
                ax.text(0.5, y_bar + 0.01 * (y_hi - y_lo),
                        f'{pstr}{sig}  (two-sided)',
                        ha='center', va='bottom', fontsize=7.5, color='#333')

    # ── Top row: hematology ───────────────────────────────────────────────────
    dot_plot_hemato(axes[0, 0], 'hct', 'HCT (%)', 'Hematocrit',
                    show_only_with_hemato=True)
    axes[0, 0].text(-0.18, 1.06, 'A', transform=axes[0, 0].transAxes,
                    fontsize=14, fontweight='bold', va='top')

    dot_plot_hemato(axes[0, 1], 'hgb', 'HGB (g/dL)', 'Hemoglobin',
                    show_only_with_hemato=True)
    axes[0, 1].text(-0.18, 1.06, 'B', transform=axes[0, 1].transAxes,
                    fontsize=14, fontweight='bold', va='top')

    dot_plot_hemato(axes[0, 2], 'rbc', 'RBC (×10⁶/μL)', 'Red Blood Cells',
                    show_only_with_hemato=True)
    axes[0, 2].text(-0.18, 1.06, 'C', transform=axes[0, 2].transAxes,
                    fontsize=14, fontweight='bold', va='top')

    # ── Bottom row: BOLD response ─────────────────────────────────────────────
    # Panel D: raw OC ΔT2* (all samples including M1_WT with open diamond)
    dot_plot_hemato(axes[1, 0], 'oc_raw',
                    'Outer cortex ΔT₂* (ms)', 'Raw OC ΔT₂*  [Post-O₂ − Air]',
                    show_stats=True, show_only_with_hemato=False)
    axes[1, 0].text(-0.18, 1.06, 'D', transform=axes[1, 0].transAxes,
                    fontsize=14, fontweight='bold', va='top')
    # Legend for open diamond
    axes[1, 0].scatter([], [], marker='D', facecolors='none',
                       edgecolors='#555', lw=1.2, s=40,
                       label='No hematology data')
    axes[1, 0].legend(fontsize=7, loc='upper left', framealpha=0.8)

    # Panel E: Hct-corrected OC ΔT2*
    dot_plot_hemato(axes[1, 1], 'oc_cor',
                    f'OC ΔT₂* × (HCT_ref / HCT_sample) (ms)',
                    f'Hct-corrected OC ΔT₂*\n(ref HCT = {hct_ref:.1f}%)',
                    show_stats=True, show_only_with_hemato=True)
    axes[1, 1].text(-0.18, 1.06, 'E', transform=axes[1, 1].transAxes,
                    fontsize=14, fontweight='bold', va='top')

    # Panel F: HCT vs OC ΔT2* scatter with regression
    ax_sc = axes[1, 2]
    sc_hct, sc_oc, sc_cols, sc_sids = [], [], [], []
    for rc in records:
        if np.isnan(rc['hct']) or np.isnan(rc['oc_raw']):
            continue
        sc_hct.append(rc['hct'])
        sc_oc.append(rc['oc_raw'])
        sc_cols.append(rc['color'])
        sc_sids.append(rc['sid'])

    for hct_v, oc_v, col, sid in zip(sc_hct, sc_oc, sc_cols, sc_sids):
        ax_sc.scatter(hct_v, oc_v, color=col, s=60, zorder=4,
                      edgecolors='white', lw=0.6)
        ax_sc.annotate(sid, (hct_v, oc_v), fontsize=6, color='#444',
                       xytext=(3, 3), textcoords='offset points')

    if len(sc_hct) >= 3:
        slope, intercept, r, p_r, _ = linregress(sc_hct, sc_oc)
        x_fit = np.linspace(min(sc_hct) - 1, max(sc_hct) + 1, 100)
        ax_sc.plot(x_fit, slope * x_fit + intercept,
                   color='#555', lw=1.4, ls='--', zorder=3,
                   label=f'r = {r:.2f},  p = {p_r:.3f}')
        ax_sc.legend(fontsize=8, loc='upper left', framealpha=0.85)

    ax_sc.set_xlabel('HCT (%)', fontsize=9)
    ax_sc.set_ylabel('Outer cortex ΔT₂* (ms)', fontsize=9)
    ax_sc.set_title('HCT vs OC ΔT₂*  [all samples with data]',
                    fontsize=9.5, style='italic', color='#333')
    ax_sc.tick_params(labelsize=8)
    ax_sc.grid(alpha=0.25)
    ax_sc.axhline(0, color='#999', lw=0.8, ls=':', zorder=1)
    ax_sc.text(-0.18, 1.06, 'F', transform=ax_sc.transAxes,
               fontsize=14, fontweight='bold', va='top')

    # Print summary
    ctrl_recs = [rc for rc in records if rc['group'] == ctrl_key and rc['has_hemato']]
    capt_recs = [rc for rc in records if rc['group'] == capt_key and rc['has_hemato']]
    print(f'\n── Hematology Summary ──────────────────────────────────────')
    for field, label in [('hct', 'HCT (%)'), ('hgb', 'HGB (g/dL)'), ('rbc', 'RBC')]:
        cv = [rc[field] for rc in ctrl_recs if not np.isnan(rc[field])]
        av = [rc[field] for rc in capt_recs if not np.isnan(rc[field])]
        print(f'  {label:12s}  Ctrl (n={len(cv)}): {np.mean(cv):.2f}±{np.std(cv):.2f}'
              f'  Capt (n={len(av)}): {np.mean(av):.2f}±{np.std(av):.2f}')
    print(f'\n── OC ΔT₂* Correction (ref HCT = {hct_ref:.1f}%) ───────────')
    for rc in sorted(records, key=lambda x: x['group']):
        tag = '' if rc['has_hemato'] else '  [no HCT data]'
        cor_str = f'{rc["oc_cor"]:.2f}' if not np.isnan(rc['oc_cor']) else 'N/A'
        print(f'  {rc["sid"]:8s}  [{rc["group"][:4]}]  '
              f'HCT={rc["hct"]:.1f}%  raw={rc["oc_raw"]:.2f} ms  '
              f'corr={cor_str} ms{tag}')

    plt.tight_layout()
    out = output_dir / 'hematology_correction'
    for fmt in ('png', 'svg', 'pdf'):
        fig.savefig(out.with_suffix(f'.{fmt}'), dpi=300, bbox_inches='tight',
                    facecolor='white')
        print(f"  ✓ Saved: {out.with_suffix(f'.{fmt}').name}")
    plt.close(fig)


# ── Entry point ────────────────────────────────────────────────────────────────

def run_comparison(exclude_samples: list = None, out_dir: Path = None,
                   hematology_csv: Path = None):
    """
    Run the full group comparison pipeline.

    Parameters
    ----------
    exclude_samples : list of str, optional
        Sample IDs to drop from their respective groups (e.g. ['174004']).
        Useful for sensitivity analyses.
    out_dir : Path, optional
        Output directory. Defaults to OUTPUT_DIR.
    hematology_csv : Path or str, optional
        Path to hematology CSV file.  If None, defaults to HEMATOLOGY_CSV.
        If the file does not exist the hematology correction figure is skipped
        silently — all other figures are generated normally.
    """
    if out_dir is None:
        out_dir = OUTPUT_DIR
    if exclude_samples is None:
        exclude_samples = []

    out_dir.mkdir(parents=True, exist_ok=True)
    label = f"(excl. {', '.join(exclude_samples)})" if exclude_samples else ""
    print(f"\nBoldPy Group Comparison  {label}")
    print(f"{'='*60}")
    print(f"Output: {out_dir}")
    if exclude_samples:
        print(f"Excluding: {exclude_samples}")

    # Build filtered group lists
    filtered_groups = {
        gname: [sid for sid in sids if sid not in exclude_samples]
        for gname, sids in GROUPS.items()
    }

    # Load all groups
    group_results = {}
    for group_name, sample_ids in filtered_groups.items():
        print(f"\nLoading {group_name}:")
        group_results[group_name] = load_group(sample_ids)

    # Get zone config from first available sample
    first_result = next(r for results in group_results.values()
                        for r in results)
    zone_config = first_result['zone_config']

    # Verify conditions
    all_conditions = set()
    for results in group_results.values():
        for r in results:
            all_conditions.update(r['conditions'].keys())
    conditions_to_plot = [c for c in CONDITION_ORDER if c in all_conditions]
    print(f"\nConditions found: {conditions_to_plot}")

    # ── Generate figures ───────────────────────────────────────────────────────

    print(f"\n{'─'*60}")
    print("Generating T2* profile (per condition)...")
    figure_metric_profiles(group_results, zone_config, 't2star', out_dir)

    print(f"\n{'─'*60}")
    print("Generating R2* profile (per condition)...")
    figure_metric_profiles(group_results, zone_config, 'r2star', out_dir)

    print(f"\n{'─'*60}")
    print("Generating perfusion baseline profile (single condition)...")
    figure_perfusion_baseline(group_results, zone_config, out_dir)

    print(f"\n{'─'*60}")
    print("Generating ΔT2* oxygen response...")
    figure_oxygen_response(group_results, zone_config, out_dir)

    print(f"\n{'─'*60}")
    print("Generating within-group oxygen challenge plots...")
    for metric in ['t2star', 'r2star']:
        figure_combined_conditions(group_results, zone_config, metric, out_dir)

    print(f"\n{'─'*60}")
    print("Generating zone bar charts...")
    for condition in conditions_to_plot:
        for metric in ['t2star', 'r2star']:
            figure_zone_barplot(group_results, zone_config, condition, metric, out_dir)

    print(f"\n{'─'*60}")
    print("Generating oxygen response index...")
    figure_oxygen_response_index(group_results, out_dir)

    print(f"\n{'─'*60}")
    print("Saving group stats JSON...")
    save_group_stats_json(group_results, zone_config, out_dir)

    print(f"\n{'─'*60}")
    print("Loading hematology data (optional)...")
    hematology = load_hematology(hematology_csv)   # empty dict if file absent
    if hematology:
        print("Generating hematology correction figure...")
        figure_hematology_correction(group_results, hematology, out_dir)
    else:
        print("  No hematology data — skipping correction figure.")

    print(f"\n{'='*60}")
    print(f"Done. Outputs in: {out_dir}")


def run(pep_path, output_dir=None, **kwargs):
    """
    Importable entry point for group_analysis.

    Parameters
    ----------
    pep_path : str or Path
        Path to PEP project_config.yaml.
    output_dir : str or Path, optional
        Override the output directory from the PEP config.
    **kwargs
        Absorbed for forward-compatibility.
    """
    load_pep_project(pep_path)
    run_comparison(out_dir=Path(output_dir) if output_dir else None)


def main():
    run_comparison()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Multi-Sample Group Comparison')
    parser.add_argument('--pep', required=True,
                        help='Path to PEP project_config.yaml')
    args = parser.parse_args()
    load_pep_project(args.pep)
    main()
