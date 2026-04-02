#!/usr/bin/env python3
"""
Heterogeneity Analysis — Within-Layer + Focal Disruption
=========================================================

Combines two complementary views of T₂* heterogeneity:

  Part 1 — Within-Layer Profile Analysis:
    Computes voxel-level heterogeneity metrics per MLCO layer from raw T₂* maps.
    Metrics: mean, std, CV, IQR, q25, q75, low_frac, very_low_frac
    → Per-condition overview figures (6 panels each)
    → Talk-ready 2-panel summary (mean vs. IQR)
    → Outer cortex bar chart (3 metrics × 3 conditions)
    → heterogeneity_stats.json

  Part 2 — Focal Disruption Analysis:
    Computes per-sample outer-cortex scalar metrics, statistical tests, and
    spatial local-CV maps to localise focal heterogeneity patches.
    → Per-animal strip plots with Mann-Whitney p-values
    → KDE distributions of outer-cortex T₂*
    → Spatial local-CV maps (group average + difference)
    → heterogeneity_statistics.json

All outputs → {output_dir}/heterogeneity/

Usage:
    cd boldpy
    python heterogeneity.py --pep code/analysis/captopril/project_config.yaml

project_config.yaml format (PEP):
    See pipeline/examples/project_config.yaml for a full template.
    Key sections: sample_table, output_dir, prepared_dir, mlco_dir, group_styles.
"""

import argparse
import json
import peppy
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
from pathlib import Path
from scipy import stats
from scipy.ndimage import uniform_filter

# ── Paths ──────────────────────────────────────────────────────────────────────

BASE       = Path(__file__).resolve().parents[3]
MLCO_DIR   = BASE / 'processed' / 'mlco'
PREP_DIR   = BASE / 'processed' / 'prepared'
ANAL_DIR   = BASE / 'processed' / 'analysis'
OUTPUT_DIR = BASE / 'processed' / 'analysis' / 'group_comparison' / 'heterogeneity'

# ── Groups ─────────────────────────────────────────────────────────────────────
# Populated at runtime via --pep project_config.yaml

GROUPS = {}


def load_pep_project(pep_path):
    """Load GROUPS, OUTPUT_DIR, PREP_DIR, MLCO_DIR from a PEP project_config.yaml."""
    global GROUPS, OUTPUT_DIR, PREP_DIR, MLCO_DIR, ANAL_DIR
    pep_path = Path(pep_path).resolve()
    project  = peppy.Project(str(pep_path))
    samples  = project.sample_table
    styles   = project.config.get('group_styles', {})

    # Reconstruct GROUPS in the full dict-of-dicts format used throughout this script
    GROUPS = {}
    for grp_id, df in samples.groupby('group', sort=False):
        style   = dict(styles.get(grp_id, {}))
        display = style.get('label', grp_id)
        GROUPS[display] = {'ids': list(df['sample_name']), **style}

    cfg  = project.config
    base = pep_path.parent

    def _resolve(key, default):
        val = cfg.get(key)
        if not val:
            return default
        p = Path(val)
        return p if p.is_absolute() else base / p

    out_base   = _resolve('group_output_dir', BASE / 'processed' / 'analysis' / 'group_comparison')
    OUTPUT_DIR = out_base / 'heterogeneity'
    ANAL_DIR   = _resolve('analysis_dir', BASE / 'processed' / 'analysis')
    PREP_DIR   = _resolve('prepared_dir', BASE / 'processed' / 'prepared')
    MLCO_DIR   = _resolve('mlco_dir',     BASE / 'processed' / 'mlco')

# ── Parameters ─────────────────────────────────────────────────────────────────

N_LAYERS    = 24
OC_LAYERS   = list(range(1, 6))    # outer cortex layers
CONDITIONS  = ['oxygen_1', 'air', 'oxygen_2']
COND_LABELS = {'oxygen_1': 'Pre-O₂ (100%)', 'air': 'Air (21%)', 'oxygen_2': 'Post-O₂ (100%)'}
LOW_T2      = 10.0   # ms — threshold for "hypoxic" voxels
VERY_LOW_T2 =  8.0   # ms — threshold for "severely hypoxic" voxels

ZONE_CONFIG = {
    'outer_cortex':  (1,  5,  '#E8F4F8'),
    'inner_cortex':  (6,  10, '#C5E3ED'),
    'cmj':           (11, 13, '#FFE5CC'),
    'outer_medulla': (14, 19, '#FFD9B3'),
    'inner_medulla': (20, 24, '#FFC999'),
}
ZONE_SHORT = {
    'outer_cortex':  'Outer\nCortex',
    'inner_cortex':  'Inner\nCortex',
    'cmj':           'CMJ',
    'outer_medulla': 'Outer\nMedulla',
    'inner_medulla': 'Inner\nMedulla',
}

# ── Data loading ───────────────────────────────────────────────────────────────

def load_maps(sid, condition):
    """Load T₂* map and MLCO bilateral mask; returns (None, None) if missing."""
    t2_path   = PREP_DIR / sid / f'{sid}_{condition}_t2star_bruker.npy'
    mlco_path = MLCO_DIR / sid / f'{sid}_mlco_bilateral.npy'
    if not t2_path.exists() or not mlco_path.exists():
        return None, None
    return (np.load(t2_path).astype('float32'),
            np.load(mlco_path).astype('int32'))

# ── Part 1: Per-layer profile analysis ─────────────────────────────────────────

def compute_layer_metrics(t2, mlco):
    """
    For each bilateral layer (1..N_LAYERS), pool left + right kidney voxels
    and compute heterogeneity metrics.  Returns {layer_num: metrics_dict}.
    """
    results = {}
    for layer in range(1, N_LAYERS + 1):
        pixels = t2[((mlco == layer) | (mlco == layer + N_LAYERS)) & (t2 > 0)]
        if len(pixels) < 5:
            continue
        mean_v = float(np.mean(pixels))
        std_v  = float(np.std(pixels, ddof=1))
        q25_v  = float(np.percentile(pixels, 25))
        q75_v  = float(np.percentile(pixels, 75))
        results[layer] = {
            'n':             len(pixels),
            'mean':          mean_v,
            'std':           std_v,
            'cv':            std_v / mean_v if mean_v > 0 else np.nan,
            'q25':           q25_v,
            'q75':           q75_v,
            'iqr':           q75_v - q25_v,
            'low_frac':      float(np.mean(pixels < LOW_T2)),
            'very_low_frac': float(np.mean(pixels < VERY_LOW_T2)),
        }
    return results


def build_group_profiles(group_info):
    """
    Build per-sample metric vectors for each condition, then compute group
    mean ± SEM.  Returns:
        {condition: {metric: {'layers', 'group_mean', 'group_sem', 'n', 'per_sample'}}}
    """
    profiles = {}
    for condition in CONDITIONS:
        per_sample = {}
        for sid in group_info['ids']:
            t2, mlco = load_maps(sid, condition)
            if t2 is None:
                print(f'    WARNING: {sid}/{condition} not found, skipping')
                continue
            per_sample[sid] = compute_layer_metrics(t2, mlco)
        if not per_sample:
            continue

        all_layers = sorted({l for lm in per_sample.values() for l in lm})
        cond_profile = {}
        for metric in ['mean', 'std', 'cv', 'q25', 'q75', 'iqr', 'low_frac', 'very_low_frac']:
            per_layer_vals = {}
            for layer in all_layers:
                vals = [per_sample[sid][layer][metric]
                        for sid in per_sample
                        if layer in per_sample[sid]
                        and not np.isnan(per_sample[sid][layer][metric])]
                if vals:
                    per_layer_vals[layer] = vals
            layers = sorted(per_layer_vals)
            gm = np.array([np.mean(per_layer_vals[l]) for l in layers])
            gs = np.array([
                np.std(per_layer_vals[l], ddof=1) / np.sqrt(len(per_layer_vals[l]))
                if len(per_layer_vals[l]) > 1 else 0.0
                for l in layers
            ])
            cond_profile[metric] = {
                'layers':     np.array(layers, dtype=int),
                'group_mean': gm,
                'group_sem':  gs,
                'n':          np.array([len(per_layer_vals[l]) for l in layers]),
                'per_sample': per_layer_vals,
            }
        profiles[condition] = cond_profile
    return profiles

# ── Profile plot helpers ────────────────────────────────────────────────────────

def add_zone_shading(ax, alpha=0.22):
    y0, y1 = ax.get_ylim()
    for zone, (l0, l1, color) in ZONE_CONFIG.items():
        ax.add_patch(Rectangle((l0 - 0.5, y0), l1 - l0 + 1, y1 - y0,
                                facecolor=color, edgecolor='none', alpha=alpha, zorder=0))
    ax.set_ylim(y0, y1)


def add_zone_labels(ax, fontsize=6.0):
    y0, y1 = ax.get_ylim()
    label_y = y0 + 0.03 * (y1 - y0)
    for zone, (l0, l1, _) in ZONE_CONFIG.items():
        ax.text((l0 + l1) / 2, label_y, ZONE_SHORT[zone],
                ha='center', va='bottom', fontsize=fontsize,
                color='#555555', style='italic', zorder=5)


def plot_profile(ax, profile, group_name, group_info, metric):
    p  = profile[metric]
    x  = p['layers']
    gm = p['group_mean']
    gs = p['group_sem']
    label = group_name.replace('\n', ' ')
    ax.plot(x, gm, color=group_info['color'], ls=group_info['ls'],
            lw=group_info['lw'], label=label, zorder=4)
    if len(group_info['ids']) > 1:
        ax.fill_between(x, gm - gs, gm + gs,
                        color=group_info['color'], alpha=0.18, zorder=3)

# ── Profile figures ─────────────────────────────────────────────────────────────

def figure_heterogeneity_overview(all_profiles, condition, output_dir):
    """6-panel per-condition overview: mean, CV, IQR, std, q25, low_frac."""
    metrics = [
        ('mean',     'T₂* Mean (ms)',               False),
        ('cv',       'CV  (std / mean)',             True),
        ('iqr',      'IQR  (ms)',                    True),
        ('std',      'T₂* Std (ms)',                 True),
        ('q25',      'T₂* Q25 (ms)',                False),
        ('low_frac', f'Fraction < {LOW_T2:.0f} ms', True),
    ]
    fig, axes = plt.subplots(3, 2, figsize=(11, 10), sharex=True)
    fig.suptitle(
        f'T₂* Within-Layer Heterogeneity  —  {COND_LABELS[condition]}\n'
        f'(bilateral pool, layers 1–{N_LAYERS} surface→center)',
        fontsize=12, fontweight='bold', y=1.01
    )
    for idx, (metric, ylabel, is_hetero) in enumerate(metrics):
        row, col = divmod(idx, 2)
        ax = axes[row][col]
        for group_name, group_info in GROUPS.items():
            if condition in all_profiles[group_name]:
                plot_profile(ax, all_profiles[group_name][condition],
                             group_name, group_info, metric)
        if is_hetero:
            ax.set_facecolor('#FFFEF5')
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_xlim(0.5, 24.5)
        ax.tick_params(labelsize=8)
        ax.grid(axis='y', alpha=0.3, zorder=1)
        if idx == 0:
            ax.legend(fontsize=8, loc='upper left')
        ax.axvspan(0.5, 5.5, alpha=0.06, color='gold', zorder=0)
        fig.canvas.draw()
        add_zone_shading(ax)
        add_zone_labels(ax)
    axes[2][0].set_xlabel('MLCO Layer (surface → center)', fontsize=9)
    axes[2][1].set_xlabel('MLCO Layer (surface → center)', fontsize=9)
    ax_cv = axes[0][1]
    ax_cv.annotate(
        'Outer cortex', xy=(3, ax_cv.get_ylim()[1] * 0.85),
        fontsize=7.5, color='#8B6914', ha='center', style='italic',
        bbox=dict(boxstyle='round,pad=0.3', fc='#FFFBE6', ec='#C8A800', alpha=0.8)
    )
    plt.tight_layout()
    out = output_dir / f'heterogeneity_overview_{condition}'
    for fmt in ('png', 'svg', 'pdf'):
        fig.savefig(out.with_suffix(f'.{fmt}'), dpi=300, bbox_inches='tight', facecolor='white')
        print(f'  ✓ {out.with_suffix(f".{fmt}").name}')
    plt.close(fig)


def figure_talk_summary(all_profiles, output_dir):
    """2-panel talk summary: T₂* mean (Air) vs. IQR (Post-O₂)."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharex=True)
    fig.suptitle(
        'BOLD MRI MLCO Analysis  —  T₂* Mean (Air) vs. IQR Heterogeneity (Post-O₂)',
        fontsize=12, fontweight='bold'
    )
    panels = [
        ('air',      'mean', 'T₂* Mean (ms)',
         'Global oxygenation  [Air 21% O₂]\n(whole-layer average)',          axes[0]),
        ('oxygen_2', 'iqr',  'T₂* IQR (ms)',
         'Focal heterogeneity  [Post-O₂ 100%]\n(interquartile range per layer)', axes[1]),
    ]
    for condition, metric, ylabel, subtitle, ax in panels:
        for group_name, group_info in GROUPS.items():
            if condition in all_profiles[group_name]:
                plot_profile(ax, all_profiles[group_name][condition],
                             group_name, group_info, metric)
        ax.set_xlabel('MLCO Layer (surface → center)', fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(subtitle, fontsize=9, style='italic', color='#444')
        ax.set_xlim(0.5, 24.5)
        ax.tick_params(labelsize=9)
        ax.grid(axis='y', alpha=0.3)
        ax.legend(fontsize=8.5, loc='upper left')
        ax.axvspan(0.5, 5.5, alpha=0.09, color='gold', zorder=0)
        fig.canvas.draw()
        add_zone_shading(ax, alpha=0.25)
        add_zone_labels(ax, fontsize=7)
    plt.tight_layout()
    out = output_dir / 'TALK_mean_vs_iqr_postO2'
    for fmt in ('png', 'svg', 'pdf'):
        fig.savefig(out.with_suffix(f'.{fmt}'), dpi=300, bbox_inches='tight', facecolor='white')
        print(f'  ✓ {out.with_suffix(f".{fmt}").name}')
    plt.close(fig)


def figure_outer_cortex_bars(all_profiles, output_dir):
    """Outer cortex (layers 1–5) bar chart: 3 heterogeneity metrics × 3 conditions."""
    oc_layers = list(range(1, 6))
    metrics   = [('cv', 'CV (std/mean)'), ('iqr', 'IQR (ms)'),
                 ('low_frac', f'Fraction < {LOW_T2:.0f} ms')]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))
    fig.suptitle('Outer Cortex (layers 1–5) — Heterogeneity Metrics by Condition',
                 fontsize=11, fontweight='bold')
    for col, (metric, ylabel) in enumerate(metrics):
        ax = axes[col]
        x, bw, n_groups = np.arange(len(CONDITIONS)), 0.35, len(GROUPS)
        for gidx, (group_name, group_info) in enumerate(GROUPS.items()):
            vals, errs = [], []
            for condition in CONDITIONS:
                if condition not in all_profiles[group_name]:
                    vals.append(np.nan); errs.append(0); continue
                p       = all_profiles[group_name][condition][metric]
                oc_m    = [p['group_mean'][i] for i, lay in enumerate(p['layers']) if lay in oc_layers]
                oc_s    = [p['group_sem'][i]  for i, lay in enumerate(p['layers']) if lay in oc_layers]
                vals.append(np.nanmean(oc_m) if oc_m else np.nan)
                n_oc = len(oc_m)
                errs.append(np.sqrt(np.nansum(np.array(oc_s)**2)) / n_oc if n_oc > 0 else 0)
            offset = (gidx - (n_groups - 1) / 2) * bw
            ax.bar(x + offset, vals, bw, label=group_name.replace('\n', ' '),
                   color=group_info['color'], alpha=0.75,
                   edgecolor=group_info['color'], linewidth=0.8)
            ax.errorbar(x + offset, vals, yerr=errs, fmt='none',
                        ecolor='#333', elinewidth=1.2, capsize=3)
        ax.set_xticks(x)
        ax.set_xticklabels([COND_LABELS[c].replace(' ', '\n') for c in CONDITIONS], fontsize=8)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(axis='y', alpha=0.3)
        ax.set_facecolor('#FFFEF5')
        if col == 0:
            ax.legend(fontsize=7.5)
    plt.tight_layout()
    out = output_dir / 'outer_cortex_heterogeneity_bars'
    for fmt in ('png', 'svg', 'pdf'):
        fig.savefig(out.with_suffix(f'.{fmt}'), dpi=300, bbox_inches='tight', facecolor='white')
        print(f'  ✓ {out.with_suffix(f".{fmt}").name}')
    plt.close(fig)


def save_profile_stats_json(all_profiles, output_dir):
    """Serialise computed profiles to JSON."""
    out_data = {}
    for group_name, cond_profiles in all_profiles.items():
        gkey = group_name.replace('\n', ' ')
        out_data[gkey] = {}
        for condition, metric_profiles in cond_profiles.items():
            out_data[gkey][condition] = {}
            for metric, p in metric_profiles.items():
                out_data[gkey][condition][metric] = {
                    'layers':     p['layers'].tolist(),
                    'group_mean': p['group_mean'].tolist(),
                    'group_sem':  p['group_sem'].tolist(),
                    'n':          p['n'].tolist(),
                }
    out = output_dir / 'heterogeneity_stats.json'
    with open(out, 'w') as f:
        json.dump(out_data, f, indent=2)
    print(f'  ✓ {out.name}')

# ── Part 2: Focal disruption helpers ───────────────────────────────────────────

def oc_pixels(t2, mlco):
    """Outer-cortex pixels (both kidneys pooled, valid T₂* only)."""
    mask = np.zeros_like(mlco, dtype=bool)
    for l in OC_LAYERS:
        mask |= (mlco == l) | (mlco == l + N_LAYERS)
    return t2[mask & (t2 > 0)]


def per_sample_metrics(condition):
    """Per-sample scalar outer-cortex metrics for one condition."""
    result = {}
    for group_name, info in GROUPS.items():
        vals = []
        for sid in info['ids']:
            t2, mlco = load_maps(sid, condition)
            if t2 is None:
                continue
            px = oc_pixels(t2, mlco)
            if len(px) < 10:
                continue
            mean_v = float(np.mean(px))
            std_v  = float(np.std(px, ddof=1))
            q25    = float(np.percentile(px, 25))
            q75    = float(np.percentile(px, 75))
            vals.append({'id': sid, 'mean': mean_v, 'std': std_v,
                         'cv': std_v / mean_v, 'iqr': q75 - q25,
                         'q25': q25, 'low_frac': float(np.mean(px < 10.0))})
        result[group_name] = vals
    return result


def mw_test(group_metrics, metric):
    """One-sided Mann-Whitney U (group B > group A)."""
    g_keys    = list(group_metrics.keys())
    vals_a    = [s[metric] for s in group_metrics[g_keys[0]]]
    vals_b    = [s[metric] for s in group_metrics[g_keys[1]]]
    if len(vals_a) < 2 or len(vals_b) < 2:
        return np.nan, np.nan
    stat, p = stats.mannwhitneyu(vals_b, vals_a, alternative='greater')
    return stat, p

# ── Focal disruption figures ────────────────────────────────────────────────────

def figure_strip_plots(output_dir):
    """Per-animal strip plots for CV, IQR, mean × 2 conditions."""
    conditions = ['air', 'oxygen_2']
    metrics    = [('cv',   'CV  (std / mean)',  'Normalised heterogeneity'),
                  ('iqr',  'IQR  (ms)',         'Absolute spread'),
                  ('mean', 'T₂* Mean  (ms)',    'Global oxygenation (reference)')]
    fig, axes = plt.subplots(len(metrics), len(conditions),
                             figsize=(9, 3.5 * len(metrics)), sharey='row')
    fig.suptitle('Outer Cortex (layers 1–5): Per-Animal Metrics  |  Mann-Whitney U p-values',
                 fontsize=12, fontweight='bold')
    for row, (metric, ylabel, _) in enumerate(metrics):
        for col, condition in enumerate(conditions):
            ax = axes[row][col]
            gm = per_sample_metrics(condition)
            _, p = mw_test(gm, metric)
            all_vals = []
            for gidx, (group_name, info) in enumerate(GROUPS.items()):
                vals  = [s[metric] for s in gm[group_name]]
                x_pos = gidx + 1
                jitter = np.random.default_rng(42).uniform(-0.12, 0.12, len(vals))
                ax.scatter([x_pos + j for j in jitter], vals,
                           color=info['color'], s=55, zorder=5, alpha=0.85,
                           edgecolors='white', linewidths=0.6)
                ax.hlines(np.median(vals), x_pos - 0.25, x_pos + 0.25,
                          color=info['color'], lw=2.5, zorder=6)
                all_vals.extend(vals)
            y_top = max(all_vals) * 1.12
            ax.annotate(
                f'p = {p:.3f}{"*" if p < 0.05 else ""}{"†" if 0.05 <= p < 0.10 else ""}',
                xy=(1.5, y_top), ha='center', fontsize=9, color='#333',
                bbox=dict(boxstyle='round,pad=0.25', fc='#F5F5F5', ec='#BBB')
            )
            ax.plot([1, 2], [y_top * 0.97, y_top * 0.97], 'k-', lw=0.8)
            ax.set_xticks([1, 2])
            ax.set_xticklabels([info.get('short', gn) for gn, info in GROUPS.items()], fontsize=9)
            ax.set_ylabel(ylabel if col == 0 else '', fontsize=9)
            ax.tick_params(labelsize=8)
            ax.grid(axis='y', alpha=0.25)
            if row == 0:
                ax.set_title(COND_LABELS[condition], fontsize=10)
    plt.tight_layout()
    out = output_dir / 'strip_plots_per_animal'
    for fmt in ('png', 'svg', 'pdf'):
        fig.savefig(out.with_suffix(f'.{fmt}'), dpi=300, bbox_inches='tight', facecolor='white')
        print(f'  ✓ {out.with_suffix(f".{fmt}").name}')
    plt.close(fig)


def figure_kde_distributions(output_dir):
    """KDE of outer-cortex T₂* voxels pooled per group."""
    from scipy.stats import gaussian_kde
    conditions = ['air', 'oxygen_2']
    fig, axes  = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.suptitle('Outer Cortex T₂* Distribution  (all voxels pooled per group)',
                 fontsize=12, fontweight='bold')
    for col, condition in enumerate(conditions):
        ax = axes[col]
        for group_name, info in GROUPS.items():
            all_px = []
            for sid in info['ids']:
                t2, mlco = load_maps(sid, condition)
                if t2 is None:
                    continue
                all_px.append(oc_pixels(t2, mlco))
            if not all_px:
                continue
            pooled = np.concatenate(all_px)
            kde    = gaussian_kde(pooled, bw_method='silverman')
            x      = np.linspace(max(0, pooled.min() - 2), pooled.max() + 2, 400)
            label  = group_name.replace('\n', ' ')
            ax.plot(x, kde(x), color=info['color'], lw=2.2,
                    ls=info.get('ls', '-'), label=label)
            ax.fill_between(x, kde(x), alpha=0.12, color=info['color'])
            ax.axvline(np.mean(pooled),   color=info['color'], ls='--', lw=1.0, alpha=0.7)
            ax.axvline(np.median(pooled), color=info['color'], ls=':',  lw=1.0, alpha=0.7)
        ax.axvline(10.0, color='gray', ls='--', lw=1.2, alpha=0.6, label='10 ms threshold')
        ax.set_xlabel('T₂* (ms)', fontsize=10)
        ax.set_ylabel('Density', fontsize=10)
        ax.set_title(COND_LABELS[condition], fontsize=10)
        ax.legend(fontsize=8, loc='upper right')
        ax.tick_params(labelsize=9)
        ax.grid(alpha=0.25)
        ax.set_xlim(0, 60)
        ax.annotate('x-axis clipped at 60 ms\n(<1% of voxels excluded)',
                    xy=(0.97, 0.95), xycoords='axes fraction',
                    fontsize=6.5, ha='right', va='top', color='#888', style='italic')
    plt.tight_layout()
    out = output_dir / 'kde_outer_cortex_distribution'
    for fmt in ('png', 'svg', 'pdf'):
        fig.savefig(out.with_suffix(f'.{fmt}'), dpi=300, bbox_inches='tight', facecolor='white')
        print(f'  ✓ {out.with_suffix(f".{fmt}").name}')
    plt.close(fig)


def compute_local_cv_map(t2, mlco, kernel=5):
    """2D local CV map via sliding window (within kidney ROI)."""
    roi    = (mlco > 0) & (t2 > 0)
    filled = np.where(roi, t2, 0.0)
    count  = uniform_filter(roi.astype('float32'), size=kernel)
    lmean  = uniform_filter(filled, size=kernel) / np.maximum(count, 1e-6)
    lmean2 = uniform_filter(filled**2, size=kernel) / np.maximum(count, 1e-6)
    lvar   = np.maximum(lmean2 - lmean**2, 0)
    lcv    = np.where(roi & (lmean > 0), np.sqrt(lvar) / lmean, np.nan)
    return lcv


def figure_spatial_cv_maps(output_dir):
    """Average spatial local-CV maps per group (Post-O₂) + difference map."""
    condition = 'oxygen_2'
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle(
        f'Spatial Local-CV Map  —  {COND_LABELS[condition]}\n'
        '(5×5 px sliding window;  outer cortex MLCO shell highlighted)',
        fontsize=11, fontweight='bold'
    )
    group_avg_maps = {}
    group_oc_masks = {}
    for group_name, info in GROUPS.items():
        cv_maps, oc_masks = [], []
        for sid in info['ids']:
            t2, mlco = load_maps(sid, condition)
            if t2 is None:
                continue
            cv_maps.append(compute_local_cv_map(t2, mlco, kernel=5))
            oc_mask = np.zeros_like(mlco, dtype=bool)
            for l in OC_LAYERS:
                oc_mask |= (mlco == l) | (mlco == l + N_LAYERS)
            oc_masks.append(oc_mask)
        if cv_maps:
            group_avg_maps[group_name] = np.nanmean(cv_maps, axis=0)
            group_oc_masks[group_name] = np.mean(oc_masks, axis=0) > 0.5

    g_keys   = list(group_avg_maps.keys())
    if len(g_keys) < 2:
        print('  Need at least 2 groups for spatial CV comparison — skipping')
        plt.close(fig); return

    map_a = group_avg_maps[g_keys[0]]
    map_b = group_avg_maps[g_keys[1]]
    diff  = map_b - map_a
    vmax  = np.nanpercentile(np.concatenate([map_a[~np.isnan(map_a)],
                                              map_b[~np.isnan(map_b)]]), 95)
    dmax  = np.nanpercentile(np.abs(diff[~np.isnan(diff)]), 95)

    short_a = GROUPS[g_keys[0]].get('short', g_keys[0])
    short_b = GROUPS[g_keys[1]].get('short', g_keys[1])
    panels  = [
        (map_a, group_oc_masks[g_keys[0]], f'{short_a} (avg)', 'hot_r', 0, vmax, False),
        (map_b, group_oc_masks[g_keys[1]], f'{short_b} (avg)', 'hot_r', 0, vmax, False),
        (diff,  None,                      f'{short_b} − {short_a} (ΔCV)',
         'RdBu_r', -dmax, dmax, True),
    ]
    for col, (data, oc_mask, title, cmap, vmin, vmax_p, is_diff) in enumerate(panels):
        ax = axes[col]
        im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax_p,
                       origin='lower', interpolation='nearest')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                     label='Local CV' if not is_diff else 'ΔCV')
        ax.set_title(title, fontsize=10); ax.axis('off')
        if oc_mask is not None:
            ax.contour(oc_mask.astype(float), levels=[0.5],
                       colors=['cyan'], linewidths=1.0, alpha=0.7)
        if is_diff:
            ax.text(0.02, 0.02,
                    f'Blue = {short_a} more heterogeneous\nRed = {short_b} more heterogeneous',
                    transform=ax.transAxes, fontsize=6.5, color='#333',
                    va='bottom', bbox=dict(fc='white', alpha=0.7, pad=2))
    plt.tight_layout()
    out = output_dir / 'spatial_cv_maps'
    for fmt in ('png', 'svg', 'pdf'):
        fig.savefig(out.with_suffix(f'.{fmt}'), dpi=300, bbox_inches='tight', facecolor='white')
        print(f'  ✓ {out.with_suffix(f".{fmt}").name}')
    plt.close(fig)


def print_and_save_stats(output_dir):
    """Print stats table and save to JSON (outer cortex, two conditions)."""
    metrics = ['mean', 'cv', 'iqr', 'low_frac']
    results = {}
    g_keys  = list(GROUPS.keys())
    label_a = g_keys[0].replace('\n', ' ')
    label_b = g_keys[1].replace('\n', ' ') if len(g_keys) > 1 else 'Group B'

    print(f'\n{"="*72}')
    print(f'OUTER CORTEX STATISTICS — Mann-Whitney U (one-sided: {label_b} > {label_a})')
    print(f'{"="*72}')

    for condition in ['air', 'oxygen_2']:
        print(f'\n  Condition: {COND_LABELS[condition]}')
        print(f'  {"Metric":<12} {label_a+" mean±SEM":>22} {label_b+" mean±SEM":>22} {"U stat":>8} {"p":>10}')
        print(f'  {"-"*74}')
        gm = per_sample_metrics(condition)
        results[condition] = {}
        for metric in metrics:
            vals_a = [s[metric] for s in gm[g_keys[0]]]
            vals_b = [s[metric] for s in gm[g_keys[1]]] if len(g_keys) > 1 else []
            ma = np.mean(vals_a); sa = np.std(vals_a, ddof=1)/np.sqrt(len(vals_a)) if len(vals_a)>1 else 0
            mb = np.mean(vals_b) if vals_b else np.nan
            sb = np.std(vals_b, ddof=1)/np.sqrt(len(vals_b)) if len(vals_b)>1 else 0
            u, p = mw_test(gm, metric)
            sig  = ' *' if p < 0.05 else (' †' if p < 0.10 else '')
            fmt  = '.3f' if metric in ('cv', 'low_frac') else '.2f'
            print(f'  {metric:<12} {ma:{fmt}}±{sa:{fmt}}{"":<10} {mb:{fmt}}±{sb:{fmt}}{"":<8} {u:>8.0f} {p:>10.4f}{sig}')
            results[condition][metric] = {
                'group_a_mean': ma, 'group_a_sem': sa, 'group_a_n': len(vals_a), 'group_a_values': vals_a,
                'group_b_mean': mb, 'group_b_sem': sb, 'group_b_n': len(vals_b), 'group_b_values': vals_b,
                'mw_u': u, 'mw_p': p, 'significant_p05': bool(p < 0.05), 'trend_p10': bool(p < 0.10),
            }
    print(f'\n  * p < 0.05   † p < 0.10  (one-sided Mann-Whitney U)')
    out = output_dir / 'heterogeneity_statistics.json'
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\n  ✓ Saved: {out.name}')
    return results

# ── Entry point ─────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f'\nBoldPy Heterogeneity Analysis')
    print(f'{"=" * 60}')
    print(f'Thresholds: low T₂* < {LOW_T2} ms, very low < {VERY_LOW_T2} ms')
    print(f'Output: {OUTPUT_DIR}\n')

    # ── Part 1: Profile analysis ──────────────────────────────────────────────
    print('── Part 1: Within-layer profile analysis ──────────────────')
    all_profiles = {}
    for group_name, group_info in GROUPS.items():
        print(f'Computing profiles — {group_name.replace(chr(10), " ")}:')
        all_profiles[group_name] = build_group_profiles(group_info)
        print()

    print('Generating per-condition overview figures...')
    for condition in CONDITIONS:
        print(f'  {COND_LABELS[condition]}:')
        figure_heterogeneity_overview(all_profiles, condition, OUTPUT_DIR)

    print('\nGenerating outer cortex bar chart...')
    figure_outer_cortex_bars(all_profiles, OUTPUT_DIR)

    print('\nSaving profile stats JSON...')
    save_profile_stats_json(all_profiles, OUTPUT_DIR)

    # ── Part 2: Focal disruption ──────────────────────────────────────────────
    print(f'\n{"=" * 60}')
    print('── Part 2: Focal disruption analysis ──────────────────────')

    print('\n[1/4] Computing outer-cortex statistics...')
    print_and_save_stats(OUTPUT_DIR)

    print('\n[2/4] Generating per-animal strip plots...')
    figure_strip_plots(OUTPUT_DIR)

    print('\n[3/4] Generating KDE distributions...')
    figure_kde_distributions(OUTPUT_DIR)

    print('\n[4/4] Generating spatial CV maps...')
    figure_spatial_cv_maps(OUTPUT_DIR)

    print(f'\n{"=" * 60}')
    print(f'Done → {OUTPUT_DIR}')


def run(pep_path, output_dir=None, **kwargs):
    """
    Importable entry point for heterogeneity.

    Parameters
    ----------
    pep_path : str or Path
        Path to PEP project_config.yaml.
    output_dir : str or Path, optional
        Override the output directory from the PEP config.
    **kwargs
        Absorbed for forward-compatibility.
    """
    global OUTPUT_DIR
    load_pep_project(pep_path)
    if output_dir is not None:
        OUTPUT_DIR = Path(output_dir) / 'heterogeneity'
    main()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Heterogeneity Analysis')
    parser.add_argument('--pep', required=True, help='Path to PEP project_config.yaml')
    args = parser.parse_args()
    load_pep_project(args.pep)
    main()
