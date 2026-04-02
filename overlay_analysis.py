#!/usr/bin/env python3
"""
Overlay Analysis — K-Means Zones + MLCO Layers
===============================================

Generates two types of per-sample 3-panel overlay figures and a group-level
k-means zone analysis summary.  K-means is computed ONCE and reused across
all three outputs.

  1. K-Means Overlay (per sample × condition):
       Left:   T₂* map (inferno)
       Middle: K-means zone assignment (solid colours, k=3)
       Right:  T₂* + semi-transparent zone overlay + contours
     → {analysis_dir}/{sample}/kmeans/{sample}_kmeans_{condition}.{png,svg}

  2. MLCO Layer Overlay (per sample × condition):
       Left:   T₂* map (inferno)
       Middle: MLCO radial layers (plasma, L1=cortex → L24=papilla)
               with static 5-zone boundary contours
       Right:  T₂* + semi-transparent MLCO layer overlay
     → {analysis_dir}/{sample}/mlco/{sample}_mlco_{condition}.{png,svg}

  3. K-Means Zone Analysis (group-level summary figure):
       A. Per-layer T₂* std profile (Post-O₂)
       B. K-means cluster assignment strip per animal
       C. Silhouette score comparison (dot plot)
       D. Superficial-zone T₂* std comparison (dot plot)
     → {output_dir}/kmeans_zone_analysis.{png,pdf,svg}

  Grid figures (all samples, per condition):
     → {output_dir}/kmeans_overlay_grid_{condition}.{png,svg}
     → {output_dir}/mlco_overlay_grid_{condition}.{png,svg}

Usage:
    cd boldpy
    python overlay_analysis.py --pep code/analysis/captopril/project_config.yaml

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
import matplotlib.cm as mcm
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch, Rectangle
from scipy.stats import mannwhitneyu
from pathlib import Path

try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
except ImportError:
    raise ImportError("scikit-learn required: pip install scikit-learn")

# ── Paths ──────────────────────────────────────────────────────────────────────

BASE         = Path(__file__).resolve().parents[3]
MLCO_DIR     = BASE / 'processed' / 'mlco'
PREP_DIR     = BASE / 'processed' / 'prepared'
ANALYSIS_DIR = BASE / 'processed' / 'analysis'
OUTPUT_DIR   = ANALYSIS_DIR / 'group_comparison'

# ── Groups ─────────────────────────────────────────────────────────────────────
# Populated at runtime via --pep project_config.yaml

GROUPS = {}


def load_pep_project(pep_path):
    """Load GROUPS, OUTPUT_DIR, PREP_DIR, MLCO_DIR from a PEP project_config.yaml."""
    global GROUPS, OUTPUT_DIR, PREP_DIR, MLCO_DIR, ANALYSIS_DIR
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

    OUTPUT_DIR   = _resolve('group_output_dir', BASE / 'processed' / 'analysis' / 'group_comparison')
    ANALYSIS_DIR = _resolve('analysis_dir', BASE / 'processed' / 'analysis')
    PREP_DIR     = _resolve('prepared_dir', BASE / 'processed' / 'prepared')
    MLCO_DIR     = _resolve('mlco_dir',     BASE / 'processed' / 'mlco')

# ── Parameters ─────────────────────────────────────────────────────────────────

N_LAYERS    = 24
N_CLUSTERS  = 3
CONDITIONS  = ['air', 'oxygen_2']
COND_LABELS = {'air': 'Air (21% O₂)', 'oxygen_2': 'Post-O₂ (100% O₂)'}

# K-means zone names (depth-ordered: shallow → deep)
ZONE_NAMES  = ['superficial', 'intermediate', 'deep']
ZONE_COLORS = ['#3498DB', '#E67E22', '#27AE60']   # blue / orange / green
ZONE_ALPHA  = 0.50

# MLCO layer colormap
LAYER_CMAP  = 'plasma'
LAYER_ALPHA = 0.50

# Static 5-zone shading config (for zone analysis figure and MLCO boundaries)
ZONE_CONFIG_SHADING = {
    'Outer\nCortex':  (1,  5,  '#D6EAF8'),
    'Inner\nCortex':  (6,  10, '#A9CCE3'),
    'CMJ':            (11, 13, '#FAD7A0'),
    'Outer\nMedulla': (14, 19, '#F5CBA7'),
    'Inner\nMedulla': (20, 24, '#EDBB99'),
}

# MLCO zone boundary definitions (for overlay annotations and contours)
ZONE_DEFS = [
    ('outer cortex',   1,  5, '#E74C3C'),
    ('inner cortex',   6, 10, '#E67E22'),
    ('CMJ',           11, 13, '#F1C40F'),
    ('outer medulla', 14, 19, '#2ECC71'),
    ('inner medulla', 20, 24, '#3498DB'),
]
ZONE_BOUNDARIES = [5, 10, 13, 19]   # draw contour after each of these layers

# ── Shared helpers ─────────────────────────────────────────────────────────────

def load_maps(sid, condition):
    t2   = np.load(PREP_DIR / sid / f'{sid}_{condition}_t2star_bruker.npy').astype('float32')
    mlco = np.load(MLCO_DIR / sid / f'{sid}_mlco_bilateral.npy').astype('int32')
    return t2, mlco


def kidney_crop_box(mlco, pad=10):
    """Bounding box of all kidney voxels with padding."""
    ys, xs = np.where(mlco > 0)
    if len(ys) == 0:
        return 0, mlco.shape[0], 0, mlco.shape[1]
    h, w = mlco.shape
    return (max(0, int(ys.min()) - pad),
            min(h, int(ys.max()) + pad + 1),
            max(0, int(xs.min()) - pad),
            min(w, int(xs.max()) + pad + 1))


def t2star_display_range(t2, mlco):
    """Robust display range: 2nd–98th percentile of kidney voxels."""
    vals = t2[(mlco > 0) & (t2 > 0)]
    if len(vals) == 0:
        return 0, 60
    return float(np.nanpercentile(vals, 2)), float(np.nanpercentile(vals, 98))


def bilateral_layer_features(t2, mlco):
    """Per-layer [median, std, depth] pooling both kidneys."""
    feats = []
    for layer in range(1, N_LAYERS + 1):
        px    = t2[((mlco == layer) | (mlco == layer + N_LAYERS)) & (t2 > 0)]
        depth = (layer - 1) / (N_LAYERS - 1)
        if len(px) < 5:
            feats.append({'layer': layer, 'median': np.nan, 'std': np.nan,
                          'depth': depth, 'n': 0})
        else:
            feats.append({'layer': layer,
                          'median': float(np.median(px)),
                          'std':    float(np.std(px, ddof=1)),
                          'depth':  depth, 'n': len(px)})
    return feats


def kmeans_cluster(layer_feats, n_clusters=N_CLUSTERS, random_state=42):
    """
    K-means on [T₂* median, T₂* std, normalised depth] per layer.
    Feature set is sensitive to within-layer heterogeneity, not just mean T₂*.
    Returns zone assignments, silhouette score, and superficial-zone summary stats.
    """
    valid = [i for i, f in enumerate(layer_feats)
             if not (np.isnan(f['median']) or np.isnan(f['std']))]
    if len(valid) < n_clusters:
        return None

    X = np.array([[layer_feats[i]['median'],
                   layer_feats[i]['std'],
                   layer_feats[i]['depth']] for i in valid])
    scaler     = StandardScaler()
    X_scaled   = scaler.fit_transform(X)
    km         = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    raw_labels = km.fit_predict(X_scaled)
    sil        = silhouette_score(X_scaled, raw_labels) if len(set(raw_labels)) > 1 else 0.0

    cluster_mean_depth = {cid: float(np.mean(X[raw_labels == cid, 2]))
                          for cid in range(n_clusters)}
    sorted_cids  = sorted(cluster_mean_depth, key=lambda c: cluster_mean_depth[c])
    cid_to_name  = {cid: ZONE_NAMES[rank] for rank, cid in enumerate(sorted_cids)}

    zone_layers     = {name: [] for name in ZONE_NAMES}
    label_per_layer = np.full(N_LAYERS, -1, dtype=int)

    for pos, vi in enumerate(valid):
        name = cid_to_name[raw_labels[pos]]
        zone_layers[name].append(layer_feats[vi]['layer'])
        label_per_layer[layer_feats[vi]['layer'] - 1] = ZONE_NAMES.index(name)

    # Assign NaN layers to nearest cluster by depth
    for i, feat in enumerate(layer_feats):
        if label_per_layer[i] == -1:
            nearest = min(cluster_mean_depth,
                          key=lambda c: abs(cluster_mean_depth[c] - feat['depth']))
            name = cid_to_name[nearest]
            zone_layers[name].append(feat['layer'])
            label_per_layer[i] = ZONE_NAMES.index(name)

    for name in zone_layers:
        zone_layers[name] = sorted(zone_layers[name])

    sup_layers   = zone_layers['superficial']
    sup_feats    = [layer_feats[l - 1] for l in sup_layers]
    sup_std_mean = float(np.nanmean([f['std'] for f in sup_feats]))

    return {
        'label_per_layer': label_per_layer,
        'zone_layers':     zone_layers,
        'silhouette':      float(sil),
        'sup_n_layers':    len(sup_layers),
        'sup_max_layer':   float(max(sup_layers)) if sup_layers else np.nan,
        'sup_std_mean':    sup_std_mean,
        'layer_feats':     layer_feats,
    }

# ── K-Means overlay helpers ────────────────────────────────────────────────────

def build_zone_index_map(mlco, label_per_layer):
    """2D integer map: -1=background, 0=superficial, 1=intermediate, 2=deep."""
    zone_map = np.full(mlco.shape, -1, dtype=np.int8)
    for layer_idx in range(N_LAYERS):
        layer_num = layer_idx + 1
        zone_idx  = int(label_per_layer[layer_idx])
        mask = (mlco == layer_num) | (mlco == layer_num + N_LAYERS)
        zone_map[mask] = zone_idx
    return zone_map


def build_rgba_overlay(zone_map, alpha=ZONE_ALPHA):
    """Convert zone_map to RGBA overlay image."""
    h, w    = zone_map.shape
    overlay = np.zeros((h, w, 4), dtype=np.float32)
    for zi, hex_color in enumerate(ZONE_COLORS):
        r, g, b, _ = mcolors.to_rgba(hex_color)
        mask = zone_map == zi
        overlay[mask, 0] = r; overlay[mask, 1] = g
        overlay[mask, 2] = b; overlay[mask, 3] = alpha
    return overlay

# ── MLCO layer overlay helpers ─────────────────────────────────────────────────

def build_layer_rgb(mlco):
    """Solid RGB: each MLCO layer mapped to plasma colormap by depth."""
    cmap = matplotlib.colormaps[LAYER_CMAP]
    rgb  = np.full((*mlco.shape, 3), 0.08, dtype=np.float32)
    for li in range(N_LAYERS):
        r, g, b, _ = cmap(li / (N_LAYERS - 1))
        mask = (mlco == li + 1) | (mlco == li + 1 + N_LAYERS)
        rgb[mask, 0] = r; rgb[mask, 1] = g; rgb[mask, 2] = b
    return rgb


def build_layer_overlay(mlco, alpha=LAYER_ALPHA):
    """Semi-transparent RGBA MLCO layer overlay."""
    cmap    = matplotlib.colormaps[LAYER_CMAP]
    overlay = np.zeros((*mlco.shape, 4), dtype=np.float32)
    for li in range(N_LAYERS):
        r, g, b, _ = cmap(li / (N_LAYERS - 1))
        mask = (mlco == li + 1) | (mlco == li + 1 + N_LAYERS)
        overlay[mask, 0] = r; overlay[mask, 1] = g
        overlay[mask, 2] = b; overlay[mask, 3] = alpha
    return overlay


def draw_zone_boundaries(ax, mlco_crop, lw=0.7, alpha=0.80):
    """White dashed contours at static 5-zone boundaries."""
    for bnd in ZONE_BOUNDARIES:
        mask = np.zeros(mlco_crop.shape, dtype=np.float32)
        for offset in (0, N_LAYERS):
            mask += (mlco_crop == bnd + offset).astype(np.float32)
        if mask.max() > 0:
            try:
                ax.contour(mask, levels=[0.5], colors=['white'],
                           linewidths=lw, linestyles=['--'], alpha=alpha)
            except Exception:
                pass

# ── K-Means per-sample figure ──────────────────────────────────────────────────

def make_kmeans_sample_figure(sid, condition, res, t2, mlco, gname, out_dir):
    """3-panel: T₂* | K-means zone solid | T₂* + zone overlay."""
    zone_map = build_zone_index_map(mlco, res['label_per_layer'])
    overlay  = build_rgba_overlay(zone_map)
    r0, r1, c0, c1 = kidney_crop_box(mlco)
    t2_crop   = t2[r0:r1, c0:c1]
    zo_crop   = zone_map[r0:r1, c0:c1]
    ov_crop   = overlay[r0:r1, c0:c1]
    mlco_crop = mlco[r0:r1, c0:c1]
    vmin, vmax = t2star_display_range(t2, mlco)
    crop_h, crop_w = t2_crop.shape

    fig_w = 13.0
    fig_h = max(3.5, (fig_w / 3) * (crop_h / crop_w) + 1.2)
    fig, axes = plt.subplots(1, 3, figsize=(fig_w, fig_h))
    fig.patch.set_facecolor('#1A1A1A')

    ginfo   = next(v for v in GROUPS.values() if sid in v['ids'])
    sil_str = f"silhouette = {res['silhouette']:.3f}"
    fig.suptitle(f'{sid}  —  {COND_LABELS[condition]}    [{gname}]    {sil_str}',
                 fontsize=10, color='white', y=1.01)

    imshow_kw = dict(origin='upper', interpolation='nearest',
                     extent=[0, crop_w, crop_h, 0])
    t2_masked = np.where(mlco_crop > 0, t2_crop, np.nan)

    # Panel 1: T₂*
    ax1 = axes[0]; ax1.set_facecolor('black')
    im1 = ax1.imshow(t2_masked, cmap='inferno', vmin=vmin, vmax=vmax, **imshow_kw)
    ax1.set_title('T₂* map', color='white', fontsize=9, pad=4); ax1.axis('off')
    cb1 = fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.03, shrink=0.85)
    cb1.set_label('T₂* (ms)', color='white', fontsize=8)
    cb1.ax.yaxis.set_tick_params(color='white', labelcolor='white', labelsize=7)

    # Panel 2: K-means zones (solid)
    ax2 = axes[1]; ax2.set_facecolor('black')
    zone_rgb = np.zeros((*t2_crop.shape, 3), dtype=np.float32)
    for zi, hex_color in enumerate(ZONE_COLORS):
        r, g, b, _ = mcolors.to_rgba(hex_color)
        m = zo_crop == zi
        zone_rgb[m, 0] = r; zone_rgb[m, 1] = g; zone_rgb[m, 2] = b
    zone_rgb[mlco_crop == 0] = 0.08
    ax2.imshow(zone_rgb, **imshow_kw)
    for zi in range(N_CLUSTERS):
        binary = (zo_crop == zi).astype(float)
        if binary.max() > 0:
            try:
                ax2.contour(binary, levels=[0.5], colors=['white'], linewidths=0.6, alpha=0.7)
            except Exception:
                pass
    ax2.set_title('K-means zones (k=3)', color='white', fontsize=9, pad=4); ax2.axis('off')
    legend_handles = [Patch(facecolor=c, edgecolor='white', linewidth=0.5, label=n)
                      for c, n in zip(ZONE_COLORS, ZONE_NAMES)]
    ax2.legend(handles=legend_handles, loc='lower right', fontsize=7,
               framealpha=0.55, facecolor='#222', edgecolor='#888', labelcolor='white')
    for zi, name in enumerate(ZONE_NAMES):
        layers = res['zone_layers'][name]
        if layers:
            ax2.text(0.02, 0.97 - zi * 0.09, f'{name}: L{min(layers)}–{max(layers)}',
                     transform=ax2.transAxes, fontsize=6.5, color='white', va='top', alpha=0.90)

    # Panel 3: Overlay
    ax3 = axes[2]; ax3.set_facecolor('black')
    ax3.imshow(t2_masked, cmap='inferno', vmin=vmin, vmax=vmax, **imshow_kw)
    ax3.imshow(ov_crop, **imshow_kw)
    for zi in range(N_CLUSTERS):
        binary = (zo_crop == zi).astype(float)
        if binary.max() > 0:
            try:
                ax3.contour(binary, levels=[0.5], colors=['white'], linewidths=0.7, alpha=0.65)
            except Exception:
                pass
    ax3.set_title(f'T₂* + zone overlay  (α={ZONE_ALPHA})', color='white', fontsize=9, pad=4)
    ax3.axis('off')

    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / f'{sid}_kmeans_{condition}'
    for fmt in ('png', 'svg'):
        fig.savefig(stem.with_suffix(f'.{fmt}'), dpi=200, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f'  ✓ {stem.name}.png/svg')

# ── MLCO per-sample figure ─────────────────────────────────────────────────────

def make_mlco_sample_figure(sid, condition, t2, mlco, gname, out_dir):
    """3-panel: T₂* | MLCO layer depth (plasma) | T₂* + MLCO overlay."""
    layer_rgb = build_layer_rgb(mlco)
    layer_ov  = build_layer_overlay(mlco)
    r0, r1, c0, c1 = kidney_crop_box(mlco)
    t2_crop     = t2[r0:r1, c0:c1]
    mlco_crop   = mlco[r0:r1, c0:c1]
    lr_crop     = layer_rgb[r0:r1, c0:c1]
    lo_crop     = layer_ov[r0:r1, c0:c1]
    vmin, vmax  = t2star_display_range(t2, mlco)
    crop_h, crop_w = t2_crop.shape

    fig_w = 13.0
    fig_h = max(3.5, (fig_w / 3) * (crop_h / crop_w) + 1.2)
    fig, axes = plt.subplots(1, 3, figsize=(fig_w, fig_h))
    fig.patch.set_facecolor('#1A1A1A')
    fig.suptitle(
        f'{sid}  —  {COND_LABELS.get(condition, condition)}    [{gname}]\n'
        f'MLCO: {N_LAYERS}-layer radial segmentation  |  cortex (L1) → papilla (L{N_LAYERS})',
        fontsize=10, color='white', y=1.02,
    )

    imshow_kw = dict(origin='upper', interpolation='nearest',
                     extent=[0, crop_w, crop_h, 0])
    t2_masked = np.where(mlco_crop > 0, t2_crop, np.nan)

    # Panel 1: T₂*
    ax1 = axes[0]; ax1.set_facecolor('black')
    im1 = ax1.imshow(t2_masked, cmap='inferno', vmin=vmin, vmax=vmax, **imshow_kw)
    ax1.set_title('T₂* map', color='white', fontsize=9, pad=4); ax1.axis('off')
    cb1 = fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.03, shrink=0.85)
    cb1.set_label('T₂* (ms)', color='white', fontsize=8)
    cb1.ax.yaxis.set_tick_params(color='white', labelcolor='white', labelsize=7)

    # Panel 2: MLCO layer depth
    ax2 = axes[1]; ax2.set_facecolor('black')
    ax2.imshow(lr_crop, origin='upper', interpolation='nearest',
               extent=[0, crop_w, crop_h, 0])
    draw_zone_boundaries(ax2, mlco_crop)
    ax2.set_title(f'MLCO layers  (k={N_LAYERS})', color='white', fontsize=9, pad=4)
    ax2.axis('off')
    sm = plt.cm.ScalarMappable(cmap=LAYER_CMAP, norm=plt.Normalize(vmin=1, vmax=N_LAYERS))
    sm.set_array([])
    cb2 = fig.colorbar(sm, ax=ax2, fraction=0.046, pad=0.03, shrink=0.85)
    cb2.set_label('Layer', color='white', fontsize=8)
    cb2.set_ticks([1, N_LAYERS])
    cb2.set_ticklabels(['1\n(cortex)', f'{N_LAYERS}\n(papilla)'])
    cb2.ax.yaxis.set_tick_params(color='white', labelcolor='white', labelsize=7)
    for i, (zname, zs, ze, zcol) in enumerate(ZONE_DEFS):
        ax2.text(0.02, 0.97 - i * 0.09, f'{zname}: L{zs}–{ze}',
                 transform=ax2.transAxes, fontsize=6.5, color=zcol, va='top', alpha=0.92)

    # Panel 3: Overlay
    ax3 = axes[2]; ax3.set_facecolor('black')
    ax3.imshow(t2_masked, cmap='inferno', vmin=vmin, vmax=vmax, **imshow_kw)
    ax3.imshow(lo_crop, origin='upper', interpolation='nearest',
               extent=[0, crop_w, crop_h, 0])
    draw_zone_boundaries(ax3, mlco_crop, lw=0.7, alpha=0.65)
    ax3.set_title(f'T₂* + MLCO overlay  (α={LAYER_ALPHA})', color='white', fontsize=9, pad=4)
    ax3.axis('off')

    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / f'{sid}_mlco_{condition}'
    for fmt in ('png', 'svg'):
        fig.savefig(stem.with_suffix(f'.{fmt}'), dpi=200, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f'  ✓ {stem.name}.png/svg')

# ── Grid figures ───────────────────────────────────────────────────────────────

def _grid_base(condition, title):
    """Create grid figure base (axes, fig) sized for all groups."""
    n_rows = len(GROUPS)
    n_cols = max(len(info['ids']) for info in GROUPS.values())
    cell_h = 3.5
    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(n_cols * 4.2, n_rows * cell_h + 1.0),
                              squeeze=False)
    fig.patch.set_facecolor('#1A1A1A')
    fig.suptitle(title, fontsize=11, color='white', y=1.02)
    return fig, axes, n_rows, n_cols


def make_kmeans_grid_figure(all_results, condition='oxygen_2'):
    """Grid of k-means overlay panels for all samples × one condition."""
    title = (f'K-Means Zone Overlay  —  {COND_LABELS[condition]}\n'
             'Feature set: [T₂* median, within-layer T₂* std, depth]  |  k = 3')
    fig, axes, n_rows, n_cols = _grid_base(condition, title)

    for row_idx, (gname, ginfo) in enumerate(GROUPS.items()):
        ids = ginfo['ids']
        for col_idx in range(n_cols):
            ax = axes[row_idx, col_idx]
            ax.set_facecolor('black')
            if col_idx >= len(ids):
                ax.axis('off'); continue
            sid = ids[col_idx]
            res = all_results.get(sid, {}).get(condition)
            if res is None:
                ax.text(0.5, 0.5, f'{sid}\n(missing)', ha='center', va='center',
                        color='white', transform=ax.transAxes); ax.axis('off'); continue
            try:
                t2, mlco = load_maps(sid, condition)
            except FileNotFoundError:
                ax.axis('off'); continue
            zone_map = build_zone_index_map(mlco, res['label_per_layer'])
            overlay  = build_rgba_overlay(zone_map)
            r0, r1, c0, c1 = kidney_crop_box(mlco)
            vmin, vmax = t2star_display_range(t2, mlco)
            t2_m  = np.where(mlco[r0:r1, c0:c1] > 0, t2[r0:r1, c0:c1], np.nan)
            ax.imshow(t2_m, cmap='inferno', vmin=vmin, vmax=vmax,
                      origin='upper', interpolation='nearest')
            ax.imshow(overlay[r0:r1, c0:c1], origin='upper', interpolation='nearest')
            zo_c = zone_map[r0:r1, c0:c1]
            for zi in range(N_CLUSTERS):
                binary = (zo_c == zi).astype(float)
                if binary.max() > 0:
                    try:
                        ax.contour(binary, levels=[0.5], colors=['white'],
                                   linewidths=0.5, alpha=0.6)
                    except Exception:
                        pass
            ax.set_title(f'{sid}\nsil={res["silhouette"]:.3f}',
                         color=ginfo['color'], fontsize=8, pad=3); ax.axis('off')
            info_lines = [f'{n}: L{min(res["zone_layers"][n])}–{max(res["zone_layers"][n])}'
                          for n in ZONE_NAMES if res['zone_layers'][n]]
            ax.text(0.02, 0.02, '\n'.join(info_lines), transform=ax.transAxes,
                    fontsize=5.5, color='white', va='bottom', alpha=0.85,
                    bbox=dict(fc='#00000055', ec='none', pad=1.5))
        axes[row_idx, 0].set_ylabel(ginfo.get('label', gname),
                                    color=ginfo['color'], fontsize=10,
                                    fontweight='bold', rotation=90, labelpad=6)
        axes[row_idx, 0].yaxis.set_label_position('left')

    legend_handles = [Patch(facecolor=c, edgecolor='white', linewidth=0.5, label=n)
                      for c, n in zip(ZONE_COLORS, ZONE_NAMES)]
    fig.legend(handles=legend_handles, loc='lower center', ncol=N_CLUSTERS,
               fontsize=9, framealpha=0.4, facecolor='#333', edgecolor='#888',
               labelcolor='white', title='K-means zone', title_fontsize=9,
               bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    cond_tag = condition.replace('_', '')
    for fmt in ('png', 'svg'):
        out = OUTPUT_DIR / f'kmeans_overlay_grid_{cond_tag}.{fmt}'
        fig.savefig(out, dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor())
        print(f'  ✓ {out.name}')
    plt.close(fig)


def make_mlco_grid_figure(all_data, condition='oxygen_2'):
    """Grid of MLCO layer overlay panels for all samples × one condition."""
    title = (f'MLCO Layer Overlay  —  {COND_LABELS.get(condition, condition)}\n'
             f'{N_LAYERS}-layer radial segmentation  |  plasma: L1 (cortex) → L{N_LAYERS} (papilla)')
    fig, axes, n_rows, n_cols = _grid_base(condition, title)

    for row_idx, (gname, ginfo) in enumerate(GROUPS.items()):
        ids = ginfo['ids']
        for col_idx in range(n_cols):
            ax = axes[row_idx, col_idx]
            ax.set_facecolor('black')
            if col_idx >= len(ids):
                ax.axis('off'); continue
            sid   = ids[col_idx]
            entry = all_data.get(sid, {}).get(condition)
            if entry is None:
                ax.text(0.5, 0.5, f'{sid}\n(missing)', ha='center', va='center',
                        color='white', transform=ax.transAxes, fontsize=8)
                ax.axis('off'); continue
            t2, mlco = entry['t2'], entry['mlco']
            r0, r1, c0, c1 = kidney_crop_box(mlco)
            vmin, vmax = t2star_display_range(t2, mlco)
            t2_m = np.where(mlco[r0:r1, c0:c1] > 0, t2[r0:r1, c0:c1], np.nan)
            ax.imshow(t2_m, cmap='inferno', vmin=vmin, vmax=vmax,
                      origin='upper', interpolation='nearest')
            ax.imshow(build_layer_overlay(mlco)[r0:r1, c0:c1],
                      origin='upper', interpolation='nearest')
            draw_zone_boundaries(ax, mlco[r0:r1, c0:c1], lw=0.5, alpha=0.6)
            ax.set_title(sid, color=ginfo['color'], fontsize=8, pad=3); ax.axis('off')
        axes[row_idx, 0].text(-0.06, 0.5, gname,
                               transform=axes[row_idx, 0].transAxes,
                               color=ginfo['color'], fontsize=8,
                               rotation=90, va='center', ha='right')
    fig.tight_layout()
    cond_stub = 'oxygen2' if condition == 'oxygen_2' else condition
    stem = OUTPUT_DIR / f'mlco_overlay_grid_{cond_stub}'
    for fmt in ('png', 'svg'):
        fig.savefig(stem.with_suffix(f'.{fmt}'), dpi=150, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
        print(f'  ✓ {stem.name}.{fmt}')
    plt.close(fig)

# ── K-Means zone analysis figure ───────────────────────────────────────────────

def group_std_profile(ids, condition):
    """Group mean ± SEM of per-layer within-layer T₂* std."""
    per_sample = {}
    for sid in ids:
        try:
            t2, mlco = load_maps(sid, condition)
            feats    = bilateral_layer_features(t2, mlco)
            per_sample[sid] = {f['layer']: f['std'] for f in feats
                               if not np.isnan(f['std'])}
        except FileNotFoundError:
            print(f'  WARNING: {sid}/{condition} missing, skipping')
    all_layers = sorted({l for lm in per_sample.values() for l in lm})
    layers, means, sems = [], [], []
    for l in all_layers:
        vals = [per_sample[sid][l] for sid in per_sample if l in per_sample[sid]]
        if not vals:
            continue
        layers.append(l)
        means.append(np.mean(vals))
        sems.append(np.std(vals, ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0)
    return np.array(layers), np.array(means), np.array(sems)


def shade_zones(ax, y0=None, y1=None, fontsize=6.5):
    if y0 is None or y1 is None:
        y0, y1 = ax.get_ylim()
    for label, (l0, l1, color) in ZONE_CONFIG_SHADING.items():
        ax.add_patch(Rectangle((l0 - 0.5, y0), l1 - l0 + 1, y1 - y0,
                                facecolor=color, edgecolor='none', alpha=0.30, zorder=0))
        ax.text((l0 + l1) / 2, y0 + 0.025 * (y1 - y0), label,
                ha='center', va='bottom', fontsize=fontsize,
                color='#555', style='italic', zorder=5)
    ax.set_ylim(y0, y1)


def dot_plot(ax, vals_a, vals_b, p_val, ylabel, title, panel_label, alt_str=''):
    """Scatter dot plot with group means and significance bracket."""
    gkeys  = list(GROUPS.keys())
    col_a  = GROUPS[gkeys[0]]['color']
    col_b  = GROUPS[gkeys[1]]['color']
    rng    = np.random.default_rng(42)
    for x, vals, col in [(0, vals_a, col_a), (1, vals_b, col_b)]:
        jitter = rng.uniform(-0.08, 0.08, len(vals))
        ax.scatter(np.full(len(vals), x) + jitter, vals,
                   color=col, s=50, zorder=4, edgecolors='white', lw=0.6)
        ax.plot([x - 0.17, x + 0.17], [np.mean(vals)] * 2, color=col, lw=2.8, zorder=5)
    ax.set_xticks([0, 1])
    def _fmt(s):
        i = s.rfind(' (')
        return s[:i] + '\n' + s[i + 1:] if i >= 0 else s
    ax.set_xticklabels([_fmt(gkeys[0]), _fmt(gkeys[1])], fontsize=8)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=9.5, style='italic', color='#333')
    ax.tick_params(labelsize=8)
    ax.grid(axis='y', alpha=0.25, zorder=1)
    y_lo, y_hi = ax.get_ylim()
    margin = 0.22 * (y_hi - y_lo)
    ax.set_ylim(y_lo, y_hi + margin)
    y_bar  = y_hi + 0.06 * (y_hi - y_lo)
    ax.plot([0, 1], [y_bar, y_bar], color='#555', lw=1.0)
    pstr   = f'p = {p_val:.3f}' if not np.isnan(p_val) else 'p = n/a'
    sig    = '*' if (not np.isnan(p_val) and p_val < 0.05) else ''
    label_str = f'{pstr}{sig}'
    if alt_str:
        label_str += f'\n({alt_str})'
    ax.text(0.5, y_bar + 0.01 * (y_hi - y_lo), label_str,
            ha='center', va='bottom', fontsize=7.5, color='#333')
    ax.text(-0.22, 1.06, panel_label, transform=ax.transAxes,
            fontsize=14, fontweight='bold', va='top')


def make_zone_figure(results):
    """4-panel k-means zone analysis summary figure using pre-computed results."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    gkeys    = list(GROUPS.keys())
    grp_a, grp_b = gkeys[0], gkeys[1]
    cond = 'oxygen_2'

    def group_vals(gname, key):
        return [results[sid][cond][key]
                for sid in GROUPS[gname]['ids']
                if results[sid].get(cond) is not None
                and not np.isnan(float(results[sid][cond][key]))]

    vals_a_sil = group_vals(grp_a, 'silhouette')
    vals_b_sil = group_vals(grp_b, 'silhouette')
    vals_a_std = group_vals(grp_a, 'sup_std_mean')
    vals_b_std = group_vals(grp_b, 'sup_std_mean')

    def mwu(a, b, alt='two-sided'):
        if len(a) >= 2 and len(b) >= 2:
            _, p = mannwhitneyu(a, b, alternative=alt)
            return float(p)
        return np.nan

    p_sil = mwu(vals_b_sil, vals_a_sil, alt='two-sided')
    p_std = mwu(vals_b_std, vals_a_std, alt='greater')

    fig = plt.figure(figsize=(16, 9.5))
    sg  = gridspec.GridSpec(2, 6, figure=fig, hspace=0.52, wspace=0.50,
                            left=0.07, right=0.97, top=0.89, bottom=0.09)
    ax_prof  = fig.add_subplot(sg[0, 0:3])
    ax_strip = fig.add_subplot(sg[0, 3:6])
    ax_sil   = fig.add_subplot(sg[1, 0:2])
    ax_std   = fig.add_subplot(sg[1, 2:4])
    ax_note  = fig.add_subplot(sg[1, 4:6])

    title_groups = '  vs.  '.join(gkeys)
    fig.suptitle(
        f'K-Means Zone Clustering  —  {title_groups}\n'
        'Feature set: [T₂* median, within-layer T₂* std, normalised depth]  |  k = 3 clusters',
        fontsize=10.5, color='#222'
    )

    # Panel A: Per-layer T₂* std profile
    for gname, ginfo in GROUPS.items():
        x, gm, gs_ = group_std_profile(ginfo['ids'], cond)
        ax_prof.plot(x, gm, color=ginfo['color'], ls=ginfo['ls'], lw=ginfo['lw'],
                     label=gname, zorder=4)
        if len(ginfo['ids']) > 1:
            ax_prof.fill_between(x, gm - gs_, gm + gs_,
                                 color=ginfo['color'], alpha=0.15, zorder=3)
    ax_prof.set_xlim(0.5, 24.5)
    ax_prof.set_xlabel('MLCO Layer  (surface → center)', fontsize=9)
    ax_prof.set_ylabel('Within-layer T₂* std (ms)', fontsize=9)
    ax_prof.set_title('Per-layer T₂* heterogeneity  [Post-O₂, 100% O₂]',
                      fontsize=9.5, style='italic', color='#333')
    ax_prof.tick_params(labelsize=8)
    ax_prof.grid(axis='y', alpha=0.25, zorder=1)
    ax_prof.legend(fontsize=7.5, loc='upper left', framealpha=0.85)
    ax_prof.axvspan(0.5, 5.5, alpha=0.10, color='gold', zorder=0)
    fig.canvas.draw()
    shade_zones(ax_prof)
    ax_prof.text(-0.08, 1.04, 'A', transform=ax_prof.transAxes,
                 fontsize=14, fontweight='bold', va='top')

    # Panel B: Cluster assignment strip
    y_pos = 0
    yticks, ytick_labels, y_label_colors = [], [], []
    for gname in gkeys:
        ginfo = GROUPS[gname]
        for sid in ginfo['ids']:
            res = results[sid].get(cond)
            if res is None:
                y_pos += 1; continue
            for layer_idx in range(N_LAYERS):
                zone_idx = int(res['label_per_layer'][layer_idx])
                ax_strip.barh(y_pos, 1, left=layer_idx + 0.5, height=0.80,
                              color=ZONE_COLORS[zone_idx],
                              edgecolor='white', linewidth=0.3)
            yticks.append(y_pos); ytick_labels.append(sid)
            y_label_colors.append(ginfo['color']); y_pos += 1
        ax_strip.axhline(y_pos - 0.5, color='#555', lw=1.2, ls='--', alpha=0.6)
        y_pos += 0.3
    ax_strip.set_xlim(0.5, N_LAYERS + 0.5)
    ax_strip.set_yticks(yticks); ax_strip.set_yticklabels(ytick_labels, fontsize=8)
    for tick, col in zip(ax_strip.get_yticklabels(), y_label_colors):
        tick.set_color(col)
    ax_strip.set_xlabel('MLCO Layer  (surface → center)', fontsize=9)
    ax_strip.set_title('K-means zone assignment per animal  [Post-O₂]',
                       fontsize=9.5, style='italic', color='#333')
    ax_strip.tick_params(labelsize=8)
    ax_strip.axvline(5.5, color='gold', lw=1.5, ls=':', alpha=0.9, zorder=5)
    ax_strip.text(5.5, y_pos + 0.1, 'OC\nboundary', ha='center', va='bottom',
                  fontsize=6.5, color='#7B5800', style='italic')
    legend_handles = [plt.Rectangle((0, 0), 1, 1, color=c, label=n)
                      for c, n in zip(ZONE_COLORS, ZONE_NAMES)]
    ax_strip.legend(handles=legend_handles, loc='lower right', fontsize=8,
                    framealpha=0.88, title='K-means zone', title_fontsize=7.5)
    ax_strip.text(-0.08, 1.04, 'B', transform=ax_strip.transAxes,
                  fontsize=14, fontweight='bold', va='top')

    # Panels C, D: Dot plots
    dot_plot(ax_sil, vals_a_sil, vals_b_sil, p_sil,
             'Silhouette score', 'Zone separation quality\n(silhouette, k=3)',
             'C', alt_str='two-sided')
    dot_plot(ax_std, vals_a_std, vals_b_std, p_std,
             'Mean T₂* std in superficial zone (ms)',
             'Outer cortex heterogeneity\n(k-means superficial cluster)',
             'D', alt_str=f'one-sided, {grp_b} > {grp_a}')

    # Panel E: Interpretation note
    ax_note.axis('off')
    n_a = len(GROUPS[grp_a]['ids']); n_b = len(GROUPS[grp_b]['ids'])
    note_text = (
        'Interpretation\n\n'
        'K-means uses within-layer T₂* std as a\n'
        'clustering feature, making it sensitive\n'
        'to focal tissue disruption — not just\n'
        'mean oxygenation differences.\n\n'
        'Elevated voxel-to-voxel T₂* variance\n'
        'in surface layers shifts superficial\n'
        'cluster membership, reducing silhouette\n'
        'and raising superficial-zone std.\n\n'
        'Panel D captures outer-cortex\n'
        'heterogeneity through the k-means lens:\n'
        'higher std in superficial cluster\n'
        'indicates focal tissue disruption.\n\n'
        f'⚠  Groups: {grp_a} (n={n_a}),\n'
        f'   {grp_b} (n={n_b}).\n'
        f'   Min achievable p (one-sided) =\n'
        f'   {1 / (n_a + n_b):.3f} with these sample sizes.'
    )
    ax_note.text(0.05, 0.97, note_text, transform=ax_note.transAxes, fontsize=8,
                 va='top', ha='left', color='#333',
                 bbox=dict(boxstyle='round,pad=0.6', fc='#F8F8F8', ec='#BBBBBB', alpha=0.90),
                 linespacing=1.55)
    ax_note.text(-0.18, 1.06, 'E', transform=ax_note.transAxes,
                 fontsize=14, fontweight='bold', va='top')

    for fmt in ('png', 'pdf', 'svg'):
        out = OUTPUT_DIR / f'kmeans_zone_analysis.{fmt}'
        fig.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
        print(f'  ✓ {out.name}')
    plt.close(fig)

    print('\n── K-Means Zone Summary (Post-O₂) ─────────────────────────')
    print(f'  Silhouette   {grp_a}: {np.mean(vals_a_sil):.3f} ± {np.std(vals_a_sil):.3f}'
          f'   {grp_b}: {np.mean(vals_b_sil):.3f} ± {np.std(vals_b_sil):.3f}'
          f'   p = {p_sil:.3f} (two-sided)')
    print(f'  Sup T₂* std  {grp_a}: {np.mean(vals_a_std):.2f} ± {np.std(vals_a_std):.2f} ms'
          f'   {grp_b}: {np.mean(vals_b_std):.2f} ± {np.std(vals_b_std):.2f} ms'
          f'   p = {p_std:.3f} (one-sided)')

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Step 1: K-means (computed once, reused for overlays + zone figure) ──
    print('\n[1/3] K-Means clustering + per-sample overlay figures...')
    all_results = {}
    all_data    = {}    # for MLCO overlays

    for gname, ginfo in GROUPS.items():
        for sid in ginfo['ids']:
            all_results[sid] = {}
            all_data[sid]    = {}
            out_km   = ANALYSIS_DIR / sid / 'kmeans'
            out_mlco = ANALYSIS_DIR / sid / 'mlco'
            for cond in CONDITIONS:
                print(f'  {sid} / {cond}...')
                try:
                    t2, mlco = load_maps(sid, cond)
                except FileNotFoundError:
                    print(f'    MISSING — skipping')
                    all_results[sid][cond] = None
                    all_data[sid][cond]    = None
                    continue
                feats = bilateral_layer_features(t2, mlco)
                res   = kmeans_cluster(feats)
                if res is None:
                    print(f'    Clustering failed')
                    all_results[sid][cond] = None
                else:
                    all_results[sid][cond] = res
                    make_kmeans_sample_figure(sid, cond, res, t2, mlco, gname, out_km)
                all_data[sid][cond] = {'t2': t2, 'mlco': mlco}
                make_mlco_sample_figure(sid, cond, t2, mlco, gname, out_mlco)

    # ── Step 2: Grid figures ──────────────────────────────────────────────────
    print('\n[2/3] Generating grid figures...')
    for cond in CONDITIONS:
        make_kmeans_grid_figure(all_results, condition=cond)
        make_mlco_grid_figure(all_data, condition=cond)

    # ── Step 3: Zone analysis summary figure ──────────────────────────────────
    print('\n[3/3] Generating zone analysis summary figure...')
    make_zone_figure(all_results)

    print(f'\nDone. Outputs in: {OUTPUT_DIR}')


def run(pep_path, output_dir=None, **kwargs):
    """
    Importable entry point for overlay_analysis.

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
        OUTPUT_DIR = Path(output_dir)
    main()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Overlay Analysis — K-Means + MLCO Layers')
    parser.add_argument('--pep', required=True, help='Path to PEP project_config.yaml')
    args = parser.parse_args()
    load_pep_project(args.pep)
    main()
