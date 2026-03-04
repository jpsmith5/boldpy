#!/usr/bin/env python3
"""
Hybrid MLCO + K-Means Clustering for Data-Driven Zone Boundaries
=================================================================

Clusters per-layer T2*/R2* statistics to automatically determine tissue zone
boundaries from the data, replacing hardcoded layer-to-zone mappings.

Inspired by Menzies et al. (2013) — uses k-means clustering on per-layer
median T2* values (not individual voxels) to find natural tissue groupings,
then maps clusters to anatomical zone names by spatial depth ordering.

Output: zone config dict in the same format as YAML-loaded configs, so all
downstream code (mlco_analysis, boldpy_plots) works unchanged.

Usage:
    from cluster_zones import cluster_and_build_zones

    zone_config, diagnostics = cluster_and_build_zones(
        t2_map=t2_map, r2_map=r2_map, mlco_mask=mlco_mask,
        n_layers=24, n_clusters=3
    )

    # Inject into tissue_zones module globals
    from tissue_zones import update_configs_from_dict
    update_configs_from_dict(zone_config)
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path

try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    from sklearn.mixture import GaussianMixture
    from sklearn.metrics import silhouette_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


# Zone name mappings for different cluster counts
def _upsample_perfusion(perfusion_map: np.ndarray, target_shape: tuple) -> np.ndarray:
    """
    Upsample perfusion map to match T2*/R2* resolution using bilinear interpolation.

    Perfusion data is typically acquired at lower resolution (e.g., 80x80) than
    T2* maps (200x200). Bilinear interpolation preserves the smooth perfusion
    gradients better than nearest-neighbor for quantitative analysis.

    Parameters
    ----------
    perfusion_map : np.ndarray
        Perfusion map at original resolution.
    target_shape : tuple
        Target (height, width) to match T2* maps.

    Returns
    -------
    np.ndarray
        Upsampled perfusion map.
    """
    if perfusion_map.shape == target_shape:
        return perfusion_map

    from scipy import ndimage

    zoom_factors = (
        target_shape[0] / perfusion_map.shape[0],
        target_shape[1] / perfusion_map.shape[1],
    )

    # Handle NaNs: interpolate the valid data, then restore NaN mask
    nan_mask = np.isnan(perfusion_map)
    if np.any(nan_mask):
        filled = perfusion_map.copy()
        filled[nan_mask] = 0.0
        upsampled = ndimage.zoom(filled, zoom_factors, order=1)  # bilinear
        nan_upsampled = ndimage.zoom(nan_mask.astype(float), zoom_factors, order=0)  # nearest
        upsampled[nan_upsampled > 0.5] = np.nan
    else:
        upsampled = ndimage.zoom(perfusion_map, zoom_factors, order=1)

    return upsampled


# Depth-ordered descriptors for cluster zones (superficial to deep)
DEPTH_DESCRIPTORS = {
    1: ['whole'],
    2: ['superficial', 'deep'],
    3: ['superficial', 'intermediate', 'deep'],
    4: ['superficial', 'mid-superficial', 'mid-deep', 'deep'],
    5: ['superficial', 'mid-superficial', 'intermediate', 'mid-deep', 'deep'],
}


def extract_layer_features(
    t2_map: np.ndarray,
    r2_map: np.ndarray,
    mlco_mask: np.ndarray,
    n_layers: int = 24,
    organ_start_layer: int = 1,
    perfusion_map: Optional[np.ndarray] = None
) -> List[Dict]:
    """
    Extract per-layer feature vectors from T2*/R2* maps and MLCO mask.

    For each layer, computes median T2*, median R2*, std T2*, pixel count,
    and normalized depth (0 = surface, 1 = center).

    Parameters
    ----------
    t2_map : np.ndarray
        T2* map (2D).
    r2_map : np.ndarray
        R2* map (2D).
    mlco_mask : np.ndarray
        MLCO layer mask (2D), integer-valued with layer numbers.
    n_layers : int
        Number of layers per organ.
    organ_start_layer : int
        Starting layer number in the mask (1 for right/single, n_layers+1 for left).
    perfusion_map : np.ndarray, optional
        Perfusion map (2D). If resolution differs from t2_map (e.g., 80x80 vs
        200x200), it will be automatically upsampled via bilinear interpolation
        to match the T2* resolution.

    Returns
    -------
    list of dict
        One dict per layer with keys: layer_idx (1-based), t2star_median,
        t2star_std, r2star_median, r2star_std, n_pixels, depth_normalized,
        and optionally perfusion_median.
    """
    # Upsample perfusion if needed
    if perfusion_map is not None and perfusion_map.shape != t2_map.shape:
        print(f"  Upsampling perfusion {perfusion_map.shape} -> {t2_map.shape} for clustering")
        perfusion_map = _upsample_perfusion(perfusion_map, t2_map.shape)

    features = []

    for layer_offset in range(n_layers):
        layer_num = organ_start_layer + layer_offset
        layer_idx = layer_offset + 1  # 1-indexed

        layer_pixels = (mlco_mask == layer_num)
        valid = layer_pixels & ~np.isnan(t2_map) & ~np.isnan(r2_map)
        n_pixels = int(np.sum(valid))

        if n_pixels == 0:
            features.append({
                'layer_idx': layer_idx,
                't2star_median': np.nan,
                't2star_std': np.nan,
                'r2star_median': np.nan,
                'r2star_std': np.nan,
                'n_pixels': 0,
                'depth_normalized': (layer_idx - 1) / max(n_layers - 1, 1),
                'perfusion_median': np.nan,
            })
            continue

        t2_values = t2_map[valid]
        r2_values = r2_map[valid]

        feat = {
            'layer_idx': layer_idx,
            't2star_median': float(np.median(t2_values)),
            't2star_std': float(np.std(t2_values)),
            'r2star_median': float(np.median(r2_values)),
            'r2star_std': float(np.std(r2_values)),
            'n_pixels': n_pixels,
            'depth_normalized': (layer_idx - 1) / max(n_layers - 1, 1),
            'perfusion_median': np.nan,
        }

        if perfusion_map is not None:
            perf_valid = valid & ~np.isnan(perfusion_map)
            if np.sum(perf_valid) > 0:
                feat['perfusion_median'] = float(np.median(perfusion_map[perf_valid]))

        features.append(feat)

    return features


def cluster_layers(
    layer_features: List[Dict],
    n_clusters: int = 3,
    features: List[str] = None,
    method: str = 'kmeans',
    random_state: int = 42
) -> Dict:
    """
    Cluster layers based on their feature vectors.

    Parameters
    ----------
    layer_features : list of dict
        Output from extract_layer_features().
    n_clusters : int
        Number of clusters (3 = cortex/medulla/papilla, 5 = 5-zone).
    features : list of str
        Feature names to use for clustering. Default: ['t2star_median', 'depth_normalized'].
    method : str
        'kmeans' or 'gmm' (Gaussian Mixture Model).
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    dict with keys:
        labels : np.ndarray of int — cluster label per layer
        centroids : np.ndarray — cluster centers in feature space
        silhouette : float — silhouette score (-1 to 1, higher = better separation)
        model : fitted sklearn model
        feature_names : list of str — features used
        feature_matrix : np.ndarray — scaled feature matrix
        scaler : StandardScaler — fitted scaler
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError(
            "scikit-learn is required for clustering. Install with: pip install scikit-learn"
        )

    if features is None:
        features = ['t2star_median', 'depth_normalized']

    # Build feature matrix, handling NaN layers
    valid_indices = []
    rows = []
    for i, feat in enumerate(layer_features):
        vals = [feat[f] for f in features]
        if any(np.isnan(v) for v in vals):
            continue
        valid_indices.append(i)
        rows.append(vals)

    if len(rows) < n_clusters:
        raise ValueError(
            f"Only {len(rows)} valid layers but {n_clusters} clusters requested. "
            "Not enough data for clustering."
        )

    X = np.array(rows)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if method == 'kmeans':
        model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        labels_valid = model.fit_predict(X_scaled)
        centroids_scaled = model.cluster_centers_
    elif method == 'gmm':
        model = GaussianMixture(
            n_components=n_clusters, random_state=random_state, n_init=5
        )
        model.fit(X_scaled)
        labels_valid = model.predict(X_scaled)
        centroids_scaled = model.means_
    else:
        raise ValueError(f"Unknown method: {method}. Use 'kmeans' or 'gmm'.")

    # Compute silhouette score
    sil = silhouette_score(X_scaled, labels_valid) if len(set(labels_valid)) > 1 else 0.0

    # Map back to all layers (NaN layers get label -1)
    all_labels = np.full(len(layer_features), -1, dtype=int)
    for idx, vi in enumerate(valid_indices):
        all_labels[vi] = labels_valid[idx]

    # Unscale centroids back to original feature space
    centroids_original = scaler.inverse_transform(centroids_scaled)

    return {
        'labels': all_labels,
        'centroids': centroids_original,
        'centroids_scaled': centroids_scaled,
        'silhouette': float(sil),
        'model': model,
        'feature_names': features,
        'feature_matrix': X,
        'feature_matrix_scaled': X_scaled,
        'scaler': scaler,
        'valid_indices': valid_indices,
    }


def assign_tissue_labels(
    layer_features: List[Dict],
    cluster_labels: np.ndarray,
    n_clusters: int
) -> Dict[str, List[int]]:
    """
    Map cluster labels to generic zone names by spatial depth ordering.

    Clusters are sorted by their mean normalized depth (ascending) and
    assigned generic names (zone_1, zone_2, ...) rather than anatomical
    terms, since k-means groupings are statistical — not validated
    anatomical boundaries.

    Parameters
    ----------
    layer_features : list of dict
        Output from extract_layer_features().
    cluster_labels : np.ndarray
        Cluster label per layer (from cluster_layers).
    n_clusters : int
        Number of clusters.

    Returns
    -------
    dict mapping zone name (str) to list of layer numbers (1-based).
    """
    # Compute mean depth per cluster
    cluster_depths = {}
    for cluster_id in range(n_clusters):
        mask = cluster_labels == cluster_id
        if not np.any(mask):
            cluster_depths[cluster_id] = float('inf')
            continue
        depths = [layer_features[i]['depth_normalized'] for i in range(len(layer_features)) if mask[i]]
        cluster_depths[cluster_id] = np.mean(depths)

    # Sort clusters by mean depth (ascending = surface to center)
    sorted_clusters = sorted(cluster_depths.keys(), key=lambda c: cluster_depths[c])

    # Generic zone names: zone_1 (shallowest) to zone_k (deepest)
    names = [f'zone_{i+1}' for i in range(n_clusters)]

    # Build cluster_id -> zone_name mapping
    cluster_to_zone = {}
    for rank, cluster_id in enumerate(sorted_clusters):
        cluster_to_zone[cluster_id] = names[rank]

    # Assign clustered layers to tissue names
    assignments = {name: [] for name in names}
    for i, feat in enumerate(layer_features):
        label = cluster_labels[i]
        if label >= 0:
            assignments[cluster_to_zone[label]].append(feat['layer_idx'])

    # Assign unassigned layers (label == -1, e.g., empty layers with no pixels)
    # to the nearest cluster by depth, so all layers 1..n are covered
    for i, feat in enumerate(layer_features):
        if cluster_labels[i] == -1:
            depth = feat['depth_normalized']
            # Find closest cluster by mean depth
            best_cluster = min(cluster_depths.keys(),
                               key=lambda c: abs(cluster_depths[c] - depth))
            assignments[cluster_to_zone[best_cluster]].append(feat['layer_idx'])

    # Sort layer indices for clean output
    for name in assignments:
        assignments[name] = sorted(assignments[name])

    return assignments


def build_zone_config(
    tissue_assignments: Dict[str, List[int]],
    n_layers: int,
    cluster_info: Dict
) -> Dict:
    """
    Build a zone config dict identical in structure to load_zone_config() output.

    Parameters
    ----------
    tissue_assignments : dict
        Output from assign_tissue_labels().
    n_layers : int
        Total number of layers per organ.
    cluster_info : dict
        Output from cluster_layers() (for metadata).

    Returns
    -------
    dict — zone config compatible with tissue_zones.validate_zone_config().
    """
    total_layers = sum(len(layers) for layers in tissue_assignments.values())

    zones = {}
    zone_names = list(tissue_assignments.keys())
    n_clusters = len(zone_names)
    descriptors = DEPTH_DESCRIPTORS.get(n_clusters,
                                         [f'rank {i+1}' for i in range(n_clusters)])
    for i, (zone_name, layers) in enumerate(tissue_assignments.items()):
        pct = round(100.0 * len(layers) / max(total_layers, 1), 1)
        zones[zone_name] = {
            'layers': layers,
            'description': f"Data-driven cluster {i+1} of {n_clusters} ({descriptors[i]})",
            'percentage': pct,
        }

    # Build aggregate zones
    aggregate_zones = _build_aggregate_zones(tissue_assignments, n_layers)

    config = {
        'metadata': {
            'organ': 'kidney',
            'n_layers': n_layers,
            'description': 'Data-driven zone boundaries via k-means clustering',
            'method': 'kmeans',
            'n_clusters': len(tissue_assignments),
            'silhouette_score': cluster_info.get('silhouette', None),
            'features_used': cluster_info.get('feature_names', []),
            'centroids': cluster_info.get('centroids', np.array([])).tolist()
                if isinstance(cluster_info.get('centroids'), np.ndarray)
                else cluster_info.get('centroids', []),
        },
        'zones': zones,
        'aggregate_zones': aggregate_zones,
    }

    return config


def _build_aggregate_zones(
    tissue_assignments: Dict[str, List[int]],
    n_layers: int
) -> Dict:
    """
    Build aggregate zones from tissue assignments using positional logic.

    Since clustered zones use generic names (zone_1, zone_2, ...) ordered
    by depth, aggregates are built positionally:
    - total_superficial: first zone (shallowest)
    - total_deep: all remaining zones (deeper)
    """
    all_layers = sorted(
        layer for layers in tissue_assignments.values() for layer in layers
    )

    aggregate = {
        'all_zones': {
            'layers': all_layers,
            'description': 'Complete kidney',
        }
    }

    zone_names = list(tissue_assignments.keys())
    n_zones = len(zone_names)

    if n_zones >= 2:
        # First zone (shallowest by depth ordering)
        superficial_layers = sorted(tissue_assignments[zone_names[0]])
        aggregate['total_superficial'] = {
            'layers': superficial_layers,
            'description': f'Superficial layers ({zone_names[0]})',
        }

        # All remaining zones (deeper)
        deep_layers = sorted(
            layer for name in zone_names[1:]
            for layer in tissue_assignments[name]
        )
        aggregate['total_deep'] = {
            'layers': deep_layers,
            'description': f'Deep layers ({", ".join(zone_names[1:])})',
        }

    return aggregate


def cluster_and_build_zones(
    t2_map: np.ndarray,
    r2_map: np.ndarray,
    mlco_mask: np.ndarray,
    n_layers: int = 24,
    n_clusters: int = 3,
    organ_start_layer: int = 1,
    perfusion_map: Optional[np.ndarray] = None,
    features: Optional[List[str]] = None,
    method: str = 'kmeans',
    random_state: int = 42
) -> Tuple[Dict, Dict]:
    """
    Public API: extract features, cluster, assign labels, build zone config.

    Parameters
    ----------
    t2_map : np.ndarray
        T2* map (2D).
    r2_map : np.ndarray
        R2* map (2D).
    mlco_mask : np.ndarray
        MLCO layer mask (2D).
    n_layers : int
        Number of layers per organ.
    n_clusters : int
        Number of tissue clusters (3 or 5 recommended).
    organ_start_layer : int
        Starting layer number in mask.
    perfusion_map : np.ndarray, optional
        Perfusion map (2D).
    features : list of str, optional
        Feature names for clustering. Default: ['t2star_median', 'depth_normalized'].
    method : str
        'kmeans' or 'gmm'.
    random_state : int
        Random seed.

    Returns
    -------
    tuple of (zone_config, diagnostics)
        zone_config : dict compatible with tissue_zones.validate_zone_config()
        diagnostics : dict with layer_features, cluster_info, tissue_assignments
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError(
            "scikit-learn is required for clustering. Install with: pip install scikit-learn"
        )

    # Step 1: Extract per-layer features (auto-upsamples perfusion if needed)
    layer_features = extract_layer_features(
        t2_map, r2_map, mlco_mask, n_layers, organ_start_layer, perfusion_map
    )

    # Step 1.5: Auto-include perfusion as a feature if available and not specified
    if features is None and perfusion_map is not None:
        # Check if any layers have valid perfusion data
        has_perfusion = any(
            not np.isnan(f['perfusion_median']) for f in layer_features
        )
        if has_perfusion:
            features = ['t2star_median', 'depth_normalized', 'perfusion_median']
            print(f"  Perfusion data available — clustering on: {features}")

    # Step 2: Cluster
    cluster_info = cluster_layers(
        layer_features, n_clusters=n_clusters, features=features,
        method=method, random_state=random_state
    )

    # Step 3: Assign tissue labels
    tissue_assignments = assign_tissue_labels(
        layer_features, cluster_info['labels'], n_clusters
    )

    # Step 4: Build zone config
    zone_config = build_zone_config(tissue_assignments, n_layers, cluster_info)

    diagnostics = {
        'layer_features': layer_features,
        'cluster_info': cluster_info,
        'tissue_assignments': tissue_assignments,
    }

    return zone_config, diagnostics


def compare_zone_configs(clustered: Dict, reference: Dict) -> Dict:
    """
    Compare clustered zone boundaries against a reference config.

    Parameters
    ----------
    clustered : dict
        Zone config from clustering.
    reference : dict
        Reference zone config (e.g., from YAML).

    Returns
    -------
    dict with per-zone Jaccard overlap and boundary shift information.
    """
    comparison = {}

    ref_zones = reference.get('zones', {})
    clust_zones = clustered.get('zones', {})

    # Compare zones that exist in both configs
    all_zone_names = set(ref_zones.keys()) | set(clust_zones.keys())

    for zone_name in all_zone_names:
        ref_layers = set(ref_zones.get(zone_name, {}).get('layers', []))
        clust_layers = set(clust_zones.get(zone_name, {}).get('layers', []))

        if not ref_layers and not clust_layers:
            continue

        intersection = ref_layers & clust_layers
        union = ref_layers | clust_layers
        jaccard = len(intersection) / len(union) if union else 0.0

        # Boundary shift: difference in min/max layer
        ref_min = min(ref_layers) if ref_layers else None
        ref_max = max(ref_layers) if ref_layers else None
        clust_min = min(clust_layers) if clust_layers else None
        clust_max = max(clust_layers) if clust_layers else None

        boundary_shift = {}
        if ref_min is not None and clust_min is not None:
            boundary_shift['lower'] = clust_min - ref_min
        if ref_max is not None and clust_max is not None:
            boundary_shift['upper'] = clust_max - ref_max

        comparison[zone_name] = {
            'jaccard': round(jaccard, 3),
            'ref_layers': sorted(ref_layers) if ref_layers else [],
            'clustered_layers': sorted(clust_layers) if clust_layers else [],
            'boundary_shift': boundary_shift,
            'in_reference': zone_name in ref_zones,
            'in_clustered': zone_name in clust_zones,
        }

    return comparison


def plot_clustering_diagnostics(
    layer_features: List[Dict],
    labels: np.ndarray,
    assignments: Dict[str, List[int]],
    centroids: np.ndarray,
    feature_names: List[str],
    reference_config: Optional[Dict] = None,
    output_path: Optional[Path] = None
):
    """
    Generate 4-panel diagnostic figure for clustering results.

    Panels:
    1. T2* profile colored by cluster assignment
    2. Feature space scatter with centroids
    3. Clustered vs hardcoded zone boundaries (if reference provided)
    4. Silhouette-style layer-cluster confidence visualization

    Parameters
    ----------
    layer_features : list of dict
        Output from extract_layer_features().
    labels : np.ndarray
        Cluster labels per layer.
    assignments : dict
        Output from assign_tissue_labels().
    centroids : np.ndarray
        Cluster centroids in original feature space.
    feature_names : list of str
        Feature names used for clustering.
    reference_config : dict, optional
        Reference zone config for comparison panel.
    output_path : Path, optional
        If provided, saves figure to this path.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib is required for plotting.")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('K-Means Zone Clustering Diagnostics', fontsize=14, fontweight='bold')

    # Color palette for clusters
    n_clusters = len(assignments)
    cmap = plt.cm.get_cmap('tab10', max(n_clusters, 3))
    zone_colors = {name: cmap(i) for i, name in enumerate(assignments.keys())}

    layer_indices = [f['layer_idx'] for f in layer_features]
    t2_medians = [f['t2star_median'] for f in layer_features]

    # --- Panel 1: T2* profile colored by cluster ---
    ax1 = axes[0, 0]
    for zone_name, zone_layers in assignments.items():
        color = zone_colors[zone_name]
        for li in zone_layers:
            idx = li - 1  # 0-based index into features
            if idx < len(t2_medians) and not np.isnan(t2_medians[idx]):
                ax1.bar(li, t2_medians[idx], color=color, edgecolor='white', linewidth=0.5)
    ax1.set_xlabel('Layer (1=surface)')
    ax1.set_ylabel('Median T2* (ms)')
    ax1.set_title('T2* Profile by Cluster')
    # Legend
    from matplotlib.patches import Patch
    legend_handles = [Patch(facecolor=color, label=zone_name) for zone_name, color in zone_colors.items()]
    ax1.legend(handles=legend_handles, fontsize=8, loc='upper right')

    # --- Panel 2: Feature space with centroids ---
    ax2 = axes[0, 1]
    # Use first two features for 2D scatter
    feat_x_name = feature_names[0] if len(feature_names) > 0 else 't2star_median'
    feat_y_name = feature_names[1] if len(feature_names) > 1 else 'depth_normalized'

    for zone_name, zone_layers in assignments.items():
        color = zone_colors[zone_name]
        xs, ys = [], []
        for li in zone_layers:
            idx = li - 1
            if idx < len(layer_features):
                x_val = layer_features[idx].get(feat_x_name, np.nan)
                y_val = layer_features[idx].get(feat_y_name, np.nan)
                if not np.isnan(x_val) and not np.isnan(y_val):
                    xs.append(x_val)
                    ys.append(y_val)
        ax2.scatter(xs, ys, color=color, label=zone_name, s=50, edgecolors='black', linewidth=0.5)

    # Plot centroids
    if centroids is not None and len(centroids) > 0:
        for i, name in enumerate(assignments.keys()):
            if i < len(centroids):
                ax2.scatter(
                    centroids[i, 0], centroids[i, 1] if centroids.shape[1] > 1 else 0,
                    marker='X', s=200, color=zone_colors[name],
                    edgecolors='black', linewidth=1.5, zorder=5
                )
    ax2.set_xlabel(feat_x_name)
    ax2.set_ylabel(feat_y_name)
    ax2.set_title('Feature Space')
    ax2.legend(fontsize=8)

    # --- Panel 3: Clustered vs hardcoded boundaries ---
    ax3 = axes[1, 0]
    n_layers = len(layer_features)
    bar_height = 0.3

    # Clustered zones (top row)
    for zone_name, zone_layers in assignments.items():
        color = zone_colors[zone_name]
        for li in zone_layers:
            ax3.barh(1, 1, left=li - 0.5, height=bar_height, color=color, edgecolor='white', linewidth=0.5)

    # Reference zones (bottom row)
    if reference_config and 'zones' in reference_config:
        ref_colors = {
            'outer_cortex': '#E8F4F8', 'inner_cortex': '#C5E3ED',
            'cmj': '#FFE5CC', 'outer_medulla': '#FFD9B3', 'inner_medulla': '#FFC999',
            'cortex': '#C5E3ED', 'medulla': '#FFD9B3', 'papilla': '#FFC999',
        }
        for zone_name, zone_info in reference_config['zones'].items():
            color = ref_colors.get(zone_name, '#CCCCCC')
            for li in zone_info['layers']:
                ax3.barh(0, 1, left=li - 0.5, height=bar_height, color=color, edgecolor='white', linewidth=0.5)
        ax3.set_yticks([0, 1])
        ax3.set_yticklabels(['Reference', 'Clustered'])
    else:
        ax3.set_yticks([1])
        ax3.set_yticklabels(['Clustered'])

    ax3.set_xlabel('Layer')
    ax3.set_title('Zone Boundaries Comparison')
    ax3.set_xlim(0.5, n_layers + 0.5)

    # --- Panel 4: Per-layer cluster membership visualization ---
    ax4 = axes[1, 1]
    # Show layers as a heatmap-like strip with zone labels
    for i, feat in enumerate(layer_features):
        li = feat['layer_idx']
        label = labels[i]
        if label >= 0:
            color = cmap(label)
        else:
            color = '#CCCCCC'
        ax4.barh(0, 1, left=li - 0.5, height=0.5, color=color, edgecolor='white', linewidth=0.5)
        # Annotate with layer number
        if n_layers <= 30:
            ax4.text(li, 0, str(li), ha='center', va='center', fontsize=7)

    ax4.set_xlabel('Layer')
    ax4.set_title('Layer-to-Cluster Assignment')
    ax4.set_yticks([])
    ax4.set_xlim(0.5, n_layers + 0.5)

    # Add zone boundary lines
    for zone_name, zone_layers in assignments.items():
        if zone_layers:
            boundary = max(zone_layers) + 0.5
            if boundary < n_layers + 0.5:
                ax4.axvline(boundary, color='red', linestyle='--', alpha=0.7, linewidth=1)

    plt.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        fig.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
        print(f"  Saved clustering diagnostics: {output_path.name}")

        # Also save SVG
        svg_path = output_path.with_suffix('.svg')
        fig.savefig(svg_path, format='svg', bbox_inches='tight', facecolor='white')

    return fig


def plot_cluster_overlay(
    t2_map: np.ndarray,
    mlco_mask: np.ndarray,
    assignments: Dict[str, List[int]],
    n_layers: int = 24,
    output_path: Optional[Path] = None,
    title: str = 'Cluster Zone Overlay',
):
    """
    Overlay clustered zone colors on the T2* map to visualize spatial correspondence.

    Produces a 3-panel figure:
      1. T2* map with zone contours
      2. Zone assignment mask (colored by cluster)
      3. Semi-transparent zone overlay on T2* map

    Parameters
    ----------
    t2_map : np.ndarray
        2D T2* map (e.g., 200x200).
    mlco_mask : np.ndarray
        2D MLCO mask with integer layer labels. Bilateral masks use 1-24 (right)
        and 25-48 (left); both sides are mapped to layers 1-n_layers.
    assignments : dict
        Zone assignments from assign_tissue_labels(), e.g.
        {'zone_1': [1,2,...], 'zone_2': [...], 'zone_3': [...]}.
    n_layers : int
        Number of layers per organ (default 24).
    output_path : Path, optional
        If provided, saves figure (PNG + SVG).
    title : str
        Figure suptitle.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib is required for plotting.")
    from matplotlib.patches import Patch
    from matplotlib.colors import to_rgba

    # Build layer -> zone name mapping (both sides for bilateral masks)
    layer_to_zone = {}
    for zone_name, layers in assignments.items():
        for li in layers:
            layer_to_zone[li] = zone_name
            # Bilateral: left kidney layers are offset by n_layers
            layer_to_zone[li + n_layers] = zone_name

    # Color palette — matches the diagnostics plot
    n_clusters = len(assignments)
    cmap = plt.cm.get_cmap('tab10', max(n_clusters, 3))
    zone_colors = {name: cmap(i) for i, name in enumerate(assignments.keys())}

    # Build zone mask image (RGBA)
    h, w = mlco_mask.shape
    zone_rgba = np.zeros((h, w, 4), dtype=np.float32)
    zone_idx_map = np.full((h, w), -1, dtype=int)  # for contour boundaries

    for y in range(h):
        for x in range(w):
            lv = int(mlco_mask[y, x])
            if lv > 0 and lv in layer_to_zone:
                zone_name = layer_to_zone[lv]
                rgba = to_rgba(zone_colors[zone_name])
                zone_rgba[y, x] = rgba
                zone_idx_map[y, x] = list(assignments.keys()).index(zone_name)

    # Prepare T2* display: mask non-kidney areas, clip for display
    t2_display = np.where(mlco_mask > 0, t2_map, np.nan)
    valid_vals = t2_display[~np.isnan(t2_display)]
    vmin = np.nanpercentile(valid_vals, 2)
    vmax = np.nanpercentile(valid_vals, 98)

    # Crop to kidney bounding box with padding for cleaner view
    ys, xs = np.where(mlco_mask > 0)
    pad = 8
    r0, r1 = max(0, ys.min() - pad), min(h, ys.max() + pad + 1)
    c0, c1 = max(0, xs.min() - pad), min(w, xs.max() + pad + 1)

    t2_crop = t2_display[r0:r1, c0:c1]
    zone_rgba_crop = zone_rgba[r0:r1, c0:c1]
    zone_idx_crop = zone_idx_map[r0:r1, c0:c1]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(title, fontsize=14, fontweight='bold')

    # --- Panel 1: T2* map with zone contours ---
    ax1 = axes[0]
    im1 = ax1.imshow(t2_crop, cmap='inferno', vmin=vmin, vmax=vmax, interpolation='nearest')
    # Draw zone boundaries as contours
    zone_names = list(assignments.keys())
    for zi in range(n_clusters):
        binary = (zone_idx_crop == zi).astype(float)
        if binary.max() > 0:
            ax1.contour(binary, levels=[0.5], colors=[zone_colors[zone_names[zi]]],
                        linewidths=2.0)
    fig.colorbar(im1, ax=ax1, label='T2* (ms)', shrink=0.8)
    ax1.set_title('T2* Map + Zone Contours')
    ax1.axis('off')

    # --- Panel 2: Zone assignment mask ---
    ax2 = axes[1]
    bg = np.ones_like(zone_rgba_crop[..., :3]) * 0.15
    ax2.imshow(bg)
    ax2.imshow(zone_rgba_crop, interpolation='nearest')
    ax2.set_title('Zone Assignment Mask')
    ax2.axis('off')

    # --- Panel 3: Semi-transparent overlay on T2* ---
    ax3 = axes[2]
    # Use inferno colormap (same as panel 1) for better contrast under overlay
    ax3.imshow(t2_crop, cmap='inferno', vmin=vmin, vmax=vmax, interpolation='nearest')
    overlay = zone_rgba_crop.copy()
    overlay[..., 3] = np.where(zone_idx_crop >= 0, 0.4, 0.0)
    ax3.imshow(overlay, interpolation='nearest')
    ax3.set_title('T2* + Zone Overlay')
    ax3.axis('off')

    # Shared legend
    legend_handles = [Patch(facecolor=zone_colors[name], edgecolor='black', label=name)
                      for name in assignments.keys()]
    fig.legend(handles=legend_handles, loc='lower center', ncol=n_clusters,
               fontsize=11, frameon=True, fancybox=True, shadow=True,
               bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])

    if output_path is not None:
        output_path = Path(output_path)
        fig.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
        print(f"  Saved cluster overlay: {output_path.name}")
        svg_path = output_path.with_suffix('.svg')
        fig.savefig(svg_path, format='svg', bbox_inches='tight', facecolor='white')

    return fig
