# Hybrid MLCO + K-Means Clustering for Data-Driven Zone Boundaries

## Overview

BoldPy v2.3.0 introduced **data-driven zone boundary detection** via k-means clustering on per-layer T2*/R2*/perfusion statistics. This replaces the need for hardcoded layer-to-zone mappings (e.g., "layers 1-5 = outer_cortex") with boundaries that adapt to each subject's anatomy.

**Motivation:** Fixed zone boundaries don't account for inter-subject anatomical variability. In pathological models (e.g., Ren-KO mice with arterial hypertrophy and medullary degradation), the tissue boundaries can shift substantially from the standard reference. K-means clustering on the MLCO layer statistics finds natural tissue groupings directly from the data.

**Approach:** Cluster the 24 per-layer median T2* values (not individual voxels, which would be too noisy) to find natural tissue groupings. Map clusters to tissue names by spatial depth ordering (shallowest = cortex, deepest = papilla/inner_medulla). Output a zone config dict in the same format as existing YAML configs, so all downstream code works unchanged.

**Reference:** Inspired by Menzies et al. (2013) — data-driven segmentation of renal tissue zones.

---

## Quick Start

### Command-Line Usage

```bash
# Basic: 3-zone clustering (cortex / medulla / papilla)
python boldpy_analyze.py \
    --config M1_config_bruker_single-region.json \
    --output-dir results/M1_clustered/ \
    --cluster-zones

# 5-zone clustering matching the standard zone scheme
python boldpy_analyze.py \
    --config M1_config_bruker_single-region.json \
    --output-dir results/M1_clustered/ \
    --cluster-zones --n-clusters 5

# Cluster on a specific condition, save config for reuse
python boldpy_analyze.py \
    --config M1_config_bruker_single-region.json \
    --output-dir results/M1_clustered/ \
    --cluster-zones --n-clusters 3 \
    --cluster-condition air \
    --save-cluster-config configs/zones/M1_clustered_k3.yaml

# Reuse a saved clustered config (no re-clustering needed)
python boldpy_analyze.py \
    --config M1_config_bruker_single-region.json \
    --output-dir results/M1_standard/ \
    --zone-config configs/zones/M1_clustered_k3.yaml
```

### Python API

```python
import numpy as np
from cluster_zones import cluster_and_build_zones, compare_zone_configs, plot_clustering_diagnostics
from tissue_zones import update_configs_from_dict, ZONE_CONFIG

# Load your data
t2_map = np.load('M1_air_t2star_bruker.npy')
r2_map = np.load('M1_air_r2star_bruker.npy')
mlco_mask = np.load('M1_mlco_bilateral.npy')
perfusion = np.load('M1_perfusion.npy')  # 80x80 — auto-upsampled to 200x200

# Run clustering (perfusion auto-included when provided)
zone_config, diagnostics = cluster_and_build_zones(
    t2_map=t2_map,
    r2_map=r2_map,
    mlco_mask=mlco_mask,
    n_layers=24,
    n_clusters=3,          # 3 or 5
    perfusion_map=perfusion # optional, auto-upsampled if shape differs
)

# Inspect results
print(f"Silhouette score: {zone_config['metadata']['silhouette_score']:.3f}")
for name, info in zone_config['zones'].items():
    print(f"  {name}: layers {info['layers']}")

# Compare against reference (hardcoded YAML)
comparison = compare_zone_configs(zone_config, ZONE_CONFIG)

# Inject into module globals — all downstream code now uses clustered zones
update_configs_from_dict(zone_config)

# Generate diagnostic figure
plot_clustering_diagnostics(
    layer_features=diagnostics['layer_features'],
    labels=diagnostics['cluster_info']['labels'],
    assignments=diagnostics['tissue_assignments'],
    centroids=diagnostics['cluster_info']['centroids'],
    feature_names=diagnostics['cluster_info']['feature_names'],
    reference_config=ZONE_CONFIG,
    output_path='cluster_diagnostics.png'
)
```

---

## CLI Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--cluster-zones` | flag | off | Enable k-means clustering for zone boundaries |
| `--n-clusters` | int | 3 | Number of tissue clusters |
| `--cluster-method` | str | `kmeans` | Clustering method: `kmeans` or `gmm` (Gaussian Mixture) |
| `--cluster-condition` | str | first | Which condition to cluster on (e.g., `air`, `oxygen_1`) |
| `--save-cluster-config` | path | none | Save clustered zone config as YAML for reuse |

---

## Zone Name Mapping

Clusters are sorted by mean normalized depth (surface to center) and assigned standard anatomical names:

| k | Zone Names (surface to center) |
|---|--------------------------------|
| 3 | `cortex`, `medulla`, `papilla` |
| 5 | `outer_cortex`, `inner_cortex`, `cmj`, `outer_medulla`, `inner_medulla` |
| other | `zone_1`, `zone_2`, ..., `zone_k` |

When k=5, zone names match the standard YAML config (`kidney_24layer.yaml`), enabling direct Jaccard overlap comparison.

---

## Clustering Features

The clustering operates on per-layer feature vectors extracted from the MLCO analysis. By default:

| Feature | Description | Always used |
|---------|-------------|-------------|
| `t2star_median` | Median T2* value across all pixels in the layer | Yes |
| `depth_normalized` | Layer position normalized to [0, 1] (surface to center) | Yes |
| `perfusion_median` | Median perfusion value (auto-included when perfusion data provided) | When available |

Additional features extracted but not used for clustering by default (available for custom feature sets):

- `t2star_std` — T2* standard deviation per layer
- `r2star_median` — Median R2* per layer
- `r2star_std` — R2* standard deviation per layer
- `n_pixels` — Number of valid pixels per layer

### Perfusion Upsampling

Perfusion data is typically acquired at lower resolution (80x80) than T2* maps (200x200). When `perfusion_map` is provided to `cluster_and_build_zones()` or `extract_layer_features()`, it is **automatically upsampled** to match the T2* resolution using bilinear interpolation (`scipy.ndimage.zoom`, order=1). NaN regions are preserved via nearest-neighbor interpolation of the NaN mask.

---

## Diagnostic Plot

The 4-panel diagnostic figure (`plot_clustering_diagnostics()`) provides:

1. **T2* Profile by Cluster** (top-left) — Bar chart of per-layer median T2* colored by cluster assignment. Shows the T2* gradient from surface to center.

2. **Feature Space** (top-right) — 2D scatter plot of the first two clustering features (typically T2* median vs. normalized depth) with cluster centroids marked as X. Reveals cluster separation in feature space.

3. **Zone Boundaries Comparison** (bottom-left) — Side-by-side strip visualization of clustered zones vs. reference (hardcoded YAML) zones. Immediately shows where boundaries shifted.

4. **Layer-to-Cluster Assignment** (bottom-right) — Layer-by-layer strip with cluster colors and boundary lines. Layer numbers annotated for quick reference.

---

## Pilot Data Results

Tested on M1 (wild-type) and M2 (Ren-KO) mouse kidney data.

### M1 (Wild-Type) — Air Condition

Clean monotonic T2* gradient from cortex (~10 ms) to papilla (~17 ms).

**k=3 (silhouette = 0.489):**

| Zone | Layers | Description |
|------|--------|-------------|
| cortex | 1-12 | Low T2*, high R2* |
| medulla | 13-20 | Intermediate values |
| papilla | 21-24 | High T2*, low R2* |

**k=5 (silhouette = 0.447):**

| Zone | Clustered Layers | Reference Layers | Jaccard |
|------|-----------------|-----------------|---------|
| outer_cortex | 3 | 1-5 | 0.20 |
| inner_cortex | 5, 6, 7, 9 | 6-10 | 0.50 |
| cmj | 8, 10, 11, 12, 13 | 11-13 | 0.60 |
| outer_medulla | 14-20 | 14-19 | 0.86 |
| inner_medulla | 21-24 | 20-24 | 0.80 |

Outer cortex Jaccard is low because layers 1-2 have no MLCO pixels (too thin at 200x200 resolution) — this is expected.

### M2 (Ren-KO) — Air Condition

Disrupted anatomy from concentric arterial/arteriolar hypertrophy with fluid-filled medullary degradation.

**k=3 with perfusion (silhouette = 0.501):**

| Zone | Layers | Key Features |
|------|--------|-------------|
| cortex | 3-14, 16 | T2* 17-27 ms, perfusion 815-1370 |
| medulla | 15, 17 | T2* 35-41 ms, perfusion 0-543 (fluid-filled!) |
| papilla | 18-24 | T2* 16-26 ms, perfusion 1310-2014 |

The clustering correctly isolates layers 15 and 17 — these show the pathological signature of **very high T2* combined with near-zero perfusion**, characteristic of fluid-filled cystic spaces in the degraded medulla from the Ren-KO phenotype. This would be invisible with hardcoded zone boundaries.

### Impact of Perfusion Data

Adding upsampled perfusion as a third clustering feature:

- **M1:** Tightens the cortex/medulla boundary (perfusion jump at layers 7-8 provides additional separating signal beyond T2* alone)
- **M2:** Critically important — perfusion reveals the pathological layers (15, 17: near-zero perfusion) that would otherwise be mixed with surrounding tissue based on T2* alone

---

## Group Comparisons with Clustered Zones

Per-sample k-means clustering assigns each sample its own zone boundaries. This means "zone_1" in M1 may cover layers 1-12 while in M2 it covers layers 1-8. Direct zone-level statistical comparison (e.g., "is cortex T2* higher in KO?") requires matching boundaries. BoldPy supports two workflows:

### Workflow A — Shared Reference Boundaries

**When to use:** Comparing zone-level statistics (T2*, R2*, perfusion) between groups.

**How it works:** Cluster one sample (or use a reference config), then apply those fixed boundaries to all samples. Zone stats are directly comparable because every sample uses the same layer-to-zone mapping.

```bash
# Step 1: Cluster a reference sample, save the config
python boldpy_analyze.py \
    --config M1_config_bruker_single-region.json \
    --output-dir results/M1_ref/ \
    --cluster-zones --n-clusters 3 \
    --save-cluster-config configs/zones/M1_reference_k3.yaml

# Step 2: Run both groups with the shared reference
python boldpy_analyze.py \
    --group1-config M1_config.json \
    --group2-config M2_config.json \
    --compare \
    --cluster-reference configs/zones/M1_reference_k3.yaml \
    --output-dir results/comparison_shared/
```

| Pros | Cons |
|------|------|
| Zone stats directly comparable | Reference boundaries may not fit pathological samples |
| Standard statistical testing applies | Choice of reference sample matters |
| Familiar zone-level analysis | May mask boundary-shift biology |

### Workflow B — Per-Sample Boundary Characterization

**When to use:** The boundary locations themselves are the scientific question (e.g., "does KO shift the cortex-medulla boundary deeper?").

**How it works:** Each sample gets its own k-means clustering. The resulting boundary positions, Jaccard overlap, and layer shifts are compared as biomarkers.

```bash
# Each sample clustered independently
python boldpy_analyze.py \
    --group1-config M1_config.json \
    --group2-config M2_config.json \
    --compare \
    --cluster-zones --n-clusters 3 \
    --output-dir results/comparison_persample/
```

The comparison output will include:
- Per-zone **Jaccard overlap** between group boundaries
- **Boundary shift** in layers (lower and upper edges)
- **Silhouette score** comparison (cluster quality)

Zone-level stats (T2*, perfusion per zone) are **not** compared because the zones cover different layers.

| Pros | Cons |
|------|------|
| Each sample gets optimal boundaries | Cannot directly compare zone-level T2*/perfusion stats |
| Boundary shift is itself a biomarker | Requires careful interpretation |
| Reveals pathology-driven zone changes | More complex output |

### Decision Table

| Question | Workflow |
|----------|----------|
| "Is cortex T2* higher in KO vs WT?" | **A** (shared reference) |
| "Does KO shift the cortex-medulla boundary?" | **B** (per-sample) |
| "Is overall tissue organization disrupted?" | **B** (per-sample) |
| "What is the mean medullary perfusion in each group?" | **A** (shared reference) |
| "Do pathological samples have different zone sizes?" | **B** (per-sample) |

### Combined Approach (Recommended)

For comprehensive analysis, run **both** workflows:

1. **Workflow A** for zone-level T2*/perfusion statistics (using a WT reference)
2. **Workflow B** for boundary characterization (are zones shifting?)

This gives you both "what's happening inside the zones" (A) and "are the zones themselves changing" (B).

### CLI Flags

| Flag | Workflow | Effect |
|------|----------|--------|
| `--cluster-zones` | B | Per-sample k-means clustering |
| `--cluster-reference PATH` | A | Apply saved YAML to all samples |
| `--save-cluster-config PATH` | (setup) | Save clustered config for reuse |

`--cluster-zones` and `--cluster-reference` are **mutually exclusive**.

### Output Differences

**Workflow A** comparison JSON includes:
- `boundaries_match: true`
- Standard `superficial_zone_comparison` with effect sizes
- Zone-level stats table in comparison plots

**Workflow B** comparison JSON includes:
- `boundaries_match: false`
- `boundary_comparison` with per-zone Jaccard and shifts
- Boundary comparison table in comparison plots (replaces zone stats)
- Dashed reference boundary lines overlaid on profile plots

---

## Architecture

### New File

**`cluster_zones.py`** — Self-contained clustering module (~460 lines)

| Function | Purpose |
|----------|---------|
| `extract_layer_features()` | Per-layer feature vectors from T2*/R2* maps + MLCO mask |
| `cluster_layers()` | StandardScaler + KMeans/GMM with silhouette scoring |
| `assign_tissue_labels()` | Maps clusters to anatomical names by depth ordering |
| `build_zone_config()` | Produces zone config dict compatible with `validate_zone_config()` |
| `cluster_and_build_zones()` | **Public API** — chains all steps, returns `(zone_config, diagnostics)` |
| `compare_zone_configs()` | Per-zone Jaccard overlap and boundary shift vs reference |
| `plot_clustering_diagnostics()` | 4-panel diagnostic figure |
| `_upsample_perfusion()` | Bilinear upsampling of perfusion maps to match T2* resolution |

### Modified Files

**`tissue_zones.py`** — Added `update_configs_from_dict(zone_config_dict)`
- Injects a programmatically-generated zone config into module globals (`ZONE_CONFIG`, `ZONE_DEFINITIONS`, `AGGREGATE_ZONES`)
- Validates the config before injection
- All downstream code that reads these globals immediately sees the new zones

**`boldpy_analyze.py`** — CLI flags + clustering integration
- Added `--cluster-zones`, `--n-clusters`, `--cluster-method`, `--cluster-condition`, `--save-cluster-config` arguments
- Added `cluster_args` parameter to `analyze_sample()` with pre-analysis clustering hook
- Fixed `calculate_oxygen_responsiveness()` — now uses `AGGREGATE_ZONES` dynamically instead of hardcoded `range(1,11)` / `range(14,25)`
- Fixed `extract_cortex_only_statistics()` — resolves cortex layers from `AGGREGATE_ZONES['total_cortex']` or first zone in config

**`boldpy_plots.py`** — Dynamic zone color support
- Added `get_zone_color(zone_name, zone_index, total_zones)` with gradient fallback for non-standard zone names
- Updated `add_zone_shading()` to use `get_zone_color()` instead of `ZONE_COLORS.get()`
- Fixed `plot_perfusion_profile()` zone summary (line 215) — iterates actual zone keys instead of hardcoded 5-name list
- Fixed `plot_mlco_comparison()` zone comparison (line 754) — iterates actual zone keys with dynamic header

**`mlco_analysis.py`** — Zone config passthrough
- Added `zone_config=None` parameter to `_analyze_mlco_single_organ()` and `_analyze_mlco_bilateral()`
- Changed `get_zone_name(layer_idx)` to `get_zone_name(layer_idx, zone_config=zone_config)` at both call sites
- Threaded `zone_config` through `analyze_mlco()` call chain
- Dynamic zone count in print label (`{n}-ZONE REGIONAL ANALYSIS`)

### Data Flow

```
cluster_and_build_zones()
    |
    ├── extract_layer_features()     # Per-layer stats from maps + mask
    |       |                        # Auto-upsamples perfusion if needed
    |       └── _upsample_perfusion()
    |
    ├── cluster_layers()             # StandardScaler + KMeans/GMM
    |
    ├── assign_tissue_labels()       # Depth-ordered cluster → name mapping
    |       |                        # Empty layers → nearest cluster
    |
    └── build_zone_config()          # Zone config dict (same as YAML format)
            |
            └── update_configs_from_dict()   # Inject into tissue_zones globals
                    |
                    ├── ZONE_CONFIG          # Used by mlco_analysis
                    ├── ZONE_DEFINITIONS     # Used by boldpy_plots
                    └── AGGREGATE_ZONES      # Used by boldpy_analyze
```

---

## Quality Metrics

### Silhouette Score

The silhouette score (-1 to +1) measures cluster separation quality:

| Score | Interpretation |
|-------|---------------|
| > 0.5 | Good separation — distinct tissue zones |
| 0.3 - 0.5 | Reasonable — zones overlap somewhat (common in biology) |
| < 0.3 | Weak — consider fewer clusters or different features |

Typical results on mouse kidney data: **0.43 - 0.58** depending on condition and cluster count.

### Jaccard Overlap

Measures agreement between clustered and reference zone boundaries (0 = no overlap, 1 = identical):

- **Outer/inner medulla**: typically 0.7-1.0 (high agreement — strong signal contrast)
- **CMJ**: typically 0.5-0.6 (moderate — transition zone inherently ambiguous)
- **Outer cortex**: often low due to empty outermost layers in the MLCO mask

---

## Handling Edge Cases

### Empty MLCO Layers

Layers with zero valid pixels (common for layers 1-2 at the outermost surface) are:
1. Excluded from clustering (assigned label = -1)
2. Assigned to the nearest cluster by normalized depth in `assign_tissue_labels()`
3. This ensures all layers 1-24 are covered, passing `validate_zone_config()`

### Perfusion Resolution Mismatch

When perfusion maps (80x80) don't match T2* resolution (200x200):
- Automatically upsampled via bilinear interpolation (`scipy.ndimage.zoom`, order=1)
- NaN masks preserved via nearest-neighbor upsampling
- No manual intervention required

### Fallback Behavior

If clustering fails (missing sklearn, insufficient data, etc.):
- Warning printed to console
- Analysis proceeds with the default YAML zone config
- No crash, no data loss

---

## Dependencies

Core clustering requires `scikit-learn` (already in the `[dev]` extras):

```bash
pip install scikit-learn
```

Required for clustering: `sklearn.cluster.KMeans`, `sklearn.mixture.GaussianMixture`, `sklearn.preprocessing.StandardScaler`, `sklearn.metrics.silhouette_score`

Perfusion upsampling requires `scipy.ndimage.zoom` (already a core dependency).
