# Scripts Reference

Complete reference for all BoldPy scripts.

---

## Core Pipeline Scripts

### `prepare_data.py`
**Purpose:** Extract and fit T2*/R2* maps from Bruker PvDatasets with intelligent frame detection

**Usage:**
```bash
# Automatic T2* frame detection (tiered: metadata → heuristic)
python prepare_data.py --input scan.PvDatasets --output-dir prepared/ --both-t2star

# Manual T2* frame specification
python prepare_data.py --input scan.PvDatasets --output-dir prepared/ --both-t2star --t2-frame 3
```

**Key Options:**
- `--input`: Single PvDatasets file
- `--input-dir`: Batch process directory
- `--output-dir`: Where to save results
- `--sample-name`: Custom output naming
- `--both-t2star`: Extract Bruker + fit custom T2*
- `--custom-t2star`: Only custom fitting
- `--no-bruker`: Skip Bruker extraction
- `--t2-frame N`: Manually specify T2* frame (1-indexed) **(NEW in v2.2.1)**
- `--extract-perfusion`: Extract perfusion from pdata/2
- `--3d`: Full 3D output
- `--slice N`: Specific slice
- `--pattern`: File pattern for batch mode

**T2* Frame Detection (NEW in v2.2.1):**

Tiered approach for robust identification:
1. **Tier 1 (Metadata):** Reads `VisuCoreFrameType` from Bruker visu_pars
2. **Tier 2 (Heuristic):** Multi-factor scoring (100-point system)
   - Mean value (40 pts): 10-30 ms optimal
   - Max value (20 pts): <100 ms optimal
   - visu_pars range (15 pts): 0-200 ms
   - Distribution (15 pts): reasonable spread
   - Coverage (10 pts): >10% non-zero
3. **Tier 3 (Manual):** User override via `--t2-frame`

**Confidence Levels:**
- HIGH (70-100 pts): Automatic use
- MEDIUM (50-69 pts): Warning issued
- LOW (<50 pts): Verification recommended

**Example Output:**
```
Tier 1: Checking Bruker metadata...
  ⚠️  Metadata uninformative (frames labeled REAL_IMAGE)
Tier 2: Using enhanced scoring heuristic...
     Frame scores:
      ★ Frame 3: 95/100 pts
           Mean 12.5 ms: 40/40 pts
           Max 89.1 ms: 20/20 pts
  ✓ Selected frame 3 (confidence: HIGH)
```

**Outputs:**
- Reference image (.npy)
- T2* maps (.npy) - Bruker and/or custom fitted
- R2* maps (.npy)
- Perfusion map (.npy, if requested)
- Metadata (.json)

---

### `roi_drawer.py`
**Purpose:** Interactive ROI mask creation

**Usage:**
```bash
python roi_drawer.py --image reference.npy --output roi_mask.npy
```

**Controls:**
- Click: Add point
- Right-click: Complete polygon
- z: Undo
- c: Clear
- Mouse wheel: Zoom
- Drag: Pan
- s: Save

**Output:** Binary mask (.npy)

---

### `generate_mlco.py`
**Purpose:** Generate Multi-Layer Concentric Object masks

**Usage:**
```bash
# Bilateral
python generate_mlco.py --mask roi.npy --split --anatomical ref.npy --n-layers 24

# Single
python generate_mlco.py --mask roi.npy --anatomical ref.npy --n-layers 24
```

**Key Options:**
- `--mask`: Input ROI mask
- `--split`: Split into bilateral components
- `--anatomical`: Reference image for visualization
- `--n-layers`: Layers per organ (default: 24)
- `--component-names`: Custom names (default: left right)
- `--no-mri-flip`: Disable MRI convention
- `--output-dir`: Save location
- `--label`: Output filename prefix

**Outputs:**
- MLCO layer mask (.npy)
- Visualization (.png)

---

### `boldpy_analyze.py`
**Purpose:** Complete MLCO analysis with metrics

**Usage:**
```bash
# Single sample
python boldpy_analyze.py --config sample.json --n-layers 24 --output-dir results/

# Group comparison
python boldpy_analyze.py --group1-config wt.json --group2-config ko.json --compare --n-layers 24 --output-dir results/
```

**Key Options:**
- `--config`: Single sample config
- `--group1-config`, `--group2-config`: Group comparison configs
- `--compare`: Perform statistical comparison
- `--n-layers`: Must match MLCO generation
- `--output-dir`: Results directory

**Outputs:**
- Complete analysis JSON
- Layer profile plots
- Perfusion plots (if available)
- Scatter plots (T2* vs perfusion)
- Triple overlay maps
- Comparison plots (if comparing groups)

---

### `boldpy_plots.py`
**Purpose:** Plotting functions (imported by boldpy_analyze.py)

**Functions:**
- `plot_mlco_profile()`: Layer-by-layer profiles
- `plot_perfusion_profile()`: Perfusion across layers
- `plot_t2star_perfusion_scatter()`: T2* vs perfusion
- `plot_triple_overlay()`: T2*/R2*/perfusion maps
- `plot_mlco_comparison()`: Group comparison plots

---

## Project Analysis Scripts

These scripts operate at the **project level** — they read pre-computed per-sample analysis JSONs produced by `boldpy_analyze.py` and generate cross-group figures and statistics. They are all configured via a shared `groups_config.json` file (see `examples/groups_config.json` for a template).

**groups_config.json format:**
```json
{
  "output_dir": "processed/analysis/my_experiment",
  "hematology_csv": "data/hematology.csv",
  "groups": {
    "Control (n=2)": {
      "ids":    ["sample_ctrl_1", "sample_ctrl_2"],
      "color":  "#E74C3C",
      "ls":     "--",
      "lw":     1.8,
      "zorder": 4,
      "label":  "Control",
      "short":  "Ctrl"
    },
    "Treatment (n=3)": {
      "ids":    ["sample_trt_1", "sample_trt_2", "sample_trt_3"],
      "color":  "#2E86C1",
      "ls":     "-",
      "lw":     2.0,
      "zorder": 3,
      "label":  "Treatment",
      "short":  "Trt"
    }
  }
}
```

All three scripts can also be used programmatically by patching their `GROUPS` and `OUTPUT_DIR` module globals directly (see `examples/` for driver script examples).

---

### `group_analysis.py`
**Purpose:** Group-level MLCO profile comparison — the project-level complement to `boldpy_analyze.py`

**Usage:**
```bash
python group_analysis.py --config groups_config.json
```

**What it does:**
- Reads per-sample `{sample_id}_complete_analysis.json` files from `processed/analysis/{id}/`
- Computes mean ± SEM per MLCO layer across all samples in each group
- Generates T2*, R2*, and perfusion comparison profiles with zone shading
- Runs Mann-Whitney U tests between groups at each layer
- Produces optional hematology (HCT) comparison figure if `hematology_csv` is provided
- Exports zone-level summary statistics as JSON

**Key Options:**
- `--config PATH`: Path to groups_config.json (required)

**Outputs** (all in `{output_dir}/`):
- `{cond}_t2star_profile.png/.svg` — T2* layer profiles, one per condition
- `{cond}_r2star_profile.png/.svg` — R2* layer profiles
- `{cond}_perfusion_profile.png/.svg` — Perfusion layer profiles
- `hematology_comparison.png` — HCT figure (if hematology_csv provided)
- `zone_summary_stats.json` — Zone-level mean ± SEM + p-values per group

---

### `overlay_analysis.py`
**Purpose:** K-means zone overlays, MLCO layer overlays, and zone statistics — consolidated in one script

**Usage:**
```bash
python overlay_analysis.py --config groups_config.json
```

**What it does** (three sequential steps):

1. **Per-sample overlay figures** — For each sample × condition, produces:
   - A 3-panel k-means figure: T2* map | k-means zone map | transparent overlay
   - A 3-panel MLCO figure: T2* map | color-coded MLCO layers | transparent overlay
   K-means is computed once per sample and reused for both figure types.

2. **Grid figures** — Cross-sample comparison grids showing all samples side-by-side per condition:
   - `kmeans_overlay_grid_{cond}.png` — All samples' k-means overlays in a grid (rows = groups, cols = samples)
   - `mlco_overlay_grid_{cond}.png` — All samples' MLCO layer overlays in the same layout

3. **Zone analysis summary** — A dot-plot figure comparing k-means zone statistics (T2* mean, T2* std, perfusion) between groups, with Mann-Whitney p-values.

**Key Options:**
- `--config PATH`: Path to groups_config.json (required)

**Outputs**:
- `{analysis_dir}/{sid}/kmeans/{sid}_kmeans_{cond}.png/.svg` — Per-sample k-means overlays
- `{analysis_dir}/{sid}/mlco/{sid}_mlco_{cond}.png/.svg` — Per-sample MLCO overlays
- `{output_dir}/kmeans_overlay_grid_{cond}.png/.svg` — K-means grid per condition
- `{output_dir}/mlco_overlay_grid_{cond}.png/.svg` — MLCO grid per condition
- `{output_dir}/kmeans_zone_analysis.png/.pdf/.svg` — Zone statistics dot-plot

---

### `heterogeneity.py`
**Purpose:** Within-layer heterogeneity profiling and focal disruption analysis

**Usage:**
```bash
python heterogeneity.py --config groups_config.json
```

**What it does** (two sequential parts):

**Part 1 — Heterogeneity Profiles:**
- Computes per-layer T2* std, CV (coefficient of variation), IQR, and skewness for every sample and condition
- Builds group-level mean ± SEM profiles across MLCO layers
- Generates overview figures (T2* std and CV vs. layer with zone shading)
- Generates a "talk summary" panel comparing air vs. oxygen conditions
- Generates outer cortex bar charts with individual data points and statistical annotations

**Part 2 — Focal Disruption:**
- Extracts outer cortex pixel-level T2* distributions per sample per condition
- Performs Mann-Whitney U test on the pooled pixel distributions
- Generates strip plots and KDE distribution plots of pixel-level T2* values
- Computes spatial local coefficient-of-variation (CV) maps (5×5 neighborhood)
- Generates spatial CV map panels showing per-sample maps and group difference maps

**Key Options:**
- `--config PATH`: Path to groups_config.json (required)

**Outputs** (all in `{output_dir}/heterogeneity/`):
- `heterogeneity_overview.png/.svg` — T2* std and CV layer profiles per group
- `talk_summary.png/.svg` — Air vs. oxygen summary panel
- `outer_cortex_bars.png/.svg` — OC T2* std bar chart with data points
- `strip_plots.png/.svg` — Per-pixel OC T2* strip plots
- `kde_distributions.png/.svg` — KDE distributions of OC pixel T2*
- `spatial_cv_maps.png/.svg` — Spatial local-CV maps per sample and group difference
- `focal_disruption_stats.json` — All statistics and p-values

---

## Supporting Scripts

### `fit_t2star.py`
**Purpose:** Standalone T2* fitting functions

**Functions:**
- `fit_single_voxel()`: Fit one pixel
- `fit_t2star_map()`: Fit entire map
- Used by prepare_data.py

---

### `mlco_analysis.py`
**Purpose:** Core MLCO analysis logic

**Functions:**
- `analyze_mlco()`: Main analysis function
- Layer statistics extraction
- Zone aggregation
- Used by boldpy_analyze.py

---

### `tissue_zones.py`
**Purpose:** Tissue quality and zone definitions

**Key Components:**
- `TISSUE_THRESHOLDS`: Viability thresholds
- `ZONE_DEFINITIONS`: 5-zone kidney anatomy
- `classify_tissue_viability()`: Per-pixel classification
- `calculate_tissue_quality()`: Region statistics

---

### `roi_format_utils.py`
**Purpose:** ROI format conversion utilities

Used by roi_drawer.py for mask handling.

