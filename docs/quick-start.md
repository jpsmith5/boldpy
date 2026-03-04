# Quick Start Guide

Get started with BoldPy's MLCO (Multi-Layer Concentric Object) analysis in 15 minutes.

---

## Prerequisites

- Python 3.8 or higher
- Bruker PvDatasets files (`.PvDatasets` format)
- Reference anatomical image for ROI drawing

---

## Installation

```bash
git clone https://github.com/yourusername/boldpy.git
cd boldpy
pip install -e .
```

Verify installation:
```bash
python -c "from boldpy.fitting.t2star_fitter import fit_t2star_map; print('✓ BoldPy installed!')"
```

---

## Complete Workflow: ROI to Results

### Step 0: Prepare Data

Extract reference image and maps from PvDatasets:

```bash
# For BOLD/T2* scans
python prepare_data.py \
    --input M1_air.PvDatasets \
    --output-dir M1_prepared/ \
    --sample-name M1_air \
    --both-t2star

# For perfusion scans
python prepare_data.py \
    --input M1_perfusion.PvDatasets \
    --output-dir M1_prepared/ \
    --sample-name M1_perfusion \
    --extract-perfusion

# For both (if combined scan)
python prepare_data.py \
    --input M1_scan.PvDatasets \
    --output-dir M1_prepared/ \
    --sample-name M1 \
    --both-t2star \
    --extract-perfusion
```

**Outputs:**
- `M1_air_reference.npy` - For ROI drawing
- `M1_air_t2star_bruker.npy` - Bruker fitted T2* (if pdata/2 exists)
- `M1_air_r2star_bruker.npy` - R2* map from Bruker T2*
- `M1_air_t2star_custom.npy` - Custom fitted T2* (if --custom-t2star or --both-t2star)
- `M1_air_r2star_custom.npy` - R2* map from custom T2*
- `M1_perfusion_perfusion.npy` - Perfusion map (if --extract-perfusion)
- `M1_air_metadata.json` - Scan info

**Options:**
```bash
# Just reference (minimal, fastest)
--no-bruker

# Custom T2* fitting (recommended for best quality)
--no-bruker --custom-t2star

# Both Bruker and custom (for comparison)
--both-t2star

# Extract perfusion (ASL/FAIR-EPI scans)
--extract-perfusion

# 3D output (full volume)
--3d

# Specific slice
--slice 10
```

**Notes:** 
- Bruker T2* extraction requires pdata/2 in your .PvDatasets file. If pdata/2 doesn't exist, use `--custom-t2star` instead.
- Perfusion extraction loads from pdata/2, Frame 5 (Bruker ASL/FAIR-EPI processing).
- Perfusion will be automatically resampled to match T2* resolution if needed.

### Step 1: Draw ROI

Create ROI using the reference image from Step 0:

```bash
python roi_drawer.py \
    --image M1_prepared/M1_air_reference.npy \
    --output M1_kidney_roi.npy
```

**Interactive controls:**
- **Click** to add points to polygon
- **Space** to close polygon
- **U** to undo last action
- **R** to redo
- **C** to clear and start over
- **+/-** to zoom in/out
- **Arrow keys** to pan when zoomed
- **S** to save

---

### Step 2: Generate MLCO Layers

Convert ROI into MLCO layers (concentric rings from cortex to medulla):

```bash
python generate_mlco.py \
    --mask M1_kidney_roi.npy \
    --split \
    --anatomical M1_prepared/M1_air_reference.npy \
    --n-layers 24 \
    --output-dir M1_mlco/ \
    --label M1
```

**Required Arguments:**
- `--mask`: Your drawn ROI mask (.npy file)
- `--anatomical`: Reference image for visualization
- `--output-dir`: Where to save results
- `--label`: Label for output files (e.g., M1, M2)

**Optional Arguments:**
- `--split`: Split bilateral organs into two components (for kidneys, lungs, etc.)
- `--n-layers`: Number of layers per organ (default: 12, use 24 for kidneys)
- `--component-names`: Custom names for split components (default: `left right`)
- `--no-mri-flip`: Disable MRI left-right flip convention
- `--min-component-size`: Minimum component size in pixels (default: 100)

**Outputs:**
- `M1_mlco_layers.npy` - MLCO mask with integer labels
- `M1_mlco_layers_visualization.png` - Visual overlay on anatomical image

**Examples:**

```bash
# Bilateral kidneys (default MRI convention)
python generate_mlco.py \
    --mask bilateral_kidney_roi.npy \
    --split \
    --anatomical reference.npy \
    --n-layers 24 \
    --output-dir results/ \
    --label M1

# Single kidney (no splitting needed)
python generate_mlco.py \
    --mask left_kidney_roi.npy \
    --anatomical reference.npy \
    --n-layers 24 \
    --output-dir results/ \
    --label M1_left

# Custom component names
python generate_mlco.py \
    --mask bilateral_roi.npy \
    --split \
    --component-names medial lateral \
    --anatomical reference.npy \
    --n-layers 12 \
    --output-dir results/ \
    --label M1
```

**MRI Convention:**
By default, `--split` applies MRI left-right flip where anatomical left appears on the image right side. Use `--no-mri-flip` if working with non-MRI data.

**Result:** 
- With `--split`: Mask with layers 1-24 (left organ), 25-48 (right organ)
- Without `--split`: Mask with layers 1-24

---

### Step 3: Create Config Files

Create JSON config for each sample with pre-computed maps:

**`m1_wt_config.json`:**
```json
{
  "id": "M1_WT",
  "t2star_maps": {
    "air": "/absolute/path/to/M1_prepared/M1_air_t2star_custom.npy",
    "oxygen_1": "/absolute/path/to/M1_prepared/M1_oxygen1_t2star_custom.npy",
    "oxygen_2": "/absolute/path/to/M1_prepared/M1_oxygen2_t2star_custom.npy"
  },
  "r2star_maps": {
    "air": "/absolute/path/to/M1_prepared/M1_air_r2star_custom.npy",
    "oxygen_1": "/absolute/path/to/M1_prepared/M1_oxygen1_r2star_custom.npy",
    "oxygen_2": "/absolute/path/to/M1_prepared/M1_oxygen2_r2star_custom.npy"
  },
  "perfusion_map": "/absolute/path/to/M1_prepared/M1_perfusion.npy",
  "mlco_mask": "/absolute/path/to/M1_mlco/M1_mlco_layers.npy"
}
```

**`m2_ko_config.json`:**
```json
{
  "id": "M2_KO",
  "t2star_maps": {
    "air": "/absolute/path/to/M2_prepared/M2_air_t2star_custom.npy",
    "oxygen_1": "/absolute/path/to/M2_prepared/M2_oxygen1_t2star_custom.npy",
    "oxygen_2": "/absolute/path/to/M2_prepared/M2_oxygen2_t2star_custom.npy"
  },
  "r2star_maps": {
    "air": "/absolute/path/to/M2_prepared/M2_air_r2star_custom.npy",
    "oxygen_1": "/absolute/path/to/M2_prepared/M2_oxygen1_r2star_custom.npy",
    "oxygen_2": "/absolute/path/to/M2_prepared/M2_oxygen2_r2star_custom.npy"
  },
  "perfusion_map": "/absolute/path/to/M2_prepared/M2_perfusion.npy",
  "mlco_mask": "/absolute/path/to/M2_mlco/M2_mlco_layers.npy"
}
```

!!! tip "Use Absolute Paths"
    Always use full absolute paths in config files to avoid path resolution issues.

!!! note "Config Requirements"
    - **Required:** `id`, `t2star_maps`, `r2star_maps`, `mlco_mask`
    - **Optional:** `perfusion_map`
    - All conditions in `t2star_maps` must also be in `r2star_maps`
    - All map files must be `.npy` files generated by `prepare_data.py`

---

### Step 4: Run Analysis

Now run the analysis using pre-computed maps from Step 0!

#### Single Sample

```bash
python boldpy_analyze.py \
    --config m1_wt_config.json \
    --n-layers 24 \
    --output-dir results/M1_WT/
```

#### Compare Groups (WT vs KO)

```bash
python boldpy_analyze.py \
    --group1-config m1_wt_config.json \
    --group2-config m2_ko_config.json \
    --compare \
    --n-layers 24 \
    --output-dir results/comparison/
```

**Parameters:**
- `--config`: Single sample config file
- `--group1-config` / `--group2-config`: Group comparison configs
- `--compare`: Perform statistical comparison between groups
- `--n-layers 24`: Number of layers per organ (must match MLCO generation)
- `--output-dir`: Where to save results

**Note:** No `--source` flag needed! Maps are already computed in Step 0.

---

### Step 5: View Results

Results are in your output directory:

```
results/comparison/
├── M1_WT_complete_analysis.json          # All WT metrics
├── M2_KO_complete_analysis.json          # All KO metrics
├── wt_vs_ko_comparison.json              # Statistical comparison
│
├── Plots (PNG/SVG/PDF)
│   ├── M1_WT_mlco_profiles.*             # WT layer profiles
│   ├── M1_WT_perfusion_profile.*         # WT perfusion plot
│   ├── M1_WT_air_triple_overlay.*        # WT triple overlay
│   │
│   ├── M2_KO_mlco_profiles.*             # KO layer profiles
│   ├── M2_KO_perfusion_profile.*         # KO perfusion plot
│   ├── M2_KO_air_triple_overlay.*        # KO triple overlay
│   │
│   └── wt_vs_ko_air_comparison.*         # Group comparison
```

---

## Understanding Results

### JSON Output

**`wt_vs_ko_comparison.json` excerpt:**
```json
{
  "cortex_comparison": {
    "air": {
      "t2star": {
        "wt_mean": 12.8,
        "ko_mean": 16.0,
        "delta": 3.2,
        "percent_change": 25.0,
        "effect_size": 1.8,
        "interpretation": "large_effect"
      },
      "perfusion": {
        "wt_mean": 275,
        "ko_mean": 195,
        "delta": -80,
        "percent_change": -29.0,
        "interpretation": "reduced"
      }
    }
  }
}
```

**Interpretation:**
- KO shows +25% T2* (elevated, possible edema/hypoxia)
- KO shows -29% perfusion (reduced blood flow)
- Effect size d=1.8 (very large effect, biologically significant)

### Key Plots

**1. MLCO Profile Plot** - Shows T2*, R2*, and perfusion across all 24 layers
   - Identifies cortex vs medulla differences
   - Shows oxygen response (air vs O2)

**2. Perfusion Profile** - Dedicated perfusion plot
   - Shows blood flow distribution
   - Identifies hypoperfused regions

**3. T2* vs Perfusion Scatter** - Quadrant analysis
   - Q1 (High T2*, High Perf): Edema
   - Q2 (Low T2*, High Perf): Viable tissue
   - Q3 (Low T2*, Low Perf): Hypoxia
   - Q4 (High T2*, Low Perf): Necrosis

**4. Comparison Plot** - WT vs KO side-by-side
   - Shows layer-by-layer differences
   - Includes cortex-only statistics
   - Effect sizes displayed

---

## Common Workflows

### Kidney Research (Current Focus)

```bash
# 24 layers: cortex (1-10), CMJ (11-13), medulla (14-24)
python generate_mlco.py --roi kidney_roi.npy --n-layers 24 --bilateral
python boldpy_analyze.py --config kidney.json --n-layers 24
```

### Brain Research (Future Application)

```bash
# 50 layers: grey matter (1-30), white matter (31-50)
python generate_mlco.py --roi brain_roi.npy --n-layers 50 --bilateral
python boldpy_analyze.py --config brain.json --n-layers 50
```

### Heart Research (Future Application)

```bash
# 12 layers: epicardium to endocardium
python generate_mlco.py --roi heart_roi.npy --n-layers 12
python boldpy_analyze.py --config heart.json --n-layers 12 --single-organ
```

---

## Format Conversion

Convert ROI masks between formats:

```bash
# NPY → JSON (for ImageJ/QuPath)
python roi_format_utils.py \
    --input M1_mlco_mask_24layers.npy \
    --to-json

# JSON → NPY
python roi_format_utils.py \
    --input M1_mlco_mask_24layers.json \
    --to-npy

# Batch convert directory
python roi_format_utils.py \
    --batch-dir masks/ \
    --from npy \
    --to json
```

---

## Troubleshooting

### "No module named 'boldpy'"
```bash
cd boldpy/
pip install -e .
```

### "FileNotFoundError"
Use absolute paths in config files:
```json
{
  "mlco_mask": "/full/path/to/M1_mlco_mask_24layers.npy"
}
```

### "No pdata/2 found"
Your scan lacks Bruker's fitted maps. Use custom fitting:
```bash
--source nonlinear_fit
```

### ROI drawer not responding
Make sure you have a GUI backend:
```bash
# Linux
sudo apt-get install python3-tk

# macOS
brew install python-tk
```

---

## Step 5: Project-Level Group Analysis

After running `boldpy_analyze.py` for all samples, three project-level scripts aggregate results across groups. All are configured via a single `groups_config.json` file (see `examples/groups_config.json` for a complete template):

```json
{
  "output_dir": "processed/analysis/my_experiment",
  "hematology_csv": null,
  "groups": {
    "Control (n=2)": {
      "ids":   ["sample_ctrl_1", "sample_ctrl_2"],
      "color": "#E74C3C", "ls": "--", "lw": 1.8, "zorder": 4,
      "label": "Control", "short": "Ctrl"
    },
    "Treatment (n=3)": {
      "ids":   ["sample_trt_1", "sample_trt_2", "sample_trt_3"],
      "color": "#2E86C1", "ls": "-", "lw": 2.0, "zorder": 3,
      "label": "Treatment", "short": "Trt"
    }
  }
}
```

```bash
# MLCO layer profiles (mean ± SEM per layer, Mann-Whitney tests, zone summary stats)
python group_analysis.py --config groups_config.json

# K-means zone overlays + MLCO layer overlays + zone statistics dot-plot
python overlay_analysis.py --config groups_config.json

# Within-layer heterogeneity profiles + focal disruption analysis
python heterogeneity.py --config groups_config.json
```

| Script | Key Outputs |
|--------|-------------|
| `group_analysis.py` | T2*/R2*/perfusion layer profiles per group, `zone_summary_stats.json` |
| `overlay_analysis.py` | Per-sample overlay figures, cross-sample grid figures, zone dot-plot |
| `heterogeneity.py` | T2* std/CV profiles, OC strip plots, spatial CV maps |

For k-means zone clustering in `boldpy_analyze.py` (per-sample or shared reference), see [K-Means Zone Clustering](kmeans-zone-clustering.md).

---

## Next Steps

- **[User Guide](user-guide.md)** — Complete workflow and feature reference
- **[Scripts Reference](scripts-reference.md)** — All scripts documented with full options
- **[K-Means Zone Clustering](kmeans-zone-clustering.md)** — Data-driven zone boundaries
- **[Examples with Data](examples-with-data.md)** — Expected outputs and interpretation
- **[Metrics Documentation](metrics-documentation.md)** — Understanding T2*, R2*, perfusion

---

## Getting Help

- **Documentation:** [https://boldpy.readthedocs.io](https://boldpy.readthedocs.io)
- **Issues:** [GitHub Issues](https://github.com/yourusername/boldpy/issues)
- **Discussions:** [GitHub Discussions](https://github.com/yourusername/boldpy/discussions)

---

**Ready to analyze!**
