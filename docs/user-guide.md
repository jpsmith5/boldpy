# Comprehensive User Guide

This guide provides in-depth information on using BoldPy v2.3.1 for BOLD MRI analysis with MLCO (Multi-Layer Concentric Object) framework.

---

## Overview

**BoldPy** is a tissue-agnostic BOLD MRI analysis framework that provides:
- **Multi-layer analysis** (MLCO) for bilateral or single organs
- **T2*/R2* quantification** with custom or Bruker-extracted maps
- **Perfusion integration** for comprehensive tissue assessment
- **Tissue quality metrics** for viability assessment
- **Statistical comparison** between groups

---

## Complete Workflow

### Step 0: Prepare Data (`prepare_data.py`)

Extract and fit T2*/R2* maps from Bruker PvDatasets files with intelligent frame detection.

**Basic Usage:**
```bash
python prepare_data.py \
    --input scan.PvDatasets \
    --output-dir prepared/ \
    --sample-name M1_air \
    --both-t2star
```

**With Manual Frame Override (NEW in v2.2.1):**
```bash
python prepare_data.py \
    --input scan.PvDatasets \
    --output-dir prepared/ \
    --sample-name M1_air \
    --both-t2star \
    --t2-frame 3
```

**What it does:**
1. Extracts reference image (first echo from pdata/1)
2. Extracts Bruker T2* map (pdata/2, intelligent frame detection)
   - **NEW:** Tiered detection (metadata → heuristic → manual)
   - Confidence scoring and detailed diagnostics
3. Fits custom T2* from raw 8-echo data (pdata/1)
4. Computes R2* maps (1000/T2*)
5. Optionally extracts perfusion (pdata/2, Frame 5)

**T2* Frame Detection (NEW in v2.2.1):**

Three-tier approach for robust identification:
- **Tier 1:** Reads Bruker `VisuCoreFrameType` metadata
- **Tier 2:** Multi-factor scoring (100-point system)
- **Tier 3:** Manual override with `--t2-frame N`

**Key Options:**
- `--both-t2star`: Extract both Bruker and fit custom T2*
- `--custom-t2star`: Only fit custom T2*
- `--no-bruker`: Skip Bruker extraction
- `--t2-frame N`: Manually specify T2* frame (1-indexed) **(NEW)**
- `--extract-perfusion`: Extract perfusion map
- `--3d`: Output full 3D volume
- `--slice N`: Extract specific slice

**Batch Processing:**
```bash
python prepare_data.py \
    --input-dir BOLD/ \
    --output-dir prepared/ \
    --pattern "*air*.PvDatasets" \
    --both-t2star
```

**Outputs:**
- `{name}_reference.npy` - Reference image for ROI drawing
- `{name}_t2star_bruker.npy` - Bruker T2* map (if available)
- `{name}_t2star_custom.npy` - Custom fitted T2*
- `{name}_r2star_bruker.npy` - Bruker R2*
- `{name}_r2star_custom.npy` - Custom R2*
- `{name}_perfusion.npy` - Perfusion map (if requested)
- `{name}_metadata.json` - Scan parameters and statistics

---

### Step 1: Draw ROI (`roi_drawer.py`)

Create region of interest mask for analysis.

**Usage:**
```bash
python roi_drawer.py \
    --image prepared/M1_air_reference.npy \
    --output M1_kidney_roi.npy
```

**Controls:**
- **Click**: Add polygon points
- **Right-click**: Complete polygon
- **z**: Undo last point
- **c**: Clear all
- **Mouse wheel**: Zoom in/out
- **Click+drag**: Pan image
- **s**: Save ROI

**Tips:**
- Draw around the entire bilateral organ (both kidneys, both lungs, etc.)
- Or draw around a single organ
- The mask will be split in the next step if bilateral

---

### Step 2: Generate MLCO Layers (`generate_mlco.py`)

Create multi-layer concentric masks for analysis.

**Bilateral Organs:**
```bash
python generate_mlco.py \
    --mask M1_kidney_roi.npy \
    --split \
    --anatomical prepared/M1_air_reference.npy \
    --n-layers 24 \
    --output-dir M1_mlco/ \
    --label M1
```

**Single Organ:**
```bash
python generate_mlco.py \
    --mask left_kidney_roi.npy \
    --anatomical prepared/M1_air_reference.npy \
    --n-layers 24 \
    --output-dir M1_mlco/ \
    --label M1_left
```

**What it does:**
1. Splits bilateral mask into two components (if `--split`)
2. Creates concentric layers from surface to center
3. Applies MRI flip convention (anatomical left = image right)
4. Generates visualization

**Key Options:**
- `--split`: Split into two components (for bilateral organs)
- `--component-names left right`: Custom naming
- `--no-mri-flip`: Disable MRI convention (for CT, etc.)
- `--min-component-size 100`: Minimum pixels per component
- `--n-layers 24`: Layers per organ (12 = coarse, 24 = standard, 48 = fine)

**Output:**
- `{label}_mlco_layers.npy` - Layer mask (1-N for organ 1, N+1-2N for organ 2)
- `{label}_visualization.png` - Visual check

---

### Step 3: Create Config Files

Create JSON config pointing to all prepared maps.

**Example - m1_wt_config.json:**
```json
{
  "id": "M1_WT",
  "t2star_maps": {
    "air": "/path/to/prepared/M1_air_t2star_custom.npy",
    "oxygen_1": "/path/to/prepared/M1_oxygen1_t2star_custom.npy",
    "oxygen_2": "/path/to/prepared/M1_oxygen2_t2star_custom.npy"
  },
  "r2star_maps": {
    "air": "/path/to/prepared/M1_air_r2star_custom.npy",
    "oxygen_1": "/path/to/prepared/M1_oxygen1_r2star_custom.npy",
    "oxygen_2": "/path/to/prepared/M1_oxygen2_r2star_custom.npy"
  },
  "perfusion_map": "/path/to/prepared/M1_perfusion.npy",
  "mlco_mask": "/path/to/M1_mlco/M1_mlco_layers.npy"
}
```

**Required Fields:**
- `id`: Sample identifier
- `t2star_maps`: Dictionary of condition names → T2* map paths
- `r2star_maps`: Dictionary matching t2star_maps keys → R2* map paths
- `mlco_mask`: Path to MLCO layer mask

**Optional Fields:**
- `perfusion_map`: Path to perfusion map

---

### Step 4: Run Analysis (`boldpy_analyze.py`)

Analyze single sample or compare groups.

**Single Sample:**
```bash
python boldpy_analyze.py \
    --config m1_wt_config.json \
    --n-layers 24 \
    --output-dir results/M1/
```

**Group Comparison:**
```bash
python boldpy_analyze.py \
    --group1-config wt_config.json \
    --group2-config ko_config.json \
    --compare \
    --n-layers 24 \
    --output-dir results/comparison/
```

**What it does:**
1. Loads all T2*/R2*/perfusion maps
2. Performs layer-by-layer MLCO analysis
3. Calculates tissue quality metrics
4. Computes oxygen responsiveness (if multiple conditions)
5. Extracts cortex-only statistics
6. Generates comprehensive plots
7. Saves JSON results

**Key Parameters:**
- `--n-layers 24`: Must match MLCO generation
- `--config`: Single sample analysis
- `--group1-config`, `--group2-config`, `--compare`: Group comparison

---

---

## Step 5: Project-Level Analysis

After running `boldpy_analyze.py` for each sample, three project-level scripts aggregate results across groups. All are configured via a `groups_config.json` file:

```bash
# Template: see examples/groups_config.json
python group_analysis.py   --config groups_config.json   # Group MLCO profiles
python overlay_analysis.py --config groups_config.json   # K-means + MLCO overlays
python heterogeneity.py    --config groups_config.json   # Heterogeneity analysis
```

**groups_config.json** specifies the output directory, optional hematology CSV, and per-group sample IDs, colors, and line styles. See `examples/groups_config.json` for a complete template.

| Script | Purpose |
|--------|---------|
| `group_analysis.py` | Mean ± SEM MLCO profiles, Mann-Whitney tests, zone summary stats |
| `overlay_analysis.py` | K-means zone overlays, MLCO layer overlays, zone statistics dot-plot |
| `heterogeneity.py` | T2* std/CV/IQR profiles, outer cortex pixel distributions, spatial CV maps |

See [Scripts Reference → Project Analysis Scripts](scripts-reference.md#project-analysis-scripts) for full documentation.

---

## Key Metrics Explained

### T2* Relaxation Time
- **Units:** milliseconds (ms)
- **Normal range:** 27-37 ms (mouse kidney cortex)
- **Interpretation:**
  - Low T2* (<40ms): Healthy tissue
  - Elevated T2* (40-60ms): Edema (fluid accumulation)
  - Very high T2* (>60ms): Likely necrosis

### R2* Relaxation Rate
- **Formula:** R2* = 1000 / T2*
- **Units:** Hz
- **Interpretation:** Inverse of T2*, higher = better oxygenation

### Perfusion
- **Units:** Relative % or mL/100g/min
- **Normal range:** 150-400 mL/100g/min (mouse kidney)
- **Interpretation:** Blood flow to tissue

### Tissue Quality (Viability)
Per-pixel classification:
- **Viable (>90%)**: Healthy tissue
- **Suspect Edema (75-90%)**: Mild damage
- **Likely Necrosis (<75%)**: Significant damage

Based on T2* thresholds:
- Viable: <40 ms
- Suspect edema: 40-60 ms
- Likely necrosis: >60 ms

### Oxygen Responsiveness
- **ΔT2*:** Change in T2* from air to oxygen
- **Positive ΔT2*:** Normal response (tissue dilates, more blood)
- **Negative ΔT2*:** Abnormal (may indicate hypoxia)

---

## 5-Zone Analysis (Kidney-Specific)

Anatomical zones from outer to inner:
1. **Cortex:** Outer 33% of layers (e.g., layers 1-8 of 24)
2. **Outer Medulla:** Next 16.7% (layers 9-12)
3. **CMJ (Cortico-Medullary Junction):** Middle layers (13-16)
4. **Inner Medulla:** Next 16.7% (layers 17-20)
5. **Papilla:** Inner 16.7% (layers 21-24)

**Note:** Currently kidney-specific but will be generalized or user-customizable in future versions.

---

## Troubleshooting

### Issue: "Config missing required fields"
**Solution:** Ensure config has `id`, `t2star_maps`, `r2star_maps`, `mlco_mask`

### Issue: "File not found"
**Solution:** Use absolute paths in config files

### Issue: "T2* and R2* maps have different shapes"
**Solution:** Re-run prepare_data.py for both maps with same parameters

### Issue: "Perfusion shape doesn't match T2*"
**Solution:** Run prepare_data.py with perfusion extraction - it will auto-resample

### Issue: "Only X frames found (need frame 5)"
**Solution:** This PvDatasets doesn't have perfusion data, skip --extract-perfusion

---

## Best Practices

1. **Always use absolute paths** in config files
2. **Use custom T2* fitting** for best quality (`--custom-t2star`)
3. **Extract perfusion separately** if resolution differs
4. **Save intermediate results** from each step
5. **Use consistent naming** across samples
6. **Run prepare_data.py once** per scan, then reuse outputs
7. **Validate MLCO layers** visually before analysis

---

## Advanced Usage

### Multiple Conditions
Add any number of conditions to your config:
```json
{
  "t2star_maps": {
    "baseline": "...",
    "drug_low": "...",
    "drug_high": "...",
    "recovery": "..."
  }
}
```

### Custom Layer Count
- 12 layers: Coarse analysis
- 24 layers: Standard (recommended)
- 48 layers: Fine detail

### Non-MRI Data
For CT or other imaging:
```bash
generate_mlco.py --mask roi.npy --no-mri-flip ...
```

---

## Output Files

### Single Sample Analysis
```
results/M1/
├── M1_complete_analysis.json          # All metrics
├── M1_tlco_profiles.png               # Layer profiles
├── M1_perfusion_profile.png           # If perfusion available
├── M1_{condition}_triple_overlay.png  # T2*/R2*/Perfusion maps
└── M1_{condition}_t2star_perfusion_scatter.png
```

### Group Comparison
```
results/comparison/
├── group1_vs_group2_comparison.json
├── group1_vs_group2_{condition}_comparison.png
├── M1_WT_complete_analysis.json
└── M2_KO_complete_analysis.json
```

---

## Getting Help

- Check [troubleshooting](#troubleshooting) section
- Review [quick-start guide](quick-start.md)
- See [metrics documentation](metrics-documentation.md)
- File issues on GitHub

