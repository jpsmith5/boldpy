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

