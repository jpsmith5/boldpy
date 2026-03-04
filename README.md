# BoldPy v2.3.1

**Renal BOLD MRI Analysis Framework with Multi-Layer Concentric Object (MLCO) Analysis,
K-Means Zone Clustering, and Group-Level Statistics**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Overview

BoldPy is a Python framework for quantitative BOLD MRI analysis using Multi-Layer Concentric
Objects (MLCO). Developed for renal oxygenation studies in mouse models, it is applicable to
any organ with concentric architecture (kidney, heart, eye).

The pipeline extracts T2*, R2*, and perfusion maps from raw MRI data, creates radial
layer-by-layer segmentations from organ surface to center, and generates quantitative profiles
enabling cortex-to-medulla-to-papilla analysis. Group-level comparisons, data-driven zone
boundary detection, and heterogeneity analysis are built in.

---

## Features

- **MLCO analysis** — 24-layer (configurable) bilateral radial segmentation from surface to center
- **T2*/R2*/perfusion quantification** — per-layer bilateral statistics, ΔT2* oxygen response
- **Intelligent T2* frame detection** — tiered: Bruker metadata → heuristic scoring → manual override
- **Data-driven zone boundaries** — k-means clustering on per-layer T2*/R2*/perfusion medians
- **Two-workflow group comparison** — shared reference (Workflow A) or per-sample boundaries (Workflow B)
- **Project-level analysis** — cross-group MLCO profiles, overlay visualization, heterogeneity metrics
- **18+ plotting functions** — PNG/PDF/SVG, dynamic zone color support
- **Bruker PvDatasets + DICOM support** — `prepare_data.py` / `prepare_dicom.py`
- **Robust missing-layer handling** — NaN fill, no crashes on incomplete data
- **YAML zone configuration** — static or data-driven zone definitions

---

## What's New in v2.3.1

### Consolidated Project Analysis Scripts

Three generic, config-driven scripts replace the previous collection of per-project analysis
scripts. All accept `--config groups_config.json` and work with any experiment:

- **`group_analysis.py`** — Cross-group MLCO profile comparison (mean ± SEM per layer,
  Mann-Whitney tests, zone-level summary statistics)
- **`overlay_analysis.py`** — K-means zone overlays + MLCO layer overlays + zone statistics
  dot-plot in one pass (k-means computed once, reused for all figure types)
- **`heterogeneity.py`** — Within-layer T2* heterogeneity profiling + focal disruption
  analysis (pixel distributions, spatial local-CV maps)

See `examples/groups_config.json` for the config template.

### K-Means Zone Clustering (v2.3.0)

Data-driven zone boundary detection replaces fixed layer-to-zone mappings:

```bash
# Per-sample clustering (Workflow B)
python boldpy_analyze.py --config sample.json --cluster-zones --n-clusters 3

# Apply one reference to all samples (Workflow A)
python boldpy_analyze.py --group1-config g1.json --group2-config g2.json \
    --compare --cluster-reference configs/zones/reference_k3.yaml
```

See [K-Means Zone Clustering](docs/kmeans-zone-clustering.md) for full details.

---

## Installation

```bash
git clone https://github.com/yourusername/boldpy
cd boldpy
pip install -e .                    # standard install
pip install -e ".[dev]"             # with pytest, black, flake8
pip install -e ".[dicom]"           # with DICOM support (pydicom)
pip install -e ".[docs]"            # with MkDocs (for building this site)
```

**Requirements:** Python >= 3.8, numpy, scipy, matplotlib, scikit-image, scikit-learn, Pillow, tqdm

---

## Quick Start

### Per-Sample Pipeline (Steps 1–4)

```bash
# Step 1: Extract maps from raw data
python prepare_data.py --input scan.PvDatasets --output-dir prepared/ --both-t2star
# or for DICOM:
python prepare_dicom.py --input /path/to/dicom/ --output-dir prepared/

# Step 2: Draw ROI interactively
python roi_drawer.py --image prepared/sample_reference.npy --output roi.npy

# Step 3: Generate MLCO layers (24-layer bilateral)
python generate_mlco.py --mask roi.npy --split \
    --anatomical prepared/sample_reference.npy \
    --n-layers 24 --output-dir mlco/ --label sample

# Step 4: Per-sample analysis
python boldpy_analyze.py --config sample_config.json \
    --n-layers 24 --output-dir results/sample/
```

**sample_config.json** (see `example_config.json` for a complete template):
```json
{
  "id": "sample_id",
  "t2star_maps": {
    "air":      "prepared/sample_air_t2star_custom.npy",
    "oxygen_1": "prepared/sample_oxygen1_t2star_custom.npy",
    "oxygen_2": "prepared/sample_oxygen2_t2star_custom.npy"
  },
  "r2star_maps": {
    "air":      "prepared/sample_air_r2star_custom.npy",
    "oxygen_1": "prepared/sample_oxygen1_r2star_custom.npy",
    "oxygen_2": "prepared/sample_oxygen2_r2star_custom.npy"
  },
  "perfusion_map": "prepared/sample_perfusion.npy",
  "mlco_mask":     "mlco/sample_mlco_layers.npy"
}
```

### Project-Level Analysis (Step 5)

After running `boldpy_analyze.py` for all samples, create a `groups_config.json`
(see `examples/groups_config.json`) and run:

```bash
python group_analysis.py   --config groups_config.json
python overlay_analysis.py --config groups_config.json
python heterogeneity.py    --config groups_config.json
```

---

## Documentation

| Document | Description |
|----------|-------------|
| [Quick Start](docs/quick-start.md) | Full 5-step workflow from raw data to group analysis |
| [User Guide](docs/user-guide.md) | Comprehensive per-sample and project-level workflow |
| [Scripts Reference](docs/scripts-reference.md) | All scripts with full options and outputs |
| [K-Means Zone Clustering](docs/kmeans-zone-clustering.md) | Data-driven zone boundaries |
| [Metrics Documentation](docs/metrics-documentation.md) | T2*, R2*, perfusion interpretation |
| [Examples with Data](docs/examples-with-data.md) | Expected outputs and result interpretation |
| [Installation](docs/installation.md) | Detailed setup and platform-specific notes |
| [Changelog](CHANGELOG.md) | Version history |

Build the documentation site locally:
```bash
pip install -e ".[docs]"
mkdocs serve
```

---

## Repository Layout

```
boldpy_v2.3.1/
├── boldpy_analyze.py              # Per-sample pipeline orchestrator
├── prepare_data.py                # Data extraction (Bruker PvDatasets)
├── prepare_dicom.py               # Data extraction (DICOM)
├── roi_drawer.py                  # Interactive ROI drawing
├── generate_mlco.py               # MLCO mask generation
├── mlco_analysis.py               # Layer-by-layer quantification
├── boldpy_plots.py                # 18+ plotting functions
├── boldpy_plots_multiregion.py    # Multiregion comparison plots
├── cluster_zones.py               # K-means zone boundary clustering
├── tissue_zones.py                # Zone config management
├── group_analysis.py              # Project: cross-group MLCO profiles
├── overlay_analysis.py            # Project: K-means + MLCO overlays
├── heterogeneity.py               # Project: heterogeneity analysis
├── fit_t2star.py                  # Standalone T2* fitting
├── roi_format_utils.py            # ROI format utilities
├── src/boldpy/                    # Python package library
├── configs/                       # Zone and threshold YAML configs
├── examples/                      # Tutorial data + groups_config.json template
└── docs/                          # Documentation (mkdocs)
```

---

## Citation

If you use BoldPy in your research, please cite:

```bibtex
@software{boldpy2026,
  title   = {BoldPy: Renal BOLD MRI Analysis Framework with MLCO and K-Means Zone Clustering},
  author  = {Your Name},
  year    = {2026},
  version = {2.3.1},
  url     = {https://github.com/yourusername/boldpy}
}
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

## Acknowledgments

Built on NumPy, SciPy, Matplotlib, scikit-image, scikit-learn.
