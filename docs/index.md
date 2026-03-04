# BoldPy Documentation

**Tissue-Agnostic BOLD MRI Analysis Framework v2.3.1**

Welcome to BoldPy, a comprehensive Python framework for analyzing Blood Oxygen Level Dependent (BOLD) MRI data across multiple tissue types.

---

## Overview

BoldPy provides tools for:

- **Intelligent T2* detection** with tiered frame identification
- **Multi-parametric BOLD analysis** (T2*, R2*, perfusion)
- **Multi-Layer Concentric Object (MLCO) analysis** with customizable layer resolution
- **Continuous whole-kidney visualization** (cortex → medulla → papilla)
- **Regional analysis** across anatomical zones
- **Statistical comparison** between experimental groups with comprehensive visualizations
- **Publication-ready visualization** with 18 plotting functions
- **Flexible data processing** (Bruker's processing or custom fitting)
- **Robust data handling** (automatic handling of missing layers)
- **Data-driven zone clustering** via k-means on per-layer T2*/R2*/perfusion medians
- **Project-level analysis scripts** for group comparison, overlay visualization, and heterogeneity analysis

---

## Key Features

### 🎯 Enhanced MLCO Analysis
- 24-layer spatial resolution (or custom) from cortex to medulla
- Bilateral kidney analysis with automatic averaging
- Integration of T2*, R2*, and perfusion measurements (automatic upsampling)
- Tissue quality assessment with configurable viability thresholds
- Missing layer handling (no crashes on incomplete data)

### 🔍 Intelligent T2* Detection (NEW in v2.2.1)
- **Tier 1:** Metadata parsing from Bruker `VisuCoreFrameType`
- **Tier 2:** Enhanced multi-factor scoring (100-point system)
- **Tier 3:** Manual override with `--t2-frame N` option
- **Confidence reporting:** HIGH, MEDIUM, LOW with detailed diagnostics

### 📊 Comprehensive Metrics
- **Cortex-only statistics** - Isolate viable tissue for accurate comparison
- **Tissue heterogeneity** - Coefficient of variation per layer
- **Oxygen responsiveness** - Air vs oxygen challenge quantification
- **Effect sizes** - Cohen's d for statistical rigor
- **Whole-kidney gradients** - Cortex-to-medulla quantification

### 🎨 Publication-Ready Plots (18 Functions)
- **Continuous profiles** - Entire kidney as one plot (NEW!)
- **Group comparisons** - Overlaid WT vs KO with perfusion (NEW!)
- MLCO profile plots with perfusion integration
- T2* vs Perfusion scatter plots with quadrant analysis
- Triple overlay plots (T2* + R2* + Perfusion)
- All plots in PNG (300 DPI), SVG, and PDF formats
- Integer-only x-axis labels for clean presentation

### 🔧 Flexible Processing
- **Bruker source** - Fast processing using Bruker's pdata/2 maps
- **Nonlinear fit** - Custom fitting from raw pdata/1 echoes
- **Automatic perfusion upsampling** - 80×80 → 200×200 bilinear interpolation
- Extensible framework for additional fitting methods

---

## What's New in v2.3.1

### Consolidated Project Analysis Scripts

Three generic, config-driven project-level analysis scripts replace the previous collection of hardcoded single-use scripts. All three accept a `--config groups_config.json` argument and work with any experiment:

- **`group_analysis.py`** — Cross-group MLCO profile comparison (T2*, R2*, perfusion mean ± SEM per layer with Mann-Whitney statistics and zone summary)
- **`overlay_analysis.py`** — K-means zone overlays + MLCO layer overlays + zone statistics dot-plot in one pass
- **`heterogeneity.py`** — Within-layer T2* heterogeneity profiling + focal disruption analysis (pixel distributions, spatial CV maps)

See `examples/groups_config.json` for the config template and [Scripts Reference](scripts-reference.md#project-analysis-scripts) for full documentation.

### K-Means Zone Clustering (v2.3.0)

Data-driven zone boundary detection via `--cluster-zones` flag in `boldpy_analyze.py`. See [K-Means Zone Clustering](kmeans-zone-clustering.md) for details.

---

## What's New in v2.2.1

### Tiered T2* Frame Detection
Intelligent three-tier system for robust frame identification:
```bash
# Automatic detection
python prepare_data.py --input scan.PvDatasets --output-dir prepared/

# Manual override if needed
python prepare_data.py --input scan.PvDatasets --output-dir prepared/ --t2-frame 3
```

### Continuous Whole-Kidney Visualization
New plotting functions show complete organ profiles:
- Single sample: `plot_whole_kidney_continuous()`
- Group comparison: `plot_whole_kidney_comparison()`
- Automated generation in `boldpy_analyze.py`

### Enhanced Robustness
- Missing layers filled with NaN (displayed as gaps)
- Layer number inference for averaged data
- Integer-only x-axis labels
- Comprehensive error diagnostics

See [Changelog](../CHANGELOG.md) for complete details.

---

## Quick Start

**Try BoldPy in 30 seconds with tutorial data:**

```bash
# Clone and install
git clone https://github.com/yourusername/boldpy
cd boldpy
pip install -e .

# Run tutorial analysis (works immediately!)
cd examples/tutorial_data/
python ../../boldpy_analyze.py \
    --config sample_config.json \
    --n-layers 12 \
    --output-dir tutorial_results/

# See results
ls tutorial_results/
```

**What you get:**
- Layer-by-layer T2*/R2* profiles
- Tissue quality metrics  
- Oxygen responsiveness analysis
- Perfusion integration

→ **[Tutorial README](../examples/tutorial_data/README.md)** for details  
→ **[Full Workflow Guide](quick-start.md)** to process your own data

---

## Use Cases

### Kidney Disease Research
```python
# Compare WT vs knockout models
# Identify cortical vs medullary pathology
# Quantify oxygen responsiveness
# Assess tissue viability
```

### Drug Efficacy Studies
```python
# Track treatment response over time
# Measure regional perfusion changes
# Quantify tissue recovery
# Generate publication figures
```

### Methodology Development
```python
# Compare fitting algorithms
# Validate imaging biomarkers
# Optimize scan protocols
# Benchmark analysis pipelines
```

---

## Example Output

### Analysis Results

**Cortex Comparison (WT vs KO):**
```
                WT          KO        Change        Effect
T2* (ms):      12.8        16.0      +3.2 (+25%)    d=1.8
Perfusion:     275         195       -80 (-29%)     -
R2* (Hz):      78          66        -12 (-15%)     -

Interpretation: Elevated T2* with reduced perfusion 
suggests tissue edema with impaired blood flow.
```

### Visualizations

![MLCO Profile](assets/mlco_profile_example.png)
*24-layer profile showing T2*, R2*, and perfusion across cortex to medulla*

![Comparison Plot](assets/comparison_example.png)
*WT vs KO comparison with cortex-only statistics and effect sizes*

---

## Documentation

### Getting Started
- **[Installation](installation.md)** — Setup instructions and requirements
- **[Quick Start](quick-start.md)** — Full 5-step workflow from raw data to group analysis
- **[User Guide](user-guide.md)** — Comprehensive per-sample and project-level workflow

### Reference
- **[Scripts Reference](scripts-reference.md)** — All scripts documented (pipeline + project-level analysis)
- **[K-Means Zone Clustering](kmeans-zone-clustering.md)** — Data-driven zone boundaries (v2.3.0+)
- **[Metrics Documentation](metrics-documentation.md)** — T2*, R2*, perfusion, and heterogeneity metrics

### Examples
- **[Examples with Data](examples-with-data.md)** — Expected outputs and result interpretation

### Changelog
- **[Changelog](../CHANGELOG.md)** — Complete version history

---

## Recent Updates

### Version 2.3.1 (2026-03-04)

**Script consolidation:**
- `group_analysis.py` — Cross-group MLCO profiles, Mann-Whitney stats, zone summary
- `overlay_analysis.py` — K-means + MLCO overlay visualization in one script
- `heterogeneity.py` — Within-layer heterogeneity + focal disruption analysis
- `examples/groups_config.json` — Template config for all project-level scripts

### Version 2.3.0 (2026-02-17)

**K-means zone clustering:**
- Data-driven zone boundary detection (`--cluster-zones`)
- Shared reference workflow (`--cluster-reference`) for valid group comparisons
- Silhouette scoring and diagnostic plots

### Version 2.2.x (2026-01-20 to 2026-01-27)

- Tiered T2* frame detection (metadata → heuristic → manual `--t2-frame N`)
- Continuous whole-kidney visualization
- Oxygen challenge plotting suite
- YAML-based zone and threshold configuration system

→ **[Full Changelog](../CHANGELOG.md)**

---

## Research Applications

BoldPy has been used in studies investigating:

- **Acute Kidney Injury** - Cortical vs medullary damage patterns
- **Diabetic Nephropathy** - Oxygen metabolism dysfunction
- **Hypertensive Nephropathy** - Vascular and perfusion changes
- **Drug Nephrotoxicity** - Regional susceptibility assessment
- **Transplant Assessment** - Graft viability monitoring

---

## Community & Support

### Getting Help

- 📖 **[Documentation](https://boldpy.readthedocs.io)** - Comprehensive guides
- 💬 **[Discussions](https://github.com/yourusername/boldpy/discussions)** - Ask questions
- 🐛 **[Issues](https://github.com/yourusername/boldpy/issues)** - Report bugs
- 📧 **Email** - your.email@institution.edu

### Contributing

Contributions welcome — bug reports, feature suggestions, and pull requests are all appreciated.
Please open an issue on [GitHub](https://github.com/yourusername/boldpy/issues) to get started.

---

## Citation

If you use BoldPy in your research, please cite:

```bibtex
@software{boldpy2026,
  author = {Your Name},
  title = {BoldPy: Tissue-Agnostic BOLD MRI Analysis Framework},
  year = {2026},
  version = {2.1.0},
  url = {https://github.com/yourusername/boldpy}
}
```

---

## License

BoldPy is released under the MIT License. See [LICENSE](https://github.com/yourusername/boldpy/blob/main/LICENSE) for details.

---

## Acknowledgments

BoldPy was developed with support from:
- [Your Institution]
- [Funding Agencies]
- The open-source scientific Python community

Built with: NumPy, SciPy, Matplotlib, scikit-image, scikit-learn

---

**Ready to get started?** → [Quick Start Guide](quick-start.md) 🚀
