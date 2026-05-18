# BoldPy PEP Integration

**Status:** Implemented — v3.0.0

BoldPy uses the [PEP (Portable Encapsulated Projects)](https://pep.databio.org) ecosystem
for sample metadata management, automated per-sample pipeline submission, checkpoint/resume,
and HPC support.

---

## Overview

| Tool | Role |
|------|------|
| [PEP](https://pep.databio.org) | Sample metadata standard (YAML + CSV) |
| [peppy](https://peppy.databio.org) | Python library to read PEP projects |
| [eido](https://eido.databio.org) | PEP schema validation |
| [looper](https://looper.databio.org) | Submits per-sample pipeline jobs (local, SLURM) |
| [PyPiper](https://pypiper.databio.org) | Pipeline execution with checkpoints and logging |
| [pipestat](https://pipestat.databio.org) | Standardized per-sample result reporting |

### Architecture

```
project_config.yaml  ──┬──► looper ──► pipeline/boldpy_pipeline.py  (PyPiper, 1 sample)
sample_table.csv     ──┘               │ Step 0: prepare_dicom/data.py
                                       │ Step 1: roi_drawer.py (manual checkpoint)
                                       │ Step 2: generate_mlco.py
                                       │ Step 3: boldpy_analyze.py
                                       └─ Step 4: pipestat report

project_config.yaml  ──► group_analysis.py      (project-level pipeline)
                    ──► overlay_analysis.py
                    └──► heterogeneity.py
```

The `groups_config.json` format is **replaced** by `project_config.yaml` + `sample_table.csv`.
Group membership (`group` column) and styling (`group_styles` section) live in the PEP config.

---

## Quick Start

### 1. Install dependencies

```bash
pip install peppy eido looper piper pipestat
# piper is the PyPI package name for pypiper
```

### 2. Create a PEP project

Copy and edit the example:
```bash
cp pipeline/examples/project_config.yaml my_experiment/
cp pipeline/examples/sample_table.csv    my_experiment/
cp pipeline/examples/looper_config.yaml  my_experiment/
# Note: templates live in pipeline/examples/, not examples/
```

Edit `sample_table.csv` — one row per sample:
```csv
sample_name,group,data_format,scan_dir,n_layers,notes
ctrl_001,control,dicom,data/ctrl_001/,24,
trt_001,treatment,dicom,data/trt_001/,24,
```

Edit `project_config.yaml` — set paths and group styles.

### 3. Validate the sample table

```bash
eido inspect my_experiment/sample_table.csv \
    -s pipeline/boldpy_pep_schema.yaml
```

### 4. Run the per-sample pipeline

```bash
# Local (sequential)
looper run my_experiment/project_config.yaml

# HPC/SLURM
looper run my_experiment/project_config.yaml --compute slurm

# Dry run (preview commands without executing)
looper run my_experiment/project_config.yaml --dry-run

# Single sample
looper run my_experiment/project_config.yaml \
    --sel-attr sample_name --sel-incl ctrl_001

# Subset by group
looper run my_experiment/project_config.yaml \
    --sel-attr group --sel-incl treatment
```

After running, **draw ROIs manually** for any sample that paused at Step 1, then re-run looper — it will resume from Step 2 automatically.

### 5. Check pipeline status

```bash
looper check my_experiment/project_config.yaml
```

### 6. Run group-level analysis

After all samples complete:
```bash
python group_analysis.py --pep my_experiment/project_config.yaml
python overlay_analysis.py --pep my_experiment/project_config.yaml
python heterogeneity.py --pep my_experiment/project_config.yaml
```

---

## Project Config (`project_config.yaml`)

```yaml
pep_version: "2.1.0"
sample_table: sample_table.csv

# Project-level paths (used by group-level scripts)
results_root: "processed"
analysis_dir: "processed/analysis"       # where per-sample _complete_analysis.json live
prepared_dir: "processed/prepared"       # where per-sample T2* .npy files live
mlco_dir:     "processed/mlco"           # where per-sample mlco_bilateral.npy live
group_output_dir: "processed/analysis/group_comparison"  # group-level figure output
hematology_csv: null                     # or "data/hematology.csv"

sample_modifiers:
  derive:
    attributes: [output_dir_sample, prepared_dir_sample, mlco_dir_sample, results_dir]
    sources:
      output_dir_sample:   "{results_root}/{sample_name}"
      prepared_dir_sample: "{results_root}/prepared/{sample_name}"
      mlco_dir_sample:     "{results_root}/mlco/{sample_name}"
      results_dir:         "{results_root}/analysis/{sample_name}"
  imply:
    - if: {n_layers: ""}
      then: {n_layers: "24"}

group_styles:
  control:
    color: "#E74C3C"
    ls: "--"
    lw: 1.8
    zorder: 4
    label: "Control (n=2)"   # display name used in figure legends
    short: "Ctrl"
  treatment:
    color: "#2E86C1"
    ls: "-"
    lw: 2.0
    zorder: 3
    label: "Treatment (n=5)"
    short: "Trt"
```

**Key features:**
- `derive` — auto-generate per-sample output paths from `sample_name`
- `imply` — default `n_layers: 24` if blank
- `group_styles` — plot styling (BoldPy-specific; PEP ignores unknown top-level keys)

---

## Sample Table (`sample_table.csv`)

| Column | Required | Description |
|--------|----------|-------------|
| `sample_name` | Yes | Unique identifier (matches directory and file prefixes) |
| `group` | Yes | Group identifier — must match a key in `group_styles` |
| `data_format` | Yes | `dicom` or `bruker` |
| `scan_dir` | Yes | Root directory of raw scan data |
| `scan_air` | No | Path to air-condition scan (auto-discovered if omitted) |
| `scan_oxygen_1` | No | Path to oxygen_1 condition scan |
| `scan_oxygen_2` | No | Path to oxygen_2 condition scan |
| `n_layers` | No | MLCO layers (default: 24 via `imply`) |
| `notes` | No | Free-text notes |

---

## Pipeline Steps and Checkpointing

`pipeline/boldpy_pipeline.py` wraps Steps 0–4 with PyPiper checkpointing:

| Step | Script | Target file | Resumable? |
|------|--------|-------------|------------|
| 0 | `prepare_dicom.py` / `prepare_data.py` | `{out}/prepared/{sid}_air_t2star_custom.npy` | Yes |
| 1 | `roi_drawer.py` (manual) | `{out}/{sid}_roi.npy` | Fails with instructions if missing |
| 2 | `generate_mlco.py` | `{out}/mlco/{sid}_mlco_bilateral.npy` | Yes |
| 3 | Config generation | `{out}/{sid}_pipeline_config.json` | Yes |
| 4 | `boldpy_analyze.py` | `{out}/results/{sid}_complete_analysis.json` | Yes |

If a step's target file already exists, PyPiper skips it. This means:
- Run `looper run` → pipeline runs Steps 0–1, then **fails** at the ROI checkpoint
- Draw ROI manually with `roi_drawer.py`
- Run `looper run` again → PyPiper skips Steps 0–1, resumes from Step 2

---

## Running Group-Level Analysis

After all per-sample jobs complete, run the three project-level scripts:

```bash
python group_analysis.py   --pep my_experiment/project_config.yaml
python overlay_analysis.py --pep my_experiment/project_config.yaml
python heterogeneity.py    --pep my_experiment/project_config.yaml
```

---

## Snakemake Compatibility

The same `project_config.yaml` can power a Snakemake workflow:

```python
# Snakefile
import peppy

project = peppy.Project("project_config.yaml")
SAMPLES = [s.sample_name for s in project.samples]

rule all:
    input:
        expand("processed/analysis/{sample}/{sample}_complete_analysis.json",
               sample=SAMPLES)
```

---

## Programmatic Access

All pipeline steps are importable modules that expose a `run()` function. The pipeline
wrappers in `pipeline/` call these functions directly — there is no subprocess overhead.
You can call any step from your own Python code in the same way:

```python
import group_analysis
group_analysis.run(pep_path="project_config.yaml", output_dir="results/group_comparison/")

import overlay_analysis
overlay_analysis.run(pep_path="project_config.yaml", output_dir="results/overlays/")

import heterogeneity
heterogeneity.run(pep_path="project_config.yaml", output_dir="results/heterogeneity/")
```

Per-sample pipeline steps follow the same pattern:

```python
import boldpy_analyze
boldpy_analyze.run(config="sample_config.json", n_layers=24, output_dir="results/sample/")
```

This makes it straightforward to embed BoldPy analysis inside a larger Python workflow,
a Jupyter notebook, or a custom batch script without invoking subprocesses.

---

## References

- [PEP specification](https://pep.databio.org/en/latest/specification/)
- [PyPiper documentation](https://pypiper.databio.org)
- [Looper documentation](https://looper.databio.org)
