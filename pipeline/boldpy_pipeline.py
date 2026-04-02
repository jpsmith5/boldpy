#!/usr/bin/env python3
"""
BoldPy Pipeline — Per-Sample Analysis (Steps 1–4)
==================================================

Second of two sample-level pipelines. Assumes data extraction (Step 0) and
ROI drawing have already been completed. Invoked by looper:

    looper run project_config.yaml \\
        --pipeline-interfaces pipeline/boldpy_pipeline_interface.yaml

Two-pipeline workflow:

    # 1. Extract data for all samples
    looper run project_config.yaml \\
        --pipeline-interfaces pipeline/boldpy_prepare_interface.yaml

    # 2. Draw ROIs manually for each sample (see printed instructions)
    #    python roi_drawer.py --image processed/{sid}/prepared/{sid}_air_reference.npy \\
    #                         --output processed/{sid}/{sid}_roi.npy

    # 3. Run analysis for all samples
    looper run project_config.yaml \\
        --pipeline-interfaces pipeline/boldpy_pipeline_interface.yaml

    # 4. Run project-level collator (group comparisons) after all samples done
    looper runp project_config.yaml

Steps
-----
  Step 1 — MLCO mask generation  (generate_mlco.run())
  Step 2 — Per-sample config     (derived from known output paths)
  Step 3 — Analysis              (boldpy_analyze.run())
  Step 4 — Result reporting      (pipestat)

Each step is skipped if its target file already exists (PyPiper checkpointing).

Precondition: fails immediately with a clear message if the ROI mask is
missing — this is a user error, not a runtime failure.
"""

import argparse
import json
import os
import sys
from pathlib import Path

# ── sys.path: allow importing from the parent boldpy_v3.0.0/ directory ────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pypiper
import generate_mlco
import boldpy_analyze

# ── Argument parsing ───────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(
    description="BoldPy per-sample analysis pipeline (Steps 1–4)"
)
parser.add_argument("--sample-name", required=True,
                    help="Unique sample identifier")
parser.add_argument("--data-format", required=True, choices=["dicom", "bruker"],
                    help="Raw data format: dicom or bruker (PvDatasets)")
parser.add_argument("--scan-dir", required=True,
                    help="Root directory for raw scan data")
parser.add_argument("--output-dir", default=None,
                    help="Per-sample base output directory (used for logs if explicit dirs not given)")
parser.add_argument("--prepared-dir", default=None,
                    help="Directory containing prepared T2*/perfusion .npy files and ROI mask")
parser.add_argument("--mlco-dir", default=None,
                    help="Directory for MLCO mask output")
parser.add_argument("--results-dir", default=None,
                    help="Directory for analysis results JSON output")
parser.add_argument("--n-layers", type=int, default=24,
                    help="Number of MLCO layers (default: 24)")
parser.add_argument("--no-perfusion", action="store_true",
                    help="Exclude perfusion from analysis config (use when perfusion data "
                         "is from an incompatible acquisition method, e.g. Bruker FAIR-EPI "
                         "vs DICOM T1-IR in the same cohort)")
parser.add_argument("--pep", default=None,
                    help="Path to project_config.yaml (used by pipestat)")
parser.add_argument("--pipestat-config", default=None,
                    help="Path to pipestat config file (provided by looper)")

parser = pypiper.add_pypiper_args(parser, groups=["pypiper", "looper"])
args = parser.parse_args()

# ── Derived paths ──────────────────────────────────────────────────────────────

sid         = args.sample_name
script_root = Path(__file__).resolve().parent.parent   # boldpy_v3.0.0/

# Resolve per-sample directories — explicit args take priority over --output-dir subdirs
out = Path(args.output_dir).resolve() if args.output_dir else None
prep_dir    = Path(args.prepared_dir).resolve() if args.prepared_dir else (out / "prepared" if out else None)
mlco_dir    = Path(args.mlco_dir).resolve()    if args.mlco_dir    else (out / "mlco"     if out else None)
results_dir = Path(args.results_dir).resolve() if args.results_dir else (out / "results"  if out else None)

if not (prep_dir and mlco_dir and results_dir):
    import sys as _sys
    print("ERROR: Provide --output-dir or all three of --prepared-dir, --mlco-dir, --results-dir",
          file=_sys.stderr)
    _sys.exit(1)

outfolder = str(results_dir)

# ── Precondition: ROI must exist ──────────────────────────────────────────────
#
# The ROI is drawn interactively — it cannot be automated. If it's missing,
# the user needs to run boldpy_prepare first and then draw the ROI manually.
# This check runs before PipelineManager is initialised so there is no
# partial pipeline state written for what is purely a usage error.

# ROI mask — check both naming conventions (roi_mask.npy is standard; roi.npy is legacy)
roi_path = prep_dir / f"{sid}_roi_mask.npy"
if not roi_path.exists():
    roi_path = prep_dir / f"{sid}_roi.npy"
ref_path  = prep_dir / f"{sid}_air_reference.npy"
if not ref_path.exists():
    ref_path = prep_dir / f"{sid}_reference.npy"   # prepare_dicom.py naming

if not roi_path.exists():
    print(
        f"\n  ERROR: ROI mask not found for '{sid}': {roi_path}\n\n"
        f"  Complete these steps first:\n\n"
        f"  1. Run the prepare pipeline if you haven't already:\n"
        f"       looper run project_config.yaml \\\n"
        f"           --pipeline-interfaces pipeline/boldpy_prepare_interface.yaml\n\n"
        f"  2. Draw the ROI:\n"
        f"       cd {script_root}\n"
        f"       python roi_drawer.py \\\n"
        f"           --image '{ref_path}' \\\n"
        f"           --output '{roi_path}'\n\n"
        f"  3. Then re-run this pipeline.\n",
        file=sys.stderr,
    )
    sys.exit(1)

# ── PipelineManager ────────────────────────────────────────────────────────────

results_dir.mkdir(parents=True, exist_ok=True)
pm = pypiper.PipelineManager(
    name="boldpy",
    outfolder=outfolder,
    args=args,
    pipestat_record_identifier=sid,
    pipestat_config=args.pipestat_config,
    pipestat_schema=str(Path(__file__).parent / "boldpy_pipestat_schema.yaml"),
)

# ── Step 1: MLCO mask generation ───────────────────────────────────────────────

target_mlco = mlco_dir / f"{sid}_mlco_bilateral.npy"

if not target_mlco.exists():
    pm.timestamp("### Step 1: MLCO mask generation")
    generate_mlco.run(
        mask=str(roi_path),
        anatomical=str(ref_path),
        output_dir=str(mlco_dir),
        label=sid,
        n_layers=args.n_layers,
        split=True,
    )
else:
    pm.timestamp("### Step 1: MLCO mask generation [SKIPPED — target exists]")

# ── Step 2: Generate per-sample analysis config ────────────────────────────────

config_path   = results_dir / f"{sid}_pipeline_config.json"
target_result = results_dir / f"{sid}_complete_analysis.json"

if True:  # always regenerate — config must reflect current prepared/ contents
    conditions = ["oxygen_1", "air", "oxygen_2"]
    t2_maps, r2_maps, perf_maps = {}, {}, {}

    perf_single = prep_dir / f"{sid}_perfusion.npy"   # single baseline measurement

    for cond in conditions:
        t2c  = prep_dir / f"{sid}_{cond}_t2star_custom.npy"
        t2b  = prep_dir / f"{sid}_{cond}_t2star_bruker.npy"
        r2c  = prep_dir / f"{sid}_{cond}_r2star_custom.npy"
        r2b  = prep_dir / f"{sid}_{cond}_r2star_bruker.npy"
        perf = prep_dir / f"{sid}_{cond}_perfusion.npy"

        t2_maps[cond]   = str(t2c)  if t2c.exists()  else str(t2b)  if t2b.exists()  else None
        r2_maps[cond]   = str(r2c)  if r2c.exists()  else str(r2b)  if r2b.exists()  else None
        # Per-condition perfusion takes priority; fall back to single baseline for air only
        if perf.exists():
            perf_maps[cond] = str(perf)
        elif cond == "air" and perf_single.exists():
            perf_maps[cond] = str(perf_single)

    # --no-perfusion: exclude when acquisition method is incompatible with the cohort
    if args.no_perfusion:
        perf_maps_clean = {}
        perf_baseline   = None
        print(f"  --no-perfusion: perfusion excluded for {sid}")
    else:
        perf_maps_clean = {c: p for c, p in perf_maps.items() if p is not None}
        perf_baseline   = perf_maps_clean.get("air") or next(iter(perf_maps_clean.values()), None)

    sample_cfg = {
        "id":             sid,
        "t2star_maps":    {c: p for c, p in t2_maps.items()   if p is not None},
        "r2star_maps":    {c: p for c, p in r2_maps.items()   if p is not None},
        "perfusion_maps": perf_maps_clean,
        "perfusion_map":  perf_baseline,
        "mlco_mask":      str(target_mlco),
        "n_layers":       args.n_layers,
        "output_dir":     str(results_dir),
    }
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as fh:
        json.dump(sample_cfg, fh, indent=2)
    print(f"  Written sample config: {config_path}")

# ── Step 3: Per-sample analysis ────────────────────────────────────────────────

if not target_result.exists():
    pm.timestamp("### Step 3: Per-sample analysis")
    boldpy_analyze.run(
        config=str(config_path),
        output_dir=str(results_dir),
        n_layers=args.n_layers,
    )
else:
    pm.timestamp("### Step 3: Per-sample analysis [SKIPPED — target exists]")

# ── Step 4: Report results via pipestat ───────────────────────────────────────

import numpy as np

with open(target_result) as fh:
    analysis = json.load(fh)

def _zone_val(metric, cond, stat="median"):
    try:
        return float(
            analysis["conditions"][cond]["bilateral"]["zones"]
            ["outer_cortex"][metric][stat]
        )
    except (KeyError, TypeError):
        return None

def _delta():
    try:
        return float(
            analysis["conditions"]["oxygen_2"]["bilateral"]["zones"]
            ["outer_cortex"]["t2star"]["delta"]
        )
    except (KeyError, TypeError):
        oc_air = _zone_val("t2star", "air")
        oc_o2  = _zone_val("t2star", "oxygen_2")
        return round(oc_o2 - oc_air, 4) if (oc_air and oc_o2) else None

def _whole_kidney_median(cond):
    try:
        layers = analysis["conditions"][cond]["bilateral"]["layers"]
        vals = [
            l["t2star"]["median"] for l in layers
            if l.get("t2star", {}).get("median") is not None
        ]
        return float(np.nanmedian(vals)) if vals else None
    except (KeyError, TypeError):
        return None

report_values = {
    k: v for k, v in {
        "outer_cortex_t2star_oxygen1":     _zone_val("t2star", "oxygen_1"),
        "outer_cortex_t2star_air":         _zone_val("t2star", "air"),
        "outer_cortex_t2star_oxygen2":     _zone_val("t2star", "oxygen_2"),
        "outer_cortex_delta_t2star":       _delta(),
        "outer_cortex_t2star_std_oxygen2": _zone_val("t2star", "oxygen_2", "std"),
        "median_t2star_air":               _whole_kidney_median("air"),
        "median_t2star_oxygen2":           _whole_kidney_median("oxygen_2"),
        "mlco_layers":                     args.n_layers,
        "analysis_json":                   str(target_result),
        "pipeline_completed":              True,
    }.items() if v is not None
}

pm.pipestat.report(record_identifier=sid, values=report_values)
print(f"  pipestat: reported {len(report_values)} metrics")

# ── Done ───────────────────────────────────────────────────────────────────────

pm.stop_pipeline()
