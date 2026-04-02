#!/usr/bin/env python3
"""
BoldPy Prepare — Data Extraction Pipeline (Step 0)
====================================================

Extracts T2*, R2*, and perfusion maps from raw MRI data for one sample.
Invoked by looper as the first of two sample-level pipelines:

    looper run project_config.yaml \\
        --pipeline-interfaces pipeline/boldpy_prepare_interface.yaml

Run this for ALL samples before drawing any ROIs. On completion it prints
the exact ``roi_drawer.py`` command for this sample so you can copy-paste it.

After drawing all ROIs, run the main pipeline:

    looper run project_config.yaml \\
        --pipeline-interfaces pipeline/boldpy_pipeline_interface.yaml

Checkpoint target:
    {output_dir}/prepared/{sample_name}_air_t2star_custom.npy

If the target exists, this pipeline is a no-op (already prepared).
"""

import argparse
import os
import sys
from pathlib import Path

# ── sys.path: allow importing from the parent boldpy_v3.0.0/ directory ────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pypiper
import prepare_data
import prepare_dicom

# ── Argument parsing ───────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(
    description="BoldPy data extraction — Step 0 only"
)
parser.add_argument("--sample-name", required=True,
                    help="Unique sample identifier")
parser.add_argument("--data-format", required=True, choices=["dicom", "bruker"],
                    help="Raw data format: dicom or bruker (PvDatasets)")
parser.add_argument("--scan-dir", required=True,
                    help="Root directory for raw scan data")
parser.add_argument("--output-dir", default=None,
                    help="Per-sample base output directory (default: same as --prepared-dir)")
parser.add_argument("--prepared-dir", default=None,
                    help="Directory for prepared output files (default: --output-dir/prepared)")
parser.add_argument("--n-layers", type=int, default=24,
                    help="Number of MLCO layers (for downstream reference; default: 24)")
# Optional per-condition scan path overrides
parser.add_argument("--scan-oxygen1", default="",
                    help="Per-condition PvDatasets subpath within scan-dir (oxygen_1)")
parser.add_argument("--scan-air",     default="",
                    help="Per-condition PvDatasets subpath within scan-dir (air)")
parser.add_argument("--scan-oxygen2", default="",
                    help="Per-condition PvDatasets subpath within scan-dir (oxygen_2)")

parser = pypiper.add_pypiper_args(parser, groups=["pypiper", "looper"])
args = parser.parse_args()

# ── Paths ──────────────────────────────────────────────────────────────────────

sid         = args.sample_name
script_root = Path(__file__).resolve().parent.parent   # boldpy_v3.0.0/

if args.prepared_dir:
    prep_dir = Path(args.prepared_dir).resolve()
    out      = Path(args.output_dir).resolve() if args.output_dir else prep_dir
elif args.output_dir:
    out      = Path(args.output_dir).resolve()
    prep_dir = out / "prepared"
else:
    raise ValueError("At least one of --output-dir or --prepared-dir is required")

# ── PipelineManager ────────────────────────────────────────────────────────────

pm = pypiper.PipelineManager(
    name="boldpy_prepare",
    outfolder=str(out),
    args=args,
    pipestat_record_identifier=sid,
)

# ── Step 0: Data extraction ────────────────────────────────────────────────────

per_condition = (
    args.data_format == "bruker"
    and args.scan_oxygen1
    and args.scan_air
    and args.scan_oxygen2
)

if args.data_format == "dicom":
    # DICOM: one call covers all conditions via auto-detection
    target = str(prep_dir / f"{sid}_air_t2star_bruker.npy")
    if not os.path.exists(target):
        pm.timestamp("### Step 0: DICOM data extraction")
        prepare_dicom.run(
            scan_dir=args.scan_dir,
            sample_name=sid,
            output_dir=str(prep_dir),
        )
    else:
        pm.timestamp("### Step 0: DICOM data extraction [SKIPPED — target exists]")

elif per_condition:
    # Bruker with explicit per-condition PvDatasets paths
    for cond, subpath in [
        ("oxygen_1", args.scan_oxygen1),
        ("air",      args.scan_air),
        ("oxygen_2", args.scan_oxygen2),
    ]:
        target = str(prep_dir / f"{sid}_{cond}_t2star_bruker.npy")
        if not os.path.exists(target):
            pm.timestamp(f"### Step 0: Bruker extraction — {cond}")
            prepare_data.run(
                scan_dir=args.scan_dir,
                sample_name=f"{sid}_{cond}",
                output_dir=str(prep_dir),
                scan_oxygen1="",
                scan_air="",
                scan_oxygen2="",
                extract_bruker=True,
                extract_custom=False,
            )
        else:
            pm.timestamp(f"### Step 0: Bruker extraction — {cond} [SKIPPED — target exists]")

else:
    # Bruker batch: process whole directory — uses both-t2star (custom fit)
    target = str(prep_dir / f"{sid}_air_t2star_custom.npy")
    if not os.path.exists(target):
        pm.timestamp("### Step 0: Bruker batch extraction")
        prepare_data.run(
            scan_dir=args.scan_dir,
            sample_name=sid,
            output_dir=str(prep_dir),
            extract_bruker=True,
            extract_custom=True,
        )
    else:
        pm.timestamp("### Step 0: Bruker batch extraction [SKIPPED — target exists]")

# ── Print ROI drawing instructions ─────────────────────────────────────────────

roi_path = prep_dir / f"{sid}_roi_mask.npy"
ref_path = prep_dir / f"{sid}_air_reference.npy"
if not ref_path.exists():
    ref_path = prep_dir / f"{sid}_reference.npy"   # prepare_dicom.py naming

print(f"\n{'─'*60}")
print(f"  Prepared: {sid}")
print(f"  Prepared dir: {prep_dir}")
if roi_path.exists():
    print(f"  ROI already exists: {roi_path}")
else:
    print(f"\n  Draw ROI next:")
    print(f"    cd {script_root}")
    print(f"    python roi_drawer.py \\")
    print(f"        --pick '{prep_dir}' \\")
    print(f"        --output '{roi_path}' \\")
    print(f"        --regions left_kidney right_kidney --title '{sid}'")
print(f"{'─'*60}\n")

# ── Done ───────────────────────────────────────────────────────────────────────

pm.stop_pipeline()
