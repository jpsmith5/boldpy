#!/usr/bin/env python3
"""
BoldPy Collate — Project-Level Analysis Pipeline
=================================================

Invoked by looper after all per-sample pipelines complete:

    looper runp -c looper_config.yaml

Runs three project-level analyses in sequence:

  Step 1 — group_analysis.py     Group comparison plots + statistics
  Step 2 — overlay_analysis.py   K-means + MLCO overlays + zone analysis
  Step 3 — heterogeneity.py      Within-layer heterogeneity + focal disruption

Each step is checkpointed: skipped if its primary output already exists.
"""

import argparse
import os
import sys
from pathlib import Path

# ── sys.path: allow importing from the parent boldpy_v3.0.0/ directory ────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import pypiper
import group_analysis
import overlay_analysis
import heterogeneity

# ── Argument parsing ───────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(
    description="BoldPy project-level collation pipeline"
)
parser.add_argument("--pep", required=True,
                    help="Path to project_config.yaml")
parser.add_argument("--output-dir", required=True,
                    help="Output directory for pipeline logs")

parser = pypiper.add_pypiper_args(parser, groups=["pypiper", "looper"])
args = parser.parse_args()

# ── Paths ──────────────────────────────────────────────────────────────────────

pep_path    = Path(args.pep).resolve()

# Resolve group output dir from PEP config (used for checkpoint targets)
with open(pep_path) as fh:
    pep_cfg = yaml.safe_load(fh)

group_out = (pep_path.parent / pep_cfg.get(
    "group_output_dir", "../../../processed/analysis/group_comparison"
)).resolve()

# ── PipelineManager ────────────────────────────────────────────────────────────

pm = pypiper.PipelineManager(
    name="boldpy_collate",
    outfolder=args.output_dir,
    args=args,
)

# ── Step 1: Group comparison ───────────────────────────────────────────────────

target_group = group_out / "group_comparison_stats.json"
if not target_group.exists():
    pm.timestamp("### Step 1: Group comparison")
    group_analysis.run(pep_path=str(pep_path))
else:
    pm.timestamp("### Step 1: Group comparison [SKIPPED — target exists]")

# ── Step 2: Overlay analysis ───────────────────────────────────────────────────

target_overlay = group_out / "kmeans_zone_analysis.png"
if not target_overlay.exists():
    pm.timestamp("### Step 2: Overlay analysis")
    overlay_analysis.run(pep_path=str(pep_path))
else:
    pm.timestamp("### Step 2: Overlay analysis [SKIPPED — target exists]")

# ── Step 3: Heterogeneity ──────────────────────────────────────────────────────

target_hetero = group_out / "heterogeneity" / "heterogeneity_statistics.json"
if not target_hetero.exists():
    pm.timestamp("### Step 3: Heterogeneity analysis")
    heterogeneity.run(pep_path=str(pep_path))
else:
    pm.timestamp("### Step 3: Heterogeneity analysis [SKIPPED — target exists]")

# ── Done ───────────────────────────────────────────────────────────────────────

pm.stop_pipeline()
