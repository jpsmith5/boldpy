#!/usr/bin/env python3
"""
BoldPy Collator — Project-Level Pipeline
=========================================

Runs cross-sample group analysis after all per-sample pipelines complete.
Invoked by looper via ``looper runp project_config.yaml``.

Three sequential steps, each with PyPiper checkpointing:

  Step 1 — Group profile comparison  (group_analysis.py)
            Target: {output_dir}/group_comparison_stats.json

  Step 2 — Overlay analysis          (overlay_analysis.py)
            Target: {output_dir}/kmeans_zone_analysis.pdf

  Step 3 — Heterogeneity analysis    (heterogeneity.py)
            Target: {output_dir}/heterogeneity/heterogeneity_statistics.json

Each step is skipped if its target file already exists, enabling resume.
Individual scripts can still be run standalone with ``--pep`` for
interactive / partial re-runs.

Architecture
------------
This script is the project-level analogue of boldpy_pipeline.py:

  Sample pipeline  (boldpy_pipeline.py)  →  per-sample: prepare → ROI → MLCO → analyze
  Project pipeline (boldpy_collator.py)  →  cross-sample: group comparisons + overlays

Corresponds to the PEPATAC collator pattern.

Usage (via looper — recommended):
    looper runp project_config.yaml
    looper runp project_config.yaml --compute slurm

Usage (direct):
    python pipeline/boldpy_collator.py \\
        --pep code/analysis/captopril/project_config.yaml \\
        --output-dir processed/analysis/captopril_collator/
"""

import argparse
import sys
from pathlib import Path

import peppy
import pypiper

# ── Argument parsing ───────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(
    description="BoldPy project-level collator (group comparisons + overlays)"
)
parser.add_argument("--pep", required=True,
                    help="Path to PEP project_config.yaml")
parser.add_argument("--output-dir", default=None,
                    help="Override output directory for collator logs (default: from pep config)")

parser = pypiper.add_pypiper_args(parser, groups=["pypiper", "looper"])
args = parser.parse_args()

# ── Resolve paths ──────────────────────────────────────────────────────────────

pep_path    = Path(args.pep).resolve()
script_root = Path(__file__).resolve().parent.parent   # boldpy_v3.0.0/

# Load project config to derive output_dir and group-level output targets
project    = peppy.Project(str(pep_path))
cfg        = project.config
pep_base   = pep_path.parent

def _resolve(key, default):
    val = cfg.get(key)
    if not val:
        return default
    p = Path(val)
    return p if p.is_absolute() else pep_base / p

group_output_dir = _resolve('output_dir', pep_base / 'processed' / 'analysis' / 'group_comparison')

# Collator log dir: either --output-dir arg or a subdir of group output
if args.output_dir:
    collator_outdir = Path(args.output_dir).resolve()
else:
    collator_outdir = group_output_dir / '_collator_logs'

collator_outdir.mkdir(parents=True, exist_ok=True)

# ── PipelineManager ────────────────────────────────────────────────────────────

pm = pypiper.PipelineManager(
    name="boldpy_collator",
    outfolder=str(collator_outdir),
    pipestat_pipeline_type="project",
    pipestat_record_identifier="summary",
    args=args,
)

pep_str = str(pep_path)

# ── Step 1: Group profile comparison ──────────────────────────────────────────

pm.run(
    cmd=(
        f"python '{script_root / 'group_analysis.py'}'"
        f" --pep '{pep_str}'"
    ),
    target=str(group_output_dir / "group_comparison_stats.json"),
    name="group_analysis",
)

# ── Step 2: K-means + MLCO overlay analysis ────────────────────────────────────

pm.run(
    cmd=(
        f"python '{script_root / 'overlay_analysis.py'}'"
        f" --pep '{pep_str}'"
    ),
    target=str(group_output_dir / "kmeans_zone_analysis.pdf"),
    name="overlay_analysis",
)

# ── Step 3: Heterogeneity analysis ─────────────────────────────────────────────

pm.run(
    cmd=(
        f"python '{script_root / 'heterogeneity.py'}'"
        f" --pep '{pep_str}'"
    ),
    target=str(group_output_dir / "heterogeneity" / "heterogeneity_statistics.json"),
    name="heterogeneity",
)

# ── Done ───────────────────────────────────────────────────────────────────────

pm.stop_pipeline()
