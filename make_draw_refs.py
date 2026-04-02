"""
Generate contrast-enhanced reference images for ROI drawing.

Always generates the full set of draw variants for a sample:
  - Raw BOLD first echo for every condition (2 contrast treatments) — usually best
  - T2* maps for every available condition (3 contrast treatments each)
  - Perfusion map (if present)
  - T2 anatomical reference (2 contrast treatments)

Saves each as a .npy in processed/prepared/{sid}/ and writes a comparison
PNG to processed/analysis/{sid}_draw_options.png.

Usage:
    python make_draw_refs.py 173811 174227
    python make_draw_refs.py --all
"""
import argparse
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from pathlib import Path

try:
    import pydicom
    PYDICOM_OK = True
except ImportError:
    PYDICOM_OK = False

BASE = Path("/mnt/c/Users/srroj/Dropbox/Docs/Personal/Jason/Downloads/gomez/lucas/BOLD_MRI")
PREP = BASE / "processed/prepared"
OUT  = BASE / "processed/analysis"

CONDITIONS = ["oxygen_1", "air", "oxygen_2"]
COND_LABEL = {"oxygen_1": "pre-O₂", "air": "air", "oxygen_2": "post-O₂"}


# ── Contrast helpers ──────────────────────────────────────────────────────────

def pct_clip(img, lo=1, hi=99):
    lo_v, hi_v = np.percentile(img, lo), np.percentile(img, hi)
    return np.clip((img - lo_v) / (hi_v - lo_v + 1e-9), 0, 1)


def local_contrast(img, sigma=10):
    """Unsharp-mask local contrast enhancement."""
    bg = gaussian_filter(img.astype(np.float32), sigma=sigma)
    return pct_clip(img - 0.7 * bg, 1, 99)


def fixed_window(img, vmax):
    return np.clip(img.astype(np.float32) / vmax, 0, 1)


# ── Per-sample variant generation ─────────────────────────────────────────────

BOLD_COND_FILE_PATTERNS = {
    'oxygen_1': ['pre', 'Pre', 'PRE'],
    'air':      ['20%'],
    'oxygen_2': ['post', 'Post', 'POST'],
}


def _find_bold_dcm(data_dir: Path, cond: str) -> Path | None:
    """Find the raw BOLD DICOM for a given condition by filename keyword."""
    for kw in BOLD_COND_FILE_PATTERNS.get(cond, []):
        matches = list(data_dir.glob(f'*_BOLD_*_{kw}_*.dcm'))
        if matches:
            return matches[0]
    return None


def generate_draw_refs(sid: str) -> dict:
    """
    Generate all draw variants for a sample.
    Returns ordered dict of {display_label: (filename_stem, array)}.
    """
    prep_dir = PREP / sid
    data_dir = BASE / "data" / sid
    variants = {}  # label -> (stem, array)

    # ── Raw BOLD first-echo (placed first — usually clearest anatomy) ──────────
    if PYDICOM_OK and data_dir.exists():
        for cond in CONDITIONS:
            dcm_path = _find_bold_dcm(data_dir, cond)
            if dcm_path is None:
                continue
            try:
                ds = pydicom.dcmread(str(dcm_path))
                frames = ds.pixel_array.astype(np.float32)
                e1 = frames[0] if frames.ndim == 3 else frames
                lbl = COND_LABEL[cond]
                variants[f"BOLD {lbl}\necho 1 (lc)"]  = (f"bold_{cond}_echo1_lc",  local_contrast(e1, sigma=8))
                variants[f"BOLD {lbl}\necho 1 (pct)"] = (f"bold_{cond}_echo1_pct", pct_clip(e1, 2, 98))
            except Exception as e:
                print(f"  Warning: could not read BOLD DICOM for {cond}: {e}")

    # ── T2* maps — three conditions × three contrast treatments ────────────────
    for cond in CONDITIONS:
        t2_path = prep_dir / f"{sid}_{cond}_t2star_bruker.npy"
        if not t2_path.exists():
            continue
        t2 = np.load(t2_path).astype(np.float32)
        lbl = COND_LABEL[cond]
        variants[f"T2* {lbl}\nlocal contrast"] = (f"t2star_{cond}_lc",  local_contrast(t2, sigma=10))
        variants[f"T2* {lbl}\n0–30 ms window"] = (f"t2star_{cond}_w30", fixed_window(t2, 30))
        variants[f"T2* {lbl}\npct stretch"]    = (f"t2star_{cond}_pct", pct_clip(t2, 2, 98))

    # Perfusion map
    perf_path = prep_dir / f"{sid}_perfusion.npy"
    if perf_path.exists():
        perf = np.load(perf_path).astype(np.float32)
        variants["Perfusion\nlocal contrast"] = ("perf_lc",  local_contrast(perf, sigma=8))
        variants["Perfusion\npct stretch"]    = ("perf_pct", pct_clip(perf, 2, 98))

    # T2 anatomical reference (256×256)
    ref_path = prep_dir / f"{sid}_reference.npy"
    if ref_path.exists():
        ref = np.load(ref_path).astype(np.float32)
        variants["T2 reference\npct stretch"]    = ("ref_pct", pct_clip(ref, 2, 98))
        variants["T2 reference\nlocal contrast"] = ("ref_lc",  local_contrast(ref, sigma=20))

    # Save all variants as .npy
    for label, (stem, arr) in variants.items():
        np.save(prep_dir / f"{sid}_draw_{stem}.npy", arr.astype(np.float32))

    return variants


def make_comparison_figure(sid: str, variants: dict) -> Path:
    """Save a comparison PNG to processed/analysis/. Returns the path."""
    n = len(variants)
    ncols = min(n, 5)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.5 * ncols, 4 * nrows))
    axes = np.array(axes).flatten()

    for i, (label, (stem, arr)) in enumerate(variants.items()):
        ax = axes[i]
        ax.imshow(arr, cmap='gray', vmin=0, vmax=1, interpolation='nearest')
        ax.set_title(f"[{i+1}] {label}", fontsize=8)
        ax.set_xlabel(f"draw_{stem}", fontsize=6, color='gray')
        ax.set_xticks([]); ax.set_yticks([])

    for ax in axes[n:]:
        ax.set_visible(False)

    fig.suptitle(
        f"{sid} — draw reference options\n"
        f"Use the filename (draw_XXXXX) with roi_drawer.py --pick or directly as the image argument",
        fontsize=10, fontweight='bold'
    )
    plt.tight_layout()
    out_path = OUT / f"{sid}_draw_options.png"
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close()
    return out_path


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('samples', nargs='*', help='Sample IDs')
    parser.add_argument('--all', action='store_true', help='Process all prepared samples')
    args = parser.parse_args()

    if args.all:
        sids = [d.name for d in sorted(PREP.iterdir())
                if d.is_dir() and (d / f"{d.name}_reference.npy").exists()]
    else:
        sids = args.samples

    if not sids:
        parser.print_help()
        return

    OUT.mkdir(parents=True, exist_ok=True)

    for sid in sids:
        print(f"\n=== {sid} ===")
        prep_dir = PREP / sid
        if not prep_dir.exists():
            print(f"  prepared/{sid}/ not found, skipping")
            continue

        variants = generate_draw_refs(sid)
        if not variants:
            print("  No data found")
            continue

        fig_path = make_comparison_figure(sid, variants)
        print(f"  Saved {len(variants)} variants → prepared/{sid}/")
        print(f"  Comparison figure → analysis/{sid}_draw_options.png")
        print()
        print("  ROI drawing commands (pick the clearest image):")
        for i, (label, (stem, arr)) in enumerate(variants.items()):
            clean = label.replace('\n', ' ')
            print(f"    [{i+1}] {clean}")
            print(f"         python roi_drawer.py $BASE/processed/prepared/{sid}/{sid}_draw_{stem}.npy \\")
            print(f"             --output $BASE/processed/prepared/{sid}/{sid}_roi_mask.npy \\")
            print(f"             --regions left_kidney right_kidney --title \"{sid}\"")


if __name__ == '__main__':
    main()
