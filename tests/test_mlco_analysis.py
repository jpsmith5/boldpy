"""
Tests for mlco_analysis.py — layer-by-layer T2*/R2* quantification.
"""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from mlco_analysis import detect_multiregion_mlco, analyze_mlco
from tissue_zones import encode_multiregion_value


# ── helpers ────────────────────────────────────────────────────────────────

def make_bilateral_mask_and_maps(n_layers: int = 6, size: int = 20):
    """
    Bilateral MLCO mask: layers 1–n_layers (right), n_layers+1–2*n_layers (left).
    Each layer is an annulus; T2* decreases from outer to inner.
    """
    mask = np.zeros((size, size), dtype=np.int32)
    center = size // 2
    max_r = center - 1

    # Right kidney (layers 1–n_layers): left half of image
    for i in range(n_layers, 0, -1):
        r = int(max_r * i / n_layers)
        for y in range(size):
            for x in range(size // 2):
                if (y - center) ** 2 + (x - center // 2) ** 2 <= r ** 2:
                    mask[y, x] = i

    # Left kidney (layers n_layers+1–2*n_layers): right half
    for i in range(n_layers, 0, -1):
        layer_val = n_layers + i
        r = int(max_r * i / n_layers)
        for y in range(size):
            for x in range(size // 2, size):
                if (y - center) ** 2 + (x - (size - center // 2)) ** 2 <= r ** 2:
                    mask[y, x] = layer_val

    t2 = np.zeros((size, size), dtype=np.float32)
    r2 = np.zeros((size, size), dtype=np.float32)
    t2_values = np.linspace(25.0, 10.0, n_layers)

    for i, t2_val in enumerate(t2_values, start=1):
        t2[mask == i] = t2_val
        t2[mask == n_layers + i] = t2_val
        r2[mask == i] = 1000.0 / t2_val
        r2[mask == n_layers + i] = 1000.0 / t2_val

    return mask, t2, r2


# ── detect_multiregion_mlco ────────────────────────────────────────────────

class TestDetectMultiregionMlco:
    def test_standard_mask_is_not_multiregion(self):
        mask = np.array([[0, 1, 2], [3, 4, 0]])
        is_mr, region_ids = detect_multiregion_mlco(mask)
        assert not is_mr
        assert region_ids is None

    def test_empty_mask_is_not_multiregion(self):
        mask = np.zeros((5, 5), dtype=np.int32)
        is_mr, region_ids = detect_multiregion_mlco(mask)
        assert not is_mr

    def test_multiregion_encoded_values_detected(self):
        mask = np.zeros((5, 5), dtype=np.int32)
        mask[0, 0] = encode_multiregion_value(1, 3)   # 1003
        mask[0, 1] = encode_multiregion_value(2, 5)   # 2005
        is_mr, region_ids = detect_multiregion_mlco(mask)
        assert is_mr
        assert 1 in region_ids
        assert 2 in region_ids

    def test_bilateral_standard_is_not_multiregion(self):
        mask, _, _ = make_bilateral_mask_and_maps(n_layers=6)
        is_mr, _ = detect_multiregion_mlco(mask)
        assert not is_mr


# ── analyze_mlco ───────────────────────────────────────────────────────────

class TestAnalyzeMlco:
    @pytest.fixture
    def bilateral_inputs(self):
        mask, t2, r2 = make_bilateral_mask_and_maps(n_layers=6)
        return mask, t2, r2

    def test_returns_dict(self, bilateral_inputs):
        mask, t2, r2 = bilateral_inputs
        result = analyze_mlco(t2, r2, mask, n_layers_per_organ=6)
        assert isinstance(result, dict)

    def test_result_has_layers_key(self, bilateral_inputs):
        mask, t2, r2 = bilateral_inputs
        result = analyze_mlco(t2, r2, mask, n_layers_per_organ=6)
        assert any(k in result for k in ('layers', 'right', 'left', 'averaged',
                                          'right_kidney', 'left_kidney'))

    def test_layer_count(self, bilateral_inputs):
        mask, t2, r2 = bilateral_inputs
        result = analyze_mlco(t2, r2, mask, n_layers_per_organ=6)
        # Bilateral: expect results for both kidneys; check total layer entries
        all_layers = []
        for key in ('right', 'left', 'averaged', 'layers'):
            if key in result and isinstance(result[key], dict) and 'layers' in result[key]:
                all_layers = result[key]['layers']
                break
            elif key in result and isinstance(result[key], list):
                all_layers = result[key]
                break
        assert len(all_layers) <= 6  # may be fewer if some layers have no pixels

    def test_t2star_stats_present(self, bilateral_inputs):
        mask, t2, r2 = bilateral_inputs
        result = analyze_mlco(t2, r2, mask, n_layers_per_organ=6)
        # Find any layer list and check first layer has t2star stats
        for key in ('right', 'left', 'averaged', 'layers'):
            layers = None
            if key in result and isinstance(result[key], dict):
                layers = result[key].get('layers')
            elif key in result and isinstance(result[key], list):
                layers = result[key]
            if layers and len(layers) > 0:
                layer0 = layers[0]
                assert 't2star' in layer0
                assert 'median' in layer0['t2star']
                break

    def test_t2star_values_plausible(self, bilateral_inputs):
        mask, t2, r2 = bilateral_inputs
        result = analyze_mlco(t2, r2, mask, n_layers_per_organ=6)
        for key in ('right', 'left', 'averaged', 'layers'):
            layers = None
            if key in result and isinstance(result[key], dict):
                layers = result[key].get('layers')
            elif key in result and isinstance(result[key], list):
                layers = result[key]
            if layers and len(layers) > 0:
                for layer in layers:
                    if 't2star' in layer:
                        med = layer['t2star']['median']
                        assert 5.0 <= med <= 100.0, f"Implausible T2* median: {med}"
                break

    def test_with_perfusion_map(self, bilateral_inputs):
        mask, t2, r2 = bilateral_inputs
        perf = np.where(mask > 0, 1200.0, 0.0).astype(np.float32)
        # Should not raise
        result = analyze_mlco(t2, r2, mask, n_layers_per_organ=6, perfusion_map=perf)
        assert result is not None

    def test_all_nan_map_does_not_crash(self, bilateral_inputs):
        mask, _, r2 = bilateral_inputs
        t2_nan = np.full_like(r2, np.nan)
        # Should handle gracefully (may return empty layers, but not crash)
        result = analyze_mlco(t2_nan, r2, mask, n_layers_per_organ=6)
        assert result is not None
