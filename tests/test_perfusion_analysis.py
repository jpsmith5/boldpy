"""
Tests for boldpy.analysis.perfusion_analysis — resample and per-layer stats.
(load_bruker_perfusion requires a real PvDatasets file and is not tested here.)
"""

import numpy as np
import pytest

from boldpy.analysis.perfusion_analysis import (
    resample_to_bold_resolution,
    analyze_perfusion_tlco_layers,
)


# ── resample_to_bold_resolution ────────────────────────────────────────────

class TestResampleToBoldResolution:
    def test_no_op_same_shape(self, capsys):
        arr = np.ones((10, 10), dtype=np.float32)
        result = resample_to_bold_resolution(arr, (10, 10))
        np.testing.assert_array_equal(result, arr)

    def test_upsample_shape(self):
        arr = np.ones((5, 5), dtype=np.float32) * 42.0
        result = resample_to_bold_resolution(arr, (10, 10))
        assert result.shape == (10, 10)

    def test_upsample_preserves_values(self):
        arr = np.full((4, 4), 100.0, dtype=np.float32)
        result = resample_to_bold_resolution(arr, (8, 8))
        assert np.allclose(result, 100.0, atol=0.5)

    def test_downsample_shape(self):
        arr = np.ones((20, 20), dtype=np.float32)
        result = resample_to_bold_resolution(arr, (10, 10))
        assert result.shape == (10, 10)

    def test_asymmetric_target_shape(self):
        arr = np.ones((4, 8), dtype=np.float32)
        result = resample_to_bold_resolution(arr, (8, 16))
        assert result.shape == (8, 16)

    def test_output_is_ndarray(self):
        arr = np.ones((5, 5), dtype=np.float32)
        result = resample_to_bold_resolution(arr, (10, 10))
        assert isinstance(result, np.ndarray)


# ── analyze_perfusion_tlco_layers ──────────────────────────────────────────

class TestAnalyzePerfusionTlcoLayers:
    @pytest.fixture
    def simple_inputs(self):
        """6-layer mask with uniform perfusion of 1200."""
        size = 30
        mask = np.zeros((size, size), dtype=np.int32)
        perf = np.zeros((size, size), dtype=np.float32)

        center = size // 2
        max_r = center - 1
        for i in range(6, 0, -1):
            r = int(max_r * i / 6)
            for y in range(size):
                for x in range(size):
                    if (y - center) ** 2 + (x - center) ** 2 <= r ** 2:
                        mask[y, x] = i
                        perf[y, x] = 1200.0

        return perf, mask

    def test_returns_dict(self, simple_inputs):
        perf, mask = simple_inputs
        result = analyze_perfusion_tlco_layers(perf, mask,
                                               kidney_start_layer=1,
                                               n_layers=6)
        assert isinstance(result, dict)

    def test_required_keys_present(self, simple_inputs):
        perf, mask = simple_inputs
        result = analyze_perfusion_tlco_layers(perf, mask,
                                               kidney_start_layer=1,
                                               n_layers=6)
        for key in ('kidney', 'n_layers', 'layers'):
            assert key in result

    def test_layer_count(self, simple_inputs):
        perf, mask = simple_inputs
        result = analyze_perfusion_tlco_layers(perf, mask,
                                               kidney_start_layer=1,
                                               n_layers=6)
        assert len(result['layers']) == 6

    def test_layer_stats_structure(self, simple_inputs):
        perf, mask = simple_inputs
        result = analyze_perfusion_tlco_layers(perf, mask,
                                               kidney_start_layer=1,
                                               n_layers=6)
        layer = result['layers'][0]
        assert 'layer' in layer
        assert 'perfusion' in layer
        assert 'median' in layer['perfusion']

    def test_perfusion_values_plausible(self, simple_inputs):
        perf, mask = simple_inputs
        result = analyze_perfusion_tlco_layers(perf, mask,
                                               kidney_start_layer=1,
                                               n_layers=6)
        for layer in result['layers']:
            assert layer['perfusion']['median'] == pytest.approx(1200.0, abs=1.0)

    def test_gradient_computed(self, simple_inputs):
        perf, mask = simple_inputs
        result = analyze_perfusion_tlco_layers(perf, mask,
                                               kidney_start_layer=1,
                                               n_layers=6)
        assert 'gradient' in result
        assert result['gradient'] is not None

    def test_layer_offset_respected(self, simple_inputs):
        perf, mask = simple_inputs
        # Make a second mask with layers starting at 7
        mask2 = np.where(mask > 0, mask + 6, 0).astype(np.int32)
        result = analyze_perfusion_tlco_layers(perf, mask2,
                                               kidney_start_layer=7,
                                               n_layers=6)
        assert len(result['layers']) == 6

    def test_zero_perfusion_pixels_skipped(self):
        size = 20
        mask = np.zeros((size, size), dtype=np.int32)
        perf = np.zeros((size, size), dtype=np.float32)
        mask[5:15, 5:15] = 1  # layer 1
        # Perfusion is all zero — should produce no valid stats
        result = analyze_perfusion_tlco_layers(perf, mask,
                                               kidney_start_layer=1,
                                               n_layers=1)
        assert len(result['layers']) == 0

    def test_kidney_label_in_result(self, simple_inputs):
        perf, mask = simple_inputs
        result = analyze_perfusion_tlco_layers(perf, mask,
                                               kidney_start_layer=1,
                                               n_layers=6,
                                               kidney_label='left_kidney')
        assert result['kidney'] == 'left_kidney'
