"""
Tests for cluster_zones.py — feature extraction and k-means zone clustering.
"""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from cluster_zones import (
    _upsample_perfusion,
    extract_layer_features,
    cluster_layers,
    cluster_and_build_zones,
    DEPTH_DESCRIPTORS,
)


# ── helpers ────────────────────────────────────────────────────────────────

def make_simple_mask_and_maps(n_layers: int = 6, size: int = 20):
    """
    Build a square mask with `n_layers` concentric rings plus matching T2*/R2* maps.

    Layer 1 = outermost (high T2*), layer n_layers = innermost (low T2*).
    Each layer is an annulus of roughly equal width.
    """
    mask = np.zeros((size, size), dtype=np.int32)
    t2 = np.zeros((size, size), dtype=np.float32)
    r2 = np.zeros((size, size), dtype=np.float32)

    center = size // 2
    max_radius = center - 1

    for layer_idx in range(n_layers, 0, -1):
        radius = int(max_radius * layer_idx / n_layers)
        for y in range(size):
            for x in range(size):
                if (y - center) ** 2 + (x - center) ** 2 <= radius ** 2:
                    mask[y, x] = layer_idx

    # Assign T2* that decreases from cortex (outer, layer 1) to centre (inner)
    t2_values = np.linspace(25.0, 10.0, n_layers)  # outer→inner
    for i, t2_val in enumerate(t2_values, start=1):
        t2[mask == i] = t2_val
        r2[mask == i] = 1000.0 / t2_val

    return mask, t2, r2


# ── _upsample_perfusion ────────────────────────────────────────────────────

class TestUpsamplePerfusion:
    def test_no_op_same_shape(self):
        arr = np.ones((10, 10))
        result = _upsample_perfusion(arr, (10, 10))
        np.testing.assert_array_equal(result, arr)

    def test_upsample_doubles_size(self):
        arr = np.ones((5, 5))
        result = _upsample_perfusion(arr, (10, 10))
        assert result.shape == (10, 10)

    def test_upsample_preserves_values(self):
        arr = np.full((4, 4), 42.0)
        result = _upsample_perfusion(arr, (8, 8))
        assert np.allclose(result, 42.0, atol=0.1)

    def test_nan_handling(self):
        arr = np.ones((4, 4))
        arr[0, 0] = np.nan
        result = _upsample_perfusion(arr, (8, 8))
        # NaN region should remain NaN, non-NaN region should remain finite
        assert np.any(np.isnan(result))
        assert np.any(np.isfinite(result))


# ── extract_layer_features ─────────────────────────────────────────────────

class TestExtractLayerFeatures:
    def test_returns_correct_count(self):
        mask, t2, r2 = make_simple_mask_and_maps(n_layers=6)
        features = extract_layer_features(t2, r2, mask, n_layers=6)
        assert len(features) == 6

    def test_layer_idx_is_1indexed(self):
        mask, t2, r2 = make_simple_mask_and_maps(n_layers=4)
        features = extract_layer_features(t2, r2, mask, n_layers=4)
        indices = [f['layer_idx'] for f in features]
        assert indices == [1, 2, 3, 4]

    def test_depth_normalized_range(self):
        mask, t2, r2 = make_simple_mask_and_maps(n_layers=6)
        features = extract_layer_features(t2, r2, mask, n_layers=6)
        depths = [f['depth_normalized'] for f in features]
        assert depths[0] == pytest.approx(0.0)
        assert depths[-1] == pytest.approx(1.0)

    def test_t2star_median_matches_assigned_value(self):
        mask, t2, r2 = make_simple_mask_and_maps(n_layers=6)
        features = extract_layer_features(t2, r2, mask, n_layers=6)
        # Layer 1 should have t2* ~ 25 ms (outermost)
        assert features[0]['t2star_median'] == pytest.approx(25.0, abs=0.5)

    def test_empty_layer_has_nan_values(self):
        mask, t2, r2 = make_simple_mask_and_maps(n_layers=4)
        # Add a phantom layer 5 that has no pixels
        features = extract_layer_features(t2, r2, mask, n_layers=5)
        assert features[4]['n_pixels'] == 0
        assert np.isnan(features[4]['t2star_median'])

    def test_with_perfusion_map(self):
        mask, t2, r2 = make_simple_mask_and_maps(n_layers=4)
        perf = np.where(mask > 0, 1200.0, 0.0).astype(np.float32)
        features = extract_layer_features(t2, r2, mask, n_layers=4, perfusion_map=perf)
        assert not np.isnan(features[0]['perfusion_median'])
        assert features[0]['perfusion_median'] == pytest.approx(1200.0, abs=1.0)

    def test_perfusion_upsampled_if_smaller(self):
        mask, t2, r2 = make_simple_mask_and_maps(n_layers=4, size=20)
        # Perfusion at half resolution
        perf_small = np.full((10, 10), 800.0, dtype=np.float32)
        features = extract_layer_features(t2, r2, mask, n_layers=4, perfusion_map=perf_small)
        # Should not crash and should return valid perfusion values
        assert features[0]['perfusion_median'] == pytest.approx(800.0, abs=10.0)


# ── cluster_layers ─────────────────────────────────────────────────────────

class TestClusterLayers:
    def _get_features(self, n_layers=12):
        mask, t2, r2 = make_simple_mask_and_maps(n_layers=n_layers, size=30)
        return extract_layer_features(t2, r2, mask, n_layers=n_layers)

    def test_returns_expected_keys(self):
        features = self._get_features()
        result = cluster_layers(features, n_clusters=3)
        for key in ('labels', 'centroids', 'silhouette', 'model', 'feature_names'):
            assert key in result

    def test_label_count_matches_layers(self):
        features = self._get_features(n_layers=12)
        result = cluster_layers(features, n_clusters=3)
        assert len(result['labels']) == 12

    def test_n_unique_labels_equals_n_clusters(self):
        features = self._get_features(n_layers=12)
        result = cluster_layers(features, n_clusters=3)
        assert len(set(result['labels'])) == 3

    def test_silhouette_in_valid_range(self):
        features = self._get_features(n_layers=12)
        result = cluster_layers(features, n_clusters=3)
        assert -1.0 <= result['silhouette'] <= 1.0

    def test_too_few_valid_layers_raises(self):
        mask, t2, r2 = make_simple_mask_and_maps(n_layers=2)
        features = extract_layer_features(t2, r2, mask, n_layers=2)
        with pytest.raises(ValueError):
            cluster_layers(features, n_clusters=5)

    def test_reproducible_with_random_state(self):
        features = self._get_features(n_layers=12)
        r1 = cluster_layers(features, n_clusters=3, random_state=42)
        r2 = cluster_layers(features, n_clusters=3, random_state=42)
        np.testing.assert_array_equal(r1['labels'], r2['labels'])


# ── cluster_and_build_zones ────────────────────────────────────────────────

class TestClusterAndBuildZones:
    def test_returns_valid_zone_config(self):
        mask, t2, r2 = make_simple_mask_and_maps(n_layers=12, size=30)
        zone_config, diagnostics = cluster_and_build_zones(
            t2_map=t2, r2_map=r2, mlco_mask=mask, n_layers=12, n_clusters=3
        )
        assert 'metadata' in zone_config
        assert 'zones' in zone_config

    def test_zone_count_matches_n_clusters(self):
        mask, t2, r2 = make_simple_mask_and_maps(n_layers=12, size=30)
        zone_config, _ = cluster_and_build_zones(
            t2_map=t2, r2_map=r2, mlco_mask=mask, n_layers=12, n_clusters=3
        )
        assert len(zone_config['zones']) == 3

    def test_all_layers_assigned(self):
        mask, t2, r2 = make_simple_mask_and_maps(n_layers=12, size=30)
        zone_config, _ = cluster_and_build_zones(
            t2_map=t2, r2_map=r2, mlco_mask=mask, n_layers=12, n_clusters=3
        )
        assigned = []
        for zone in zone_config['zones'].values():
            assigned.extend(zone['layers'])
        assert sorted(assigned) == list(range(1, 13))

    def test_diagnostics_has_silhouette(self):
        mask, t2, r2 = make_simple_mask_and_maps(n_layers=12, size=30)
        _, diagnostics = cluster_and_build_zones(
            t2_map=t2, r2_map=r2, mlco_mask=mask, n_layers=12, n_clusters=3
        )
        assert 'cluster_info' in diagnostics
        assert 'silhouette' in diagnostics['cluster_info']

    def test_depth_descriptors_cover_expected_k(self):
        for k in (2, 3, 4, 5):
            assert k in DEPTH_DESCRIPTORS
            assert len(DEPTH_DESCRIPTORS[k]) == k
