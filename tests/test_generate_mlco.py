"""
Tests for generate_mlco.py — MLCO mask generation from binary ROI masks.
"""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from generate_mlco import (
    detect_mask_type,
    split_mask,
    generate_mlco_layers,
    generate_bilateral_mlco_layers,
)


# ── helpers ────────────────────────────────────────────────────────────────

def make_circle_mask(size: int = 40, radius: int = 15, center=None) -> np.ndarray:
    """Binary mask with a single filled circle."""
    if center is None:
        center = (size // 2, size // 2)
    mask = np.zeros((size, size), dtype=bool)
    for y in range(size):
        for x in range(size):
            if (y - center[0]) ** 2 + (x - center[1]) ** 2 <= radius ** 2:
                mask[y, x] = True
    return mask


def make_bilateral_circle_mask(size: int = 60, radius: int = 10) -> np.ndarray:
    """Binary mask with two well-separated circles (left and right of centre)."""
    mask = np.zeros((size, size), dtype=bool)
    cy = size // 2
    # Left circle at 1/4 of width, right circle at 3/4 — gap of size//2 - 2*radius
    for y in range(size):
        for x in range(size):
            if (y - cy) ** 2 + (x - size // 4) ** 2 <= radius ** 2:
                mask[y, x] = True
    for y in range(size):
        for x in range(size):
            if (y - cy) ** 2 + (x - 3 * size // 4) ** 2 <= radius ** 2:
                mask[y, x] = True
    return mask


# ── detect_mask_type ───────────────────────────────────────────────────────

class TestDetectMaskType:
    def test_single_circle_is_binary(self):
        mask = make_circle_mask().astype(np.uint8)
        info = detect_mask_type(mask)
        assert info['type'] == 'binary'
        assert not info['is_multi_region']

    def test_bilateral_circles_detected(self):
        mask = make_bilateral_circle_mask().astype(np.uint8)
        info = detect_mask_type(mask)
        assert info['type'] == 'bilateral'
        assert info['is_bilateral']

    def test_empty_mask_raises(self):
        mask = np.zeros((10, 10), dtype=np.uint8)
        with pytest.raises(ValueError, match='empty'):
            detect_mask_type(mask)

    def test_multi_label_mask(self):
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[1:4, 1:4] = 1
        mask[6:9, 6:9] = 2
        info = detect_mask_type(mask)
        assert info['type'] == 'multi-region'
        assert info['is_multi_region']
        assert info['n_regions'] == 2


# ── split_mask ─────────────────────────────────────────────────────────────

class TestSplitMask:
    def test_splits_into_two_components(self):
        mask = make_bilateral_circle_mask().astype(np.uint8)
        result = split_mask(mask, apply_mri_flip=False)
        assert len(result) == 2

    def test_component_names_respected(self):
        mask = make_bilateral_circle_mask().astype(np.uint8)
        result = split_mask(mask, component_names=('alpha', 'beta'), apply_mri_flip=False)
        assert 'alpha' in result
        assert 'beta' in result

    def test_components_are_disjoint(self):
        mask = make_bilateral_circle_mask().astype(np.uint8)
        result = split_mask(mask, apply_mri_flip=False)
        keys = list(result.keys())
        overlap = result[keys[0]] & result[keys[1]]
        assert not np.any(overlap)

    def test_components_cover_full_mask(self):
        mask = make_bilateral_circle_mask().astype(np.uint8)
        result = split_mask(mask, apply_mri_flip=False)
        keys = list(result.keys())
        combined = result[keys[0]] | result[keys[1]]
        np.testing.assert_array_equal(combined, mask.astype(bool))

    def test_mri_flip_swaps_assignment(self):
        mask = make_bilateral_circle_mask().astype(np.uint8)
        no_flip = split_mask(mask, component_names=('left', 'right'), apply_mri_flip=False)
        with_flip = split_mask(mask, component_names=('left', 'right'), apply_mri_flip=True)
        # With MRI flip, left and right should be swapped
        np.testing.assert_array_equal(no_flip['left'], with_flip['right'])
        np.testing.assert_array_equal(no_flip['right'], with_flip['left'])

    def test_single_component_raises(self):
        mask = make_circle_mask().astype(np.uint8)
        with pytest.raises(ValueError):
            split_mask(mask, min_size=1)

    def test_min_size_filters_small_components(self):
        mask = make_bilateral_circle_mask().astype(np.uint8)
        # min_size larger than any component should raise
        with pytest.raises(ValueError):
            split_mask(mask, min_size=99999)


# ── generate_mlco_layers ───────────────────────────────────────────────────

class TestGenerateMlcoLayers:
    @pytest.fixture
    def circle_mask(self):
        return make_circle_mask(size=40, radius=15)

    def test_output_shape_matches_input(self, circle_mask):
        layer_mask = generate_mlco_layers(circle_mask, n_layers=6)
        assert layer_mask.shape == circle_mask.shape

    def test_background_is_zero(self, circle_mask):
        layer_mask = generate_mlco_layers(circle_mask, n_layers=6)
        assert np.all(layer_mask[~circle_mask] == 0)

    def test_all_foreground_pixels_assigned(self, circle_mask):
        layer_mask = generate_mlco_layers(circle_mask, n_layers=6)
        assert np.all(layer_mask[circle_mask] > 0)

    def test_layer_count_matches_n_layers(self, circle_mask):
        n = 6
        layer_mask = generate_mlco_layers(circle_mask, n_layers=n)
        unique = set(np.unique(layer_mask)) - {0}
        assert unique == set(range(1, n + 1))

    def test_layer_1_is_outermost(self, circle_mask):
        layer_mask = generate_mlco_layers(circle_mask, n_layers=6)
        # Layer 1 pixels should be closer to edge than layer 6 pixels
        from scipy.ndimage import distance_transform_edt
        dist = distance_transform_edt(circle_mask)
        mean_dist_l1 = dist[layer_mask == 1].mean()
        mean_dist_l6 = dist[layer_mask == 6].mean()
        assert mean_dist_l1 < mean_dist_l6

    def test_erosion_method_works(self, circle_mask):
        layer_mask = generate_mlco_layers(circle_mask, n_layers=4, method='erosion')
        unique = set(np.unique(layer_mask)) - {0}
        assert len(unique) >= 1  # at least one layer assigned
        assert np.all(layer_mask[~circle_mask] == 0)

    def test_24_layers_requested(self):
        # Use a large enough mask so all 24 layers have sufficient depth
        large_mask = make_circle_mask(size=120, radius=50)
        layer_mask = generate_mlco_layers(large_mask, n_layers=24)
        unique = set(np.unique(layer_mask)) - {0}
        assert len(unique) == 24


# ── generate_bilateral_mlco_layers ────────────────────────────────────────

class TestGenerateBilateralMlcoLayers:
    @pytest.fixture
    def bilateral_setup(self):
        bilateral = make_bilateral_circle_mask(size=50, radius=10)
        components = split_mask(bilateral.astype(np.uint8),
                                component_names=('left', 'right'),
                                apply_mri_flip=False)
        return bilateral, components['left'], components['right']

    def test_output_shape(self, bilateral_setup):
        bilateral, left, right = bilateral_setup
        result = generate_bilateral_mlco_layers(bilateral, left, right, n_layers=6)
        assert result.shape == bilateral.shape

    def test_right_layers_numbered_first(self, bilateral_setup):
        bilateral, left, right = bilateral_setup
        result = generate_bilateral_mlco_layers(bilateral, left, right, n_layers=6)
        right_vals = set(np.unique(result[right])) - {0}
        assert right_vals == set(range(1, 7))

    def test_left_layers_offset(self, bilateral_setup):
        bilateral, left, right = bilateral_setup
        result = generate_bilateral_mlco_layers(bilateral, left, right, n_layers=6)
        left_vals = set(np.unique(result[left])) - {0}
        assert left_vals == set(range(7, 13))

    def test_components_do_not_overlap(self, bilateral_setup):
        bilateral, left, right = bilateral_setup
        result = generate_bilateral_mlco_layers(bilateral, left, right, n_layers=6)
        right_pixels = (result >= 1) & (result <= 6)
        left_pixels = (result >= 7) & (result <= 12)
        assert not np.any(right_pixels & left_pixels)

    def test_all_foreground_assigned(self, bilateral_setup):
        bilateral, left, right = bilateral_setup
        result = generate_bilateral_mlco_layers(bilateral, left, right, n_layers=6)
        assert np.all(result[bilateral] > 0)
