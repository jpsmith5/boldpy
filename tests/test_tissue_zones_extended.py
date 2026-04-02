"""
Tests for tissue_zones.py — tissue classification, zone lookup, effect size,
update_configs_from_dict, and interpret functions.
"""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import tissue_zones
from tissue_zones import (
    get_zone_name,
    get_zone_layers,
    classify_tissue_viability,
    calculate_tissue_quality,
    calculate_effect_size,
    interpret_effect_size,
    interpret_tissue_state,
    update_configs_from_dict,
    load_zone_config,
)


# ── get_zone_name ──────────────────────────────────────────────────────────

class TestGetZoneName:
    def test_layer_1_is_outer_cortex(self):
        assert get_zone_name(1) == 'outer_cortex'

    def test_layer_5_is_outer_cortex(self):
        assert get_zone_name(5) == 'outer_cortex'

    def test_layer_6_is_inner_cortex(self):
        assert get_zone_name(6) == 'inner_cortex'

    def test_layer_11_is_cmj(self):
        assert get_zone_name(11) == 'cmj'

    def test_layer_14_is_outer_medulla(self):
        assert get_zone_name(14) == 'outer_medulla'

    def test_layer_20_is_inner_medulla(self):
        assert get_zone_name(20) == 'inner_medulla'

    def test_layer_24_is_inner_medulla(self):
        assert get_zone_name(24) == 'inner_medulla'

    def test_unknown_layer_returns_unknown(self):
        assert get_zone_name(99) == 'unknown'

    def test_custom_config_respected(self):
        custom = {
            'metadata': {'n_layers': 4},
            'zones': {
                'zone_a': {'layers': [1, 2]},
                'zone_b': {'layers': [3, 4]},
            }
        }
        assert get_zone_name(1, zone_config=custom) == 'zone_a'
        assert get_zone_name(3, zone_config=custom) == 'zone_b'


# ── get_zone_layers ────────────────────────────────────────────────────────

class TestGetZoneLayers:
    def test_outer_cortex_range(self):
        layers = get_zone_layers('outer_cortex')
        assert 1 in layers
        assert 5 in layers

    def test_inner_medulla_range(self):
        layers = get_zone_layers('inner_medulla')
        assert 24 in layers

    def test_unknown_zone_raises(self):
        with pytest.raises(ValueError):
            get_zone_layers('nonexistent_zone')

    def test_returns_range_type(self):
        result = get_zone_layers('outer_cortex')
        assert isinstance(result, range)


# ── classify_tissue_viability ──────────────────────────────────────────────

class TestClassifyTissueViability:
    def test_normal_t2star_is_viable(self):
        result = classify_tissue_viability(t2star=12.0, region='cortex')
        assert result == 'viable'

    def test_very_high_t2star_is_necrosis(self):
        # T2* >100ms is clearly necrotic/fluid
        result = classify_tissue_viability(t2star=150.0, region='cortex')
        assert 'necrosis' in result

    def test_moderately_elevated_t2star_is_edema(self):
        result = classify_tissue_viability(t2star=45.0, region='cortex')
        assert result == 'suspect_edema'

    def test_necrosis_with_low_perfusion_is_high_confidence(self):
        result = classify_tissue_viability(t2star=150.0, perfusion=20.0, region='cortex')
        assert 'high_conf' in result

    def test_all_regions_return_valid_string(self):
        for region in ('cortex', 'cmj', 'medulla'):
            result = classify_tissue_viability(t2star=15.0, region=region)
            assert isinstance(result, str)
            assert len(result) > 0


# ── calculate_tissue_quality ───────────────────────────────────────────────

class TestCalculateTissueQuality:
    def _make_viable_region(self, size=20):
        t2 = np.full((size, size), 12.0, dtype=np.float32)
        mask = np.ones((size, size), dtype=bool)
        return t2, mask

    def test_all_viable_tissue(self):
        t2, mask = self._make_viable_region()
        quality = calculate_tissue_quality(t2, mask)
        assert quality['viable_pct'] == pytest.approx(100.0)
        assert quality['likely_necrosis_pct'] == pytest.approx(0.0)

    def test_all_necrotic_tissue(self):
        t2 = np.full((10, 10), 200.0, dtype=np.float32)
        mask = np.ones((10, 10), dtype=bool)
        quality = calculate_tissue_quality(t2, mask, region='cortex')
        assert quality['likely_necrosis_pct'] > 0.0

    def test_empty_mask_returns_zeros(self):
        t2 = np.zeros((5, 5), dtype=np.float32)
        mask = np.zeros((5, 5), dtype=bool)
        quality = calculate_tissue_quality(t2, mask)
        assert quality['viable_pct'] == 0
        assert quality['n_pixels'] == 0

    def test_required_keys_present(self):
        t2, mask = self._make_viable_region()
        quality = calculate_tissue_quality(t2, mask)
        for key in ('viable_pct', 'suspect_edema_pct', 'likely_necrosis_pct',
                    'n_pixels', 'tissue_quality_score'):
            assert key in quality

    def test_percentages_sum_to_100(self):
        t2 = np.array([[12.0, 45.0], [150.0, 15.0]], dtype=np.float32)
        mask = np.ones((2, 2), dtype=bool)
        quality = calculate_tissue_quality(t2, mask)
        total = quality['viable_pct'] + quality['suspect_edema_pct'] + quality['likely_necrosis_pct']
        assert total == pytest.approx(100.0, abs=1.0)

    def test_tissue_quality_score_between_0_and_1(self):
        t2, mask = self._make_viable_region()
        quality = calculate_tissue_quality(t2, mask)
        assert 0.0 <= quality['tissue_quality_score'] <= 1.0


# ── calculate_effect_size / interpret_effect_size ──────────────────────────

class TestEffectSize:
    def test_identical_groups_gives_zero(self):
        d = calculate_effect_size(10.0, 2.0, 10.0, 2.0)
        assert d == pytest.approx(0.0)

    def test_sign_positive_when_group2_higher(self):
        d = calculate_effect_size(10.0, 2.0, 15.0, 2.0)
        assert d > 0

    def test_sign_negative_when_group2_lower(self):
        d = calculate_effect_size(15.0, 2.0, 10.0, 2.0)
        assert d < 0

    def test_zero_std_returns_zero(self):
        d = calculate_effect_size(10.0, 0.0, 15.0, 0.0)
        assert d == 0.0

    def test_large_difference_gives_large_d(self):
        d = calculate_effect_size(10.0, 1.0, 20.0, 1.0)
        assert abs(d) > 0.8

    def test_interpret_negligible(self):
        result = interpret_effect_size(0.1)
        assert 'negligible' in result

    def test_interpret_small(self):
        result = interpret_effect_size(0.35)
        assert 'small' in result

    def test_interpret_medium(self):
        result = interpret_effect_size(0.65)
        assert 'medium' in result

    def test_interpret_large(self):
        result = interpret_effect_size(1.5)
        assert 'large' in result

    def test_interpret_direction_up(self):
        result = interpret_effect_size(1.0)
        assert '↑' in result

    def test_interpret_direction_down(self):
        result = interpret_effect_size(-1.0)
        assert '↓' in result


# ── interpret_tissue_state ─────────────────────────────────────────────────

class TestInterpretTissueState:
    def _quality(self, viable=100, edema=0, necrosis=0):
        return {
            'viable_pct': viable,
            'suspect_edema_pct': edema,
            'likely_necrosis_pct': necrosis,
            'n_pixels': 100,
            'tissue_quality_score': viable / 100,
        }

    def test_healthy_tissue_returns_viable_string(self):
        result = interpret_tissue_state(12.0, 300.0, self._quality(100, 0, 0), 'outer_cortex')
        assert 'Viable' in result or 'viable' in result

    def test_severe_damage_flagged(self):
        result = interpret_tissue_state(100.0, 20.0, self._quality(10, 10, 80), 'outer_cortex')
        assert 'damage' in result or 'necrotic' in result or 'Severe' in result

    def test_reduced_perfusion_flagged(self):
        result = interpret_tissue_state(12.0, 80.0, self._quality(95, 5, 0), 'outer_cortex')
        assert 'perfusion' in result or 'ischemia' in result or 'viable' in result.lower()

    def test_no_perfusion_does_not_crash(self):
        result = interpret_tissue_state(12.0, None, self._quality(100, 0, 0), 'outer_cortex')
        assert isinstance(result, str)


# ── update_configs_from_dict ───────────────────────────────────────────────

class TestUpdateConfigsFromDict:
    def _make_4layer_config(self):
        return {
            'metadata': {'n_layers': 4, 'name': 'test_4layer'},
            'zones': {
                'superficial': {'layers': [1, 2]},
                'deep': {'layers': [3, 4]},
            }
        }

    def test_updates_zone_config_global(self):
        config = self._make_4layer_config()
        update_configs_from_dict(config)
        assert tissue_zones.ZONE_CONFIG['metadata']['n_layers'] == 4

    def test_zone_name_lookup_uses_new_config(self):
        config = self._make_4layer_config()
        update_configs_from_dict(config)
        assert get_zone_name(1) == 'superficial'
        assert get_zone_name(4) == 'deep'

    def test_invalid_config_raises(self):
        bad_config = {'metadata': {'n_layers': 3}, 'zones': {'z': {'layers': [1, 2, 99]}}}
        with pytest.raises(ValueError):
            update_configs_from_dict(bad_config)

    def test_restores_default_config(self):
        """Ensure we can restore the default 24-layer config after tests."""
        default = load_zone_config()
        update_configs_from_dict(default)
        assert get_zone_name(1) == 'outer_cortex'
        assert get_zone_name(24) == 'inner_medulla'
