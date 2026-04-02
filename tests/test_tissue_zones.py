"""
Tests for tissue_zones.py — zone config loading, validation, and encoding.
"""

import sys
import os
import pytest

# tissue_zones.py lives at repo root, not inside the boldpy package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tissue_zones import (
    decode_multiregion_value,
    encode_multiregion_value,
    is_multiregion_config,
    load_zone_config,
    validate_zone_config,
)


# ── encode / decode ────────────────────────────────────────────────────────

class TestMultiregionEncoding:
    def test_encode_decode_roundtrip(self):
        for region_id in (1, 2, 3):
            for layer_num in (1, 12, 24):
                encoded = encode_multiregion_value(region_id, layer_num)
                decoded_region, decoded_layer = decode_multiregion_value(encoded)
                assert decoded_region == region_id
                assert decoded_layer == layer_num

    def test_encode_formula(self):
        assert encode_multiregion_value(2, 5) == 2005

    def test_decode_known_value(self):
        region, layer = decode_multiregion_value(3012)
        assert region == 3
        assert layer == 12

    def test_layer_1_region_1(self):
        assert encode_multiregion_value(1, 1) == 1001

    def test_large_region_id(self):
        encoded = encode_multiregion_value(10, 24)
        region, layer = decode_multiregion_value(encoded)
        assert region == 10
        assert layer == 24


# ── is_multiregion_config ──────────────────────────────────────────────────

class TestIsMultiregionConfig:
    def test_single_region_config_is_false(self):
        config = {'metadata': {'n_layers': 24}, 'zones': {}}
        assert not is_multiregion_config(config)

    def test_config_with_regions_key_is_true(self):
        config = {'metadata': {}, 'regions': {}}
        assert is_multiregion_config(config)

    def test_config_with_mode_multi_region_is_true(self):
        config = {'metadata': {'mode': 'multi_region'}}
        assert is_multiregion_config(config)

    def test_empty_config_is_false(self):
        assert not is_multiregion_config({'metadata': {}})


# ── load_zone_config ───────────────────────────────────────────────────────

class TestLoadZoneConfig:
    def test_loads_default_config(self):
        config = load_zone_config()
        assert 'metadata' in config
        assert 'zones' in config

    def test_default_has_24_layers(self):
        config = load_zone_config()
        assert config['metadata']['n_layers'] == 24

    def test_default_has_expected_zones(self):
        config = load_zone_config()
        zone_names = set(config['zones'].keys())
        expected = {'outer_cortex', 'inner_cortex', 'cmj', 'outer_medulla', 'inner_medulla'}
        assert expected.issubset(zone_names)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_zone_config(tmp_path / 'nonexistent.yaml')


# ── validate_zone_config ───────────────────────────────────────────────────

class TestValidateZoneConfig:
    def _make_valid_config(self):
        return {
            'metadata': {'n_layers': 4, 'name': 'test'},
            'zones': {
                'zone_a': {'layers': [1, 2]},
                'zone_b': {'layers': [3, 4]},
            }
        }

    def test_valid_config_passes(self):
        validate_zone_config(self._make_valid_config())  # should not raise

    def test_missing_metadata_raises(self):
        config = {'zones': {'z': {'layers': [1]}}}
        with pytest.raises(ValueError, match='metadata'):
            validate_zone_config(config)

    def test_missing_zones_raises(self):
        config = {'metadata': {'n_layers': 1}}
        with pytest.raises(ValueError, match='zones'):
            validate_zone_config(config)

    def test_missing_n_layers_raises(self):
        config = {'metadata': {}, 'zones': {'z': {'layers': [1]}}}
        with pytest.raises(ValueError, match='n_layers'):
            validate_zone_config(config)

    def test_duplicate_layer_raises(self):
        config = {
            'metadata': {'n_layers': 3},
            'zones': {
                'zone_a': {'layers': [1, 2]},
                'zone_b': {'layers': [2, 3]},  # layer 2 duplicated
            }
        }
        with pytest.raises(ValueError):
            validate_zone_config(config)

    def test_out_of_range_layer_raises(self):
        config = {
            'metadata': {'n_layers': 3},
            'zones': {
                'zone_a': {'layers': [1, 2, 3, 99]},  # layer 99 invalid
            }
        }
        with pytest.raises(ValueError):
            validate_zone_config(config)

    def test_missing_layer_raises(self):
        config = {
            'metadata': {'n_layers': 4},
            'zones': {
                'zone_a': {'layers': [1, 2]},
                # layer 3 and 4 never assigned
            }
        }
        with pytest.raises(ValueError):
            validate_zone_config(config)

    def test_all_24_layers_covered(self):
        config = load_zone_config()
        # Default config should already pass validation
        validate_zone_config(config)
        all_layers = []
        for zone in config['zones'].values():
            all_layers.extend(zone['layers'])
        assert sorted(all_layers) == list(range(1, 25))
