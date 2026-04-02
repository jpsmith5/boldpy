"""
Tests for cluster_zones.compare_zone_configs — Jaccard overlap and boundary
shift computation between two zone configs (used in Workflow B comparisons).
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from cluster_zones import compare_zone_configs


# ── helpers ────────────────────────────────────────────────────────────────

def make_config(zone_layers: dict) -> dict:
    """Build a minimal zone config from a {zone_name: [layers]} dict."""
    all_layers = [l for layers in zone_layers.values() for l in layers]
    return {
        'metadata': {'n_layers': max(all_layers)},
        'zones': {name: {'layers': layers} for name, layers in zone_layers.items()}
    }


# ── basic structure ────────────────────────────────────────────────────────

class TestCompareZoneConfigsStructure:
    def test_returns_dict(self):
        a = make_config({'zone_a': [1, 2, 3], 'zone_b': [4, 5, 6]})
        b = make_config({'zone_a': [1, 2, 3], 'zone_b': [4, 5, 6]})
        result = compare_zone_configs(a, b)
        assert isinstance(result, dict)

    def test_keys_are_zone_names(self):
        a = make_config({'cortex': [1, 2], 'medulla': [3, 4]})
        b = make_config({'cortex': [1, 2], 'medulla': [3, 4]})
        result = compare_zone_configs(a, b)
        assert 'cortex' in result
        assert 'medulla' in result

    def test_each_entry_has_required_keys(self):
        a = make_config({'zone_a': [1, 2, 3]})
        b = make_config({'zone_a': [1, 2, 3]})
        result = compare_zone_configs(a, b)
        entry = result['zone_a']
        for key in ('jaccard', 'ref_layers', 'clustered_layers',
                    'boundary_shift', 'in_reference', 'in_clustered'):
            assert key in entry

    def test_zone_only_in_reference_flagged(self):
        ref = make_config({'zone_a': [1, 2], 'zone_b': [3, 4]})
        clust = make_config({'zone_a': [1, 2]})  # zone_b missing
        result = compare_zone_configs(clust, ref)
        assert result['zone_b']['in_reference']
        assert not result['zone_b']['in_clustered']

    def test_zone_only_in_clustered_flagged(self):
        ref = make_config({'zone_a': [1, 2]})
        clust = make_config({'zone_a': [1, 2], 'zone_b': [3, 4]})
        result = compare_zone_configs(clust, ref)
        assert not result['zone_b']['in_reference']
        assert result['zone_b']['in_clustered']


# ── Jaccard overlap ────────────────────────────────────────────────────────

class TestJaccardOverlap:
    def test_identical_configs_jaccard_1(self):
        config = make_config({'zone_a': [1, 2, 3, 4, 5]})
        result = compare_zone_configs(config, config)
        assert result['zone_a']['jaccard'] == pytest.approx(1.0)

    def test_no_overlap_jaccard_0(self):
        ref = make_config({'zone_a': [1, 2, 3]})
        clust = make_config({'zone_a': [4, 5, 6]})
        result = compare_zone_configs(clust, ref)
        assert result['zone_a']['jaccard'] == pytest.approx(0.0)

    def test_partial_overlap(self):
        # ref: [1,2,3,4], clust: [3,4,5,6] → intersection=[3,4], union=[1,2,3,4,5,6]
        ref = make_config({'zone_a': [1, 2, 3, 4]})
        clust = make_config({'zone_a': [3, 4, 5, 6]})
        result = compare_zone_configs(clust, ref)
        assert result['zone_a']['jaccard'] == pytest.approx(2 / 6, abs=0.001)

    def test_jaccard_bounded_0_to_1(self):
        ref = make_config({'zone_a': [1, 2, 3], 'zone_b': [4, 5, 6]})
        clust = make_config({'zone_a': [2, 3, 4], 'zone_b': [5, 6, 7]})
        result = compare_zone_configs(clust, ref)
        for zone_name, entry in result.items():
            assert 0.0 <= entry['jaccard'] <= 1.0


# ── boundary shift ─────────────────────────────────────────────────────────

class TestBoundaryShift:
    def test_no_shift_when_identical(self):
        config = make_config({'zone_a': [1, 2, 3, 4, 5]})
        result = compare_zone_configs(config, config)
        shift = result['zone_a']['boundary_shift']
        assert shift.get('lower', 0) == 0
        assert shift.get('upper', 0) == 0

    def test_lower_boundary_shift_positive(self):
        # clustered starts 2 layers deeper than reference
        ref = make_config({'zone_a': [1, 2, 3, 4]})
        clust = make_config({'zone_a': [3, 4, 5, 6]})
        result = compare_zone_configs(clust, ref)
        assert result['zone_a']['boundary_shift']['lower'] == 2

    def test_upper_boundary_shift_negative(self):
        # clustered ends 1 layer shallower
        ref = make_config({'zone_a': [1, 2, 3, 4, 5]})
        clust = make_config({'zone_a': [1, 2, 3, 4]})
        result = compare_zone_configs(clust, ref)
        assert result['zone_a']['boundary_shift']['upper'] == -1

    def test_shift_computed_per_zone(self):
        ref = make_config({'cortex': [1, 2, 3], 'medulla': [4, 5, 6]})
        clust = make_config({'cortex': [1, 2, 4], 'medulla': [5, 6, 7]})
        result = compare_zone_configs(clust, ref)
        # Cortex lower boundary: same (both start at 1)
        assert result['cortex']['boundary_shift']['lower'] == 0
        # Medulla lower boundary: shifts from 4 to 5 (+1)
        assert result['medulla']['boundary_shift']['lower'] == 1


# ── ref_layers / clustered_layers in output ────────────────────────────────

class TestOutputLayers:
    def test_ref_layers_sorted(self):
        ref = make_config({'zone_a': [3, 1, 2]})
        clust = make_config({'zone_a': [1, 2, 3]})
        result = compare_zone_configs(clust, ref)
        assert result['zone_a']['ref_layers'] == [1, 2, 3]

    def test_clustered_layers_sorted(self):
        ref = make_config({'zone_a': [1, 2, 3]})
        clust = make_config({'zone_a': [5, 3, 4]})
        result = compare_zone_configs(clust, ref)
        assert result['zone_a']['clustered_layers'] == [3, 4, 5]

    def test_missing_zone_has_empty_layers(self):
        ref = make_config({'zone_a': [1, 2], 'zone_b': [3, 4]})
        clust = make_config({'zone_a': [1, 2]})
        result = compare_zone_configs(clust, ref)
        assert result['zone_b']['clustered_layers'] == []
