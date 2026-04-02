"""
Tests for boldpy.fitting.t2star_fitter
"""

import numpy as np
import pytest

from boldpy.fitting.t2star_fitter import (
    monoexponential_decay,
    fit_pixel_t2star,
    fit_t2star_map,
    compute_r2star_map,
    validate_fit_quality,
)
from helpers import ECHO_TIMES, make_t2star_signal


# ── monoexponential_decay ──────────────────────────────────────────────────

class TestMonoexponentialDecay:
    def test_at_te_zero_equals_s0(self):
        result = monoexponential_decay(np.array([0.0]), s0=500.0, t2star=20.0)
        assert result[0] == pytest.approx(500.0)

    def test_decay_shape(self, echo_times):
        signal = monoexponential_decay(echo_times, s0=1000.0, t2star=20.0)
        assert np.all(np.diff(signal) < 0)

    def test_known_value(self):
        # S(TE=20ms) = S0 * exp(-1) when T2*=20ms
        result = monoexponential_decay(np.array([20.0]), s0=1000.0, t2star=20.0)
        assert result[0] == pytest.approx(1000.0 / np.e, rel=1e-6)

    def test_vectorised_over_te(self):
        te = np.array([5.0, 10.0, 20.0])
        result = monoexponential_decay(te, s0=100.0, t2star=10.0)
        assert result.shape == (3,)


# ── fit_pixel_t2star ───────────────────────────────────────────────────────

class TestFitPixelT2star:
    def test_recovers_t2star_clean(self, echo_times, clean_signal_20ms):
        t2star, s0, r2, success = fit_pixel_t2star(echo_times, clean_signal_20ms)
        assert success
        assert t2star == pytest.approx(20.0, abs=0.5)
        assert s0 == pytest.approx(1000.0, rel=0.01)
        assert r2 > 0.999

    def test_recovers_t2star_noisy(self, echo_times, noisy_signal_15ms):
        t2star, s0, r2, success = fit_pixel_t2star(echo_times, noisy_signal_15ms)
        assert success
        assert t2star == pytest.approx(15.0, abs=2.0)
        assert r2 > 0.95

    def test_zero_signal_fails(self, echo_times):
        signal = np.zeros(len(echo_times))
        t2star, s0, r2, success = fit_pixel_t2star(echo_times, signal)
        assert not success
        assert t2star == 0.0

    def test_negative_signal_fails(self, echo_times):
        signal = np.full(len(echo_times), -10.0)
        _, _, _, success = fit_pixel_t2star(echo_times, signal)
        assert not success

    def test_nan_signal_fails(self, echo_times):
        signal = make_t2star_signal(20.0)
        signal[2] = np.nan
        _, _, _, success = fit_pixel_t2star(echo_times, signal)
        assert not success

    def test_inf_signal_fails(self, echo_times):
        signal = make_t2star_signal(20.0)
        signal[0] = np.inf
        _, _, _, success = fit_pixel_t2star(echo_times, signal)
        assert not success

    def test_returns_four_values(self, echo_times, clean_signal_20ms):
        result = fit_pixel_t2star(echo_times, clean_signal_20ms)
        assert len(result) == 4

    def test_r2_bounded(self, echo_times, clean_signal_20ms):
        _, _, r2, success = fit_pixel_t2star(echo_times, clean_signal_20ms)
        assert success
        assert 0.0 <= r2 <= 1.0


# ── fit_t2star_map ─────────────────────────────────────────────────────────

class TestFitT2starMap:
    def test_output_keys(self, echo_times, synthetic_echo_data):
        results = fit_t2star_map(synthetic_echo_data, echo_times, show_progress=False)
        for key in ('t2star_map', 's0_map', 'r2_map', 'success_map'):
            assert key in results

    def test_output_shapes(self, echo_times, synthetic_echo_data):
        results = fit_t2star_map(synthetic_echo_data, echo_times, show_progress=False)
        assert results['t2star_map'].shape == (10, 10)
        assert results['success_map'].dtype == bool

    def test_background_pixels_not_fitted(self, echo_times, synthetic_echo_data):
        results = fit_t2star_map(synthetic_echo_data, echo_times, show_progress=False)
        assert not results['success_map'][0, 0]
        assert results['t2star_map'][0, 0] == 0.0

    def test_foreground_pixels_fitted(self, echo_times, synthetic_echo_data):
        results = fit_t2star_map(synthetic_echo_data, echo_times, show_progress=False)
        assert results['success_map'][5, 5]
        assert results['t2star_map'][5, 5] == pytest.approx(20.0, abs=1.5)

    def test_echo_time_mismatch_raises(self, synthetic_echo_data):
        bad_times = np.array([3.0, 6.0])  # wrong length
        with pytest.raises(ValueError):
            fit_t2star_map(synthetic_echo_data, bad_times, show_progress=False)

    def test_min_signal_threshold(self, echo_times, synthetic_echo_data):
        results = fit_t2star_map(synthetic_echo_data, echo_times,
                                 min_signal=9999.0, show_progress=False)
        assert not np.any(results['success_map'])


# ── compute_r2star_map ─────────────────────────────────────────────────────

class TestComputeR2starMap:
    def test_r2star_formula(self):
        t2star = np.array([[20.0, 25.0], [0.0, 50.0]])
        r2star = compute_r2star_map(t2star)
        assert r2star[0, 0] == pytest.approx(1000.0 / 20.0)
        assert r2star[0, 1] == pytest.approx(1000.0 / 25.0)
        assert r2star[1, 1] == pytest.approx(1000.0 / 50.0)

    def test_zero_t2star_gives_zero_r2star(self):
        t2star = np.array([[0.0, 10.0]])
        r2star = compute_r2star_map(t2star)
        assert r2star[0, 0] == 0.0

    def test_output_shape_matches_input(self):
        t2star = np.ones((5, 8)) * 20.0
        r2star = compute_r2star_map(t2star)
        assert r2star.shape == (5, 8)

    def test_no_division_by_zero(self):
        t2star = np.zeros((4, 4))
        r2star = compute_r2star_map(t2star)
        assert np.all(np.isfinite(r2star))


# ── validate_fit_quality ───────────────────────────────────────────────────

class TestValidateFitQuality:
    @pytest.fixture
    def good_fit_results(self, echo_times, synthetic_echo_data):
        return fit_t2star_map(synthetic_echo_data, echo_times, show_progress=False)

    def test_required_keys_present(self, good_fit_results):
        quality = validate_fit_quality(good_fit_results)
        for key in ('n_total_pixels', 'n_fitted_pixels', 'pct_fitted',
                    't2_mean', 't2_median', 't2_std', 'r2_mean', 'pct_good_fits'):
            assert key in quality

    def test_pct_fitted_in_range(self, good_fit_results):
        quality = validate_fit_quality(good_fit_results)
        assert 0.0 <= quality['pct_fitted'] <= 100.0

    def test_foreground_pixels_have_good_r2(self, good_fit_results):
        quality = validate_fit_quality(good_fit_results, r2_threshold=0.9)
        assert quality['pct_good_fits'] > 80.0

    def test_t2_median_plausible(self, good_fit_results):
        quality = validate_fit_quality(good_fit_results)
        assert quality['t2_median'] == pytest.approx(20.0, abs=2.0)

    def test_empty_success_map(self):
        empty_results = {
            't2star_map': np.zeros((5, 5)),
            'r2_map': np.zeros((5, 5)),
            'success_map': np.zeros((5, 5), dtype=bool),
        }
        quality = validate_fit_quality(empty_results)
        assert quality['n_fitted_pixels'] == 0
        assert quality['pct_fitted'] == 0.0
