"""
Shared pytest fixtures for BoldPy test suite.
"""

import numpy as np
import pytest

from helpers import ECHO_TIMES, make_t2star_signal


# ── T2* fitting fixtures ───────────────────────────────────────────────────

@pytest.fixture
def echo_times():
    return ECHO_TIMES


@pytest.fixture
def clean_signal_20ms():
    """Noiseless signal with T2*=20 ms, S0=1000."""
    return make_t2star_signal(t2star_ms=20.0, s0=1000.0)


@pytest.fixture
def noisy_signal_15ms():
    """Low-noise signal with T2*=15 ms, S0=800."""
    return make_t2star_signal(t2star_ms=15.0, s0=800.0, noise_std=5.0)


@pytest.fixture
def synthetic_echo_data():
    """
    Small (7-echo, 10×10) multi-echo array.
    Centre 6×6 pixels have T2*=20 ms, S0=1000; background = 0.
    """
    rng = np.random.default_rng(0)
    data = np.zeros((len(ECHO_TIMES), 10, 10), dtype=np.float32)
    for i, te in enumerate(ECHO_TIMES):
        signal = 1000.0 * np.exp(-te / 20.0)
        noise = rng.normal(0, 2.0, (6, 6)).astype(np.float32)
        data[i, 2:8, 2:8] = signal + noise
    return data


# ── MLCO / zone fixtures ───────────────────────────────────────────────────

@pytest.fixture
def synthetic_mlco_mask_6layers():
    """
    10×10 mask with 6 concentric layers (3 per kidney, bilateral).
    Layers 1-3 = right, 4-6 = left.
    """
    mask = np.zeros((10, 10), dtype=np.int32)
    mask[1:9, 1:9] = 1
    mask[2:8, 2:8] = 2
    mask[3:7, 3:7] = 3
    mask[1:9, 1:5] = 4
    mask[2:8, 2:5] = 5
    mask[3:7, 3:5] = 6
    return mask


@pytest.fixture
def synthetic_t2star_map(synthetic_mlco_mask_6layers):
    """T2* map with a cortex-to-medulla gradient (outer layers high, inner low)."""
    mask = synthetic_mlco_mask_6layers
    t2 = np.zeros((10, 10), dtype=np.float32)
    for layer_val, t2_val in [(1, 20.0), (2, 18.0), (3, 15.0),
                               (4, 20.0), (5, 18.0), (6, 15.0)]:
        t2[mask == layer_val] = t2_val
    return t2


@pytest.fixture
def synthetic_r2star_map(synthetic_t2star_map):
    """R2* = 1000/T2* (valid pixels only)."""
    r2 = np.zeros_like(synthetic_t2star_map)
    valid = synthetic_t2star_map > 0
    r2[valid] = 1000.0 / synthetic_t2star_map[valid]
    return r2
