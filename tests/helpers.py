"""
Shared test helpers and synthetic data generators.
"""

import numpy as np

ECHO_TIMES = np.array([3.0, 6.5, 10.0, 13.5, 17.0, 20.5, 24.0])  # ms, typical Bruker MGE


def make_t2star_signal(t2star_ms: float, s0: float = 1000.0,
                       noise_std: float = 0.0,
                       rng: np.random.Generator = None) -> np.ndarray:
    """Generate synthetic mono-exponential T2* signal."""
    signal = s0 * np.exp(-ECHO_TIMES / t2star_ms)
    if noise_std > 0:
        if rng is None:
            rng = np.random.default_rng(42)
        signal = signal + rng.normal(0, noise_std, signal.shape)
    return signal
