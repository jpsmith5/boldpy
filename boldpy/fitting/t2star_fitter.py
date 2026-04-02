"""
T2* Fitting from Multi-Echo Data
=================================

Proper T2* fitting using actual echo times from Bruker method files.
"""

import numpy as np
from scipy.optimize import curve_fit
from typing import Tuple, Optional, Dict
import warnings


def monoexponential_decay(te: np.ndarray, s0: float, t2star: float) -> np.ndarray:
    """
    Monoexponential T2* decay model
    
    S(TE) = S0 × exp(-TE/T2*)
    
    Parameters:
    -----------
    te : array-like
        Echo times in milliseconds
    s0 : float
        Signal intensity at TE=0
    t2star : float
        T2* relaxation time in milliseconds
        
    Returns:
    --------
    signal : array-like
        Signal intensities at given echo times
    """
    return s0 * np.exp(-te / t2star)


def fit_pixel_t2star(echo_times: np.ndarray, 
                     signal: np.ndarray,
                     bounds: Tuple[Tuple[float, float], Tuple[float, float]] = ((0, 5), (np.inf, 2000)),
                     initial_t2star: float = 30.0) -> Tuple[float, float, float, bool]:
    """
    Fit T2* for a single pixel
    
    Parameters:
    -----------
    echo_times : array
        Echo times in milliseconds
    signal : array
        Signal intensities for each echo
    bounds : tuple of tuples
        Fitting bounds: ((s0_min, t2star_min), (s0_max, t2star_max))
    initial_t2star : float
        Initial guess for T2* in milliseconds
        
    Returns:
    --------
    t2star : float
        Fitted T2* value in milliseconds
    s0 : float
        Fitted signal intensity at TE=0
    r2 : float
        R² goodness of fit
    success : bool
        Whether fitting succeeded
    """
    # Check for invalid signal
    if np.any(signal <= 0) or np.any(~np.isfinite(signal)):
        return 0.0, 0.0, 0.0, False
    
    try:
        # Initial guess: S0 = first echo, T2* = initial_t2star
        p0 = [signal[0], initial_t2star]
        
        # Fit with bounds
        popt, pcov = curve_fit(
            monoexponential_decay,
            echo_times,
            signal,
            p0=p0,
            bounds=bounds,
            maxfev=1000
        )
        
        s0, t2star = popt
        
        # Calculate R² (goodness of fit)
        fitted = monoexponential_decay(echo_times, s0, t2star)
        residuals = signal - fitted
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((signal - np.mean(signal))**2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        
        return t2star, s0, r2, True
        
    except Exception as e:
        # Fitting failed
        return 0.0, 0.0, 0.0, False


def fit_t2star_map(echo_data: np.ndarray,
                   echo_times: np.ndarray,
                   bounds: Tuple[Tuple[float, float], Tuple[float, float]] = ((0, 5), (np.inf, 2000)),
                   initial_t2star: float = 30.0,
                   min_signal: float = 10.0,
                   show_progress: bool = True) -> Dict[str, np.ndarray]:
    """
    Fit T2* map from multi-echo data
    
    Parameters:
    -----------
    echo_data : ndarray
        Multi-echo data, shape (n_echoes, height, width)
    echo_times : array
        Echo times in milliseconds
    bounds : tuple
        Fitting bounds for (S0, T2*)
    initial_t2star : float
        Initial guess for T2* in ms
    min_signal : float
        Minimum signal threshold for fitting
    show_progress : bool
        Whether to print progress updates
        
    Returns:
    --------
    results : dict
        Dictionary containing:
        - 't2star_map': T2* map (ms)
        - 's0_map': Signal intensity map
        - 'r2_map': R² goodness of fit map
        - 'success_map': Boolean map of successful fits
    """
    n_echoes, height, width = echo_data.shape
    
    if len(echo_times) != n_echoes:
        raise ValueError(f"Number of echo times ({len(echo_times)}) must match "
                        f"number of echoes ({n_echoes})")
    
    # Initialize output arrays
    t2star_map = np.zeros((height, width), dtype=np.float32)
    s0_map = np.zeros((height, width), dtype=np.float32)
    r2_map = np.zeros((height, width), dtype=np.float32)
    success_map = np.zeros((height, width), dtype=bool)
    
    # Total pixels
    total_pixels = height * width
    fitted_pixels = 0
    failed_pixels = 0
    
    if show_progress:
        print(f"\nFitting {total_pixels:,} pixels...")
        progress_interval = max(1, total_pixels // 20)  # 20 updates
    
    # Fit each pixel
    for y in range(height):
        for x in range(width):
            # Progress update
            pixel_idx = y * width + x
            if show_progress and pixel_idx % progress_interval == 0:
                pct = 100 * pixel_idx / total_pixels
                print(f"  Progress: {pct:.0f}% ({pixel_idx:,}/{total_pixels:,} pixels)", end='\r')
            
            # Extract signal for this pixel
            signal = echo_data[:, y, x]
            
            # Check if signal is above threshold
            if signal[0] < min_signal:
                continue
            
            # Fit
            t2star, s0, r2, success = fit_pixel_t2star(
                echo_times, signal, bounds, initial_t2star
            )
            
            if success:
                t2star_map[y, x] = t2star
                s0_map[y, x] = s0
                r2_map[y, x] = r2
                success_map[y, x] = True
                fitted_pixels += 1
            else:
                failed_pixels += 1
    
    if show_progress:
        print(f"  Progress: 100% ({total_pixels:,}/{total_pixels:,} pixels)")
        print(f"\n✓ Fitting complete:")
        print(f"    Fitted: {fitted_pixels:,} pixels ({100*fitted_pixels/total_pixels:.1f}%)")
        print(f"    Failed: {failed_pixels:,} pixels ({100*failed_pixels/total_pixels:.1f}%)")
    
    return {
        't2star_map': t2star_map,
        's0_map': s0_map,
        'r2_map': r2_map,
        'success_map': success_map,
        'echo_times': echo_times,
        'bounds': bounds
    }


def compute_r2star_map(t2star_map: np.ndarray) -> np.ndarray:
    """
    Compute R2* map from T2* map
    
    R2* = 1000 / T2*  (converts ms to Hz)
    
    Parameters:
    -----------
    t2star_map : ndarray
        T2* map in milliseconds
        
    Returns:
    --------
    r2star_map : ndarray
        R2* map in Hz
    """
    r2star_map = np.zeros_like(t2star_map)
    
    # Avoid division by zero
    valid = t2star_map > 0
    r2star_map[valid] = 1000.0 / t2star_map[valid]
    
    return r2star_map


def validate_fit_quality(fit_results: Dict[str, np.ndarray],
                        r2_threshold: float = 0.7) -> Dict[str, any]:
    """
    Validate T2* fit quality
    
    Parameters:
    -----------
    fit_results : dict
        Output from fit_t2star_map()
    r2_threshold : float
        Minimum R² for good fit quality
        
    Returns:
    --------
    quality : dict
        Quality metrics and diagnostics
    """
    t2star_map = fit_results['t2star_map']
    r2_map = fit_results['r2_map']
    success_map = fit_results['success_map']
    
    # Extract valid fits
    valid = success_map & (t2star_map > 0)
    t2_valid = t2star_map[valid]
    r2_valid = r2_map[valid]
    
    # Quality metrics
    quality = {
        'n_total_pixels': success_map.size,
        'n_fitted_pixels': np.sum(valid),
        'pct_fitted': 100 * np.sum(valid) / success_map.size,
        
        # T2* statistics
        't2_mean': np.mean(t2_valid) if len(t2_valid) > 0 else 0,
        't2_median': np.median(t2_valid) if len(t2_valid) > 0 else 0,
        't2_std': np.std(t2_valid) if len(t2_valid) > 0 else 0,
        't2_min': np.min(t2_valid) if len(t2_valid) > 0 else 0,
        't2_max': np.max(t2_valid) if len(t2_valid) > 0 else 0,
        
        # Fit quality (R²)
        'r2_mean': np.mean(r2_valid) if len(r2_valid) > 0 else 0,
        'r2_median': np.median(r2_valid) if len(r2_valid) > 0 else 0,
        'pct_good_fits': 100 * np.sum(r2_valid > r2_threshold) / len(r2_valid) if len(r2_valid) > 0 else 0,
        
        # Ceiling/floor effects
        'pct_at_floor': 100 * np.sum(t2_valid <= 10) / len(t2_valid) if len(t2_valid) > 0 else 0,
        'pct_at_ceiling': 100 * np.sum(t2_valid >= 1950) / len(t2_valid) if len(t2_valid) > 0 else 0,
    }
    
    return quality
