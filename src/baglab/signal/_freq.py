"""Frequency-domain analysis utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd


def fft(
    x: pd.Series | np.ndarray,
    fs: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the single-sided FFT magnitude spectrum.

    Parameters
    ----------
    x : pd.Series | np.ndarray
        Input signal.
    fs : float
        Sampling frequency in Hz.

    Returns
    -------
    freq : np.ndarray
        Positive frequency bins in Hz.
    amplitude : np.ndarray
        Magnitude spectrum (absolute value, normalised by N).

    """
    x_arr = np.asarray(x, dtype=float)
    n = len(x_arr)

    spectrum = np.fft.rfft(x_arr)
    freq = np.fft.rfftfreq(n, d=1.0 / fs)
    amplitude = np.abs(spectrum) * 2.0 / n

    # DC component should not be doubled
    amplitude[0] /= 2.0

    return freq, amplitude
