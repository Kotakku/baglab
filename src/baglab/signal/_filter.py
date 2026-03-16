"""Frequency-domain filters (Butterworth lowpass, etc.)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def lowpass(
    x: pd.Series | np.ndarray,
    cutoff: float,
    fs: float,
    order: int = 4,
) -> pd.Series | np.ndarray:
    """Apply a zero-phase Butterworth lowpass filter.

    Parameters
    ----------
    x : pd.Series | np.ndarray
        Input signal.
    cutoff : float
        Cutoff frequency in Hz.
    fs : float
        Sampling frequency in Hz.
    order : int, optional
        Filter order (default 4).

    Returns
    -------
    pd.Series | np.ndarray
        Filtered signal, same type as *x*.

    """
    try:
        from scipy.signal import butter, sosfiltfilt
    except ImportError as exc:
        raise ImportError(
            "scipy is required for lowpass(). Install it with: pip install baglab[analysis]"
        ) from exc

    nyq = 0.5 * fs
    sos = butter(order, cutoff / nyq, btype="low", output="sos")
    filtered = sosfiltfilt(sos, np.asarray(x))

    if isinstance(x, pd.Series):
        return pd.Series(filtered, index=x.index, name=x.name)
    return filtered
