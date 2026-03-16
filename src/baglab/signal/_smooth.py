"""Smoothing filters."""

from __future__ import annotations

import pandas as pd


def moving_average(x: pd.Series, window: int) -> pd.Series:
    """Centered moving average filter.

    Parameters
    ----------
    x : pd.Series
        Input signal.
    window : int
        Number of points in the averaging window.

    Returns
    -------
    pd.Series
        Smoothed signal with ``NaN`` at edges where the full window
        is not available (centered).

    """
    return x.rolling(window=window, center=True, min_periods=window).mean()
