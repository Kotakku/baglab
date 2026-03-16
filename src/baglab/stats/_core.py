"""Core statistical functions."""

from __future__ import annotations

import numpy as np
import pandas as pd


def rms(x: pd.Series | np.ndarray) -> float:
    """Root Mean Square of a signal.

    Parameters
    ----------
    x : pd.Series | np.ndarray
        Input signal.

    Returns
    -------
    float
        Root mean square value: ``sqrt(mean(x**2))``.

    """
    arr = np.asarray(x, dtype=float)
    return float(np.sqrt(np.mean(arr ** 2)))


def describe(x: pd.Series | np.ndarray) -> dict[str, float]:
    """Summary statistics for a signal.

    Parameters
    ----------
    x : pd.Series | np.ndarray
        Input signal.

    Returns
    -------
    dict[str, float]
        Dictionary with keys ``"mean"``, ``"std"``, ``"min"``, ``"max"``,
        ``"median"``, ``"p5"``, ``"p95"``, ``"rms"``.

    """
    arr = np.asarray(x, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=0)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "median": float(np.median(arr)),
        "p5": float(np.percentile(arr, 5)),
        "p95": float(np.percentile(arr, 95)),
        "rms": rms(arr),
    }
