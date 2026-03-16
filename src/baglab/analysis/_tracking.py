"""Tracking error statistics."""

from __future__ import annotations

import numpy as np
import pandas as pd


def tracking_error(
    actual: pd.Series,
    ref: pd.Series,
    t: pd.Series | None = None,
) -> dict[str, float | pd.Series]:
    """Compute tracking error statistics.

    Parameters
    ----------
    actual : pd.Series
        Actual values.
    ref : pd.Series
        Reference / command values (same length as *actual*).
    t : pd.Series | None, optional
        Time in seconds.  Reserved for future time-weighted metrics.

    Returns
    -------
    dict
        Dictionary with keys ``error`` (pd.Series), ``mean``, ``max``,
        ``rms``, and ``std``.
    """
    error = actual.values - ref.values
    error_series = pd.Series(error, index=actual.index, name="error")

    return {
        "error": error_series,
        "mean": float(np.mean(error)),
        "max": float(np.max(np.abs(error))),
        "rms": float(np.sqrt(np.mean(error ** 2))),
        "std": float(np.std(error, ddof=0)),
    }
