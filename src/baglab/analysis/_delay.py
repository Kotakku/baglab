"""Transport-delay estimation via cross-correlation."""

from __future__ import annotations

import numpy as np
import pandas as pd


def delay_estimate(
    cmd: pd.Series,
    response: pd.Series,
    t: pd.Series,
) -> float:
    """Estimate dead time / transport delay using cross-correlation.

    Parameters
    ----------
    cmd : pd.Series
        Command signal.
    response : pd.Series
        Response signal (same length as *cmd*).
    t : pd.Series
        Time in seconds (must be uniformly sampled, or approximately so).

    Returns
    -------
    float
        Estimated delay in seconds.

    Raises
    ------
    ImportError
        If *scipy* is not installed.
    """
    try:
        from scipy.signal import correlate
    except ImportError as exc:
        raise ImportError(
            "scipy is required for delay_estimate(). "
            "Install it with: pip install baglab[analysis]"
        ) from exc

    c = np.asarray(cmd, dtype=float)
    r = np.asarray(response, dtype=float)
    t_arr = np.asarray(t, dtype=float)

    dt = float(np.mean(np.diff(t_arr)))

    corr = correlate(r, c, mode="full")
    lags = np.arange(-len(c) + 1, len(c)) * dt

    peak_idx = int(np.argmax(corr))
    return float(lags[peak_idx])
