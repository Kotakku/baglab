"""Step response analysis."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def stepinfo(
    actual: pd.Series,
    target: float | pd.Series,
    t: pd.Series,
) -> dict[str, float]:
    """Compute step response metrics.

    Parameters
    ----------
    actual : pd.Series
        Actual response signal.
    target : float | pd.Series
        Target / setpoint value.  If a Series, the final value is used as the
        steady-state target.
    t : pd.Series
        Time in seconds.

    Returns
    -------
    dict[str, float]
        Dictionary with keys ``overshoot``, ``settling_time``, ``rise_time``,
        and ``steady_state_error``.

    Notes
    -----
    Logic follows MATLAB ``stepinfo`` / python-control ``step_info``.
    """
    y = np.asarray(actual, dtype=float)
    t_arr = np.asarray(t, dtype=float)

    if isinstance(target, pd.Series):
        final_target = float(target.iloc[-1])
    else:
        final_target = float(target)

    # Initial and final values of the response
    y0 = y[0]
    yf = final_target

    dy = yf - y0  # expected total change

    # --- Steady-state error ---
    steady_state_error = abs(y[-1] - yf)

    # --- Overshoot ---
    if abs(dy) < 1e-12:
        # No step – cannot compute meaningful metrics
        overshoot = 0.0
    elif dy > 0:
        peak = np.max(y)
        overshoot = max((peak - yf) / abs(dy) * 100.0, 0.0)
    else:
        trough = np.min(y)
        overshoot = max((yf - trough) / abs(dy) * 100.0, 0.0)

    # --- Rise time (10 % to 90 % of final value) ---
    rise_time: float
    if abs(dy) < 1e-12:
        rise_time = float("nan")
    else:
        low_thresh = y0 + 0.1 * dy
        high_thresh = y0 + 0.9 * dy
        if dy > 0:
            low_cross = np.where(y >= low_thresh)[0]
            high_cross = np.where(y >= high_thresh)[0]
        else:
            low_cross = np.where(y <= low_thresh)[0]
            high_cross = np.where(y <= high_thresh)[0]

        if len(low_cross) > 0 and len(high_cross) > 0:
            rise_time = float(t_arr[high_cross[0]] - t_arr[low_cross[0]])
        else:
            rise_time = float("nan")

    # --- Settling time (2 % band around final value) ---
    settling_time: float
    if abs(dy) < 1e-12:
        settling_time = float("nan")
    else:
        tol = 0.02 * abs(dy)
        outside = np.where(np.abs(y - yf) > tol)[0]
        if len(outside) == 0:
            # Already within band from the start
            settling_time = float(t_arr[0])
        else:
            last_outside = outside[-1]
            if last_outside < len(t_arr) - 1:
                settling_time = float(t_arr[last_outside + 1])
            else:
                # Never settles
                settling_time = float("nan")

    return {
        "overshoot": overshoot,
        "settling_time": settling_time,
        "rise_time": rise_time,
        "steady_state_error": steady_state_error,
    }
