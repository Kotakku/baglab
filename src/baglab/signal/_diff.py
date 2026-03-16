"""Numerical differentiation and integration."""

from __future__ import annotations

import numpy as np
import pandas as pd


def diff(x: pd.Series, t: pd.Series) -> pd.Series:
    """Numerical differentiation using finite differences.

    Uses forward difference for interior points and backward difference
    for the last point.  The first element is ``NaN`` because there is
    no preceding sample.

    Parameters
    ----------
    x : pd.Series
        Signal values.
    t : pd.Series
        Time in seconds (same length as *x*).

    Returns
    -------
    pd.Series
        dx/dt, same length as *x*.  First element is ``NaN``.

    """
    x_arr = np.asarray(x, dtype=float)
    t_arr = np.asarray(t, dtype=float)

    dx = np.diff(x_arr)
    dt = np.diff(t_arr)
    dxdt = dx / dt

    result = np.empty_like(x_arr)
    result[0] = np.nan
    result[1:] = dxdt

    return pd.Series(result, index=x.index, name=x.name)


def integrate(x: pd.Series, t: pd.Series) -> pd.Series:
    """Cumulative trapezoidal integration.

    Parameters
    ----------
    x : pd.Series
        Signal values.
    t : pd.Series
        Time in seconds (same length as *x*).

    Returns
    -------
    pd.Series
        Cumulative integral.  First element is ``0.0``.

    """
    x_arr = np.asarray(x, dtype=float)
    t_arr = np.asarray(t, dtype=float)

    dt = np.diff(t_arr)
    avg = 0.5 * (x_arr[:-1] + x_arr[1:])
    cumul = np.concatenate(([0.0], np.cumsum(avg * dt)))

    return pd.Series(cumul, index=x.index, name=x.name)
