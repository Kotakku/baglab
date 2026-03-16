"""Angle normalization and wrapping utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd


def normalize_angle(angle):
    """Normalize angle to [-pi, pi].

    Parameters
    ----------
    angle : float | np.ndarray | pd.Series
        Angle in radians.

    Returns
    -------
    float | np.ndarray | pd.Series
        Same type as input.

    """
    result = (np.asarray(angle) + np.pi) % (2 * np.pi) - np.pi
    if isinstance(angle, pd.Series):
        return pd.Series(result, index=angle.index, name=angle.name)
    if np.ndim(angle) == 0:
        return float(result)
    return result


def angle_diff(a, b):
    """Shortest signed angle difference ``a - b``, wrapped to [-pi, pi].

    Parameters
    ----------
    a, b : float | np.ndarray | pd.Series
        Angles in radians.

    Returns
    -------
    float | np.ndarray | pd.Series
        Same type as *a*.

    """
    result = (np.asarray(a) - np.asarray(b) + np.pi) % (2 * np.pi) - np.pi
    if isinstance(a, pd.Series):
        return pd.Series(result, index=a.index)
    if np.ndim(a) == 0:
        return float(result)
    return result


def unwrap(angle: pd.Series) -> pd.Series:
    """Remove discontinuities in angle series (Series-aware ``np.unwrap``).

    Parameters
    ----------
    angle : pd.Series
        Angle in radians, possibly with ±pi jumps.

    Returns
    -------
    pd.Series

    """
    return pd.Series(
        np.unwrap(np.asarray(angle)),
        index=angle.index,
        name=angle.name,
    )
