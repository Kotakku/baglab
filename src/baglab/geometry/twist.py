"""Twist / velocity operations."""

from __future__ import annotations

import numpy as np
import pandas as pd

from baglab.geometry._common import FieldInput, _is_scalar, _to_df
from baglab.geometry.quaternion import quat_to_yaw


def twist_to_speed(linear: FieldInput) -> pd.Series | float:
    """Compute scalar speed from linear velocity (x, y, z).

    Parameters
    ----------
    linear : FieldGroup | DataFrame | object
        Must contain fields/attributes ``x``, ``y``, ``z``.

    Returns
    -------
    float | pd.Series
        Speed = sqrt(x² + y² + z²).

    """
    scalar = _is_scalar(linear)
    d = _to_df(linear)
    result = np.sqrt(d["x"].values ** 2 + d["y"].values ** 2 + d["z"].values ** 2)
    if scalar:
        return float(result[0])
    return pd.Series(result, index=d.index, name="speed")


def twist_to_speed_2d(linear: FieldInput) -> pd.Series | float:
    """Compute 2D scalar speed from linear velocity (x, y).

    Parameters
    ----------
    linear : FieldGroup | DataFrame | object
        Must contain fields/attributes ``x``, ``y``.

    Returns
    -------
    float | pd.Series
        Speed = sqrt(x² + y²).

    """
    scalar = _is_scalar(linear)
    d = _to_df(linear)
    result = np.hypot(d["x"].values, d["y"].values)
    if scalar:
        return float(result[0])
    return pd.Series(result, index=d.index, name="speed_2d")


def velocity_from_pose(position: FieldInput, time: pd.Series) -> pd.Series:
    """Estimate scalar velocity by differentiating position over time.

    Parameters
    ----------
    position : FieldGroup | DataFrame
        Must contain fields ``x``, ``y``.
    time : pd.Series
        Time in seconds (float).

    Returns
    -------
    pd.Series
        Velocity (distance / dt).

    """
    d = _to_df(position)
    dx = np.diff(d["x"].values, prepend=np.nan)
    dy = np.diff(d["y"].values, prepend=np.nan)
    dt = np.diff(time.values, prepend=np.nan)
    speed = np.hypot(dx, dy) / dt
    return pd.Series(speed, index=d.index, name="velocity")


def yaw_rate_from_pose(orientation: FieldInput, time: pd.Series) -> pd.Series:
    """Estimate yaw rate by differentiating yaw over time.

    Parameters
    ----------
    orientation : FieldGroup | DataFrame
        Quaternion fields ``x``, ``y``, ``z``, ``w``.
    time : pd.Series
        Time in seconds (float).

    Returns
    -------
    pd.Series
        Yaw rate in rad/s.

    """
    yaw = np.unwrap(quat_to_yaw(orientation).values)
    dyaw = np.diff(yaw, prepend=np.nan)
    dt = np.diff(time.values, prepend=np.nan)
    return pd.Series(dyaw / dt, index=time.index, name="yaw_rate")
