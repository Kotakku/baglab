"""Quaternion <-> Euler angle conversions."""

from __future__ import annotations

import numpy as np
import pandas as pd

from baglab.geometry._common import FieldInput, _is_scalar, _to_df


def quat_to_yaw(orientation: FieldInput) -> pd.Series | float:
    """Convert quaternion (x, y, z, w) to yaw angle in radians.

    Parameters
    ----------
    orientation : FieldGroup | DataFrame | object
        Must contain columns/fields/attributes ``x``, ``y``, ``z``, ``w``.
        Accepts a single ROS msg (returns float) or a DataFrame/FieldGroup
        (returns Series).

    Returns
    -------
    float | pd.Series
        Yaw angle in radians.

    """
    scalar = _is_scalar(orientation)
    d = _to_df(orientation)
    qx, qy, qz, qw = d["x"], d["y"], d["z"], d["w"]
    result = pd.Series(
        np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy**2 + qz**2)),
        index=d.index,
        name="yaw",
    )
    return float(result.iloc[0]) if scalar else result


def quat_to_rpy(orientation: FieldInput) -> pd.DataFrame | dict[str, float]:
    """Convert quaternion (x, y, z, w) to roll, pitch, yaw in radians.

    Parameters
    ----------
    orientation : FieldGroup | DataFrame | object
        Must contain columns/fields/attributes ``x``, ``y``, ``z``, ``w``.
        Accepts a single ROS msg (returns dict) or a DataFrame/FieldGroup
        (returns DataFrame).

    Returns
    -------
    dict[str, float] | pd.DataFrame
        ``{"roll": …, "pitch": …, "yaw": …}`` for scalar input,
        or DataFrame with columns ``roll``, ``pitch``, ``yaw``.

    """
    scalar = _is_scalar(orientation)
    d = _to_df(orientation)
    qx, qy, qz, qw = d["x"], d["y"], d["z"], d["w"]

    # roll (x-axis rotation)
    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx**2 + qy**2)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis rotation)
    sinp = 2.0 * (qw * qy - qz * qx)
    pitch = np.where(np.abs(sinp) >= 1, np.copysign(np.pi / 2, sinp), np.arcsin(sinp))

    # yaw (z-axis rotation)
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy**2 + qz**2)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    if scalar:
        return {
            "roll": float(np.asarray(roll).flat[0]),
            "pitch": float(np.asarray(pitch).flat[0]),
            "yaw": float(np.asarray(yaw).flat[0]),
        }
    return pd.DataFrame({"roll": roll, "pitch": pitch, "yaw": yaw}, index=d.index)


def yaw_to_quat(yaw) -> pd.DataFrame | dict[str, float]:
    """Convert yaw angle (radians) to quaternion with zero roll and pitch.

    Parameters
    ----------
    yaw : float | pd.Series
        Yaw angle in radians.

    Returns
    -------
    dict[str, float] | pd.DataFrame
        ``{"x": …, "y": …, "z": …, "w": …}`` for scalar input,
        or DataFrame with columns ``x``, ``y``, ``z``, ``w``.

    """
    scalar = np.ndim(yaw) == 0
    half = np.asarray(yaw) * 0.5
    if scalar:
        return {"x": 0.0, "y": 0.0, "z": float(np.sin(half)), "w": float(np.cos(half))}
    return pd.DataFrame(
        {
            "x": np.zeros(len(yaw)),
            "y": np.zeros(len(yaw)),
            "z": np.sin(half),
            "w": np.cos(half),
        },
        index=yaw.index,
    )


def rpy_to_quat(roll, pitch, yaw) -> pd.DataFrame | dict[str, float]:
    """Convert roll, pitch, yaw (radians) to quaternion.

    Parameters
    ----------
    roll, pitch, yaw : float | pd.Series
        Euler angles in radians.

    Returns
    -------
    dict[str, float] | pd.DataFrame
        ``{"x": …, "y": …, "z": …, "w": …}`` for scalar input,
        or DataFrame with columns ``x``, ``y``, ``z``, ``w``.

    """
    scalar = np.ndim(yaw) == 0
    cr = np.cos(np.asarray(roll) * 0.5)
    sr = np.sin(np.asarray(roll) * 0.5)
    cp = np.cos(np.asarray(pitch) * 0.5)
    sp = np.sin(np.asarray(pitch) * 0.5)
    cy = np.cos(np.asarray(yaw) * 0.5)
    sy = np.sin(np.asarray(yaw) * 0.5)

    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    w = cr * cp * cy + sr * sp * sy

    if scalar:
        return {"x": float(x), "y": float(y), "z": float(z), "w": float(w)}
    return pd.DataFrame({"x": x, "y": y, "z": z, "w": w}, index=yaw.index)
