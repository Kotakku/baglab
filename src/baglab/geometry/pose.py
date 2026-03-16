"""Pose composite operations."""

from __future__ import annotations

import numpy as np
import pandas as pd

from baglab.geometry._common import FieldInput, _to_df
from baglab.geometry.quaternion import quat_to_yaw


def pose_to_xyyaw(pose) -> pd.DataFrame | dict[str, float]:
    """Extract x, y, yaw from a Pose.

    Parameters
    ----------
    pose : FieldGroup | DataFrame | object
        Accepted formats:

        - DataFrame with ``position.x``, ``position.y``,
          ``orientation.x/y/z/w`` columns.
        - FieldGroup pointing at a Pose level.
        - ROS msg Pose object with ``position`` and ``orientation``
          attributes (returns dict).

    Returns
    -------
    dict[str, float] | pd.DataFrame
        ``{"x": …, "y": …, "yaw": …}`` for scalar input,
        or DataFrame with columns ``x``, ``y``, ``yaw``.

    """
    from baglab.io import FieldGroup

    if isinstance(pose, (FieldGroup, pd.DataFrame)):
        d = _to_df(pose)
        yaw = quat_to_yaw(
            d[["orientation.x", "orientation.y", "orientation.z", "orientation.w"]].rename(
                columns=lambda c: c.split(".")[-1]
            )
        )
        return pd.DataFrame(
            {"x": d["position.x"], "y": d["position.y"], "yaw": yaw},
            index=d.index,
        )

    # Scalar: ROS msg Pose with .position and .orientation attributes
    return {
        "x": float(pose.position.x),
        "y": float(pose.position.y),
        "yaw": quat_to_yaw(pose.orientation),
    }


def _extract_xyyaw(src):
    """Extract x, y, yaw arrays from various input formats.

    Supports:
    - DataFrame with ``position.x`` + ``orientation.x/y/z/w`` (Pose format)
    - DataFrame with ``x``, ``y``, ``yaw`` columns (xyyaw format)
    - dict/object with ``x``, ``y``, ``yaw`` keys/attributes (scalar)

    Returns (x, y, yaw, index, is_scalar).
    """
    # DataFrame — check column format
    if isinstance(src, pd.DataFrame):
        if "position.x" in src.columns:
            # Pose format
            x = src["position.x"].values
            y = src["position.y"].values
            yaw = quat_to_yaw(
                src[["orientation.x", "orientation.y", "orientation.z", "orientation.w"]].rename(
                    columns=lambda c: c.split(".")[-1]
                )
            ).values
            return x, y, yaw, src.index, False
        else:
            # xyyaw format
            return src["x"].values, src["y"].values, src["yaw"].values, src.index, False

    # FieldGroup — delegate to _to_df then recurse
    from baglab.io import FieldGroup
    if isinstance(src, FieldGroup):
        return _extract_xyyaw(_to_df(src))

    # Scalar: dict or object with x, y, yaw
    if isinstance(src, dict):
        return (
            np.array([src["x"]]),
            np.array([src["y"]]),
            np.array([src["yaw"]]),
            None,
            True,
        )
    if hasattr(src, "x") and hasattr(src, "y") and hasattr(src, "yaw"):
        return (
            np.array([src.x]),
            np.array([src.y]),
            np.array([src.yaw]),
            None,
            True,
        )

    raise TypeError(
        f"Cannot extract x, y, yaw from {type(src).__name__}. "
        "Expected a Pose DataFrame, xyyaw DataFrame/dict, or object with x/y/yaw attributes."
    )


def pose_error(
    actual_pose,
    ref_pose,
    frame: str = "xy",
):
    """Compute pose error between actual and reference poses.

    Returns along-track, cross-track, and yaw errors.  The decomposition axes
    depend on *frame*.

    Parameters
    ----------
    actual_pose, ref_pose : FieldGroup | DataFrame | dict | object
        Pose data.  Accepted formats:

        - DataFrame with ``position.x``, ``position.y``,
          ``orientation.x/y/z/w`` (Pose format).
        - DataFrame with ``x``, ``y``, ``yaw`` columns (xyyaw format).
        - dict or object with ``x``, ``y``, ``yaw`` keys/attributes
          (scalar — returns dict).
    frame : str
        Coordinate frame for along/cross decomposition:

        - ``"xy"`` (default): world axes. along = dx, cross = dy.
        - ``"ref_heading"``: reference yaw direction.
        - ``"ref_path"``: tangent of the reference path (estimated from
          consecutive positions).

    Returns
    -------
    pd.DataFrame | dict[str, float]
        Columns/keys: ``along``, ``cross``, ``yaw``.

    """
    act_x, act_y, act_yaw, act_idx, act_scalar = _extract_xyyaw(actual_pose)
    ref_x, ref_y, ref_yaw, ref_idx, ref_scalar = _extract_xyyaw(ref_pose)
    scalar = act_scalar and ref_scalar

    dx = act_x - ref_x
    dy = act_y - ref_y
    yaw_err = (act_yaw - ref_yaw + np.pi) % (2 * np.pi) - np.pi

    if frame == "xy":
        along = dx
        cross = dy
    elif frame == "ref_heading":
        cos_r = np.cos(ref_yaw)
        sin_r = np.sin(ref_yaw)
        along = dx * cos_r + dy * sin_r
        cross = -dx * sin_r + dy * cos_r
    elif frame == "ref_path":
        tangent_x = np.gradient(ref_x)
        tangent_y = np.gradient(ref_y)
        path_yaw = np.arctan2(tangent_y, tangent_x)
        cos_p = np.cos(path_yaw)
        sin_p = np.sin(path_yaw)
        along = dx * cos_p + dy * sin_p
        cross = -dx * sin_p + dy * cos_p
    else:
        raise ValueError(f"Unknown frame: {frame!r}. Use 'xy', 'ref_heading', or 'ref_path'.")

    if scalar:
        return {
            "along": float(along[0]),
            "cross": float(cross[0]),
            "yaw": float(yaw_err[0]),
        }

    index = act_idx if act_idx is not None else ref_idx
    return pd.DataFrame(
        {"along": along, "cross": cross, "yaw": yaw_err},
        index=index,
    )
