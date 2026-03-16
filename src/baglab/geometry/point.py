"""Position / Point operations."""

from __future__ import annotations

import numpy as np
import pandas as pd

from baglab.geometry._common import FieldInput, _is_scalar, _to_df


def to_xy(position: FieldInput) -> pd.DataFrame | dict[str, float]:
    """Extract x, y from a position/point.

    Parameters
    ----------
    position : FieldGroup | DataFrame | object
        Must contain fields/attributes ``x``, ``y``.

    Returns
    -------
    dict[str, float] | pd.DataFrame
        ``{"x": …, "y": …}`` for scalar input, or DataFrame with
        columns ``x``, ``y``.

    """
    scalar = _is_scalar(position)
    d = _to_df(position)
    if scalar:
        return {"x": float(d["x"].iloc[0]), "y": float(d["y"].iloc[0])}
    return pd.DataFrame({"x": d["x"], "y": d["y"]}, index=d.index)


def to_xyz(position: FieldInput) -> pd.DataFrame | dict[str, float]:
    """Extract x, y, z from a position/point.

    Parameters
    ----------
    position : FieldGroup | DataFrame | object
        Must contain fields/attributes ``x``, ``y``, ``z``.

    Returns
    -------
    dict[str, float] | pd.DataFrame
        ``{"x": …, "y": …, "z": …}`` for scalar input, or DataFrame with
        columns ``x``, ``y``, ``z``.

    """
    scalar = _is_scalar(position)
    d = _to_df(position)
    if scalar:
        return {"x": float(d["x"].iloc[0]), "y": float(d["y"].iloc[0]), "z": float(d["z"].iloc[0])}
    return pd.DataFrame({"x": d["x"], "y": d["y"], "z": d["z"]}, index=d.index)


def distance_2d(pos1: FieldInput, pos2: FieldInput) -> pd.Series | float:
    """Euclidean distance between two positions in the xy-plane.

    Parameters
    ----------
    pos1, pos2 : FieldGroup | DataFrame | object
        Must contain fields/attributes ``x``, ``y``.

    Returns
    -------
    float | pd.Series

    """
    scalar = _is_scalar(pos1) and _is_scalar(pos2)
    d1, d2 = _to_df(pos1), _to_df(pos2)
    dx = d1["x"].values - d2["x"].values
    dy = d1["y"].values - d2["y"].values
    result = np.hypot(dx, dy)
    if scalar:
        return float(result[0])
    return pd.Series(result, index=d1.index, name="distance_2d")


def distance_3d(pos1: FieldInput, pos2: FieldInput) -> pd.Series | float:
    """Euclidean distance between two positions in 3D.

    Parameters
    ----------
    pos1, pos2 : FieldGroup | DataFrame | object
        Must contain fields/attributes ``x``, ``y``, ``z``.

    Returns
    -------
    float | pd.Series

    """
    scalar = _is_scalar(pos1) and _is_scalar(pos2)
    d1, d2 = _to_df(pos1), _to_df(pos2)
    dx = d1["x"].values - d2["x"].values
    dy = d1["y"].values - d2["y"].values
    dz = d1["z"].values - d2["z"].values
    result = np.sqrt(dx**2 + dy**2 + dz**2)
    if scalar:
        return float(result[0])
    return pd.Series(result, index=d1.index, name="distance_3d")


def cumulative_distance(position: FieldInput) -> pd.Series:
    """Cumulative travel distance from consecutive xy positions.

    Parameters
    ----------
    position : FieldGroup | DataFrame
        Must contain fields ``x``, ``y``.

    Returns
    -------
    pd.Series
        Cumulative distance starting from 0.

    """
    d = _to_df(position)
    dx = np.diff(d["x"].values, prepend=d["x"].values[0])
    dy = np.diff(d["y"].values, prepend=d["y"].values[0])
    return pd.Series(
        np.cumsum(np.hypot(dx, dy)), index=d.index, name="cumulative_distance"
    )
