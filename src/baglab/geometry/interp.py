"""Interpolation and time alignment utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd


def match_by_time(
    t_query: pd.Series | np.ndarray,
    t_source: pd.Series | np.ndarray,
) -> np.ndarray:
    """Find the nearest source index for each query time.

    Uses binary search (``np.searchsorted``) for O(N log M) performance.

    Parameters
    ----------
    t_query : pd.Series | ndarray
        Query times (need not be sorted).
    t_source : pd.Series | ndarray
        Source times (must be monotonically increasing).

    Returns
    -------
    np.ndarray
        Integer indices into *t_source*, one per element in *t_query*.

    """
    q = np.asarray(t_query, dtype=float)
    s = np.asarray(t_source, dtype=float)

    idx = np.searchsorted(s, q, side="right")
    idx = np.clip(idx, 1, len(s) - 1)

    # Compare distance to left and right neighbours
    left = idx - 1
    right = idx
    mask = np.abs(q - s[left]) <= np.abs(q - s[right])
    return np.where(mask, left, right)


def align_time(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    time_col: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Align two DataFrames by time via linear interpolation.

    The output DataFrames share the same time index (the union of both input
    indices, restricted to the overlapping time range).  Numeric columns are
    interpolated linearly; non-numeric columns are forward-filled.

    Parameters
    ----------
    df1, df2 : pd.DataFrame
        DataFrames to align.  If *time_col* is ``None``, the existing index
        (assumed to be a DatetimeIndex or numeric) is used.  Otherwise the
        named column is used as the time axis.
    time_col : str | None
        Column name to use as time.  When provided, this column is set as
        the index before interpolation.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        Aligned copies of *df1* and *df2*.

    """
    a = df1.copy()
    b = df2.copy()

    if time_col is not None:
        a = a.set_index(time_col)
        b = b.set_index(time_col)

    # Sort by index
    a = a.sort_index()
    b = b.sort_index()

    # Common time range
    t_start = max(a.index[0], b.index[0])
    t_end = min(a.index[-1], b.index[-1])

    # Union index within overlapping range
    common = a.index.union(b.index)
    common = common[(common >= t_start) & (common <= t_end)]

    def _reindex(df: pd.DataFrame, idx: pd.Index) -> pd.DataFrame:
        # Include all original index points for accurate interpolation,
        # then restrict to the requested index.
        full_idx = df.index.union(idx).sort_values()
        result = df.reindex(full_idx)
        numeric_cols = result.select_dtypes(include="number").columns
        if len(numeric_cols) > 0:
            result[numeric_cols] = result[numeric_cols].interpolate(method="index")
        non_numeric = result.columns.difference(numeric_cols)
        if len(non_numeric) > 0:
            result[non_numeric] = result[non_numeric].ffill()
        return result.reindex(idx)

    return _reindex(a, common), _reindex(b, common)


def resample(df: pd.DataFrame, dt: float, time_col: str | None = None) -> pd.DataFrame:
    """Resample a DataFrame at uniform time intervals.

    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    dt : float
        Desired time step in seconds.
    time_col : str | None
        Column name to use as time.  If ``None``, the index is used
        (must be numeric, e.g. from :func:`~baglab.stamp_to_sec`).

    Returns
    -------
    pd.DataFrame
        Resampled data with uniform time index.

    """
    d = df.copy()
    if time_col is not None:
        d = d.set_index(time_col)
    d = d.sort_index()

    new_index = np.arange(d.index[0], d.index[-1], dt)
    new_index_pd = pd.Index(new_index, name=d.index.name)
    result = d.reindex(d.index.union(new_index_pd))

    numeric_cols = result.select_dtypes(include="number").columns
    if len(numeric_cols) > 0:
        result[numeric_cols] = result[numeric_cols].interpolate(method="index")
    non_numeric = result.columns.difference(numeric_cols)
    if len(non_numeric) > 0:
        result[non_numeric] = result[non_numeric].ffill()

    return result.loc[new_index_pd]


def interp_pose(
    df: pd.DataFrame,
    target_time: pd.Series | np.ndarray,
    position_cols: tuple[str, str] = ("position.x", "position.y"),
    orientation_cols: tuple[str, str, str, str] = (
        "orientation.x",
        "orientation.y",
        "orientation.z",
        "orientation.w",
    ),
) -> pd.DataFrame:
    """Interpolate pose data at target times.

    Position is linearly interpolated.  Orientation is interpolated via
    Spherical Linear Interpolation (SLERP) of quaternions.

    Parameters
    ----------
    df : pd.DataFrame
        Must have a numeric index (time in seconds) and contain the
        position and orientation columns.
    target_time : array-like
        Times at which to interpolate.
    position_cols : tuple[str, str]
        Column names for position x and y.
    orientation_cols : tuple[str, str, str, str]
        Column names for quaternion x, y, z, w.

    Returns
    -------
    pd.DataFrame
        Interpolated pose with columns matching *position_cols* and
        *orientation_cols*.

    """
    src_t = df.index.values.astype(float)
    tgt_t = np.asarray(target_time, dtype=float)

    # Interpolate position linearly
    px = np.interp(tgt_t, src_t, df[position_cols[0]].values)
    py = np.interp(tgt_t, src_t, df[position_cols[1]].values)

    # Interpolate quaternion via SLERP
    quats = np.column_stack(
        [df[c].values for c in orientation_cols]
    )  # (N, 4) as x,y,z,w
    interp_q = _slerp_series(src_t, quats, tgt_t)

    result = pd.DataFrame(
        {
            position_cols[0]: px,
            position_cols[1]: py,
            orientation_cols[0]: interp_q[:, 0],
            orientation_cols[1]: interp_q[:, 1],
            orientation_cols[2]: interp_q[:, 2],
            orientation_cols[3]: interp_q[:, 3],
        },
        index=pd.Index(tgt_t, name=df.index.name),
    )
    return result


def _slerp_series(
    src_t: np.ndarray, quats: np.ndarray, tgt_t: np.ndarray
) -> np.ndarray:
    """SLERP quaternion interpolation at target times.

    Parameters
    ----------
    src_t : ndarray, shape (M,)
        Source times (sorted).
    quats : ndarray, shape (M, 4)
        Source quaternions (x, y, z, w).
    tgt_t : ndarray, shape (N,)
        Target times.

    Returns
    -------
    ndarray, shape (N, 4)

    """
    indices = np.searchsorted(src_t, tgt_t, side="right") - 1
    indices = np.clip(indices, 0, len(src_t) - 2)

    t0 = src_t[indices]
    t1 = src_t[indices + 1]
    dt = t1 - t0
    # Avoid division by zero for duplicate timestamps
    dt = np.where(dt == 0, 1.0, dt)
    alpha = (tgt_t - t0) / dt
    alpha = np.clip(alpha, 0.0, 1.0)

    q0 = quats[indices]
    q1 = quats[indices + 1]

    # Ensure shortest path
    dot = np.sum(q0 * q1, axis=1)
    q1 = np.where(dot[:, None] < 0, -q1, q1)
    dot = np.abs(dot)

    # For nearly identical quaternions, use linear interpolation
    linear_mask = dot > 0.9995
    theta = np.arccos(np.clip(dot, -1.0, 1.0))
    sin_theta = np.sin(theta)
    # Avoid division by zero
    sin_theta = np.where(sin_theta == 0, 1.0, sin_theta)

    s0 = np.sin((1.0 - alpha) * theta) / sin_theta
    s1 = np.sin(alpha * theta) / sin_theta

    # Fall back to linear interpolation where quaternions are nearly parallel
    s0 = np.where(linear_mask, 1.0 - alpha, s0)
    s1 = np.where(linear_mask, alpha, s1)

    result = s0[:, None] * q0 + s1[:, None] * q1
    # Normalize
    norms = np.linalg.norm(result, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return result / norms
