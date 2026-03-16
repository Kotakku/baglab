"""Timestamp utilities for rosbag DataFrames."""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd


def stamp_to_sec(
    df: pd.DataFrame,
    relative: bool = False,
    stamp: str = "header.stamp",
) -> pd.Series:
    """Combine stamp sec and nanosec columns into float seconds.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing ``{stamp}.sec`` and ``{stamp}.nanosec`` columns.
    relative : bool
        If True, return time relative to the first message (starting from 0).
    stamp : str
        Column prefix for the builtin_interfaces/Time field.
        Default ``"header.stamp"``.

    Returns
    -------
    pd.Series
        Timestamp in seconds as float64.

    """
    t = df[f"{stamp}.sec"] + df[f"{stamp}.nanosec"] * 1e-9
    if relative:
        t = t - t.iloc[0]
    return t


def reindex_by_stamp(df: pd.DataFrame) -> pd.DataFrame:
    """Replace index with header.stamp as DatetimeIndex.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing ``header.stamp.sec`` and ``header.stamp.nanosec`` columns.

    Returns
    -------
    pd.DataFrame
        New DataFrame with publish-time index. Original is not modified.

    """
    result = df.copy()
    ns = df["header.stamp.sec"] * 10**9 + df["header.stamp.nanosec"]
    result.index = pd.to_datetime(ns.astype("int64"), unit="ns")
    return result


def time_slice(
    df: pd.DataFrame,
    t_start: float,
    t_end: float,
) -> pd.DataFrame:
    """Slice a DatetimeIndex DataFrame by relative time in seconds.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a DatetimeIndex.
    t_start : float
        Start time in seconds relative to ``df.index[0]``.
    t_end : float
        End time in seconds relative to ``df.index[0]``.

    Returns
    -------
    pd.DataFrame
        Subset of *df* within the requested time window.
        The original DataFrame is not modified.

    """
    origin = df.index[0]
    start = origin + pd.Timedelta(seconds=t_start)
    end = origin + pd.Timedelta(seconds=t_end)
    return df.loc[start:end].copy()


def find_time(
    series: pd.Series,
    condition: Callable[[pd.Series], pd.Series],
) -> float:
    """Return the relative time of the first element satisfying *condition*.

    Parameters
    ----------
    series : pd.Series
        Series with a DatetimeIndex.
    condition : Callable[[pd.Series], pd.Series]
        A callable that accepts the Series values and returns a boolean Series.
        For example ``lambda x: x >= 0.5``.

    Returns
    -------
    float
        Seconds from ``series.index[0]`` to the first matching element.

    Raises
    ------
    ValueError
        If no element satisfies the condition.

    """
    mask = condition(series)
    if not mask.any():
        raise ValueError("No element satisfies the condition.")
    first_idx = series.index[mask.values][0]
    return (first_idx - series.index[0]).total_seconds()
