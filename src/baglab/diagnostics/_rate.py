"""Topic rate and gap analysis utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd


def topic_rate(timestamps: pd.Series | pd.DatetimeIndex) -> dict[str, float | int]:
    """Compute topic publish rate statistics.

    Parameters
    ----------
    timestamps : pd.Series | pd.DatetimeIndex
        Timestamps as float seconds (e.g., from ``stamp_to_sec``) or as a
        ``pd.DatetimeIndex``.

    Returns
    -------
    dict[str, float | int]
        Dictionary with keys ``"mean_hz"``, ``"std_hz"``, ``"min_dt"``,
        ``"max_dt"``, ``"count"``, ``"duration"``.

    """
    if isinstance(timestamps, pd.DatetimeIndex):
        ts = np.array(
            (timestamps - timestamps[0]).total_seconds(), dtype=float
        )
    else:
        ts = np.asarray(timestamps, dtype=float)

    count = len(ts)
    if count < 2:
        return {
            "mean_hz": 0.0,
            "std_hz": 0.0,
            "min_dt": 0.0,
            "max_dt": 0.0,
            "count": count,
            "duration": 0.0,
        }

    dt = np.diff(ts)
    duration = float(ts[-1] - ts[0])
    mean_dt = float(np.mean(dt))
    mean_hz = 1.0 / mean_dt if mean_dt > 0 else 0.0
    std_dt = float(np.std(dt, ddof=0))
    # Propagate rate std via delta method: std_hz ≈ std_dt / mean_dt^2
    std_hz = std_dt / (mean_dt ** 2) if mean_dt > 0 else 0.0

    return {
        "mean_hz": mean_hz,
        "std_hz": std_hz,
        "min_dt": float(np.min(dt)),
        "max_dt": float(np.max(dt)),
        "count": count,
        "duration": duration,
    }


def message_gaps(
    timestamps: pd.Series,
    expected_rate: float,
) -> pd.DataFrame:
    """Find gaps (missing messages) in a topic.

    A gap is defined as a period longer than ``1.5 * (1 / expected_rate)``.

    Parameters
    ----------
    timestamps : pd.Series
        Timestamps as float seconds.
    expected_rate : float
        Expected publish rate in Hz.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ``"start"``, ``"end"``, ``"duration"``,
        ``"expected_count"`` for each detected gap.

    """
    ts = np.asarray(timestamps, dtype=float)
    if len(ts) < 2:
        return pd.DataFrame(columns=["start", "end", "duration", "expected_count"])

    dt = np.diff(ts)
    expected_period = 1.0 / expected_rate
    threshold = 1.5 * expected_period
    gap_mask = dt > threshold

    if not np.any(gap_mask):
        return pd.DataFrame(columns=["start", "end", "duration", "expected_count"])

    gap_indices = np.where(gap_mask)[0]
    rows: list[dict[str, float | int]] = []
    for idx in gap_indices:
        start = float(ts[idx])
        end = float(ts[idx + 1])
        duration = end - start
        expected_count = int(np.round(duration * expected_rate))
        rows.append(
            {
                "start": start,
                "end": end,
                "duration": duration,
                "expected_count": expected_count,
            }
        )

    return pd.DataFrame(rows, columns=["start", "end", "duration", "expected_count"])
