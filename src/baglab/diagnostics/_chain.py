"""Pipeline latency chain analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd


def latency_chain(
    timestamps_dict: dict[str, pd.Series],
) -> pd.DataFrame:
    """Visualize pipeline latency across multiple topics.

    For each topic, computes rate statistics and relative timing.

    Parameters
    ----------
    timestamps_dict : dict[str, pd.Series]
        Mapping of topic name to timestamp series (float seconds).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ``"topic"``, ``"mean_hz"``, ``"mean_dt"``,
        ``"std_dt"``, ``"min_dt"``, ``"max_dt"``, sorted by topic name.

    """
    rows: list[dict[str, str | float]] = []
    for topic_name in sorted(timestamps_dict.keys()):
        ts = np.asarray(timestamps_dict[topic_name], dtype=float)
        if len(ts) < 2:
            rows.append(
                {
                    "topic": topic_name,
                    "mean_hz": 0.0,
                    "mean_dt": 0.0,
                    "std_dt": 0.0,
                    "min_dt": 0.0,
                    "max_dt": 0.0,
                }
            )
            continue

        dt = np.diff(ts)
        mean_dt = float(np.mean(dt))
        rows.append(
            {
                "topic": topic_name,
                "mean_hz": 1.0 / mean_dt if mean_dt > 0 else 0.0,
                "mean_dt": mean_dt,
                "std_dt": float(np.std(dt, ddof=0)),
                "min_dt": float(np.min(dt)),
                "max_dt": float(np.max(dt)),
            }
        )

    return pd.DataFrame(
        rows, columns=["topic", "mean_hz", "mean_dt", "std_dt", "min_dt", "max_dt"]
    )
