"""Publish-to-receive delay analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd


def topic_delay(
    pub_stamp: pd.Series,
    recv_stamp: pd.Series,
) -> dict[str, pd.Series | float]:
    """Compute publish-to-receive delay statistics.

    Parameters
    ----------
    pub_stamp : pd.Series
        Publish timestamps as float seconds (header.stamp converted).
    recv_stamp : pd.Series
        Receive timestamps as float seconds (index converted).

    Returns
    -------
    dict[str, pd.Series | float]
        Dictionary with keys ``"delay"`` (per-message ``pd.Series``),
        ``"mean"``, ``"max"``, ``"std"``, ``"median"``.

    """
    pub = np.asarray(pub_stamp, dtype=float)
    recv = np.asarray(recv_stamp, dtype=float)
    delay_arr = recv - pub
    delay = pd.Series(delay_arr, index=pub_stamp.index if hasattr(pub_stamp, "index") else None)

    return {
        "delay": delay,
        "mean": float(np.mean(delay_arr)),
        "max": float(np.max(delay_arr)),
        "std": float(np.std(delay_arr, ddof=0)),
        "median": float(np.median(delay_arr)),
    }
