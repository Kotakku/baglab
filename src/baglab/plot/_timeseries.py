"""Time-series plotting."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import matplotlib.axes
    import pandas as pd


def timeseries(
    t: pd.Series | np.ndarray,
    *series: pd.Series | np.ndarray,
    labels: list[str] | None = None,
    ylabel: str | None = None,
    title: str | None = None,
    ax: matplotlib.axes.Axes | None = None,
) -> matplotlib.axes.Axes:
    """Plot one or more time series on the same axes.

    Parameters
    ----------
    t : pd.Series | np.ndarray
        Time axis.
    *series : pd.Series | np.ndarray
        One or more data series to plot against *t*.
    labels : list[str] | None
        Legend labels.  If provided, a legend is shown.
    ylabel : str | None
        Y-axis label.
    title : str | None
        Axes title.
    ax : matplotlib.axes.Axes | None
        Axes to draw on.  A new figure is created when *None*.

    Returns
    -------
    matplotlib.axes.Axes
        The axes with the plot.

    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots()

    for i, s in enumerate(series):
        label = labels[i] if labels is not None and i < len(labels) else None
        ax.plot(t, s, label=label)

    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)
    if labels is not None:
        ax.legend()
    ax.set_xlabel("time [s]")
    ax.grid(True)

    return ax
