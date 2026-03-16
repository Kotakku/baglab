"""2-D trajectory plotting."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import matplotlib.axes
    import pandas as pd


def plot_xy_trajectory(
    x: pd.Series | np.ndarray,
    y: pd.Series | np.ndarray,
    ref_x: pd.Series | np.ndarray | None = None,
    ref_y: pd.Series | np.ndarray | None = None,
    label: str = "actual",
    ref_label: str = "reference",
    title: str | None = None,
    ax: matplotlib.axes.Axes | None = None,
) -> matplotlib.axes.Axes:
    """Plot a 2-D trajectory with optional reference path.

    Parameters
    ----------
    x, y : pd.Series | np.ndarray
        Actual path coordinates.
    ref_x, ref_y : pd.Series | np.ndarray | None
        Optional reference path coordinates.
    label : str
        Label for the actual path.
    ref_label : str
        Label for the reference path.
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

    x_arr = np.asarray(x)
    y_arr = np.asarray(y)

    ax.plot(x_arr, y_arr, label=label)
    ax.plot(x_arr[0], y_arr[0], "go", markersize=8, label="start")
    ax.plot(x_arr[-1], y_arr[-1], "rs", markersize=8, label="end")

    if ref_x is not None and ref_y is not None:
        ax.plot(np.asarray(ref_x), np.asarray(ref_y), "--", label=ref_label)

    if title is not None:
        ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    ax.grid(True)
    ax.legend()

    return ax
