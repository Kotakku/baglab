"""Error band plotting."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import matplotlib.axes
    import pandas as pd


def error_band(
    t: pd.Series | np.ndarray,
    error: pd.Series | np.ndarray,
    sigma: float | None = None,
    label: str | None = None,
    title: str | None = None,
    ax: matplotlib.axes.Axes | None = None,
) -> matplotlib.axes.Axes:
    """Plot error time series with optional +/-sigma shaded band.

    Parameters
    ----------
    t : pd.Series | np.ndarray
        Time axis.
    error : pd.Series | np.ndarray
        Error values.
    sigma : float | None
        Standard deviation for the band.  Computed from *error* when *None*.
    label : str | None
        Label for the error line.
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

    t_arr = np.asarray(t)
    err_arr = np.asarray(error, dtype=float)

    if sigma is None:
        sigma = float(np.std(err_arr))

    ax.plot(t_arr, err_arr, label=label)
    ax.axhline(0, color="k", linewidth=0.8)
    ax.fill_between(
        t_arr,
        -sigma * np.ones_like(t_arr),
        sigma * np.ones_like(t_arr),
        alpha=0.2,
        color="orange",
        label=f"\u00b1\u03c3 ({sigma:.3g})",
    )

    if title is not None:
        ax.set_title(title)
    if label is not None:
        ax.legend()
    else:
        ax.legend(loc="upper right")
    ax.set_xlabel("time [s]")
    ax.grid(True)

    return ax
