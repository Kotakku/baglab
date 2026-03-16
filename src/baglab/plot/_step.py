"""Step response plotting."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import matplotlib.axes
    import pandas as pd


def plot_step_response(
    t: pd.Series | np.ndarray,
    actual: pd.Series | np.ndarray,
    target: float | pd.Series | np.ndarray,
    title: str | None = None,
    ax: matplotlib.axes.Axes | None = None,
) -> matplotlib.axes.Axes:
    """Plot step response with target line and settling band.

    A horizontal line is drawn at the target value and a shaded region
    indicates the +/-2 % settling band around the target.

    Parameters
    ----------
    t : pd.Series | np.ndarray
        Time axis.
    actual : pd.Series | np.ndarray
        Measured response.
    target : float | pd.Series | np.ndarray
        Target (setpoint) value.  If scalar, a horizontal line is drawn.
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
    actual_arr = np.asarray(actual)

    ax.plot(t_arr, actual_arr, label="actual")

    target_scalar = float(np.asarray(target).flat[0]) if np.ndim(target) == 0 else None

    if target_scalar is not None:
        ax.axhline(target_scalar, color="k", linestyle="--", label="target")
        band = abs(target_scalar) * 0.02
        ax.axhspan(
            target_scalar - band,
            target_scalar + band,
            alpha=0.15,
            color="green",
            label="\u00b12 % band",
        )
    else:
        target_arr = np.asarray(target)
        ax.plot(t_arr, target_arr, "k--", label="target")
        band = np.abs(target_arr) * 0.02
        ax.fill_between(
            t_arr,
            target_arr - band,
            target_arr + band,
            alpha=0.15,
            color="green",
            label="\u00b12 % band",
        )

    if title is not None:
        ax.set_title(title)
    ax.set_xlabel("time [s]")
    ax.grid(True)
    ax.legend()

    return ax
