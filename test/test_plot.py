"""Tests for baglab.plot module."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from baglab.plot import error_band, step_response_plot, timeseries, xy_trajectory


@pytest.fixture(autouse=True)
def _close_figs():
    """Close all figures after each test."""
    yield
    plt.close("all")


# ---- timeseries ----


class TestTimeseries:
    def test_single_series(self):
        t = np.linspace(0, 1, 50)
        y = np.sin(t)
        ax = timeseries(t, y)
        assert isinstance(ax, matplotlib.axes.Axes)

    def test_multiple_series_with_labels(self):
        t = np.linspace(0, 1, 50)
        y1 = np.sin(t)
        y2 = np.cos(t)
        ax = timeseries(t, y1, y2, labels=["sin", "cos"], ylabel="value", title="trig")
        assert isinstance(ax, matplotlib.axes.Axes)
        legend = ax.get_legend()
        assert legend is not None
        assert len(legend.get_texts()) == 2

    def test_custom_ax(self):
        fig, ax_in = plt.subplots()
        t = np.linspace(0, 1, 20)
        ax_out = timeseries(t, t**2, ax=ax_in)
        assert ax_out is ax_in

    def test_pandas_inputs(self):
        t = pd.Series(np.linspace(0, 1, 30))
        y = pd.Series(np.random.randn(30))
        ax = timeseries(t, y, labels=["data"])
        assert isinstance(ax, matplotlib.axes.Axes)


# ---- xy_trajectory ----


class TestXYTrajectory:
    def test_basic_path(self):
        x = np.linspace(0, 10, 100)
        y = np.sin(x)
        ax = xy_trajectory(x, y)
        assert isinstance(ax, matplotlib.axes.Axes)

    def test_with_reference(self):
        x = np.linspace(0, 10, 100)
        y = np.sin(x)
        ref_x = x
        ref_y = np.sin(x) + 0.1
        ax = xy_trajectory(x, y, ref_x=ref_x, ref_y=ref_y, title="traj")
        assert isinstance(ax, matplotlib.axes.Axes)

    def test_equal_aspect(self):
        x = np.array([0.0, 1.0, 2.0])
        y = np.array([0.0, 1.0, 0.0])
        ax = xy_trajectory(x, y)
        # matplotlib may return "equal" or 1.0 depending on version
        assert ax.get_aspect() in ("equal", 1.0)


# ---- step_response_plot ----


class TestStepResponsePlot:
    def test_basic(self):
        t = np.linspace(0, 5, 200)
        actual = 1 - np.exp(-t)
        ax = step_response_plot(t, actual, target=1.0, title="step")
        assert isinstance(ax, matplotlib.axes.Axes)

    def test_series_target(self):
        t = np.linspace(0, 5, 200)
        actual = 1 - np.exp(-t)
        target = np.ones_like(t) * 1.0
        ax = step_response_plot(t, actual, target=target)
        assert isinstance(ax, matplotlib.axes.Axes)


# ---- error_band ----


class TestErrorBand:
    def test_basic(self):
        t = np.linspace(0, 5, 100)
        err = np.random.randn(100) * 0.1
        ax = error_band(t, err, sigma=0.1, label="err")
        assert isinstance(ax, matplotlib.axes.Axes)

    def test_auto_sigma(self):
        t = np.linspace(0, 5, 100)
        err = np.random.randn(100) * 0.3
        ax = error_band(t, err, title="auto sigma")
        assert isinstance(ax, matplotlib.axes.Axes)

    def test_pandas_input(self):
        t = pd.Series(np.linspace(0, 2, 50))
        err = pd.Series(np.random.randn(50) * 0.2)
        ax = error_band(t, err)
        assert isinstance(ax, matplotlib.axes.Axes)
