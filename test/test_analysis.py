"""Tests for baglab.analysis module."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from baglab.analysis import delay_estimate, stepinfo, tracking_error


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _first_order_step(tau: float, dt: float = 0.001, duration: float = 5.0) -> tuple[pd.Series, pd.Series]:
    """Generate a first-order step response: y(t) = 1 - exp(-t/tau)."""
    t = np.arange(0, duration, dt)
    y = 1.0 - np.exp(-t / tau)
    return pd.Series(y), pd.Series(t)


def _second_order_underdamped(
    wn: float = 10.0,
    zeta: float = 0.3,
    dt: float = 0.001,
    duration: float = 5.0,
) -> tuple[pd.Series, pd.Series]:
    """Generate a second-order underdamped step response."""
    t = np.arange(0, duration, dt)
    wd = wn * np.sqrt(1 - zeta ** 2)
    y = 1.0 - np.exp(-zeta * wn * t) * (
        np.cos(wd * t) + (zeta / np.sqrt(1 - zeta ** 2)) * np.sin(wd * t)
    )
    return pd.Series(y), pd.Series(t)


# ---------------------------------------------------------------------------
# stepinfo tests
# ---------------------------------------------------------------------------

class TestStepinfo:
    def test_first_order_no_overshoot(self):
        """First-order system should have zero overshoot."""
        y, t = _first_order_step(tau=0.5)
        info = stepinfo(y, target=1.0, t=t)
        assert info["overshoot"] == pytest.approx(0.0, abs=0.1)
        assert info["steady_state_error"] < 0.01

    def test_second_order_underdamped_overshoot(self):
        """Second-order underdamped system should have known overshoot."""
        zeta = 0.3
        y, t = _second_order_underdamped(zeta=zeta)
        info = stepinfo(y, target=1.0, t=t)
        # Theoretical overshoot for 2nd order: exp(-pi*zeta/sqrt(1-zeta^2)) * 100
        theoretical = math.exp(-math.pi * zeta / math.sqrt(1 - zeta ** 2)) * 100
        assert info["overshoot"] == pytest.approx(theoretical, rel=0.05)

    def test_rise_time_ramp_like(self):
        """Rise time for a ramp-like response that crosses 10% and 90%."""
        dt = 0.001
        t = np.arange(0, 2.0, dt)
        # Linear ramp from 0 to 1 over [0, 1], then constant 1
        y = np.clip(t, 0.0, 1.0)
        info = stepinfo(pd.Series(y), target=1.0, t=pd.Series(t))
        # 10% at t=0.1, 90% at t=0.9 → rise_time ≈ 0.8
        assert info["rise_time"] == pytest.approx(0.8, abs=0.01)

    def test_settling_time_first_order(self):
        """First-order system settling time should be about 4*tau (2% criterion)."""
        tau = 0.5
        y, t = _first_order_step(tau=tau, duration=10.0)
        info = stepinfo(y, target=1.0, t=t)
        # Theoretical: -tau*ln(0.02) ≈ 3.91*tau
        theoretical_settling = -tau * math.log(0.02)
        assert info["settling_time"] == pytest.approx(theoretical_settling, rel=0.05)

    def test_target_as_series(self):
        """When target is a Series, final value should be used."""
        y, t = _first_order_step(tau=0.5)
        target_series = pd.Series(np.linspace(0.5, 1.0, len(y)))
        info = stepinfo(y, target=target_series, t=t)
        # Should use 1.0 as the final target
        assert info["steady_state_error"] < 0.01

    def test_no_step_returns_nan(self):
        """If target == initial value (no step), rise/settling should be NaN."""
        t = pd.Series(np.arange(0, 1.0, 0.01))
        y = pd.Series(np.ones(len(t)))
        info = stepinfo(y, target=1.0, t=t)
        assert math.isnan(info["rise_time"])
        assert math.isnan(info["settling_time"])
        assert info["overshoot"] == 0.0


# ---------------------------------------------------------------------------
# tracking_error tests
# ---------------------------------------------------------------------------

class TestTrackingError:
    def test_zero_error(self):
        """When actual == ref, all metrics should be zero."""
        n = 100
        vals = pd.Series(np.sin(np.linspace(0, 2 * np.pi, n)))
        result = tracking_error(vals, vals)
        assert result["mean"] == pytest.approx(0.0, abs=1e-12)
        assert result["max"] == pytest.approx(0.0, abs=1e-12)
        assert result["rms"] == pytest.approx(0.0, abs=1e-12)
        assert result["std"] == pytest.approx(0.0, abs=1e-12)
        assert len(result["error"]) == n

    def test_known_error(self):
        """Check metrics for a known constant offset."""
        n = 100
        ref = pd.Series(np.zeros(n))
        actual = pd.Series(np.ones(n) * 3.0)
        result = tracking_error(actual, ref)
        assert result["mean"] == pytest.approx(3.0)
        assert result["max"] == pytest.approx(3.0)
        assert result["rms"] == pytest.approx(3.0)
        assert result["std"] == pytest.approx(0.0, abs=1e-12)

    def test_rms_calculation(self):
        """RMS of [1, -1, 1, -1, ...] should be 1.0."""
        n = 100
        ref = pd.Series(np.zeros(n))
        error_vals = np.ones(n)
        error_vals[1::2] = -1.0
        actual = pd.Series(error_vals)
        result = tracking_error(actual, ref)
        assert result["rms"] == pytest.approx(1.0)
        assert result["mean"] == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# delay_estimate tests
# ---------------------------------------------------------------------------

class TestDelayEstimate:
    def test_shifted_sine(self):
        """Shifted sine wave should yield correct delay."""
        dt = 0.001
        duration = 2.0
        delay_true = 0.05  # 50 ms
        t = np.arange(0, duration, dt)
        freq = 5.0
        cmd = pd.Series(np.sin(2 * np.pi * freq * t))
        response = pd.Series(np.sin(2 * np.pi * freq * (t - delay_true)))
        estimated = delay_estimate(cmd, response, pd.Series(t))
        assert estimated == pytest.approx(delay_true, abs=2 * dt)

    def test_zero_delay(self):
        """Identical signals should yield ~0 delay."""
        dt = 0.001
        t = np.arange(0, 1.0, dt)
        sig = pd.Series(np.sin(2 * np.pi * 3.0 * t))
        estimated = delay_estimate(sig, sig, pd.Series(t))
        assert estimated == pytest.approx(0.0, abs=2 * dt)
