"""Tests for baglab.signal module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from baglab.signal import diff, fft, integrate, lowpass, moving_average


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sine_series(freq: float, fs: float, duration: float) -> tuple[pd.Series, pd.Series]:
    """Return (x, t) for a pure sine wave."""
    t_arr = np.arange(0, duration, 1.0 / fs)
    x_arr = np.sin(2 * np.pi * freq * t_arr)
    t = pd.Series(t_arr)
    x = pd.Series(x_arr)
    return x, t


# ---------------------------------------------------------------------------
# lowpass
# ---------------------------------------------------------------------------

class TestLowpass:
    def test_removes_high_frequency_noise(self):
        """A 5 Hz sine + 50 Hz noise filtered at 10 Hz should recover the sine."""
        fs = 500.0
        t_arr = np.arange(0, 2.0, 1.0 / fs)
        clean = np.sin(2 * np.pi * 5.0 * t_arr)
        noisy = clean + 0.5 * np.sin(2 * np.pi * 50.0 * t_arr)

        x = pd.Series(noisy)
        filtered = lowpass(x, cutoff=10.0, fs=fs, order=4)

        # Trim edges to avoid filter transient effects
        mid = slice(100, -100)
        np.testing.assert_allclose(filtered.values[mid], clean[mid], atol=0.05)

    def test_returns_series_for_series_input(self):
        x = pd.Series(np.random.randn(200))
        result = lowpass(x, cutoff=10.0, fs=100.0)
        assert isinstance(result, pd.Series)
        assert len(result) == len(x)

    def test_returns_ndarray_for_ndarray_input(self):
        x = np.random.randn(200)
        result = lowpass(x, cutoff=10.0, fs=100.0)
        assert isinstance(result, np.ndarray)


# ---------------------------------------------------------------------------
# diff
# ---------------------------------------------------------------------------

class TestDiff:
    def test_linear_gives_constant_derivative(self):
        """d/dt of (2*t + 3) should be ~2 everywhere (except first element)."""
        t = pd.Series(np.linspace(0, 1, 100))
        x = pd.Series(2.0 * t.values + 3.0)

        result = diff(x, t)

        assert np.isnan(result.iloc[0])
        np.testing.assert_allclose(result.values[1:], 2.0, atol=1e-10)

    def test_quadratic_gives_linear_derivative(self):
        """d/dt of t^2 should be ~2*t."""
        t = pd.Series(np.linspace(0, 2, 1000))
        x = pd.Series(t.values ** 2)

        result = diff(x, t)

        # Compare from index 1 onward; backward diff = (x[i]-x[i-1])/(t[i]-t[i-1])
        expected = 2.0 * t.values[1:]
        np.testing.assert_allclose(result.values[1:], expected, atol=0.01)

    def test_output_length_matches_input(self):
        t = pd.Series(np.linspace(0, 1, 50))
        x = pd.Series(np.sin(t.values))
        assert len(diff(x, t)) == len(x)


# ---------------------------------------------------------------------------
# integrate
# ---------------------------------------------------------------------------

class TestIntegrate:
    def test_constant_gives_linear(self):
        """Integral of constant c over time should be c*t."""
        t = pd.Series(np.linspace(0, 5, 500))
        c = 3.0
        x = pd.Series(np.full(len(t), c))

        result = integrate(x, t)

        expected = c * t.values
        np.testing.assert_allclose(result.values, expected, atol=0.05)

    def test_first_element_is_zero(self):
        t = pd.Series(np.linspace(0, 1, 100))
        x = pd.Series(np.ones(100))
        result = integrate(x, t)
        assert result.iloc[0] == 0.0

    def test_roundtrip_with_diff(self):
        """integrate(diff(x)) should approximately recover x (up to constant)."""
        t = pd.Series(np.linspace(0, 2, 500))
        x = pd.Series(np.sin(2 * np.pi * t.values))

        dx = diff(x, t)
        # Replace first NaN so integration works
        dx.iloc[0] = dx.iloc[1]
        recovered = integrate(dx, t)

        # Shift by initial value
        recovered_shifted = recovered + x.iloc[0]
        np.testing.assert_allclose(
            recovered_shifted.values[10:], x.values[10:], atol=0.05
        )


# ---------------------------------------------------------------------------
# moving_average
# ---------------------------------------------------------------------------

class TestMovingAverage:
    def test_constant_signal_unchanged(self):
        x = pd.Series(np.full(100, 5.0))
        result = moving_average(x, window=11)
        # Interior (non-NaN) values should be 5.0
        valid = result.dropna()
        np.testing.assert_allclose(valid.values, 5.0, atol=1e-12)

    def test_output_length_matches_input(self):
        x = pd.Series(np.random.randn(100))
        result = moving_average(x, window=7)
        assert len(result) == len(x)

    def test_edges_are_nan(self):
        x = pd.Series(np.random.randn(50))
        result = moving_average(x, window=11)
        # With centered window=11, first 5 and last 5 should be NaN
        assert result.iloc[:5].isna().all()
        assert result.iloc[-5:].isna().all()


# ---------------------------------------------------------------------------
# fft
# ---------------------------------------------------------------------------

class TestFFT:
    def test_single_frequency_peak(self):
        """FFT of a 10 Hz sine should peak near 10 Hz."""
        fs = 200.0
        freq_target = 10.0
        t_arr = np.arange(0, 2.0, 1.0 / fs)
        x = pd.Series(np.sin(2 * np.pi * freq_target * t_arr))

        freq, amp = fft(x, fs)

        peak_idx = np.argmax(amp)
        assert abs(freq[peak_idx] - freq_target) < 1.0  # within 1 Hz

    def test_returns_only_positive_frequencies(self):
        fs = 100.0
        x = np.random.randn(256)
        freq, amp = fft(x, fs)
        assert freq[0] >= 0.0
        assert np.all(freq >= 0.0)

    def test_ndarray_input(self):
        x = np.sin(np.linspace(0, 4 * np.pi, 256))
        freq, amp = fft(x, fs=128.0)
        assert isinstance(freq, np.ndarray)
        assert isinstance(amp, np.ndarray)
        assert len(freq) == len(amp)
