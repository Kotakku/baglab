"""Tests for baglab.stats module."""

import numpy as np
import pandas as pd
import pytest

from baglab.stats import describe, rms


# ===========================================================================
# rms
# ===========================================================================


class TestRms:
    def test_known_values(self):
        """rms([1, -1, 1, -1]) == 1.0."""
        assert rms(pd.Series([1, -1, 1, -1])) == pytest.approx(1.0)

    def test_constant_signal(self):
        """rms of a constant signal equals the absolute value of that constant."""
        assert rms(pd.Series([3.0, 3.0, 3.0, 3.0])) == pytest.approx(3.0)
        assert rms(pd.Series([-5.0, -5.0, -5.0])) == pytest.approx(5.0)

    def test_sine_wave(self):
        """rms of a sine wave with amplitude A is A / sqrt(2)."""
        amplitude = 4.0
        t = np.linspace(0, 2 * np.pi, 10_000, endpoint=False)
        signal = amplitude * np.sin(t)
        expected = amplitude / np.sqrt(2)
        assert rms(signal) == pytest.approx(expected, rel=1e-4)

    def test_numpy_array_input(self):
        arr = np.array([3.0, 4.0])
        # rms = sqrt((9 + 16) / 2) = sqrt(12.5)
        assert rms(arr) == pytest.approx(np.sqrt(12.5))

    def test_single_element(self):
        assert rms(pd.Series([7.0])) == pytest.approx(7.0)


# ===========================================================================
# describe
# ===========================================================================


class TestDescribe:
    def test_returns_all_expected_keys(self):
        result = describe(pd.Series([1.0, 2.0, 3.0]))
        expected_keys = {"mean", "std", "min", "max", "median", "p5", "p95", "rms"}
        assert set(result.keys()) == expected_keys

    def test_values_correct_for_known_input(self):
        x = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        result = describe(x)
        assert result["mean"] == pytest.approx(3.0)
        assert result["min"] == pytest.approx(1.0)
        assert result["max"] == pytest.approx(5.0)
        assert result["median"] == pytest.approx(3.0)
        assert result["std"] == pytest.approx(np.std([1, 2, 3, 4, 5], ddof=0))
        assert result["p5"] == pytest.approx(np.percentile([1, 2, 3, 4, 5], 5))
        assert result["p95"] == pytest.approx(np.percentile([1, 2, 3, 4, 5], 95))
        assert result["rms"] == pytest.approx(
            np.sqrt(np.mean(np.array([1, 2, 3, 4, 5]) ** 2))
        )

    def test_numpy_array_input(self):
        arr = np.array([10.0, 20.0, 30.0])
        result = describe(arr)
        assert result["mean"] == pytest.approx(20.0)
        assert result["min"] == pytest.approx(10.0)
        assert result["max"] == pytest.approx(30.0)
        assert result["median"] == pytest.approx(20.0)

    def test_constant_array(self):
        result = describe(np.array([5.0, 5.0, 5.0]))
        assert result["mean"] == pytest.approx(5.0)
        assert result["std"] == pytest.approx(0.0)
        assert result["min"] == pytest.approx(5.0)
        assert result["max"] == pytest.approx(5.0)
        assert result["rms"] == pytest.approx(5.0)

    def test_all_values_are_float(self):
        result = describe(pd.Series([1, 2, 3]))
        for v in result.values():
            assert isinstance(v, float)
