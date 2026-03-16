"""Tests for time_slice and find_time utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from baglab.io.stamp import find_time, time_slice


def _make_df(n: int = 100, freq_hz: float = 10.0) -> pd.DataFrame:
    """Create a DataFrame with a DatetimeIndex at *freq_hz* Hz."""
    origin = pd.Timestamp("2024-01-01")
    index = pd.to_datetime(
        [origin + pd.Timedelta(seconds=i / freq_hz) for i in range(n)]
    )
    return pd.DataFrame({"data": np.arange(n, dtype=float)}, index=index)


# -- time_slice tests --------------------------------------------------------


class TestTimeSlice:
    def test_basic_slicing(self):
        df = _make_df(100, freq_hz=10.0)
        sliced = time_slice(df, 1.0, 3.0)
        assert len(sliced) > 0
        assert len(sliced) < len(df)

    def test_returns_correct_subset(self):
        df = _make_df(100, freq_hz=10.0)
        sliced = time_slice(df, 1.0, 2.0)
        # 10 Hz -> rows at 1.0, 1.1, ..., 2.0 => 11 rows
        assert len(sliced) == 11

    def test_float_boundaries(self):
        df = _make_df(100, freq_hz=10.0)
        sliced = time_slice(df, 0.5, 1.5)
        # rows at 0.5, 0.6, ..., 1.5 => 11 rows
        assert len(sliced) == 11

    def test_does_not_modify_original(self):
        df = _make_df(50, freq_hz=10.0)
        original_len = len(df)
        _ = time_slice(df, 1.0, 2.0)
        assert len(df) == original_len

    def test_empty_result(self):
        df = _make_df(10, freq_hz=10.0)
        sliced = time_slice(df, 100.0, 200.0)
        assert len(sliced) == 0


# -- find_time tests ---------------------------------------------------------


class TestFindTime:
    def test_finds_first_matching_time(self):
        df = _make_df(100, freq_hz=10.0)
        t = find_time(df["data"], lambda x: x >= 50.0)
        assert t == pytest.approx(5.0)

    def test_raises_when_no_match(self):
        df = _make_df(100, freq_hz=10.0)
        with pytest.raises(ValueError, match="No element"):
            find_time(df["data"], lambda x: x > 9999.0)

    def test_first_element_returns_zero(self):
        df = _make_df(100, freq_hz=10.0)
        t = find_time(df["data"], lambda x: x >= 0.0)
        assert t == pytest.approx(0.0)


# -- combination test --------------------------------------------------------


class TestFindTimeSliceCombination:
    def test_find_then_slice(self):
        df = _make_df(200, freq_hz=10.0)
        t0 = find_time(df["data"], lambda x: x >= 100.0)
        sliced = time_slice(df, t0 - 0.5, t0 + 1.0)
        # t0 is 10.0s; slice from 9.5 to 11.0
        assert len(sliced) > 0
        assert sliced["data"].iloc[0] == pytest.approx(95.0)
        assert sliced["data"].iloc[-1] == pytest.approx(110.0)
