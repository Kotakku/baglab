"""Tests for baglab.diagnostics module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from baglab.diagnostics import latency_chain, message_gaps, topic_delay, topic_rate


# ---------------------------------------------------------------------------
# topic_rate
# ---------------------------------------------------------------------------

class TestTopicRate:
    def test_uniform_100hz(self):
        """Uniform 100 Hz timestamps should yield mean_hz close to 100."""
        n = 1001
        ts = pd.Series(np.arange(n) / 100.0)
        result = topic_rate(ts)

        assert result["count"] == n
        assert result["mean_hz"] == pytest.approx(100.0, rel=1e-6)
        assert result["std_hz"] == pytest.approx(0.0, abs=1e-9)
        assert result["min_dt"] == pytest.approx(0.01, rel=1e-6)
        assert result["max_dt"] == pytest.approx(0.01, rel=1e-6)
        assert result["duration"] == pytest.approx(10.0, rel=1e-6)

    def test_datetimeindex_input(self):
        """DatetimeIndex input should be handled correctly."""
        n = 501
        base = pd.Timestamp("2025-01-01")
        idx = pd.DatetimeIndex([base + pd.Timedelta(seconds=i / 50.0) for i in range(n)])
        result = topic_rate(idx)

        assert result["count"] == n
        assert result["mean_hz"] == pytest.approx(50.0, rel=1e-3)
        assert result["duration"] == pytest.approx(10.0, rel=1e-3)

    def test_single_timestamp(self):
        """Single timestamp should return zero rate."""
        ts = pd.Series([1.0])
        result = topic_rate(ts)

        assert result["count"] == 1
        assert result["mean_hz"] == 0.0
        assert result["duration"] == 0.0


# ---------------------------------------------------------------------------
# topic_delay
# ---------------------------------------------------------------------------

class TestTopicDelay:
    def test_constant_delay(self):
        """Known constant delay should be recovered exactly."""
        n = 100
        pub = pd.Series(np.arange(n, dtype=float))
        recv = pub + 0.05
        result = topic_delay(pub, recv)

        assert result["mean"] == pytest.approx(0.05, abs=1e-12)
        assert result["max"] == pytest.approx(0.05, abs=1e-12)
        assert result["std"] == pytest.approx(0.0, abs=1e-12)
        assert result["median"] == pytest.approx(0.05, abs=1e-12)
        assert len(result["delay"]) == n

    def test_variable_delay_statistics(self):
        """Variable delay should yield correct statistics."""
        pub = pd.Series([0.0, 1.0, 2.0, 3.0])
        recv = pd.Series([0.1, 1.2, 2.1, 3.3])
        result = topic_delay(pub, recv)

        delays = np.array([0.1, 0.2, 0.1, 0.3])
        assert result["mean"] == pytest.approx(float(np.mean(delays)), abs=1e-12)
        assert result["max"] == pytest.approx(0.3, abs=1e-12)
        assert result["std"] == pytest.approx(float(np.std(delays, ddof=0)), abs=1e-12)
        assert result["median"] == pytest.approx(float(np.median(delays)), abs=1e-12)


# ---------------------------------------------------------------------------
# latency_chain
# ---------------------------------------------------------------------------

class TestLatencyChain:
    def test_multiple_topics(self):
        """Multiple topics should produce correct DataFrame sorted by name."""
        ts_a = pd.Series(np.arange(101) / 100.0)  # 100 Hz
        ts_b = pd.Series(np.arange(51) / 50.0)    # 50 Hz

        result = latency_chain({"/topic_b": ts_b, "/topic_a": ts_a})

        assert list(result.columns) == ["topic", "mean_hz", "mean_dt", "std_dt", "min_dt", "max_dt"]
        assert list(result["topic"]) == ["/topic_a", "/topic_b"]
        assert result.iloc[0]["mean_hz"] == pytest.approx(100.0, rel=1e-6)
        assert result.iloc[1]["mean_hz"] == pytest.approx(50.0, rel=1e-6)
        assert result.iloc[0]["mean_dt"] == pytest.approx(0.01, rel=1e-6)
        assert result.iloc[1]["mean_dt"] == pytest.approx(0.02, rel=1e-6)


# ---------------------------------------------------------------------------
# message_gaps
# ---------------------------------------------------------------------------

class TestMessageGaps:
    def test_no_gaps_uniform(self):
        """Uniform data at expected rate should produce no gaps."""
        ts = pd.Series(np.arange(1000) / 100.0)
        result = message_gaps(ts, expected_rate=100.0)

        assert len(result) == 0
        assert list(result.columns) == ["start", "end", "duration", "expected_count"]

    def test_inserted_gap_detected(self):
        """A single inserted gap should be detected."""
        # 100 Hz for 1 second, then skip 0.5 seconds, then 100 Hz again
        t1 = np.arange(100) / 100.0          # 0.00 .. 0.99
        t2 = np.arange(100) / 100.0 + 1.5    # 1.50 .. 2.49
        ts = pd.Series(np.concatenate([t1, t2]))

        result = message_gaps(ts, expected_rate=100.0)

        assert len(result) == 1
        row = result.iloc[0]
        assert row["start"] == pytest.approx(0.99, rel=1e-6)
        assert row["end"] == pytest.approx(1.50, rel=1e-6)
        assert row["duration"] == pytest.approx(0.51, rel=1e-3)

    def test_expected_count_reasonable(self):
        """expected_count should reflect how many messages fit in the gap."""
        t1 = np.arange(50) / 10.0   # 10 Hz, 0..4.9
        t2 = np.arange(50) / 10.0 + 10.0  # 10 Hz, 10..14.9
        ts = pd.Series(np.concatenate([t1, t2]))

        result = message_gaps(ts, expected_rate=10.0)

        assert len(result) == 1
        row = result.iloc[0]
        # Gap from 4.9 to 10.0 = 5.1 seconds at 10 Hz => ~51 messages
        assert row["expected_count"] == pytest.approx(51, abs=1)
