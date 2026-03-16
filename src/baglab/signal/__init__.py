"""Signal processing utilities for time-series data."""

from baglab.signal._diff import diff, integrate
from baglab.signal._filter import lowpass
from baglab.signal._freq import fft
from baglab.signal._smooth import moving_average

__all__ = [
    "diff",
    "fft",
    "integrate",
    "lowpass",
    "moving_average",
]
