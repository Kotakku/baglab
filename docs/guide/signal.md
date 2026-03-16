# Signal Processing

The `baglab.signal` module provides common signal processing operations.

## Lowpass filter

```python
import baglab

filtered = baglab.lowpass(series, cutoff=5.0, fs=100.0)
filtered = baglab.lowpass(series, cutoff=5.0, fs=100.0, order=6)
```

## Differentiation and integration

```python
velocity = baglab.diff(position, t)
position = baglab.integrate(velocity, t)
```

## Moving average

```python
smoothed = baglab.moving_average(series, window=10)
```

## FFT

```python
freq, amplitude = baglab.fft(series, fs=100.0)
```
