# Analysis

The `baglab.analysis` module provides control-engineering analysis tools.

## Step response info

```python
import baglab

info = baglab.stepinfo(response, t)
# Returns: rise_time, settling_time, overshoot, steady_state_value, ...
```

## Tracking error

```python
error = baglab.tracking_error(command, response, t)
```

## Dead-time estimation

```python
delay = baglab.delay_estimate(command, response, fs=100.0)
```
