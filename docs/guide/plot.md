# Plotting

The `baglab.plot` module provides ready-made plots for common visualizations.

All functions use matplotlib and return `(fig, ax)` tuples for further customization.

## Time series

```python
import baglab

fig, ax = baglab.plot.timeseries(t, data, ylabel="velocity [m/s]")
```

## Trajectory

```python
fig, ax = baglab.plot.trajectory(x, y)
fig, ax = baglab.plot.trajectory(x, y, color_by=speed)  # color by scalar
```

## Step response

```python
fig, ax = baglab.plot.step_response(t, command, response)
```

## Error band

```python
fig, ax = baglab.plot.error_band(t, error, label="tracking error [m]")
```
