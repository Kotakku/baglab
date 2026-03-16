# baglab

ROS2 rosbag data analysis library. Load, explore, and visualize rosbag data with MATLAB-like simplicity using pandas DataFrames.

## Features

- **Lazy loading** — open a bag and read only the topics you need
- **Dot-access** — navigate message fields like `msg.twist.linear.x`
- **Geometry** — quaternion/euler conversion, distance, pose interpolation
- **Signal processing** — lowpass filter, FFT, differentiation, integration
- **Diagnostics** — topic rate, message gaps, latency chain analysis
- **Plotting** — time-series, trajectory, step response, error band

## Installation

```bash
pip install baglab
```

For development:

```bash
git clone https://github.com/Kotakku/baglab.git
cd baglab
pip install -e ".[dev]"
```

### Prerequisites

- Python 3.10+
- ROS2 (Humble / Jazzy)

## Usage

```python
import baglab

# Load a rosbag (lazy loading)
bag = baglab.load("path/to/rosbag")

# List topics
print(bag.topics)
# {"/motor/angle": "sensor_msgs/msg/JointState", ...}

# Access topic data (loaded on first access)
twist_df = bag["/cmd_vel"]

# Dot-access for message fields
twist_df.msg.twist.linear.x         # Series
twist_df.msg.twist.linear.df        # DataFrame (columns: [x, y, z])

# Timestamps
t = baglab.stamp_to_sec(twist_df)                  # absolute [s]
t = baglab.stamp_to_sec(twist_df, relative=True)   # relative (starts at 0)

# Reindex by publication timestamp
twist_df = baglab.reindex_by_stamp(twist_df)
```

### Plot example

```python
import baglab
import matplotlib.pyplot as plt

bag = baglab.load("path/to/rosbag")
twist_df = bag["/cmd_vel"]

t = baglab.stamp_to_sec(twist_df, relative=True)
vel = twist_df.msg.twist.linear.df

fig, axes = plt.subplots(3, 1, sharex=True)
for ax, axis_name in zip(axes, ["x", "y", "z"]):
    ax.plot(t, vel[axis_name])
    ax.set_ylabel(axis_name)
axes[-1].set_xlabel("time [s]")
plt.show()
```

## Documentation

[https://kotakku.github.io/baglab/](https://kotakku.github.io/baglab/)

## License

Apache-2.0
