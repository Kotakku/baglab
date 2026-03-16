# Getting Started

## Prerequisites

- Python 3.10+
- ROS2 environment (Humble / Jazzy)

## Installation

### From PyPI

```bash
pip install baglab
```

### From source

```bash
git clone https://github.com/Kotakku/baglab.git
cd baglab
pip install -e ".[dev]"
```

## Basic usage

### Load a rosbag

```python
import baglab

bag = baglab.load("path/to/rosbag")
print(bag.topics)
# {"/motor/angle": "sensor_msgs/msg/JointState", ...}
```

### Access topic data

Topic data is loaded lazily — only when you access it:

```python
twist_df = bag["/cmd_vel"]
```

### Dot-access for message fields

```python
twist_df.msg.twist.linear.x         # pandas Series
twist_df.msg.twist.linear.df        # DataFrame with columns [x, y, z]
```

### Timestamps

```python
t = baglab.stamp_to_sec(twist_df)                  # absolute time [s]
t = baglab.stamp_to_sec(twist_df, relative=True)   # relative time (starts at 0)
```

### Reindex by stamp

```python
twist_df = baglab.reindex_by_stamp(twist_df)
```

### Time slicing

```python
sliced = baglab.time_slice(twist_df, t_start=1.0, t_end=5.0)
```

## Plot example

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
