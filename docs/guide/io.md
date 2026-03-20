# I/O & Bag Loading

## Loading a rosbag

```python
import baglab

bag = baglab.load("path/to/rosbag")
```

`load()` returns a `Bag` object. Topic data is loaded lazily — the actual
deserialization happens when you first access a topic.

## Listing topics

```python
print(bag.topics)
# {"/cmd_vel": "geometry_msgs/msg/Twist", "/odom": "nav_msgs/msg/Odometry", ...}
```

## Accessing topic data

```python
df = bag["/cmd_vel"]
```

Each row corresponds to one message. Nested message fields are flattened into
columns using dot-separated names (e.g. `twist.linear.x`).

## Dot-access with MsgAccessor

The `.msg` accessor lets you navigate fields using attribute access:

```python
df.msg.twist.linear.x         # pd.Series
df.msg.twist.linear.df        # DataFrame with [x, y, z]
```

The `FieldGroup` object returned by intermediate access (e.g. `df.msg.twist.linear`)
supports `.df` to get a DataFrame and direct attribute access for leaf fields.

## Timestamps

```python
# Absolute time in seconds (float64)
t = baglab.stamp_to_sec(df)

# Relative time starting from 0
t = baglab.stamp_to_sec(df, relative=True)

# With custom stamp prefix
t = baglab.stamp_to_sec(df, prefix="header.stamp")
```

## Reindexing and slicing

```python
# Replace index with publication timestamp
df = baglab.reindex_by_stamp(df)

# Slice by relative time
sliced = baglab.time_slice(df, t_start=1.0, t_end=5.0)

# Find index nearest to a given time
idx = baglab.find_time(df, 3.5)
```

## Backends

baglab ships with two reading backends:

| Backend | Formats | ROS 2 dependency | Speed |
|---|---|---|---|
| **mcap** (`baglab-mcap-backend`) | mcap | None | Fast |
| **rosbags** (built-in) | mcap, db3 | None | Slow |

### Default behavior

With `backend="auto"` (the default), the backend is selected automatically:

1. If `baglab-mcap-backend` is installed → **mcap**
2. Otherwise → **rosbags**

```python
# Automatic selection (recommended)
bag = baglab.load("path/to/rosbag")

# Explicit selection
bag = baglab.load("path/to/rosbag", backend="mcap")
bag = baglab.load("path/to/rosbag", backend="rosbags")
```

### mcap backend

A C++ backend specialized for the MCAP format. It uses the
[foxglove/mcap](https://github.com/foxglove/mcap) C++ library and a
custom CDR deserializer to achieve fast reading without any ROS 2 dependency.

- Zero-copy I/O via mmap
- Parses ros2msg schemas embedded in MCAP files directly (no rosidl needed)
- Self-contained CDR (Common Data Representation) deserializer

```bash
# Installation
sudo apt install liblz4-dev libzstd-dev
pip install --no-build-isolation ./baglab_mcap_backend
```

```python
print(baglab.has_mcap_backend())  # True / False
```

!!! note
    The mcap backend only supports the `ros2msg` schema encoding.
    Messages with `ros2idl` schemas will raise an error with a suggestion
    to switch to the rosbags backend. All standard ROS 2 messages use
    `ros2msg`, so this is rarely an issue in practice.

### rosbags backend

Uses the pure-Python [rosbags](https://gitlab.com/ternaris/rosbags) library.
It supports both mcap and db3 formats and requires no additional installation.

Use this backend when working with db3 bags or custom messages that
only have `ros2idl` schema definitions.

## Preload (batch reading)

When reading multiple topics, **preload** can significantly speed up loading.
With the mcap backend, all requested topics are read in a single pass
through the file.

### Via the `topics` argument of `load()`

```python
# Pass a list of topic names for batch reading
bag = baglab.load("path/to/rosbag", topics=[
    "/sensing/imu/imu_data",
    "/vehicle/status/velocity_status",
    "/localization/twist_estimator/twist_with_covariance",
])

# Subsequent access returns instantly from cache
df = bag["/sensing/imu/imu_data"]
```

### Via `bag.preload()`

```python
bag = baglab.load("path/to/rosbag")

# Batch-read topics once you know which ones you need
bag.preload(["/topic_a", "/topic_b", "/topic_c"])

# Preloaded topics return instantly; others are loaded on demand
df_a = bag["/topic_a"]       # instant (preloaded)
df_x = bag["/other_topic"]   # on-demand
```

!!! tip
    Measured on a 20 GB bag with ~600 topics:

    - **Sequential** (`bag[topic]` x 100): 5.2 s
    - **Preload** (batch): 0.5 s (**10x faster**)

    Use preload when working with many topics.

### Dict form (field selection)

The dict form is still supported. It loads only the specified fields
per topic (does not use the single-pass batch scan).

```python
bag = baglab.load("path/to/rosbag", topics={
    "/sensing/imu/imu_data": ["header.stamp.sec", "header.stamp.nanosec"],
})
```

## Custom message types (db3 bags)

mcap bags embed `.msg` definitions inside the file, so custom types work
out of the box.  db3 (SQLite3) bags do **not** carry type definitions, so
custom message types must be registered before
the data can be deserialized.

### Option 1 — pass workspace paths to `load()`

The simplest approach. Pass the workspace `src` directory (or any directory
tree containing `<pkg>/msg/*.msg` files) via the `msg_paths` argument:

```python
import baglab

bag = baglab.load(
    "path/to/rosbag",
    msg_paths=["~/my_ws/src"],
)
```

Multiple paths can be specified:

```python
bag = baglab.load(
    "path/to/rosbag",
    msg_paths=["~/my_ws/src", "~/custom_msgs_ws/src"],
)
```

### Option 2 — `register_msg_types()` for fine-grained control

If you need to register types into a specific typestore or reuse it across
multiple bags, use `register_msg_types()` directly:

```python
from rosbags.typesys import Stores, get_typestore
from baglab.io.typesys import register_msg_types

typestore = get_typestore(Stores.LATEST)
n = register_msg_types(typestore, "~/my_ws/src")
print(f"Registered {n} types")
```

!!! note
    The directory must follow the standard ROS2 layout
    `<package_name>/msg/<MsgName>.msg`.  Nested structures are resolved
    recursively, so pointing at a workspace `src/` directory works.
