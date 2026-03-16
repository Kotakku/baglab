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
