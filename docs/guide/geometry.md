# Geometry

The `baglab.geometry` module provides conversions and computations for
`geometry_msgs` types stored in DataFrames.

## Quaternion / Euler conversion

```python
import baglab

odom = bag["/odom"]
orientation = odom.msg.pose.pose.orientation

yaw = baglab.quat_to_yaw(orientation)           # pd.Series
rpy = baglab.quat_to_rpy(orientation)            # DataFrame [roll, pitch, yaw]
quat = baglab.rpy_to_quat(rpy)                   # DataFrame [x, y, z, w]
quat = baglab.yaw_to_quat(yaw)                   # DataFrame [x, y, z, w]
```

## Distance and trajectory

```python
position = odom.msg.pose.pose.position

d = baglab.distance_2d(position)                 # Euclidean distance from origin
d = baglab.cumulative_distance(position)          # Cumulative path length
```

## Pose utilities

```python
xyyaw = baglab.pose_to_xyyaw(odom.msg.pose.pose)   # DataFrame [x, y, yaw]

err = baglab.pose_error(actual, desired)             # DataFrame [x, y, yaw]
```

## Speed from twist / pose

```python
speed = baglab.twist_to_speed(odom.msg.twist.twist)
speed_2d = baglab.twist_to_speed_2d(odom.msg.twist.twist)

vel = baglab.velocity_from_pose(position, t)
yaw_rate = baglab.yaw_rate_from_pose(orientation, t)
```

## Interpolation and alignment

```python
# Interpolate pose at arbitrary timestamps
pose_interp = baglab.interp_pose(pose_df, target_times)

# Resample to uniform intervals
resampled = baglab.resample(df, dt=0.01)

# Align two DataFrames by timestamp
a_aligned, b_aligned = baglab.align_time(df_a, df_b)

# Match rows by nearest timestamp
matched = baglab.match_by_time(df_a, df_b, tolerance=0.05)
```

## Angle utilities

```python
diff = baglab.angle_diff(theta1, theta2)         # Shortest angular difference
theta = baglab.normalize_angle(theta)             # Wrap to [-pi, pi]
theta = baglab.unwrap(theta)                      # Phase unwrap
```

## NumPy conversion

```python
xy = baglab.to_xy(position)                      # ndarray (N, 2)
xyz = baglab.to_xyz(position)                     # ndarray (N, 3)
arr_2d = baglab.to_numpy_2d(position)             # ndarray (N, 2)
arr_3d = baglab.to_numpy_3d(position)             # ndarray (N, 3)
```
