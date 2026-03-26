"""baglab - ROS2 rosbag data analysis library.

Quick start
-----------

Load a rosbag and access topic data as a pandas DataFrame::

    import baglab

    bag = baglab.load("path/to/rosbag")
    df = bag["/cmd_vel"]                        # full topic (lazy, cached)
    df = bag["/cmd_vel", ["twist.linear.x"]]    # field selection (not cached)

Access nested message fields with dot notation::

    vel = df.msg.twist.linear.x     # -> pd.Series
    vel = df.msg.twist.linear.df    # -> pd.DataFrame with columns [x, y, z]

Timestamp helpers::

    t = baglab.stamp_to_sec(df, relative=True)  # header.stamp -> float seconds
    t = baglab.recv_time_to_sec(df, relative=True)  # receive time -> float seconds

    # align multiple topics to a common t=0
    t1, t2 = baglab.align_origin(t1, t2)

Batch preloading (single-pass read on mcap backend)::

    bag = baglab.load("path/to/rosbag", topics=["/cmd_vel", "/odom"])

Array field expansion (e.g. JointState)::

    pos = baglab.explode_array(df["position"], names=["j1", "j2", "j3"])
"""

from baglab import plot as plot
from baglab.plot import plot_error_band, plot_step_response, plot_timeseries, plot_xy_trajectory
from baglab.analysis import delay_estimate, stepinfo, tracking_error
from baglab.diagnostics import latency_chain, message_gaps, topic_delay, topic_rate
from baglab.io import MsgAccessor, FieldGroup, align_origin, clear_cache, explode_array, find_bags, find_time, has_mcap_backend, load, recv_time_to_sec, reindex_by_stamp, stamp_to_sec, time_slice
from baglab.signal import diff, fft, integrate, lowpass, moving_average
from baglab.stats import describe, rms
from baglab.tui import select_bag
from baglab.geometry import (
    align_time,
    angle_diff,
    cumulative_distance,
    distance_2d,
    distance_3d,
    interp_pose,
    match_by_time,
    normalize_angle,
    pose_error,
    pose_to_xyyaw,
    quat_to_rpy,
    quat_to_yaw,
    resample,
    rpy_to_quat,
    to_numpy_2d,
    to_numpy_3d,
    to_xy,
    to_xyz,
    twist_to_speed,
    twist_to_speed_2d,
    unwrap,
    velocity_from_pose,
    yaw_rate_from_pose,
    yaw_to_quat,
)

__all__ = [
    "MsgAccessor",
    "FieldGroup",
    "align_origin",
    "clear_cache",
    "explode_array",
    "find_bags",
    "find_time",
    "has_mcap_backend",
    "load",
    "recv_time_to_sec",
    "reindex_by_stamp",
    "stamp_to_sec",
    "time_slice",
    "align_time",
    "angle_diff",
    "cumulative_distance",
    "distance_2d",
    "distance_3d",
    "interp_pose",
    "match_by_time",
    "normalize_angle",
    "pose_error",
    "pose_to_xyyaw",
    "quat_to_rpy",
    "quat_to_yaw",
    "resample",
    "rpy_to_quat",
    "to_numpy_2d",
    "to_numpy_3d",
    "to_xy",
    "to_xyz",
    "twist_to_speed",
    "twist_to_speed_2d",
    "unwrap",
    "velocity_from_pose",
    "yaw_rate_from_pose",
    "yaw_to_quat",
    "describe",
    "rms",
    "diff",
    "fft",
    "integrate",
    "lowpass",
    "moving_average",
    "delay_estimate",
    "stepinfo",
    "tracking_error",
    "latency_chain",
    "message_gaps",
    "topic_delay",
    "topic_rate",
    "select_bag",
    "plot_error_band",
    "plot_step_response",
    "plot_timeseries",
    "plot_xy_trajectory",
]
