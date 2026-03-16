"""Geometry utilities for ROS2 message data in DataFrames.

Provides conversions and computations commonly needed when working with
geometry_msgs types (Quaternion, Point, Pose, Twist, etc.) stored as
DataFrames or :class:`~baglab.io.FieldGroup` objects.
"""

from baglab.geometry._common import FieldInput
from baglab.geometry.angle import angle_diff, normalize_angle, unwrap
from baglab.geometry.interp import align_time, interp_pose, match_by_time, resample
from baglab.geometry.numpy_conv import to_numpy_2d, to_numpy_3d
from baglab.geometry.point import (
    cumulative_distance,
    distance_2d,
    distance_3d,
    to_xy,
    to_xyz,
)
from baglab.geometry.pose import pose_error, pose_to_xyyaw
from baglab.geometry.quaternion import quat_to_rpy, quat_to_yaw, rpy_to_quat, yaw_to_quat
from baglab.geometry.twist import (
    twist_to_speed,
    twist_to_speed_2d,
    velocity_from_pose,
    yaw_rate_from_pose,
)

__all__ = [
    "FieldInput",
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
]
