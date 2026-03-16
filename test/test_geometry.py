"""Tests for baglab.geometry module."""

import numpy as np
import pandas as pd
import pytest

import baglab
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
from baglab.io import FieldGroup


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_quat_df(yaws: list[float]) -> pd.DataFrame:
    """Create a DataFrame with quaternion columns from yaw angles."""
    half = np.array(yaws) * 0.5
    return pd.DataFrame(
        {
            "x": np.zeros(len(yaws)),
            "y": np.zeros(len(yaws)),
            "z": np.sin(half),
            "w": np.cos(half),
        }
    )


def _make_pose_df(
    xs: list[float], ys: list[float], yaws: list[float]
) -> pd.DataFrame:
    """Create a DataFrame that looks like a Pose (position + orientation)."""
    half = np.array(yaws) * 0.5
    return pd.DataFrame(
        {
            "position.x": xs,
            "position.y": ys,
            "position.z": np.zeros(len(xs)),
            "orientation.x": np.zeros(len(xs)),
            "orientation.y": np.zeros(len(xs)),
            "orientation.z": np.sin(half),
            "orientation.w": np.cos(half),
        }
    )


def _make_point_df(xs, ys, zs=None) -> pd.DataFrame:
    d = {"x": xs, "y": ys}
    if zs is not None:
        d["z"] = zs
    return pd.DataFrame(d)


class _Vec:
    """Minimal mock for ROS msg objects with x/y/z/w attributes."""

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


# ===========================================================================
# Scalar input support
# ===========================================================================


class TestScalarInput:
    """Functions accepting FieldInput should also accept scalar-like objects."""

    def test_quat_to_yaw_scalar(self):
        o = _Vec(x=0.0, y=0.0, z=0.0, w=1.0)
        result = quat_to_yaw(o)
        assert isinstance(result, float)
        assert result == pytest.approx(0.0)

    def test_quat_to_yaw_scalar_known(self):
        yaw = np.pi / 4
        o = _Vec(x=0.0, y=0.0, z=np.sin(yaw / 2), w=np.cos(yaw / 2))
        assert quat_to_yaw(o) == pytest.approx(yaw)

    def test_quat_to_rpy_scalar(self):
        o = _Vec(x=0.0, y=0.0, z=0.0, w=1.0)
        result = quat_to_rpy(o)
        assert isinstance(result, dict)
        assert result["roll"] == pytest.approx(0.0)
        assert result["pitch"] == pytest.approx(0.0)
        assert result["yaw"] == pytest.approx(0.0)

    def test_to_xy_scalar(self):
        p = _Vec(x=1.0, y=2.0, z=3.0)
        result = to_xy(p)
        assert isinstance(result, dict)
        assert result == {"x": 1.0, "y": 2.0}

    def test_to_xyz_scalar(self):
        p = _Vec(x=1.0, y=2.0, z=3.0)
        result = to_xyz(p)
        assert isinstance(result, dict)
        assert result == {"x": 1.0, "y": 2.0, "z": 3.0}

    def test_distance_2d_scalar(self):
        p1 = _Vec(x=0.0, y=0.0)
        p2 = _Vec(x=3.0, y=4.0)
        result = distance_2d(p1, p2)
        assert isinstance(result, float)
        assert result == pytest.approx(5.0)

    def test_distance_3d_scalar(self):
        p1 = _Vec(x=0.0, y=0.0, z=0.0)
        p2 = _Vec(x=1.0, y=2.0, z=2.0)
        result = distance_3d(p1, p2)
        assert isinstance(result, float)
        assert result == pytest.approx(3.0)

    def test_twist_to_speed_scalar(self):
        v = _Vec(x=3.0, y=4.0, z=0.0)
        result = twist_to_speed(v)
        assert isinstance(result, float)
        assert result == pytest.approx(5.0)

    def test_twist_to_speed_2d_scalar(self):
        v = _Vec(x=3.0, y=4.0)
        result = twist_to_speed_2d(v)
        assert isinstance(result, float)
        assert result == pytest.approx(5.0)

    def test_pose_error_scalar_dict(self):
        act = {"x": 3.0, "y": 5.0, "yaw": 0.1}
        ref = {"x": 1.0, "y": 2.0, "yaw": 0.0}
        err = pose_error(act, ref, frame="xy")
        assert isinstance(err, dict)
        assert err["along"] == pytest.approx(2.0)
        assert err["cross"] == pytest.approx(3.0)
        assert err["yaw"] == pytest.approx(0.1)

    def test_pose_error_scalar_ref_heading(self):
        act = {"x": 1.0, "y": 1.0, "yaw": np.pi / 2}
        ref = {"x": 0.0, "y": 0.0, "yaw": np.pi / 2}
        err = pose_error(act, ref, frame="ref_heading")
        assert isinstance(err, dict)
        assert err["along"] == pytest.approx(1.0, abs=1e-10)
        assert err["cross"] == pytest.approx(-1.0, abs=1e-10)

    def test_pose_error_scalar_object(self):
        act = _Vec(x=1.0, y=0.0, yaw=0.0)
        ref = _Vec(x=0.0, y=0.0, yaw=0.0)
        err = pose_error(act, ref, frame="xy")
        assert isinstance(err, dict)
        assert err["along"] == pytest.approx(1.0)
        assert err["cross"] == pytest.approx(0.0)

    def test_yaw_to_quat_scalar(self):
        q = yaw_to_quat(np.pi / 4)
        assert isinstance(q, dict)
        assert q["x"] == pytest.approx(0.0)
        assert q["y"] == pytest.approx(0.0)
        # roundtrip
        assert quat_to_yaw(_Vec(**q)) == pytest.approx(np.pi / 4)

    def test_rpy_to_quat_scalar(self):
        q = rpy_to_quat(0.0, 0.0, np.pi / 6)
        assert isinstance(q, dict)
        assert set(q.keys()) == {"x", "y", "z", "w"}
        rpy = quat_to_rpy(_Vec(**q))
        assert rpy["yaw"] == pytest.approx(np.pi / 6)

    def test_pose_to_xyyaw_scalar(self):
        pose = _Vec(
            position=_Vec(x=1.0, y=2.0, z=0.0),
            orientation=_Vec(x=0.0, y=0.0, z=0.0, w=1.0),
        )
        result = pose_to_xyyaw(pose)
        assert isinstance(result, dict)
        assert result["x"] == pytest.approx(1.0)
        assert result["y"] == pytest.approx(2.0)
        assert result["yaw"] == pytest.approx(0.0)

    def test_normalize_angle_scalar(self):
        assert normalize_angle(2 * np.pi) == pytest.approx(0.0, abs=1e-10)
        assert isinstance(normalize_angle(0.5), float)

    def test_angle_diff_scalar(self):
        result = angle_diff(np.pi - 0.1, -np.pi + 0.1)
        assert isinstance(result, float)
        assert result == pytest.approx(-0.2, abs=1e-10)


# ===========================================================================
# Quaternion <-> Euler
# ===========================================================================


class TestQuatToYaw:
    def test_identity_quaternion(self):
        df = _make_quat_df([0.0])
        result = quat_to_yaw(df)
        assert result.iloc[0] == pytest.approx(0.0)

    def test_known_yaws(self):
        yaws = [0.0, np.pi / 4, np.pi / 2, -np.pi / 4, np.pi]
        df = _make_quat_df(yaws)
        result = quat_to_yaw(df)
        for i, expected in enumerate(yaws):
            assert result.iloc[i] == pytest.approx(expected, abs=1e-10)

    def test_returns_series(self):
        df = _make_quat_df([0.0, 1.0])
        result = quat_to_yaw(df)
        assert isinstance(result, pd.Series)
        assert result.name == "yaw"

    def test_preserves_index(self):
        df = _make_quat_df([0.0, 1.0])
        df.index = [10, 20]
        result = quat_to_yaw(df)
        assert list(result.index) == [10, 20]

    def test_field_group_input(self):
        """FieldGroup wrapping a DataFrame should also work."""
        inner = pd.DataFrame(
            {
                "orientation.x": [0.0],
                "orientation.y": [0.0],
                "orientation.z": [0.0],
                "orientation.w": [1.0],
            }
        )
        fg = FieldGroup(inner, "orientation")
        result = quat_to_yaw(fg)
        assert result.iloc[0] == pytest.approx(0.0)


class TestQuatToRpy:
    def test_identity(self):
        df = _make_quat_df([0.0])
        result = quat_to_rpy(df)
        assert list(result.columns) == ["roll", "pitch", "yaw"]
        assert result["roll"].iloc[0] == pytest.approx(0.0)
        assert result["pitch"].iloc[0] == pytest.approx(0.0)
        assert result["yaw"].iloc[0] == pytest.approx(0.0)

    def test_yaw_only_roundtrip(self):
        yaws = [0.0, np.pi / 6, -np.pi / 3]
        df = _make_quat_df(yaws)
        result = quat_to_rpy(df)
        np.testing.assert_allclose(result["yaw"].values, yaws, atol=1e-10)
        np.testing.assert_allclose(result["roll"].values, 0.0, atol=1e-10)
        np.testing.assert_allclose(result["pitch"].values, 0.0, atol=1e-10)


class TestYawToQuat:
    def test_roundtrip(self):
        yaws = pd.Series([0.0, np.pi / 4, -np.pi / 2, np.pi])
        q = yaw_to_quat(yaws)
        recovered = quat_to_yaw(q)
        np.testing.assert_allclose(recovered.values, yaws.values, atol=1e-10)

    def test_columns(self):
        q = yaw_to_quat(pd.Series([0.0]))
        assert list(q.columns) == ["x", "y", "z", "w"]


class TestRpyToQuat:
    def test_roundtrip(self):
        roll = pd.Series([0.1, -0.2, 0.3])
        pitch = pd.Series([0.05, 0.1, -0.15])
        yaw = pd.Series([0.5, -1.0, 2.0])
        q = rpy_to_quat(roll, pitch, yaw)
        rpy = quat_to_rpy(q)
        np.testing.assert_allclose(rpy["roll"].values, roll.values, atol=1e-10)
        np.testing.assert_allclose(rpy["pitch"].values, pitch.values, atol=1e-10)
        np.testing.assert_allclose(rpy["yaw"].values, yaw.values, atol=1e-10)


# ===========================================================================
# Angle utilities
# ===========================================================================


class TestNormalizeAngle:
    def test_within_range(self):
        s = pd.Series([0.0, np.pi / 2, -np.pi / 2])
        result = normalize_angle(s)
        np.testing.assert_allclose(result.values, s.values, atol=1e-10)

    def test_wraps_positive(self):
        s = pd.Series([2 * np.pi, 3 * np.pi])
        result = normalize_angle(s)
        np.testing.assert_allclose(result.values, [0.0, -np.pi], atol=1e-10)

    def test_wraps_negative(self):
        s = pd.Series([-2 * np.pi, -3 * np.pi])
        result = normalize_angle(s)
        np.testing.assert_allclose(result.values, [0.0, -np.pi], atol=1e-10)


class TestAngleDiff:
    def test_simple(self):
        a = pd.Series([np.pi / 2])
        b = pd.Series([0.0])
        result = angle_diff(a, b)
        assert result.iloc[0] == pytest.approx(np.pi / 2)

    def test_wrapping(self):
        a = pd.Series([np.pi - 0.1])
        b = pd.Series([-np.pi + 0.1])
        result = angle_diff(a, b)
        assert result.iloc[0] == pytest.approx(-0.2, abs=1e-10)


class TestUnwrap:
    def test_removes_jumps(self):
        # Simulate angle that crosses pi boundary
        angles = pd.Series([3.0, 3.1, -3.1, -3.0])
        result = unwrap(angles)
        # After unwrapping, should be monotonically increasing
        assert (result.diff().dropna() > 0).all()


# ===========================================================================
# Position / Point operations
# ===========================================================================


class TestToXy:
    def test_columns(self):
        df = _make_point_df([1.0, 2.0], [3.0, 4.0], [5.0, 6.0])
        result = to_xy(df)
        assert list(result.columns) == ["x", "y"]
        assert len(result) == 2

    def test_values(self):
        df = _make_point_df([1.0], [2.0])
        result = to_xy(df)
        assert result["x"].iloc[0] == 1.0
        assert result["y"].iloc[0] == 2.0


class TestToXyz:
    def test_columns(self):
        df = _make_point_df([1.0], [2.0], [3.0])
        result = to_xyz(df)
        assert list(result.columns) == ["x", "y", "z"]


class TestDistance2d:
    def test_known_distance(self):
        p1 = _make_point_df([0.0], [0.0])
        p2 = _make_point_df([3.0], [4.0])
        result = distance_2d(p1, p2)
        assert result.iloc[0] == pytest.approx(5.0)

    def test_zero_distance(self):
        p = _make_point_df([1.0], [2.0])
        result = distance_2d(p, p)
        assert result.iloc[0] == pytest.approx(0.0)


class TestDistance3d:
    def test_known_distance(self):
        p1 = _make_point_df([0.0], [0.0], [0.0])
        p2 = _make_point_df([1.0], [2.0], [2.0])
        result = distance_3d(p1, p2)
        assert result.iloc[0] == pytest.approx(3.0)


class TestCumulativeDistance:
    def test_straight_line(self):
        df = _make_point_df([0.0, 1.0, 2.0, 3.0], [0.0, 0.0, 0.0, 0.0])
        result = cumulative_distance(df)
        np.testing.assert_allclose(result.values, [0.0, 1.0, 2.0, 3.0])

    def test_starts_at_zero(self):
        df = _make_point_df([5.0, 6.0], [5.0, 5.0])
        result = cumulative_distance(df)
        assert result.iloc[0] == pytest.approx(0.0)


# ===========================================================================
# Pose composite operations
# ===========================================================================


class TestPoseToXyyaw:
    def test_columns(self):
        df = _make_pose_df([1.0], [2.0], [0.5])
        result = pose_to_xyyaw(df)
        assert list(result.columns) == ["x", "y", "yaw"]

    def test_values(self):
        yaw_val = np.pi / 4
        df = _make_pose_df([1.0], [2.0], [yaw_val])
        result = pose_to_xyyaw(df)
        assert result["x"].iloc[0] == pytest.approx(1.0)
        assert result["y"].iloc[0] == pytest.approx(2.0)
        assert result["yaw"].iloc[0] == pytest.approx(yaw_val)


class TestPoseError:
    def test_xy_frame_no_offset(self):
        pose = _make_pose_df([1.0], [2.0], [0.0])
        err = pose_error(pose, pose, frame="xy")
        assert err["along"].iloc[0] == pytest.approx(0.0)
        assert err["cross"].iloc[0] == pytest.approx(0.0)
        assert err["yaw"].iloc[0] == pytest.approx(0.0)

    def test_xy_frame_offset(self):
        actual = _make_pose_df([3.0], [5.0], [0.1])
        ref = _make_pose_df([1.0], [2.0], [0.0])
        err = pose_error(actual, ref, frame="xy")
        assert err["along"].iloc[0] == pytest.approx(2.0)
        assert err["cross"].iloc[0] == pytest.approx(3.0)
        assert err["yaw"].iloc[0] == pytest.approx(0.1)

    def test_ref_heading_frame(self):
        # ref heading = pi/2 (pointing in +y direction)
        actual = _make_pose_df([1.0], [1.0], [np.pi / 2])
        ref = _make_pose_df([0.0], [0.0], [np.pi / 2])
        err = pose_error(actual, ref, frame="ref_heading")
        # dx=1, dy=1, ref_yaw=pi/2
        # along = dx*cos(pi/2) + dy*sin(pi/2) = 0 + 1 = 1
        # cross = -dx*sin(pi/2) + dy*cos(pi/2) = -1 + 0 = -1
        assert err["along"].iloc[0] == pytest.approx(1.0, abs=1e-10)
        assert err["cross"].iloc[0] == pytest.approx(-1.0, abs=1e-10)

    def test_ref_path_frame(self):
        # Reference path going in +x direction
        ref = _make_pose_df([0.0, 1.0, 2.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
        actual = _make_pose_df([0.0, 1.0, 2.0], [0.5, 0.5, 0.5], [0.0, 0.0, 0.0])
        err = pose_error(actual, ref, frame="ref_path")
        # Path tangent is along +x, so cross = dy = 0.5
        np.testing.assert_allclose(err["cross"].values, [0.5, 0.5, 0.5], atol=1e-10)
        np.testing.assert_allclose(err["along"].values, [0.0, 0.0, 0.0], atol=1e-10)

    def test_invalid_frame_raises(self):
        pose = _make_pose_df([0.0], [0.0], [0.0])
        with pytest.raises(ValueError, match="Unknown frame"):
            pose_error(pose, pose, frame="invalid")

    def test_yaw_wrapping(self):
        actual = _make_pose_df([0.0], [0.0], [np.pi - 0.1])
        ref = _make_pose_df([0.0], [0.0], [-np.pi + 0.1])
        err = pose_error(actual, ref)
        assert err["yaw"].iloc[0] == pytest.approx(-0.2, abs=1e-10)


# ===========================================================================
# Twist / velocity
# ===========================================================================


class TestTwistToSpeed:
    def test_known_speed(self):
        df = pd.DataFrame({"x": [3.0], "y": [4.0], "z": [0.0]})
        result = twist_to_speed(df)
        assert result.iloc[0] == pytest.approx(5.0)

    def test_3d_speed(self):
        df = pd.DataFrame({"x": [1.0], "y": [2.0], "z": [2.0]})
        result = twist_to_speed(df)
        assert result.iloc[0] == pytest.approx(3.0)


class TestTwistToSpeed2d:
    def test_known_speed(self):
        df = pd.DataFrame({"x": [3.0], "y": [4.0]})
        result = twist_to_speed_2d(df)
        assert result.iloc[0] == pytest.approx(5.0)


class TestVelocityFromPose:
    def test_constant_speed(self):
        pos = _make_point_df([0.0, 1.0, 2.0, 3.0], [0.0, 0.0, 0.0, 0.0])
        t = pd.Series([0.0, 1.0, 2.0, 3.0])
        result = velocity_from_pose(pos, t)
        # First element is NaN (no previous point)
        np.testing.assert_allclose(result.values[1:], [1.0, 1.0, 1.0])


class TestYawRateFromPose:
    def test_constant_yaw_rate(self):
        yaws = [0.0, 0.1, 0.2, 0.3]
        quat_df = _make_quat_df(yaws)
        t = pd.Series([0.0, 1.0, 2.0, 3.0])
        result = yaw_rate_from_pose(quat_df, t)
        np.testing.assert_allclose(result.values[1:], [0.1, 0.1, 0.1], atol=1e-10)


# ===========================================================================
# numpy conversion
# ===========================================================================


class TestToNumpy:
    def test_2d_shape(self):
        df = _make_point_df([1.0, 2.0], [3.0, 4.0])
        result = to_numpy_2d(df)
        assert result.shape == (2, 2)
        np.testing.assert_array_equal(result, [[1.0, 3.0], [2.0, 4.0]])

    def test_3d_shape(self):
        df = _make_point_df([1.0], [2.0], [3.0])
        result = to_numpy_3d(df)
        assert result.shape == (1, 3)
        np.testing.assert_array_equal(result, [[1.0, 2.0, 3.0]])


# ===========================================================================
# Interpolation / time alignment
# ===========================================================================


class TestMatchByTime:
    def test_exact_match(self):
        source = pd.Series([0.0, 1.0, 2.0, 3.0])
        query = pd.Series([0.0, 1.0, 2.0])
        result = match_by_time(query, source)
        np.testing.assert_array_equal(result, [0, 1, 2])

    def test_nearest(self):
        source = pd.Series([0.0, 1.0, 2.0, 3.0])
        query = pd.Series([0.4, 1.6, 2.9])
        result = match_by_time(query, source)
        np.testing.assert_array_equal(result, [0, 2, 3])

    def test_before_first(self):
        source = pd.Series([1.0, 2.0, 3.0])
        query = pd.Series([-5.0])
        result = match_by_time(query, source)
        np.testing.assert_array_equal(result, [0])

    def test_after_last(self):
        source = pd.Series([1.0, 2.0, 3.0])
        query = pd.Series([100.0])
        result = match_by_time(query, source)
        np.testing.assert_array_equal(result, [2])

    def test_midpoint_tie(self):
        # Equidistant: should pick left (<=)
        source = pd.Series([0.0, 2.0])
        query = pd.Series([1.0])
        result = match_by_time(query, source)
        np.testing.assert_array_equal(result, [0])

    def test_numpy_input(self):
        source = np.array([0.0, 1.0, 2.0])
        query = np.array([0.7, 1.3])
        result = match_by_time(query, source)
        np.testing.assert_array_equal(result, [1, 1])


class TestAlignTime:
    def test_aligned_length(self):
        idx1 = pd.RangeIndex(5)
        idx2 = pd.RangeIndex(5)
        df1 = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0]}, index=idx1)
        df2 = pd.DataFrame({"b": [10.0, 20.0, 30.0, 40.0, 50.0]}, index=idx2)
        a, b = align_time(df1, df2)
        assert len(a) == len(b)

    def test_interpolation(self):
        df1 = pd.DataFrame({"a": [0.0, 10.0]}, index=[0.0, 1.0])
        df2 = pd.DataFrame({"b": [0.0, 100.0]}, index=[0.0, 0.5])
        a, b = align_time(df1, df2)
        # At t=0.5, df1["a"] should be interpolated to 5.0
        assert a.loc[0.5, "a"] == pytest.approx(5.0)
        # At t=0.5, df2["b"] should be 100.0 (original value)
        assert b.loc[0.5, "b"] == pytest.approx(100.0)

    def test_with_time_col(self):
        df1 = pd.DataFrame({"t": [0.0, 1.0], "a": [0.0, 10.0]})
        df2 = pd.DataFrame({"t": [0.0, 0.5], "b": [0.0, 5.0]})
        a, b = align_time(df1, df2, time_col="t")
        assert len(a) == len(b)


class TestResample:
    def test_uniform_output(self):
        df = pd.DataFrame({"a": [0.0, 10.0, 20.0]}, index=[0.0, 1.0, 2.0])
        result = resample(df, dt=0.5)
        expected_index = [0.0, 0.5, 1.0, 1.5]
        np.testing.assert_allclose(result.index.values, expected_index)

    def test_interpolated_values(self):
        df = pd.DataFrame({"a": [0.0, 10.0]}, index=[0.0, 1.0])
        result = resample(df, dt=0.25)
        np.testing.assert_allclose(
            result["a"].values, [0.0, 2.5, 5.0, 7.5], atol=1e-10
        )


class TestInterpPose:
    def test_position_interpolation(self):
        df = pd.DataFrame(
            {
                "position.x": [0.0, 10.0],
                "position.y": [0.0, 10.0],
                "orientation.x": [0.0, 0.0],
                "orientation.y": [0.0, 0.0],
                "orientation.z": [0.0, 0.0],
                "orientation.w": [1.0, 1.0],
            },
            index=[0.0, 1.0],
        )
        result = interp_pose(df, [0.5])
        assert result["position.x"].iloc[0] == pytest.approx(5.0)
        assert result["position.y"].iloc[0] == pytest.approx(5.0)

    def test_quaternion_slerp(self):
        # Interpolate between yaw=0 and yaw=pi/2
        half1 = 0.0
        half2 = np.pi / 4
        df = pd.DataFrame(
            {
                "position.x": [0.0, 1.0],
                "position.y": [0.0, 0.0],
                "orientation.x": [0.0, 0.0],
                "orientation.y": [0.0, 0.0],
                "orientation.z": [np.sin(half1), np.sin(half2)],
                "orientation.w": [np.cos(half1), np.cos(half2)],
            },
            index=[0.0, 1.0],
        )
        result = interp_pose(df, [0.5])
        # Midpoint should be yaw=pi/4
        mid_yaw = np.arctan2(
            2.0 * result["orientation.w"].iloc[0] * result["orientation.z"].iloc[0],
            1.0 - 2.0 * result["orientation.z"].iloc[0] ** 2,
        )
        assert mid_yaw == pytest.approx(np.pi / 4, abs=1e-6)
