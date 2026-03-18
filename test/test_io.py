"""Tests for baglab.io module."""

import pandas as pd
import pytest

import baglab
from baglab.io import Bag, clear_cache


class TestLoad:
    """Tests for baglab.load()."""

    def test_returns_bag(self, test_bag_path):
        bag = baglab.load(test_bag_path)
        assert isinstance(bag, Bag)
        bag.close()

    def test_eager_load(self, test_bag_path):
        bag = baglab.load(
            test_bag_path,
            topics={"/test/joint_state": ["position", "effort"]},
        )
        assert "/test/joint_state" in bag._cache
        assert list(bag._cache["/test/joint_state"].columns) == ["position", "effort"]
        bag.close()

    def test_load_nonexistent_path(self):
        with pytest.raises(Exception):
            baglab.load("/nonexistent/path")


class TestBagTopics:
    """Tests for Bag.topics property."""

    def test_topics_returns_dict(self, bag):
        topics = bag.topics
        assert isinstance(topics, dict)
        assert len(topics) == 2

    def test_topic_names(self, bag):
        assert "/test/joint_state" in bag.topics
        assert "/test/twist" in bag.topics

    def test_topic_types(self, bag):
        assert bag.topics["/test/joint_state"] == "sensor_msgs/msg/JointState"
        assert bag.topics["/test/twist"] == "geometry_msgs/msg/TwistStamped"


class TestFieldExpansion:
    """Tests for automatic field expansion."""

    def test_joint_state_columns(self, bag):
        df = bag["/test/joint_state"]
        expected = [
            "header.stamp.sec",
            "header.stamp.nanosec",
            "header.frame_id",
            "name",
            "position",
            "velocity",
            "effort",
        ]
        assert list(df.columns) == expected

    def test_twist_nested_expansion(self, bag):
        df = bag["/test/twist"]
        expected = [
            "header.stamp.sec",
            "header.stamp.nanosec",
            "header.frame_id",
            "twist.linear.x",
            "twist.linear.y",
            "twist.linear.z",
            "twist.angular.x",
            "twist.angular.y",
            "twist.angular.z",
        ]
        assert list(df.columns) == expected

    def test_array_fields_not_expanded(self, bag):
        """Array fields (position, velocity, effort) should be single columns, not expanded."""
        df = bag["/test/joint_state"]
        assert "position" in df.columns
        # Each cell should contain a list/array, not a scalar
        assert hasattr(df["position"].iloc[0], "__len__")


class TestLazyLoading:
    """Tests for lazy loading behavior."""

    def test_no_cache_on_load(self, test_bag_path):
        bag = baglab.load(test_bag_path)
        assert len(bag._cache) == 0
        bag.close()

    def test_cache_populated_on_access(self, bag):
        _ = bag["/test/joint_state"]
        assert "/test/joint_state" in bag._cache

    def test_cached_returns_same_object(self, bag):
        df1 = bag["/test/joint_state"]
        df2 = bag["/test/joint_state"]
        assert df1 is df2

    def test_field_selection_not_cached(self, bag):
        _ = bag["/test/joint_state", ["position"]]
        assert "/test/joint_state" not in bag._cache


class TestFieldSelection:
    """Tests for explicit field selection via bag[topic, fields]."""

    def test_select_single_field(self, bag):
        df = bag["/test/twist", ["twist.linear.x"]]
        assert list(df.columns) == ["twist.linear.x"]

    def test_select_multiple_fields(self, bag):
        df = bag["/test/joint_state", ["position", "effort"]]
        assert list(df.columns) == ["position", "effort"]

    def test_invalid_topic_raises(self, bag):
        with pytest.raises(KeyError, match="not found"):
            bag["/nonexistent/topic"]


class TestDataContent:
    """Tests for correctness of loaded data."""

    def test_row_count(self, bag):
        df = bag["/test/joint_state"]
        assert df.shape[0] > 100  # 5s at 100Hz = ~500 rows

    def test_index_is_datetime(self, bag):
        df = bag["/test/joint_state"]
        assert isinstance(df.index, pd.DatetimeIndex)

    def test_stamp_sec_is_integer(self, bag):
        df = bag["/test/joint_state"]
        assert df["header.stamp.sec"].dtype in ("int32", "int64", "uint32")

    def test_twist_values_are_float(self, bag):
        df = bag["/test/twist"]
        assert df["twist.linear.x"].dtype == "float64"


class TestStampToSec:
    """Tests for baglab.stamp_to_sec()."""

    def test_returns_series(self, bag):
        df = bag["/test/joint_state"]
        result = baglab.stamp_to_sec(df)
        assert isinstance(result, pd.Series)

    def test_dtype_float64(self, bag):
        df = bag["/test/joint_state"]
        result = baglab.stamp_to_sec(df)
        assert result.dtype == "float64"

    def test_values_are_positive(self, bag):
        df = bag["/test/joint_state"]
        result = baglab.stamp_to_sec(df)
        assert (result > 0).all()

    def test_monotonically_increasing(self, bag):
        df = bag["/test/joint_state"]
        result = baglab.stamp_to_sec(df)
        assert (result.diff().dropna() >= 0).all()

    def test_relative_starts_at_zero(self, bag):
        df = bag["/test/joint_state"]
        result = baglab.stamp_to_sec(df, relative=True)
        assert result.iloc[0] == pytest.approx(0.0)

    def test_relative_values_small(self, bag):
        df = bag["/test/joint_state"]
        result = baglab.stamp_to_sec(df, relative=True)
        # test data is ~5 seconds
        assert result.iloc[-1] < 10.0

    def test_relative_monotonically_increasing(self, bag):
        df = bag["/test/joint_state"]
        result = baglab.stamp_to_sec(df, relative=True)
        assert (result.diff().dropna() >= 0).all()


class TestRecvTimeToSec:
    """Tests for baglab.recv_time_to_sec()."""

    def test_returns_series(self, bag):
        df = bag["/test/joint_state"]
        result = baglab.recv_time_to_sec(df)
        assert isinstance(result, pd.Series)

    def test_dtype_float64(self, bag):
        df = bag["/test/joint_state"]
        result = baglab.recv_time_to_sec(df)
        assert result.dtype == "float64"

    def test_values_are_positive(self, bag):
        df = bag["/test/joint_state"]
        result = baglab.recv_time_to_sec(df)
        assert (result > 0).all()

    def test_monotonically_increasing(self, bag):
        df = bag["/test/joint_state"]
        result = baglab.recv_time_to_sec(df)
        assert (result.diff().dropna() >= 0).all()

    def test_relative_starts_at_zero(self, bag):
        df = bag["/test/joint_state"]
        result = baglab.recv_time_to_sec(df, relative=True)
        assert result.iloc[0] == pytest.approx(0.0)

    def test_relative_values_small(self, bag):
        df = bag["/test/joint_state"]
        result = baglab.recv_time_to_sec(df, relative=True)
        assert result.iloc[-1] < 10.0

    def test_relative_monotonically_increasing(self, bag):
        df = bag["/test/joint_state"]
        result = baglab.recv_time_to_sec(df, relative=True)
        assert (result.diff().dropna() >= 0).all()

    def test_differs_from_stamp_to_sec(self, bag):
        df = bag["/test/joint_state"]
        recv = baglab.recv_time_to_sec(df)
        stamp = baglab.stamp_to_sec(df)
        assert not (recv == stamp).all()

    def test_works_without_stamp_columns(self, bag):
        df = bag["/test/twist"]
        result = baglab.recv_time_to_sec(df, relative=True)
        assert result.iloc[0] == pytest.approx(0.0)


class TestAlignOrigin:
    """Tests for baglab.align_origin()."""

    def test_returns_tuple(self, bag):
        df = bag["/test/joint_state"]
        t1 = baglab.stamp_to_sec(df)
        t2 = baglab.recv_time_to_sec(df)
        result = baglab.align_origin(t1, t2)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_earliest_starts_at_zero(self, bag):
        df = bag["/test/joint_state"]
        t1 = baglab.stamp_to_sec(df)
        t2 = baglab.recv_time_to_sec(df)
        r1, r2 = baglab.align_origin(t1, t2)
        assert min(r1.iloc[0], r2.iloc[0]) == pytest.approx(0.0)

    def test_relative_offsets_preserved(self, bag):
        js = bag["/test/joint_state"]
        tw = bag["/test/twist"]
        t1 = baglab.stamp_to_sec(js)
        t2 = baglab.stamp_to_sec(tw)
        r1, r2 = baglab.align_origin(t1, t2)
        # difference between first elements should be preserved
        original_diff = t1.iloc[0] - t2.iloc[0]
        aligned_diff = r1.iloc[0] - r2.iloc[0]
        assert original_diff == pytest.approx(aligned_diff)

    def test_single_series(self, bag):
        df = bag["/test/joint_state"]
        t = baglab.stamp_to_sec(df)
        (r,) = baglab.align_origin(t)
        assert r.iloc[0] == pytest.approx(0.0)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            baglab.align_origin()


class TestExplodeArray:
    """Tests for baglab.explode_array()."""

    def test_returns_dataframe(self, bag):
        df = bag["/test/joint_state"]
        result = baglab.explode_array(df.msg.position)
        assert isinstance(result, pd.DataFrame)

    def test_preserves_index(self, bag):
        df = bag["/test/joint_state"]
        result = baglab.explode_array(df.msg.position)
        assert (result.index == df.index).all()

    def test_column_count_matches_array_length(self, bag):
        df = bag["/test/joint_state"]
        result = baglab.explode_array(df.msg.position)
        expected_len = len(df["position"].iloc[0])
        assert result.shape[1] == expected_len

    def test_custom_names(self, bag):
        df = bag["/test/joint_state"]
        n = len(df["position"].iloc[0])
        names = [f"j{i}" for i in range(n)]
        result = baglab.explode_array(df.msg.position, names=names)
        assert list(result.columns) == names

    def test_default_integer_names(self, bag):
        df = bag["/test/joint_state"]
        result = baglab.explode_array(df.msg.position)
        assert all(isinstance(c, int) for c in result.columns)

    def test_wrong_names_length_raises(self, bag):
        df = bag["/test/joint_state"]
        with pytest.raises(ValueError, match="names has"):
            baglab.explode_array(df.msg.position, names=["a", "b"])

    def test_values_match_original(self, bag):
        df = bag["/test/joint_state"]
        result = baglab.explode_array(df.msg.position)
        for i in range(min(5, len(df))):
            original = list(df["position"].iloc[i])
            expanded = list(result.iloc[i])
            assert original == expanded


class TestReindexByStamp:
    """Tests for baglab.reindex_by_stamp()."""

    def test_returns_new_dataframe(self, bag):
        df = bag["/test/joint_state"]
        result = baglab.reindex_by_stamp(df)
        assert result is not df

    def test_original_unchanged(self, bag):
        df = bag["/test/joint_state"]
        original_index = df.index.copy()
        _ = baglab.reindex_by_stamp(df)
        assert (df.index == original_index).all()

    def test_index_is_datetime(self, bag):
        df = bag["/test/joint_state"]
        result = baglab.reindex_by_stamp(df)
        assert isinstance(result.index, pd.DatetimeIndex)

    def test_index_differs_from_original(self, bag):
        df = bag["/test/joint_state"]
        result = baglab.reindex_by_stamp(df)
        # receive time and publish time should differ slightly
        assert not (df.index == result.index).all()

    def test_columns_preserved(self, bag):
        df = bag["/test/joint_state"]
        result = baglab.reindex_by_stamp(df)
        assert list(result.columns) == list(df.columns)


class TestContextManager:
    """Tests for Bag as context manager."""

    def test_with_statement(self, test_bag_path):
        with baglab.load(test_bag_path) as bag:
            assert len(bag.topics) == 2
        assert bag._closed

    def test_double_close_safe(self, test_bag_path):
        bag = baglab.load(test_bag_path)
        bag.close()
        bag.close()  # should not raise


class TestMsgAccessor:
    """Tests for df.msg accessor (ROS2-style dot access)."""

    def test_leaf_access_returns_series(self, bag):
        df = bag["/test/twist"]
        result = df.msg.twist.linear.x
        assert isinstance(result, pd.Series)
        assert len(result) == len(df)

    def test_intermediate_returns_field_group(self, bag):
        df = bag["/test/twist"]
        from baglab.io import FieldGroup
        assert isinstance(df.msg.twist.linear, FieldGroup)
        assert isinstance(df.msg.twist, FieldGroup)

    def test_field_group_df_strips_prefix(self, bag):
        df = bag["/test/twist"]
        result = df.msg.twist.linear.df
        assert list(result.columns) == ["x", "y", "z"]

    def test_field_group_df_deeper_prefix(self, bag):
        df = bag["/test/twist"]
        result = df.msg.twist.df
        expected = ["linear.x", "linear.y", "linear.z",
                    "angular.x", "angular.y", "angular.z"]
        assert list(result.columns) == expected

    def test_field_group_df_preserves_index(self, bag):
        df = bag["/test/twist"]
        result = df.msg.twist.linear.df
        assert (result.index == df.index).all()

    def test_leaf_preserves_index(self, bag):
        df = bag["/test/twist"]
        result = df.msg.twist.linear.x
        assert (result.index == df.index).all()

    def test_bracket_notation(self, bag):
        df = bag["/test/twist"]
        result = df.msg["twist"]["linear"]["x"]
        assert isinstance(result, pd.Series)

    def test_nonexistent_field_raises(self, bag):
        df = bag["/test/twist"]
        with pytest.raises(AttributeError, match="No fields matching"):
            df.msg.nonexistent

    def test_nonexistent_nested_raises(self, bag):
        df = bag["/test/twist"]
        with pytest.raises(AttributeError, match="No fields matching"):
            df.msg.twist.nonexistent

    def test_header_access(self, bag):
        df = bag["/test/twist"]
        stamp_df = df.msg.header.stamp.df
        assert list(stamp_df.columns) == ["sec", "nanosec"]

    def test_field_group_repr(self, bag):
        df = bag["/test/twist"]
        r = repr(df.msg.twist.linear)
        assert "FieldGroup(" in r
        assert "twist.linear" in r


class TestParquetCache:
    """Tests for disk cache (Parquet / pickle)."""

    def test_cache_dir_created(self, writable_bag):
        _ = writable_bag["/test/twist"]
        cache_dir = writable_bag._path / ".baglab_cache"
        assert cache_dir.is_dir()

    def test_cache_files_exist(self, writable_bag):
        _ = writable_bag["/test/twist"]
        cache_dir = writable_bag._path / ".baglab_cache"
        assert (cache_dir / "test__twist.parquet").exists()
        assert (cache_dir / "test__twist.meta.json").exists()

    def test_cached_matches_fresh(self, writable_bag):
        df_fresh = writable_bag["/test/twist"]
        # Clear in-memory cache to force disk read
        writable_bag._cache.clear()
        df_cached = writable_bag["/test/twist"]
        pd.testing.assert_frame_equal(df_fresh, df_cached)

    def test_cross_session_cache(self, test_bag_path, tmp_path):
        import shutil
        dst = tmp_path / "bag"
        shutil.copytree(test_bag_path, dst)
        # Session 1
        b1 = baglab.load(dst)
        df1 = b1["/test/twist"]
        b1.close()
        # Session 2
        b2 = baglab.load(dst)
        df2 = b2["/test/twist"]
        b2.close()
        pd.testing.assert_frame_equal(df1, df2)

    def test_cache_invalidation(self, writable_bag):
        _ = writable_bag["/test/twist"]
        # Tamper with fingerprint
        import json
        meta_path = writable_bag._path / ".baglab_cache" / "test__twist.meta.json"
        meta = json.loads(meta_path.read_text())
        meta["bag_files"] = {"fake.db3": {"mtime_ns": 0, "size": 0}}
        meta_path.write_text(json.dumps(meta))
        assert not writable_bag._is_cache_valid("/test/twist")

    def test_array_columns_roundtrip(self, writable_bag):
        df_fresh = writable_bag["/test/joint_state"]
        writable_bag._cache.clear()
        df_cached = writable_bag["/test/joint_state"]
        # Array column values should survive round-trip
        for i in range(min(5, len(df_fresh))):
            assert list(df_fresh["position"].iloc[i]) == list(df_cached["position"].iloc[i])

    def test_clear_cache(self, writable_bag):
        _ = writable_bag["/test/twist"]
        cache_dir = writable_bag._path / ".baglab_cache"
        assert cache_dir.exists()
        clear_cache(writable_bag._path)
        assert not cache_dir.exists()

    def test_field_selection_still_uncached(self, writable_bag):
        _ = writable_bag["/test/joint_state", ["position"]]
        assert "/test/joint_state" not in writable_bag._cache


class TestRepr:
    """Tests for Bag.__repr__."""

    def test_repr(self, bag):
        r = repr(bag)
        assert "Bag(" in r
        assert "2 topics" in r
