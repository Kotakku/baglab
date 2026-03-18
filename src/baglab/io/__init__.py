"""I/O module for rosbag loading, field access, and timestamp utilities."""

from baglab.io.accessor import FieldGroup, MsgAccessor, explode_array
from baglab.io.bag import Bag, clear_cache, load
from baglab.io.stamp import align_origin, find_time, recv_time_to_sec, reindex_by_stamp, stamp_to_sec, time_slice
from baglab.io.typesys import register_msg_types

__all__ = [
    "Bag",
    "align_origin",
    "clear_cache",
    "explode_array",
    "FieldGroup",
    "MsgAccessor",
    "find_time",
    "load",
    "recv_time_to_sec",
    "register_msg_types",
    "reindex_by_stamp",
    "stamp_to_sec",
    "time_slice",
]
