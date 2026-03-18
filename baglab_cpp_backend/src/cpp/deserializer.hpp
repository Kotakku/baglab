#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <variant>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

namespace baglab_cpp_backend {

/// Holds columnar data for all requested fields + timestamps.
/// Key: field name (dot notation) or "__timestamps__"
/// Value: py::object (numpy array or Python list)
using ColumnarResult = std::map<std::string, py::object>;

/// Deserialize raw messages and extract fields into columnar data.
///
/// @param msg_type  ROS 2 message type (e.g. "geometry_msgs/msg/TwistStamped")
/// @param messages  Raw serialized messages with timestamps
/// @param field_paths  Requested field paths in dot notation. Empty = all fields.
/// @return  Columnar data including "__timestamps__" key.
ColumnarResult deserialize_to_columns(
    const std::string& msg_type,
    const std::vector<struct RawMessage>& messages,
    const std::vector<std::string>& field_paths);

}  // namespace baglab_cpp_backend
