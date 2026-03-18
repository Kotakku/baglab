#include "reader.hpp"
#include "deserializer.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

/// Read a single topic and return columnar data.
///
/// Returns a dict:
///   "__timestamps__" -> numpy int64 array (nanoseconds)
///   "field.path"     -> numpy array or Python list
///
/// If field_paths is empty, all fields are returned.
static py::dict read_topic(
    const std::string& bag_path,
    const std::string& topic_name,
    const std::vector<std::string>& field_paths)
{
    auto msg_type = baglab_cpp_backend::get_topic_type(bag_path, topic_name);
    auto messages = baglab_cpp_backend::read_raw_messages(bag_path, topic_name);
    auto columns = baglab_cpp_backend::deserialize_to_columns(msg_type, messages, field_paths);

    py::dict result;
    for (auto& [key, val] : columns) {
        result[py::str(key)] = std::move(val);
    }
    return result;
}

/// Get topic metadata from a bag.
static py::dict py_get_topics(const std::string& bag_path) {
    auto topics = baglab_cpp_backend::get_topics(bag_path);
    py::dict result;
    for (const auto& [name, type] : topics) {
        result[py::str(name)] = py::str(type);
    }
    return result;
}

PYBIND11_MODULE(_core, m) {
    m.doc() = "baglab C++ acceleration backend";

    m.def("get_topics", &py_get_topics,
          py::arg("bag_path"),
          "Get {topic_name: message_type} from a rosbag.");

    m.def("read_topic", &read_topic,
          py::arg("bag_path"),
          py::arg("topic_name"),
          py::arg("field_paths") = std::vector<std::string>{},
          "Read a topic and return columnar data as a dict of numpy arrays.");
}
