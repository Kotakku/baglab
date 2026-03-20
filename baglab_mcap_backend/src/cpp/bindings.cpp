#include "mcap_reader.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

static py::dict py_read_topic(
    const std::string& path,
    const std::string& topic_name,
    const std::vector<std::string>& field_paths)
{
    auto columns = baglab_mcap::read_topic(path, topic_name, field_paths);
    py::dict result;
    for (auto& [key, val] : columns) {
        result[py::str(key)] = std::move(val);
    }
    return result;
}

static py::dict py_get_topics(const std::string& path) {
    auto topics = baglab_mcap::get_topics(path);
    py::dict result;
    for (const auto& [name, info] : topics) {
        result[py::str(name)] = py::str(info.msg_type);
    }
    return result;
}

static py::dict bag_reader_read_topic(
    baglab_mcap::BagReader& self,
    const std::string& topic_name,
    const std::vector<std::string>& field_paths)
{
    auto columns = self.read_topic(topic_name, field_paths);
    py::dict result;
    for (auto& [key, val] : columns) {
        result[py::str(key)] = std::move(val);
    }
    return result;
}

PYBIND11_MODULE(_core, m) {
    m.doc() = "baglab MCAP backend (no ROS 2 dependency)";

    m.def("get_topics", &py_get_topics,
          py::arg("bag_path"),
          "Get {topic_name: message_type} from an MCAP file.");

    m.def("read_topic", &py_read_topic,
          py::arg("bag_path"),
          py::arg("topic_name"),
          py::arg("field_paths") = std::vector<std::string>{},
          "Read a topic and return columnar data as a dict of numpy arrays.");

    py::class_<baglab_mcap::BagReader>(m, "BagReader")
        .def(py::init<const std::string&>(), py::arg("bag_path"))
        .def("read_topic", &bag_reader_read_topic,
             py::arg("topic_name"),
             py::arg("field_paths") = std::vector<std::string>{})
        .def("read_topics", [](baglab_mcap::BagReader& self,
                               const std::vector<std::string>& topic_names,
                               const std::map<std::string, std::vector<std::string>>& field_paths) {
            auto results = self.read_topics(topic_names, field_paths);
            py::dict py_results;
            for (auto& [topic, columns] : results) {
                py::dict topic_dict;
                for (auto& [key, val] : columns) {
                    topic_dict[py::str(key)] = std::move(val);
                }
                py_results[py::str(topic)] = std::move(topic_dict);
            }
            return py_results;
        },
             py::arg("topic_names"),
             py::arg("field_paths") = std::map<std::string, std::vector<std::string>>{})
        .def("topics", &baglab_mcap::BagReader::topics)
        .def("close", &baglab_mcap::BagReader::close);
}
