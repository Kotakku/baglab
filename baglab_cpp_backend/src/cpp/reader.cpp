#include "reader.hpp"

#include <rosbag2_cpp/reader.hpp>
#include <rosbag2_storage/storage_options.hpp>
#include <rosbag2_storage/storage_filter.hpp>
#include <stdexcept>

namespace baglab_cpp_backend {

std::map<std::string, std::string> get_topics(const std::string& bag_path) {
    rosbag2_cpp::Reader reader;
    rosbag2_storage::StorageOptions opts;
    opts.uri = bag_path;
    reader.open(opts);

    auto metadata = reader.get_metadata();
    std::map<std::string, std::string> result;
    for (const auto& info : metadata.topics_with_message_count) {
        result[info.topic_metadata.name] = info.topic_metadata.type;
    }
    return result;
}

std::string get_topic_type(const std::string& bag_path, const std::string& topic_name) {
    auto topics = get_topics(bag_path);
    auto it = topics.find(topic_name);
    if (it == topics.end()) {
        throw std::runtime_error("Topic '" + topic_name + "' not found in bag");
    }
    return it->second;
}

std::vector<RawMessage> read_raw_messages(
    const std::string& bag_path,
    const std::string& topic_name)
{
    rosbag2_cpp::Reader reader;
    rosbag2_storage::StorageOptions opts;
    opts.uri = bag_path;
    reader.open(opts);

    rosbag2_storage::StorageFilter filter;
    filter.topics = {topic_name};
    reader.set_filter(filter);

    std::vector<RawMessage> messages;
    while (reader.has_next()) {
        auto msg = reader.read_next();
        RawMessage raw;
        raw.timestamp_ns = msg->time_stamp;
        auto& ser = msg->serialized_data;
        raw.data.assign(
            ser->buffer,
            ser->buffer + ser->buffer_length);
        messages.push_back(std::move(raw));
    }
    return messages;
}

}  // namespace baglab_cpp_backend
