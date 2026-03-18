#pragma once

#include <map>
#include <string>
#include <vector>
#include <cstdint>

namespace baglab_cpp_backend {

/// Topic metadata: {topic_name -> message_type}
std::map<std::string, std::string> get_topics(const std::string& bag_path);

/// A single serialized message with its receive timestamp.
struct RawMessage {
    int64_t timestamp_ns;
    std::vector<uint8_t> data;
};

/// Read all serialized messages for a topic.
std::vector<RawMessage> read_raw_messages(
    const std::string& bag_path,
    const std::string& topic_name);

/// Get the message type name for a topic.
std::string get_topic_type(const std::string& bag_path, const std::string& topic_name);

}  // namespace baglab_cpp_backend
