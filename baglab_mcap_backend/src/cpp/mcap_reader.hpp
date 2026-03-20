#pragma once

#include "cdr_deserializer.hpp"
#include "schema_parser.hpp"

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace baglab_mcap {

/// Topic metadata from MCAP summary.
struct TopicInfo {
    std::string name;
    std::string msg_type;       // e.g. "geometry_msgs/msg/TwistStamped"
    std::string schema_text;    // ros2msg schema
    std::string schema_encoding;
    uint16_t channel_id = 0;
    uint64_t message_count = 0;
};

/// Get topic metadata from an MCAP file.
/// Accepts either a directory (with metadata.yaml) or a direct .mcap path.
std::map<std::string, TopicInfo> get_topics(const std::string& path);

/// Read a single topic and return columnar data (opens/closes file each call).
ColumnarResult read_topic(
    const std::string& path,
    const std::string& topic_name,
    const std::vector<std::string>& field_paths);

// Forward declaration
class MmapReader;

/// Persistent reader that keeps the MCAP file open for sequential per-topic reads.
/// mmap + readSummary is done once in the constructor.
class BagReader {
public:
    explicit BagReader(const std::string& path);
    ~BagReader();

    BagReader(const BagReader&) = delete;
    BagReader& operator=(const BagReader&) = delete;

    /// Read a single topic. Can be called repeatedly for different topics
    /// without re-opening the file or re-reading the summary.
    ColumnarResult read_topic(
        const std::string& topic_name,
        const std::vector<std::string>& field_paths);

    /// Read multiple topics in a single pass through the file.
    /// Much faster than calling read_topic() repeatedly.
    /// Returns {topic_name -> columnar data}.
    std::map<std::string, ColumnarResult> read_topics(
        const std::vector<std::string>& topic_names,
        const std::map<std::string, std::vector<std::string>>& topic_field_paths);

    /// Return {topic_name -> message_type} for all topics.
    std::map<std::string, std::string> topics() const;

    void close();

private:
    std::unique_ptr<MmapReader> mmap_reader_;
    struct Impl;
    std::unique_ptr<Impl> impl_;
    std::map<std::string, TopicInfo> topic_infos_;
};

}  // namespace baglab_mcap
