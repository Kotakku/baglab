#define MCAP_IMPLEMENTATION
#include "mcap_reader.hpp"

#include <mcap/reader.hpp>

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

namespace fs = std::filesystem;

namespace baglab_mcap {

// ---------------------------------------------------------------------------
// Memory-mapped IReadable for zero-copy I/O
// ---------------------------------------------------------------------------

class MmapReader final : public mcap::IReadable {
public:
    MmapReader(const std::string& path) {
        fd_ = ::open(path.c_str(), O_RDONLY);
        if (fd_ < 0) {
            throw std::runtime_error("Cannot open file: " + path);
        }
        struct stat st;
        if (::fstat(fd_, &st) != 0) {
            ::close(fd_);
            throw std::runtime_error("Cannot stat file: " + path);
        }
        size_ = static_cast<uint64_t>(st.st_size);
        data_ = static_cast<const std::byte*>(
            ::mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd_, 0));
        if (data_ == MAP_FAILED) {
            ::close(fd_);
            throw std::runtime_error("mmap failed for: " + path);
        }
        ::madvise(const_cast<std::byte*>(data_), size_, MADV_SEQUENTIAL);
    }

    ~MmapReader() override {
        if (data_ && data_ != MAP_FAILED) {
            ::munmap(const_cast<std::byte*>(data_), size_);
        }
        if (fd_ >= 0) {
            ::close(fd_);
        }
    }

    MmapReader(const MmapReader&) = delete;
    MmapReader& operator=(const MmapReader&) = delete;

    uint64_t size() const override { return size_; }

    uint64_t read(std::byte** output, uint64_t offset, uint64_t size) override {
        if (offset >= size_) {
            return 0;
        }
        uint64_t available = std::min(size, size_ - offset);
        *output = const_cast<std::byte*>(data_ + offset);
        return available;
    }

private:
    int fd_ = -1;
    const std::byte* data_ = nullptr;
    uint64_t size_ = 0;
};

// ---------------------------------------------------------------------------
// Path resolution
// ---------------------------------------------------------------------------

static std::string resolve_mcap_path(const std::string& path) {
    fs::path p(path);
    if (fs::is_regular_file(p) && p.extension() == ".mcap") {
        return path;
    }
    if (fs::is_directory(p)) {
        for (const auto& entry : fs::directory_iterator(p)) {
            if (entry.path().extension() == ".mcap") {
                return entry.path().string();
            }
        }
        throw std::runtime_error("No .mcap file found in directory: " + path);
    }
    throw std::runtime_error("Path is not a file or directory: " + path);
}

// ---------------------------------------------------------------------------
// Helper: collect TopicInfo from an open McapReader
// ---------------------------------------------------------------------------

static std::map<std::string, TopicInfo> collect_topic_infos(mcap::McapReader& reader) {
    std::map<std::string, TopicInfo> result;
    auto channels = reader.channels();
    auto schemas = reader.schemas();
    const auto& stats_opt = reader.statistics();

    for (const auto& [ch_id, ch_ptr] : channels) {
        TopicInfo info;
        info.name = ch_ptr->topic;
        info.channel_id = ch_id;

        auto schema_it = schemas.find(ch_ptr->schemaId);
        if (schema_it != schemas.end()) {
            info.msg_type = schema_it->second->name;
            info.schema_encoding = schema_it->second->encoding;
            const auto& data = schema_it->second->data;
            info.schema_text = std::string(
                reinterpret_cast<const char*>(data.data()), data.size());
        }

        if (stats_opt.has_value()) {
            auto count_it = stats_opt->channelMessageCounts.find(ch_id);
            if (count_it != stats_opt->channelMessageCounts.end()) {
                info.message_count = count_it->second;
            }
        }

        result[info.name] = std::move(info);
    }
    return result;
}

// ---------------------------------------------------------------------------
// Helper: read a single topic given an open McapReader + TopicInfo
// ---------------------------------------------------------------------------

static ColumnarResult read_topic_from_reader(
    mcap::McapReader& reader,
    const TopicInfo& info,
    const std::vector<std::string>& field_paths)
{
    if (info.schema_encoding != "ros2msg") {
        throw std::runtime_error(
            "Unsupported schema encoding '" + info.schema_encoding +
            "' for topic " + info.name +
            ". Only 'ros2msg' is supported. "
            "'ros2idl' schemas require the rosbags backend.");
    }

    auto layouts = parse_ros2msg_schema(info.schema_text, info.msg_type);
    auto plan = build_extraction_plan(info.msg_type, layouts, field_paths);

    std::vector<Column> columns(plan.column_names.size());
    for (size_t i = 0; i < columns.size(); ++i) {
        columns[i].type = plan.column_types[i];
        columns[i].is_list = plan.column_is_list[i];
        if (!plan.column_is_list[i]) {
            reserve_column(columns[i], info.message_count);
        }
    }

    std::vector<int64_t> timestamps;
    timestamps.reserve(info.message_count);

    {
        std::string target_topic = info.name;
        mcap::ReadMessageOptions opts;
        opts.topicFilter = [&target_topic](std::string_view topic) {
            return topic == target_topic;
        };

        auto messages = reader.readMessages(
            [](const mcap::Status&) {},
            opts);

        for (const auto& msg_view : messages) {
            if (msg_view.message.data == nullptr || msg_view.message.dataSize == 0) {
                continue;
            }

            timestamps.push_back(
                static_cast<int64_t>(msg_view.message.logTime));

            CdrReader cdr(
                reinterpret_cast<const uint8_t*>(msg_view.message.data),
                static_cast<size_t>(msg_view.message.dataSize));

            try {
                execute_plan(cdr, plan.steps, columns);
            } catch (const std::exception&) {
                timestamps.pop_back();
            }
        }
    }

    if (timestamps.empty()) {
        ColumnarResult result;
        result["__timestamps__"] = py::array_t<int64_t>(0);
        return result;
    }

    return build_columnar_result(timestamps, plan, columns);
}

// ---------------------------------------------------------------------------
// get_topics (standalone)
// ---------------------------------------------------------------------------

std::map<std::string, TopicInfo> get_topics(const std::string& path) {
    std::string mcap_path = resolve_mcap_path(path);

    MmapReader mmap_reader(mcap_path);
    mcap::McapReader reader;
    auto status = reader.open(mmap_reader);
    if (!status.ok()) {
        throw std::runtime_error("Failed to open MCAP: " + status.message);
    }

    auto summary = reader.readSummary(mcap::ReadSummaryMethod::NoFallbackScan);
    if (!summary.ok()) {
        throw std::runtime_error("Failed to read MCAP summary: " + summary.message);
    }

    auto result = collect_topic_infos(reader);
    reader.close();
    return result;
}

// ---------------------------------------------------------------------------
// read_topic (standalone — opens/closes each call)
// ---------------------------------------------------------------------------

ColumnarResult read_topic(
    const std::string& path,
    const std::string& topic_name,
    const std::vector<std::string>& field_paths)
{
    std::string mcap_path = resolve_mcap_path(path);

    MmapReader mmap_reader(mcap_path);
    mcap::McapReader reader;
    auto status = reader.open(mmap_reader);
    if (!status.ok()) {
        throw std::runtime_error("Failed to open MCAP: " + status.message);
    }

    auto summary_status = reader.readSummary(mcap::ReadSummaryMethod::NoFallbackScan);
    if (!summary_status.ok()) {
        throw std::runtime_error("Failed to read summary: " + summary_status.message);
    }

    auto infos = collect_topic_infos(reader);
    auto it = infos.find(topic_name);
    if (it == infos.end()) {
        throw std::runtime_error("Topic '" + topic_name + "' not found in MCAP file");
    }

    auto result = read_topic_from_reader(reader, it->second, field_paths);
    reader.close();
    return result;
}

// ---------------------------------------------------------------------------
// BagReader (persistent — open once, read many topics)
// ---------------------------------------------------------------------------

struct BagReader::Impl {
    mcap::McapReader reader;
};

BagReader::BagReader(const std::string& path) {
    std::string mcap_path = resolve_mcap_path(path);

    mmap_reader_ = std::make_unique<MmapReader>(mcap_path);
    impl_ = std::make_unique<Impl>();

    auto status = impl_->reader.open(*mmap_reader_);
    if (!status.ok()) {
        throw std::runtime_error("Failed to open MCAP: " + status.message);
    }

    auto summary = impl_->reader.readSummary(mcap::ReadSummaryMethod::NoFallbackScan);
    if (!summary.ok()) {
        throw std::runtime_error("Failed to read MCAP summary: " + summary.message);
    }

    topic_infos_ = collect_topic_infos(impl_->reader);
}

BagReader::~BagReader() {
    close();
}

ColumnarResult BagReader::read_topic(
    const std::string& topic_name,
    const std::vector<std::string>& field_paths)
{
    if (!impl_) {
        throw std::runtime_error("BagReader is closed");
    }

    auto it = topic_infos_.find(topic_name);
    if (it == topic_infos_.end()) {
        throw std::runtime_error("Topic '" + topic_name + "' not found in MCAP file");
    }

    return read_topic_from_reader(impl_->reader, it->second, field_paths);
}

std::map<std::string, ColumnarResult> BagReader::read_topics(
    const std::vector<std::string>& topic_names,
    const std::map<std::string, std::vector<std::string>>& topic_field_paths)
{
    if (!impl_) {
        throw std::runtime_error("BagReader is closed");
    }

    // Per-topic state for single-pass extraction
    struct TopicState {
        std::string topic_name;
        ExtractionPlan plan;
        std::vector<Column> columns;
        std::vector<int64_t> timestamps;
    };

    // channel_id → TopicState index for fast dispatch
    std::map<uint16_t, size_t> channel_to_state;
    std::vector<TopicState> states;
    std::set<std::string> target_topics;

    for (const auto& topic_name : topic_names) {
        auto it = topic_infos_.find(topic_name);
        if (it == topic_infos_.end()) continue;
        const auto& info = it->second;
        if (info.schema_encoding != "ros2msg") continue;

        try {
            auto layouts = parse_ros2msg_schema(info.schema_text, info.msg_type);

            auto fp_it = topic_field_paths.find(topic_name);
            std::vector<std::string> fields =
                (fp_it != topic_field_paths.end()) ? fp_it->second : std::vector<std::string>{};
            auto plan = build_extraction_plan(info.msg_type, layouts, fields);

            TopicState state;
            state.topic_name = topic_name;
            state.plan = std::move(plan);
            state.columns.resize(state.plan.column_names.size());
            for (size_t i = 0; i < state.columns.size(); ++i) {
                state.columns[i].type = state.plan.column_types[i];
                state.columns[i].is_list = state.plan.column_is_list[i];
                if (!state.plan.column_is_list[i]) {
                    reserve_column(state.columns[i], info.message_count);
                }
            }
            state.timestamps.reserve(info.message_count);

            size_t idx = states.size();
            channel_to_state[info.channel_id] = idx;
            target_topics.insert(topic_name);
            states.push_back(std::move(state));
        } catch (const std::exception&) {
            // Skip topics with parse errors
            continue;
        }
    }

    // Single pass through all messages, dispatching by channel_id
    {
        mcap::ReadMessageOptions opts;
        opts.topicFilter = [&target_topics](std::string_view topic) {
            return target_topics.count(std::string(topic)) > 0;
        };

        auto messages = impl_->reader.readMessages(
            [](const mcap::Status&) {},
            opts);

        for (const auto& msg_view : messages) {
            auto ch_it = channel_to_state.find(msg_view.channel->id);
            if (ch_it == channel_to_state.end()) continue;

            auto& state = states[ch_it->second];

            if (msg_view.message.data == nullptr || msg_view.message.dataSize == 0) {
                continue;
            }

            state.timestamps.push_back(
                static_cast<int64_t>(msg_view.message.logTime));

            CdrReader cdr(
                reinterpret_cast<const uint8_t*>(msg_view.message.data),
                static_cast<size_t>(msg_view.message.dataSize));

            try {
                execute_plan(cdr, state.plan.steps, state.columns);
            } catch (const std::exception&) {
                state.timestamps.pop_back();
            }
        }
    }

    // Build results
    std::map<std::string, ColumnarResult> results;
    for (auto& state : states) {
        if (state.timestamps.empty()) {
            ColumnarResult result;
            result["__timestamps__"] = py::array_t<int64_t>(0);
            results[state.topic_name] = std::move(result);
        } else {
            results[state.topic_name] = build_columnar_result(
                state.timestamps, state.plan, state.columns);
        }
    }

    return results;
}

std::map<std::string, std::string> BagReader::topics() const {
    std::map<std::string, std::string> result;
    for (const auto& [name, info] : topic_infos_) {
        result[name] = info.msg_type;
    }
    return result;
}

void BagReader::close() {
    if (impl_) {
        impl_->reader.close();
        impl_.reset();
    }
    mmap_reader_.reset();
    topic_infos_.clear();
}

}  // namespace baglab_mcap
