#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace baglab_mcap {

enum class PrimitiveType : uint8_t {
    BOOL,
    INT8,
    UINT8,
    INT16,
    UINT16,
    INT32,
    UINT32,
    INT64,
    UINT64,
    FLOAT32,
    FLOAT64,
    STRING,
};

/// Size in bytes for fixed-size primitives. 0 for STRING.
size_t primitive_size(PrimitiveType t);

/// CDR alignment for a primitive type.
size_t primitive_alignment(PrimitiveType t);

struct FieldDef {
    std::string name;
    PrimitiveType type{};
    bool is_primitive = false;
    bool is_array = false;       // fixed-size array
    bool is_sequence = false;    // dynamic-size sequence
    size_t array_size = 0;       // >0 for fixed array
    std::string nested_type;     // e.g. "geometry_msgs/msg/Vector3"
};

struct MessageLayout {
    std::string full_name;
    std::vector<FieldDef> fields;
};

/// Parse ros2msg-format schema text into a map of message layouts.
/// root_type is e.g. "geometry_msgs/msg/TwistStamped".
/// The schema text contains the root definition followed by
/// sub-definitions separated by "===...".
std::map<std::string, MessageLayout> parse_ros2msg_schema(
    const std::string& schema_text,
    const std::string& root_type);

}  // namespace baglab_mcap
