#include "schema_parser.hpp"

#include <algorithm>
#include <sstream>
#include <stdexcept>

namespace baglab_mcap {

size_t primitive_size(PrimitiveType t) {
    switch (t) {
        case PrimitiveType::BOOL:
        case PrimitiveType::INT8:
        case PrimitiveType::UINT8:   return 1;
        case PrimitiveType::INT16:
        case PrimitiveType::UINT16:  return 2;
        case PrimitiveType::INT32:
        case PrimitiveType::UINT32:
        case PrimitiveType::FLOAT32: return 4;
        case PrimitiveType::INT64:
        case PrimitiveType::UINT64:
        case PrimitiveType::FLOAT64: return 8;
        case PrimitiveType::STRING:  return 0;
    }
    return 0;
}

size_t primitive_alignment(PrimitiveType t) {
    size_t s = primitive_size(t);
    return s == 0 ? 4 : s;  // string aligned to 4 (for length prefix)
}

// Map ROS 2 type name string to PrimitiveType
static bool try_parse_primitive(const std::string& type_str, PrimitiveType& out) {
    static const std::map<std::string, PrimitiveType> map = {
        {"bool",    PrimitiveType::BOOL},
        {"int8",    PrimitiveType::INT8},
        {"uint8",   PrimitiveType::UINT8},
        {"byte",    PrimitiveType::UINT8},
        {"char",    PrimitiveType::UINT8},
        {"int16",   PrimitiveType::INT16},
        {"uint16",  PrimitiveType::UINT16},
        {"int32",   PrimitiveType::INT32},
        {"uint32",  PrimitiveType::UINT32},
        {"int64",   PrimitiveType::INT64},
        {"uint64",  PrimitiveType::UINT64},
        {"float32", PrimitiveType::FLOAT32},
        {"float64", PrimitiveType::FLOAT64},
        {"string",  PrimitiveType::STRING},
        {"wstring", PrimitiveType::STRING},
    };
    auto it = map.find(type_str);
    if (it != map.end()) {
        out = it->second;
        return true;
    }
    return false;
}

static std::string trim(const std::string& s) {
    auto start = s.find_first_not_of(" \t\r\n");
    if (start == std::string::npos) return "";
    auto end = s.find_last_not_of(" \t\r\n");
    return s.substr(start, end - start + 1);
}

/// Resolve short type name like "Twist" to full "geometry_msgs/msg/Twist"
/// using the available layouts map. If already fully qualified, return as-is.
static std::string resolve_type(
    const std::string& type_str,
    const std::string& root_pkg,
    const std::map<std::string, MessageLayout>& layouts)
{
    // Already fully qualified
    if (type_str.find('/') != std::string::npos) {
        return type_str;
    }
    // Try same package
    std::string candidate = root_pkg + "/msg/" + type_str;
    if (layouts.count(candidate)) {
        return candidate;
    }
    // Search all layouts
    for (const auto& [name, _] : layouts) {
        auto pos = name.rfind('/');
        if (pos != std::string::npos && name.substr(pos + 1) == type_str) {
            return name;
        }
    }
    return candidate;  // best guess
}

static MessageLayout parse_block(const std::string& block) {
    MessageLayout layout;
    std::istringstream ss(block);
    std::string line;

    while (std::getline(ss, line)) {
        // Strip inline comments (e.g. "float64[<=3] dimensions  # comment")
        auto comment_pos = line.find('#');
        if (comment_pos != std::string::npos) {
            line = line.substr(0, comment_pos);
        }
        line = trim(line);
        if (line.empty()) continue;

        // Parse "type name" or "type[] name" or "type[N] name"
        // Also handle "type<=N] name" for bounded sequences
        std::istringstream ls(line);
        std::string type_str, field_name;
        ls >> type_str >> field_name;
        if (field_name.empty()) continue;

        // Skip constants (e.g. "uint8 BOX=1", "uint8 TRAILER = 4")
        // Constants have '=' in field_name or a third token containing '='
        if (field_name.find('=') != std::string::npos) continue;
        // Handle "uint8 NAME = VALUE" (space around =)
        std::string rest;
        ls >> rest;
        if (!rest.empty() && rest[0] == '=') continue;

        FieldDef field;
        field.name = field_name;

        // Check for array suffix
        auto bracket = type_str.find('[');
        std::string base_type = type_str;
        if (bracket != std::string::npos) {
            base_type = type_str.substr(0, bracket);
            auto close = type_str.find(']', bracket);
            std::string inner = type_str.substr(bracket + 1, close - bracket - 1);
            if (inner.empty()) {
                field.is_sequence = true;
            } else if (inner.find("<=") != std::string::npos) {
                field.is_sequence = true;  // bounded sequence
            } else {
                field.is_array = true;
                field.array_size = std::stoull(inner);
            }
        }

        // Check for bounded string "string<=N"
        auto leq = base_type.find("<=");
        if (leq != std::string::npos) {
            base_type = base_type.substr(0, leq);
        }

        PrimitiveType prim;
        if (try_parse_primitive(base_type, prim)) {
            field.is_primitive = true;
            field.type = prim;
        } else {
            field.is_primitive = false;
            field.nested_type = base_type;
        }

        layout.fields.push_back(std::move(field));
    }

    return layout;
}

std::map<std::string, MessageLayout> parse_ros2msg_schema(
    const std::string& schema_text,
    const std::string& root_type)
{
    std::map<std::string, MessageLayout> layouts;

    // Built-in types that may not be in the schema text
    {
        MessageLayout time_layout;
        time_layout.full_name = "builtin_interfaces/msg/Time";
        time_layout.fields.push_back({"sec", PrimitiveType::INT32, true, false, false, 0, ""});
        time_layout.fields.push_back({"nanosec", PrimitiveType::UINT32, true, false, false, 0, ""});
        layouts["builtin_interfaces/msg/Time"] = time_layout;

        MessageLayout dur_layout;
        dur_layout.full_name = "builtin_interfaces/msg/Duration";
        dur_layout.fields = time_layout.fields;
        layouts["builtin_interfaces/msg/Duration"] = dur_layout;
    }

    // Split on separator lines (=== or lines of only '=' chars)
    std::vector<std::pair<std::string, std::string>> blocks;  // (type_name, block_text)
    std::istringstream ss(schema_text);
    std::string line;
    std::string current_block;
    std::string current_type = root_type;

    while (std::getline(ss, line)) {
        std::string trimmed = trim(line);
        // Separator: line of '=' chars (at least 3)
        if (trimmed.size() >= 3 && trimmed.find_first_not_of('=') == std::string::npos) {
            if (!current_block.empty()) {
                blocks.push_back({current_type, current_block});
                current_block.clear();
            }
            current_type.clear();
            continue;
        }
        // MSG: header
        if (trimmed.substr(0, 4) == "MSG:") {
            current_type = trim(trimmed.substr(4));
            continue;
        }
        current_block += line + "\n";
    }
    if (!current_block.empty()) {
        blocks.push_back({current_type, current_block});
    }

    for (auto& [type_name, block_text] : blocks) {
        auto layout = parse_block(block_text);
        layout.full_name = type_name;
        layouts[type_name] = std::move(layout);
    }

    // Resolve short type names in nested_type fields
    std::string root_pkg = root_type.substr(0, root_type.find('/'));
    for (auto& [_, layout] : layouts) {
        for (auto& field : layout.fields) {
            if (!field.is_primitive) {
                field.nested_type = resolve_type(field.nested_type, root_pkg, layouts);
            }
        }
    }

    return layouts;
}

}  // namespace baglab_mcap
