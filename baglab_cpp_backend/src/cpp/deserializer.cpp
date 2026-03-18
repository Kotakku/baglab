#include "deserializer.hpp"
#include "reader.hpp"

#include <algorithm>
#include <cstring>
#include <memory>
#include <stdexcept>

#include <rosidl_typesupport_introspection_cpp/field_types.hpp>
#include <rosidl_typesupport_introspection_cpp/message_introspection.hpp>
#include <rosidl_typesupport_introspection_cpp/identifier.hpp>
#include <rosidl_typesupport_cpp/identifier.hpp>
#include <rcpputils/find_library.hpp>
#include <rcpputils/shared_library.hpp>
#include <rmw/rmw.h>

#include <pybind11/stl.h>

namespace baglab_cpp_backend {

using MembersPtr = const rosidl_typesupport_introspection_cpp::MessageMembers*;
using MemberPtr = const rosidl_typesupport_introspection_cpp::MessageMember*;

// ---------------------------------------------------------------------------
// Typesupport loading
// ---------------------------------------------------------------------------

static std::pair<std::string, std::string> parse_msg_type(const std::string& msg_type) {
    auto first_slash = msg_type.find('/');
    auto last_slash = msg_type.rfind('/');
    if (first_slash == std::string::npos || last_slash == first_slash) {
        throw std::runtime_error("Invalid message type format: " + msg_type);
    }
    return {msg_type.substr(0, first_slash), msg_type.substr(last_slash + 1)};
}

static MembersPtr load_introspection(const std::string& msg_type) {
    auto [pkg, name] = parse_msg_type(msg_type);

    std::string lib_name = pkg + "__rosidl_typesupport_introspection_cpp";
    auto lib_path = rcpputils::find_library_path(lib_name);
    if (lib_path.empty()) {
        throw std::runtime_error(
            "Cannot find typesupport library for " + msg_type +
            " (looked for " + lib_name + ")");
    }

    auto lib = std::make_shared<rcpputils::SharedLibrary>(lib_path);

    std::string symbol =
        "rosidl_typesupport_introspection_cpp__get_message_type_support_handle__" +
        pkg + "__msg__" + name;

    if (!lib->has_symbol(symbol)) {
        throw std::runtime_error("Symbol not found: " + symbol);
    }

    using GetTSFunc = const rosidl_message_type_support_t* (*)();
    auto func = reinterpret_cast<GetTSFunc>(lib->get_symbol(symbol));
    auto ts = func();

    static std::vector<std::shared_ptr<rcpputils::SharedLibrary>> kept_libs;
    kept_libs.push_back(lib);

    return static_cast<MembersPtr>(ts->data);
}

static const rosidl_message_type_support_t* load_typesupport(const std::string& msg_type) {
    auto [pkg, name] = parse_msg_type(msg_type);

    std::string lib_name = pkg + "__rosidl_typesupport_cpp";
    auto lib_path = rcpputils::find_library_path(lib_name);
    if (lib_path.empty()) {
        throw std::runtime_error("Cannot find typesupport_cpp library for " + msg_type);
    }

    auto lib = std::make_shared<rcpputils::SharedLibrary>(lib_path);

    std::string symbol =
        "rosidl_typesupport_cpp__get_message_type_support_handle__" +
        pkg + "__msg__" + name;

    if (!lib->has_symbol(symbol)) {
        throw std::runtime_error("Symbol not found: " + symbol);
    }

    using GetTSFunc = const rosidl_message_type_support_t* (*)();
    auto func = reinterpret_cast<GetTSFunc>(lib->get_symbol(symbol));

    static std::vector<std::shared_ptr<rcpputils::SharedLibrary>> kept_libs;
    kept_libs.push_back(lib);

    return func();
}

// ---------------------------------------------------------------------------
// Field access plan
// ---------------------------------------------------------------------------

namespace field_type = rosidl_typesupport_introspection_cpp;

/// Describes how to access one leaf field from the deserialized message buffer.
struct FieldAccessor {
    std::string name;               // dot-notation name
    std::vector<size_t> offsets;    // chain of offsets from root to leaf
    uint8_t type_id;                // rosidl field type of the leaf
    bool is_array;                  // true if SEQUENCE or ARRAY
    size_t array_size;              // >0 for fixed-size arrays, 0 for sequences
    // Pointer to the MessageMember for this field (needed for size_function/get_const_function)
    MemberPtr member = nullptr;
};

/// Recursively collect field accessors.
static void collect_fields(
    MembersPtr members,
    const std::string& prefix,
    std::vector<size_t> offset_chain,
    const std::vector<std::string>& filter,  // empty = all
    std::vector<FieldAccessor>& out)
{
    for (uint32_t i = 0; i < members->member_count_; ++i) {
        const auto& m = members->members_[i];
        std::string path = prefix.empty() ? m.name_ : prefix + "." + m.name_;
        auto chain = offset_chain;
        chain.push_back(m.offset_);

        bool is_sequence = m.is_array_;

        if (!is_sequence && m.type_id_ == field_type::ROS_TYPE_MESSAGE && m.members_) {
            auto nested = static_cast<MembersPtr>(m.members_->data);
            collect_fields(nested, path, chain, filter, out);
        } else {
            if (!filter.empty()) {
                bool found = false;
                for (const auto& f : filter) {
                    if (f == path) { found = true; break; }
                }
                if (!found) continue;
            }

            FieldAccessor acc;
            acc.name = path;
            acc.offsets = chain;
            acc.type_id = m.type_id_;
            acc.is_array = is_sequence;
            acc.array_size = m.is_array_ ? m.array_size_ : 0;
            acc.member = &m;
            out.push_back(std::move(acc));
        }
    }
}

// ---------------------------------------------------------------------------
// Field value extraction
// ---------------------------------------------------------------------------

static const uint8_t* resolve_ptr(const uint8_t* base, const std::vector<size_t>& offsets) {
    const uint8_t* ptr = base;
    for (auto off : offsets) {
        ptr = ptr + off;
    }
    return ptr;
}

template <typename T>
static py::array_t<T> build_numpy(const std::vector<T>& vec) {
    return py::array_t<T>(
        {static_cast<py::ssize_t>(vec.size())},
        vec.data());
}

template <typename T>
static void append_scalar(const uint8_t* ptr, std::vector<T>& col) {
    T val;
    std::memcpy(&val, ptr, sizeof(T));
    col.push_back(val);
}

/// Extract a scalar value from a void* element pointer.
static py::object extract_element_value(const void* elem_ptr, uint8_t type_id) {
    switch (type_id) {
        case field_type::ROS_TYPE_FLOAT:
            return py::cast(*static_cast<const float*>(elem_ptr));
        case field_type::ROS_TYPE_DOUBLE:
            return py::cast(*static_cast<const double*>(elem_ptr));
        case field_type::ROS_TYPE_INT8:
            return py::cast(*static_cast<const int8_t*>(elem_ptr));
        case field_type::ROS_TYPE_INT16:
            return py::cast(*static_cast<const int16_t*>(elem_ptr));
        case field_type::ROS_TYPE_INT32:
            return py::cast(*static_cast<const int32_t*>(elem_ptr));
        case field_type::ROS_TYPE_INT64:
            return py::cast(*static_cast<const int64_t*>(elem_ptr));
        case field_type::ROS_TYPE_UINT8:
            return py::cast(*static_cast<const uint8_t*>(elem_ptr));
        case field_type::ROS_TYPE_UINT16:
            return py::cast(*static_cast<const uint16_t*>(elem_ptr));
        case field_type::ROS_TYPE_UINT32:
            return py::cast(*static_cast<const uint32_t*>(elem_ptr));
        case field_type::ROS_TYPE_UINT64:
            return py::cast(*static_cast<const uint64_t*>(elem_ptr));
        case field_type::ROS_TYPE_BOOLEAN:
            return py::cast(*static_cast<const bool*>(elem_ptr));
        case field_type::ROS_TYPE_STRING:
            return py::cast(*static_cast<const std::string*>(elem_ptr));
        default:
            return py::none();
    }
}

/// Extract an array/sequence field using rosidl introspection accessors.
static py::list extract_sequence(const void* field_ptr, const FieldAccessor& acc) {
    py::list result;
    size_t count;

    if (acc.array_size > 0) {
        // Fixed-size array
        count = acc.array_size;
    } else {
        // Dynamic sequence: use size_function
        count = acc.member->size_function(field_ptr);
    }

    for (size_t i = 0; i < count; ++i) {
        const void* elem = acc.member->get_const_function(field_ptr, i);
        result.append(extract_element_value(elem, acc.type_id));
    }
    return result;
}

// ---------------------------------------------------------------------------
// Main deserialize function
// ---------------------------------------------------------------------------

ColumnarResult deserialize_to_columns(
    const std::string& msg_type,
    const std::vector<RawMessage>& messages,
    const std::vector<std::string>& field_paths)
{
    if (messages.empty()) {
        ColumnarResult result;
        result["__timestamps__"] = py::array_t<int64_t>(0);
        return result;
    }

    auto introspection = load_introspection(msg_type);
    auto ts_handle = load_typesupport(msg_type);

    std::vector<FieldAccessor> accessors;
    collect_fields(introspection, "", {}, field_paths, accessors);

    if (accessors.empty() && !field_paths.empty()) {
        throw std::runtime_error("No matching fields found for the requested paths");
    }

    size_t n = messages.size();

    struct Column {
        uint8_t type_id;
        bool is_array;
        std::vector<float> f32;
        std::vector<double> f64;
        std::vector<int8_t> i8;
        std::vector<int16_t> i16;
        std::vector<int32_t> i32;
        std::vector<int64_t> i64;
        std::vector<uint8_t> u8;
        std::vector<uint16_t> u16;
        std::vector<uint32_t> u32;
        std::vector<uint64_t> u64;
        std::vector<bool> bools;
        py::list pylist;
    };

    std::vector<Column> columns(accessors.size());
    for (size_t i = 0; i < accessors.size(); ++i) {
        columns[i].type_id = accessors[i].type_id;
        columns[i].is_array = accessors[i].is_array;
    }

    std::vector<int64_t> timestamps;
    timestamps.reserve(n);

    size_t msg_size = introspection->size_of_;
    std::vector<uint8_t> msg_buf(msg_size);

    for (const auto& raw : messages) {
        timestamps.push_back(raw.timestamp_ns);

        introspection->init_function(msg_buf.data(),
            rosidl_runtime_cpp::MessageInitialization::ALL);

        rcutils_uint8_array_t serialized;
        serialized.buffer = const_cast<uint8_t*>(raw.data.data());
        serialized.buffer_length = raw.data.size();
        serialized.buffer_capacity = raw.data.size();
        serialized.allocator = rcutils_get_default_allocator();

        auto ret = rmw_deserialize(&serialized, ts_handle, msg_buf.data());
        if (ret != RMW_RET_OK) {
            introspection->fini_function(msg_buf.data());
            throw std::runtime_error("rmw_deserialize failed");
        }

        for (size_t i = 0; i < accessors.size(); ++i) {
            const auto& acc = accessors[i];
            auto& col = columns[i];
            const uint8_t* ptr = resolve_ptr(msg_buf.data(), acc.offsets);

            if (acc.is_array) {
                col.pylist.append(extract_sequence(ptr, acc));
                continue;
            }

            switch (acc.type_id) {
                case field_type::ROS_TYPE_FLOAT:
                    append_scalar<float>(ptr, col.f32); break;
                case field_type::ROS_TYPE_DOUBLE:
                    append_scalar<double>(ptr, col.f64); break;
                case field_type::ROS_TYPE_INT8:
                    append_scalar<int8_t>(ptr, col.i8); break;
                case field_type::ROS_TYPE_INT16:
                    append_scalar<int16_t>(ptr, col.i16); break;
                case field_type::ROS_TYPE_INT32:
                    append_scalar<int32_t>(ptr, col.i32); break;
                case field_type::ROS_TYPE_INT64:
                    append_scalar<int64_t>(ptr, col.i64); break;
                case field_type::ROS_TYPE_UINT8:
                    append_scalar<uint8_t>(ptr, col.u8); break;
                case field_type::ROS_TYPE_UINT16:
                    append_scalar<uint16_t>(ptr, col.u16); break;
                case field_type::ROS_TYPE_UINT32:
                    append_scalar<uint32_t>(ptr, col.u32); break;
                case field_type::ROS_TYPE_UINT64:
                    append_scalar<uint64_t>(ptr, col.u64); break;
                case field_type::ROS_TYPE_BOOLEAN:
                    col.bools.push_back(*reinterpret_cast<const bool*>(ptr)); break;
                case field_type::ROS_TYPE_STRING: {
                    auto* s = reinterpret_cast<const std::string*>(ptr);
                    col.pylist.append(*s);
                    break;
                }
                default:
                    col.pylist.append(py::none());
                    break;
            }
        }

        introspection->fini_function(msg_buf.data());
    }

    // Build result dict
    ColumnarResult result;
    result["__timestamps__"] = build_numpy(timestamps);

    for (size_t i = 0; i < accessors.size(); ++i) {
        const auto& acc = accessors[i];
        auto& col = columns[i];
        const std::string& name = acc.name;

        if (acc.is_array || acc.type_id == field_type::ROS_TYPE_STRING) {
            result[name] = col.pylist;
        } else {
            switch (acc.type_id) {
                case field_type::ROS_TYPE_FLOAT:   result[name] = build_numpy(col.f32); break;
                case field_type::ROS_TYPE_DOUBLE:  result[name] = build_numpy(col.f64); break;
                case field_type::ROS_TYPE_INT8:    result[name] = build_numpy(col.i8); break;
                case field_type::ROS_TYPE_INT16:   result[name] = build_numpy(col.i16); break;
                case field_type::ROS_TYPE_INT32:   result[name] = build_numpy(col.i32); break;
                case field_type::ROS_TYPE_INT64:   result[name] = build_numpy(col.i64); break;
                case field_type::ROS_TYPE_UINT8:   result[name] = build_numpy(col.u8); break;
                case field_type::ROS_TYPE_UINT16:  result[name] = build_numpy(col.u16); break;
                case field_type::ROS_TYPE_UINT32:  result[name] = build_numpy(col.u32); break;
                case field_type::ROS_TYPE_UINT64:  result[name] = build_numpy(col.u64); break;
                case field_type::ROS_TYPE_BOOLEAN: {
                    auto arr = py::array_t<bool>(static_cast<py::ssize_t>(col.bools.size()));
                    auto buf = arr.mutable_unchecked<1>();
                    for (size_t j = 0; j < col.bools.size(); ++j) {
                        buf(j) = col.bools[j];
                    }
                    result[name] = arr;
                    break;
                }
                default:
                    result[name] = col.pylist;
                    break;
            }
        }
    }

    return result;
}

}  // namespace baglab_cpp_backend
