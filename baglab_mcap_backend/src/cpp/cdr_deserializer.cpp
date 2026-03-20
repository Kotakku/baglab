#include "cdr_deserializer.hpp"

#include <algorithm>
#include <cstring>
#include <stdexcept>

#include <pybind11/stl.h>

namespace baglab_mcap {

// ---------------------------------------------------------------------------
// CdrReader
// ---------------------------------------------------------------------------

CdrReader::CdrReader(const uint8_t* data, size_t size)
    : data_(data + 4), size_(size - 4)
{
    if (size < 4) {
        throw std::runtime_error("CDR data too short for encapsulation header");
    }
    // CDR encapsulation header: 2 bytes identifier + 2 bytes options
    //   XCDR1: 0x0000 (CDR_BE), 0x0001 (CDR_LE)
    //   XCDR2: 0x0006..0x000b (not yet supported)
    uint16_t encapsulation_id = (static_cast<uint16_t>(data[0]) << 8) | data[1];
    switch (encapsulation_id) {
        case 0x0000:  // CDR_BE (XCDR1, big-endian)
            little_endian_ = false;
            break;
        case 0x0001:  // CDR_LE (XCDR1, little-endian)
            little_endian_ = true;
            break;
        default:
        {
            char hex[8];
            std::snprintf(hex, sizeof(hex), "0x%04x", encapsulation_id);
            throw std::runtime_error(
                std::string("Unsupported CDR encapsulation ID: ") + hex +
                ". XCDR2 (0x0006-0x000b) is not supported. "
                "This may indicate a bag recorded with a newer ROS 2 version.");
        }
    }
    pos_ = 0;
}

void CdrReader::align(size_t n) {
    if (n > 1) {
        size_t rem = pos_ % n;
        if (rem != 0) {
            pos_ += n - rem;
        }
    }
    if (pos_ > size_) {
        throw std::runtime_error("CDR alignment past end of buffer");
    }
}

void CdrReader::skip(size_t n) {
    pos_ += n;
    if (pos_ > size_) {
        throw std::runtime_error("CDR skip past end of buffer");
    }
}

bool CdrReader::read_bool() {
    if (pos_ >= size_) throw std::runtime_error("CDR read past end");
    return data_[pos_++] != 0;
}

int8_t CdrReader::read_int8() {
    if (pos_ >= size_) throw std::runtime_error("CDR read past end");
    return static_cast<int8_t>(data_[pos_++]);
}

uint8_t CdrReader::read_uint8() {
    if (pos_ >= size_) throw std::runtime_error("CDR read past end");
    return data_[pos_++];
}

int16_t CdrReader::read_int16() {
    align(2);
    int16_t val;
    std::memcpy(&val, data_ + pos_, 2);
    pos_ += 2;
    return val;
}

uint16_t CdrReader::read_uint16() {
    align(2);
    uint16_t val;
    std::memcpy(&val, data_ + pos_, 2);
    pos_ += 2;
    return val;
}

int32_t CdrReader::read_int32() {
    align(4);
    int32_t val;
    std::memcpy(&val, data_ + pos_, 4);
    pos_ += 4;
    return val;
}

uint32_t CdrReader::read_uint32() {
    align(4);
    uint32_t val;
    std::memcpy(&val, data_ + pos_, 4);
    pos_ += 4;
    return val;
}

int64_t CdrReader::read_int64() {
    align(8);
    int64_t val;
    std::memcpy(&val, data_ + pos_, 8);
    pos_ += 8;
    return val;
}

uint64_t CdrReader::read_uint64() {
    align(8);
    uint64_t val;
    std::memcpy(&val, data_ + pos_, 8);
    pos_ += 8;
    return val;
}

float CdrReader::read_float32() {
    align(4);
    float val;
    std::memcpy(&val, data_ + pos_, 4);
    pos_ += 4;
    return val;
}

double CdrReader::read_float64() {
    align(8);
    double val;
    std::memcpy(&val, data_ + pos_, 8);
    pos_ += 8;
    return val;
}

std::string CdrReader::read_string() {
    uint32_t len = read_uint32();  // includes null terminator
    if (len == 0) return "";
    if (pos_ + len > size_) {
        throw std::runtime_error("CDR string read past end of buffer");
    }
    std::string result(reinterpret_cast<const char*>(data_ + pos_), len - 1);
    pos_ += len;
    return result;
}

uint32_t CdrReader::read_sequence_length() {
    return read_uint32();
}

// ---------------------------------------------------------------------------
// Scalar read/skip helpers
// ---------------------------------------------------------------------------

static void read_scalar_into(CdrReader& reader, PrimitiveType type, Column& col) {
    switch (type) {
        case PrimitiveType::BOOL:    col.bools.push_back(reader.read_bool()); break;
        case PrimitiveType::INT8:    col.i8.push_back(reader.read_int8()); break;
        case PrimitiveType::UINT8:   col.u8.push_back(reader.read_uint8()); break;
        case PrimitiveType::INT16:   col.i16.push_back(reader.read_int16()); break;
        case PrimitiveType::UINT16:  col.u16.push_back(reader.read_uint16()); break;
        case PrimitiveType::INT32:   col.i32.push_back(reader.read_int32()); break;
        case PrimitiveType::UINT32:  col.u32.push_back(reader.read_uint32()); break;
        case PrimitiveType::INT64:   col.i64.push_back(reader.read_int64()); break;
        case PrimitiveType::UINT64:  col.u64.push_back(reader.read_uint64()); break;
        case PrimitiveType::FLOAT32: col.f32.push_back(reader.read_float32()); break;
        case PrimitiveType::FLOAT64: col.f64.push_back(reader.read_float64()); break;
        case PrimitiveType::STRING:  col.pylist.append(reader.read_string()); break;
    }
}

static void skip_scalar(CdrReader& reader, PrimitiveType type) {
    if (type == PrimitiveType::STRING) {
        reader.read_string();  // must parse to skip
    } else {
        size_t sz = primitive_size(type);
        reader.align(primitive_alignment(type));
        reader.skip(sz);
    }
}

static py::object read_scalar_element(CdrReader& reader, PrimitiveType type) {
    switch (type) {
        case PrimitiveType::BOOL:    return py::cast(reader.read_bool());
        case PrimitiveType::INT8:    return py::cast(reader.read_int8());
        case PrimitiveType::UINT8:   return py::cast(reader.read_uint8());
        case PrimitiveType::INT16:   return py::cast(reader.read_int16());
        case PrimitiveType::UINT16:  return py::cast(reader.read_uint16());
        case PrimitiveType::INT32:   return py::cast(reader.read_int32());
        case PrimitiveType::UINT32:  return py::cast(reader.read_uint32());
        case PrimitiveType::INT64:   return py::cast(reader.read_int64());
        case PrimitiveType::UINT64:  return py::cast(reader.read_uint64());
        case PrimitiveType::FLOAT32: return py::cast(reader.read_float32());
        case PrimitiveType::FLOAT64: return py::cast(reader.read_float64());
        case PrimitiveType::STRING:  return py::cast(reader.read_string());
    }
    return py::none();
}

// ---------------------------------------------------------------------------
// Extraction plan builder
// ---------------------------------------------------------------------------

using Step = ExtractionPlan::Step;
using Action = Step::Action;

static void build_steps(
    const std::string& type_name,
    const std::string& prefix,
    const std::map<std::string, MessageLayout>& layouts,
    const std::set<std::string>& targets,
    bool extract_all,
    std::vector<Step>& steps,
    std::vector<std::string>& col_names,
    std::vector<PrimitiveType>& col_types,
    std::vector<bool>& col_is_list)
{
    auto it = layouts.find(type_name);
    if (it == layouts.end()) {
        throw std::runtime_error("Unknown message type: " + type_name);
    }
    const auto& layout = it->second;

    for (const auto& field : layout.fields) {
        std::string dot_path = prefix.empty() ? field.name : prefix + "." + field.name;

        if (field.is_primitive) {
            bool want = extract_all || targets.count(dot_path);

            if (field.is_array) {
                Step step;
                step.type = field.type;
                step.array_size = field.array_size;
                if (want) {
                    step.action = Action::READ_ARRAY;
                    step.column_index = col_names.size();
                    col_names.push_back(dot_path);
                    col_types.push_back(field.type);
                    col_is_list.push_back(true);
                } else {
                    step.action = Action::SKIP_ARRAY;
                }
                steps.push_back(std::move(step));
            } else if (field.is_sequence) {
                Step step;
                step.type = field.type;
                if (want) {
                    step.action = Action::READ_SEQUENCE;
                    step.column_index = col_names.size();
                    col_names.push_back(dot_path);
                    col_types.push_back(field.type);
                    col_is_list.push_back(true);
                } else {
                    step.action = Action::SKIP_SEQUENCE;
                }
                steps.push_back(std::move(step));
            } else if (field.type == PrimitiveType::STRING) {
                Step step;
                step.type = PrimitiveType::STRING;
                if (want) {
                    step.action = Action::READ_STRING;
                    step.column_index = col_names.size();
                    col_names.push_back(dot_path);
                    col_types.push_back(PrimitiveType::STRING);
                    col_is_list.push_back(false);
                } else {
                    step.action = Action::SKIP_STRING;
                }
                steps.push_back(std::move(step));
            } else {
                Step step;
                step.type = field.type;
                if (want) {
                    step.action = Action::READ_SCALAR;
                    step.column_index = col_names.size();
                    col_names.push_back(dot_path);
                    col_types.push_back(field.type);
                    col_is_list.push_back(false);
                } else {
                    step.action = Action::SKIP_SCALAR;
                }
                steps.push_back(std::move(step));
            }
        } else {
            // Nested message (non-array)
            if (field.is_array || field.is_sequence) {
                // Array/sequence of messages — store as pylist
                bool want = extract_all || targets.count(dot_path);
                Step step;
                step.action = want ? Action::READ_SEQUENCE : Action::SKIP_SEQUENCE;
                step.type = PrimitiveType::UINT8;  // placeholder
                step.array_size = field.is_array ? field.array_size : 0;
                if (want) {
                    step.column_index = col_names.size();
                    col_names.push_back(dot_path);
                    col_types.push_back(PrimitiveType::UINT8);
                    col_is_list.push_back(true);
                }
                // Build sub-steps for iterating through nested message elements.
                if (want) {
                    // READ: build sub-steps that extract all fields so we can
                    // deserialize each element into a Python dict.
                    build_steps(field.nested_type, "", layouts, {}, true,
                               step.sub_steps, step.sub_column_names,
                               step.sub_column_types, step.sub_column_is_list);
                } else {
                    // SKIP: sub-steps only advance the CDR cursor.
                    std::vector<std::string> skip_names;
                    std::vector<PrimitiveType> skip_types;
                    std::vector<bool> skip_is_list;
                    build_steps(field.nested_type, "", layouts, {}, false,
                               step.sub_steps, skip_names, skip_types, skip_is_list);
                }
                steps.push_back(std::move(step));
            } else {
                // Non-array nested message → flatten into parent
                Step step;
                step.action = Action::ENTER_NESTED;
                build_steps(field.nested_type, dot_path, layouts, targets, extract_all,
                           step.sub_steps, col_names, col_types, col_is_list);
                steps.push_back(std::move(step));
            }
        }
    }
}

ExtractionPlan build_extraction_plan(
    const std::string& root_type,
    const std::map<std::string, MessageLayout>& layouts,
    const std::vector<std::string>& target_fields)
{
    ExtractionPlan plan;
    std::set<std::string> targets(target_fields.begin(), target_fields.end());
    bool extract_all = targets.empty();

    build_steps(root_type, "", layouts, targets, extract_all,
                plan.steps, plan.column_names, plan.column_types, plan.column_is_list);

    return plan;
}

// ---------------------------------------------------------------------------
// Plan execution
// ---------------------------------------------------------------------------

void execute_plan(
    CdrReader& reader,
    const std::vector<Step>& steps,
    std::vector<Column>& columns)
{
    for (const auto& step : steps) {
        switch (step.action) {
            case Action::READ_SCALAR:
                read_scalar_into(reader, step.type, columns[step.column_index]);
                break;

            case Action::SKIP_SCALAR:
                skip_scalar(reader, step.type);
                break;

            case Action::READ_STRING:
                columns[step.column_index].pylist.append(reader.read_string());
                break;

            case Action::SKIP_STRING:
                reader.read_string();
                break;

            case Action::READ_ARRAY: {
                py::list arr;
                reader.align(primitive_alignment(step.type));
                for (size_t i = 0; i < step.array_size; ++i) {
                    arr.append(read_scalar_element(reader, step.type));
                }
                columns[step.column_index].pylist.append(std::move(arr));
                break;
            }

            case Action::SKIP_ARRAY: {
                reader.align(primitive_alignment(step.type));
                size_t sz = primitive_size(step.type);
                reader.skip(sz * step.array_size);
                break;
            }

            case Action::READ_SEQUENCE: {
                uint32_t count = reader.read_sequence_length();
                if (step.sub_steps.empty()) {
                    // Primitive sequence
                    py::list seq;
                    for (uint32_t i = 0; i < count; ++i) {
                        seq.append(read_scalar_element(reader, step.type));
                    }
                    columns[step.column_index].pylist.append(std::move(seq));
                } else if (step.sub_column_names.empty()) {
                    // Message sequence with SKIP sub-steps (no column metadata)
                    // — just advance the CDR cursor and store None.
                    py::list seq;
                    for (uint32_t i = 0; i < count; ++i) {
                        execute_plan(reader, step.sub_steps, columns);
                        seq.append(py::none());
                    }
                    columns[step.column_index].pylist.append(std::move(seq));
                } else {
                    // Message sequence — deserialize each element into a dict.
                    py::list seq;
                    size_t ncols = step.sub_column_names.size();
                    for (uint32_t i = 0; i < count; ++i) {
                        std::vector<Column> temp_cols(ncols);
                        execute_plan(reader, step.sub_steps, temp_cols);
                        py::dict d;
                        for (size_t j = 0; j < ncols; ++j) {
                            const auto& name = step.sub_column_names[j];
                            auto stype = step.sub_column_types[j];
                            bool is_list = step.sub_column_is_list[j];
                            py::object val;
                            if (is_list || stype == PrimitiveType::STRING) {
                                if (temp_cols[j].pylist.size() > 0) {
                                    val = py::object(temp_cols[j].pylist[0]);
                                } else {
                                    val = py::none();
                                }
                            } else {
                                switch (stype) {
                                    case PrimitiveType::BOOL:
                                        val = temp_cols[j].bools.empty() ? py::none() : py::cast(static_cast<bool>(temp_cols[j].bools[0])); break;
                                    case PrimitiveType::INT8:
                                        val = temp_cols[j].i8.empty() ? py::none() : py::cast(temp_cols[j].i8[0]); break;
                                    case PrimitiveType::UINT8:
                                        val = temp_cols[j].u8.empty() ? py::none() : py::cast(temp_cols[j].u8[0]); break;
                                    case PrimitiveType::INT16:
                                        val = temp_cols[j].i16.empty() ? py::none() : py::cast(temp_cols[j].i16[0]); break;
                                    case PrimitiveType::UINT16:
                                        val = temp_cols[j].u16.empty() ? py::none() : py::cast(temp_cols[j].u16[0]); break;
                                    case PrimitiveType::INT32:
                                        val = temp_cols[j].i32.empty() ? py::none() : py::cast(temp_cols[j].i32[0]); break;
                                    case PrimitiveType::UINT32:
                                        val = temp_cols[j].u32.empty() ? py::none() : py::cast(temp_cols[j].u32[0]); break;
                                    case PrimitiveType::INT64:
                                        val = temp_cols[j].i64.empty() ? py::none() : py::cast(temp_cols[j].i64[0]); break;
                                    case PrimitiveType::UINT64:
                                        val = temp_cols[j].u64.empty() ? py::none() : py::cast(temp_cols[j].u64[0]); break;
                                    case PrimitiveType::FLOAT32:
                                        val = temp_cols[j].f32.empty() ? py::none() : py::cast(temp_cols[j].f32[0]); break;
                                    case PrimitiveType::FLOAT64:
                                        val = temp_cols[j].f64.empty() ? py::none() : py::cast(temp_cols[j].f64[0]); break;
                                    default:
                                        val = py::none(); break;
                                }
                            }
                            d[py::cast(name)] = val;
                        }
                        seq.append(std::move(d));
                    }
                    columns[step.column_index].pylist.append(std::move(seq));
                }
                break;
            }

            case Action::SKIP_SEQUENCE: {
                uint32_t count = reader.read_sequence_length();
                if (step.sub_steps.empty()) {
                    // Primitive sequence
                    size_t sz = primitive_size(step.type);
                    if (sz > 0) {
                        reader.align(primitive_alignment(step.type));
                        reader.skip(sz * count);
                    } else {
                        // String sequence
                        for (uint32_t i = 0; i < count; ++i) {
                            reader.read_string();
                        }
                    }
                } else {
                    // Message sequence
                    for (uint32_t i = 0; i < count; ++i) {
                        execute_plan(reader, step.sub_steps, columns);
                    }
                }
                break;
            }

            case Action::ENTER_NESTED:
                execute_plan(reader, step.sub_steps, columns);
                break;
        }
    }
}

// ---------------------------------------------------------------------------
// Build final result
// ---------------------------------------------------------------------------

template <typename T>
static py::array_t<T> to_numpy(const std::vector<T>& vec) {
    // Must copy: the vector will be destroyed after this function returns.
    auto arr = py::array_t<T>(static_cast<py::ssize_t>(vec.size()));
    std::memcpy(arr.mutable_data(), vec.data(), vec.size() * sizeof(T));
    return arr;
}

ColumnarResult build_columnar_result(
    const std::vector<int64_t>& timestamps,
    const ExtractionPlan& plan,
    std::vector<Column>& columns)
{
    ColumnarResult result;
    result["__timestamps__"] = to_numpy(timestamps);

    for (size_t i = 0; i < plan.column_names.size(); ++i) {
        const auto& name = plan.column_names[i];
        auto& col = columns[i];
        auto type = plan.column_types[i];
        bool is_list = plan.column_is_list[i];

        if (is_list || type == PrimitiveType::STRING) {
            result[name] = col.pylist;
        } else {
            switch (type) {
                case PrimitiveType::FLOAT32: result[name] = to_numpy(col.f32); break;
                case PrimitiveType::FLOAT64: result[name] = to_numpy(col.f64); break;
                case PrimitiveType::INT8:    result[name] = to_numpy(col.i8); break;
                case PrimitiveType::INT16:   result[name] = to_numpy(col.i16); break;
                case PrimitiveType::INT32:   result[name] = to_numpy(col.i32); break;
                case PrimitiveType::INT64:   result[name] = to_numpy(col.i64); break;
                case PrimitiveType::UINT8:   result[name] = to_numpy(col.u8); break;
                case PrimitiveType::UINT16:  result[name] = to_numpy(col.u16); break;
                case PrimitiveType::UINT32:  result[name] = to_numpy(col.u32); break;
                case PrimitiveType::UINT64:  result[name] = to_numpy(col.u64); break;
                case PrimitiveType::BOOL: {
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

}  // namespace baglab_mcap
