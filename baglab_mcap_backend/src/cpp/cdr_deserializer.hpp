#pragma once

#include "schema_parser.hpp"

#include <cstdint>
#include <map>
#include <set>
#include <string>
#include <variant>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

namespace baglab_mcap {

/// Columnar result: field name -> numpy array or Python list.
using ColumnarResult = std::map<std::string, py::object>;

/// Column storage for accumulating field values across messages.
struct Column {
    PrimitiveType type{};
    bool is_list = false;  // array/sequence → stored as Python list of lists
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

/// Pre-reserve column capacity for expected message count.
inline void reserve_column(Column& col, uint64_t n) {
    switch (col.type) {
        case PrimitiveType::FLOAT32: col.f32.reserve(n); break;
        case PrimitiveType::FLOAT64: col.f64.reserve(n); break;
        case PrimitiveType::INT8:    col.i8.reserve(n); break;
        case PrimitiveType::INT16:   col.i16.reserve(n); break;
        case PrimitiveType::INT32:   col.i32.reserve(n); break;
        case PrimitiveType::INT64:   col.i64.reserve(n); break;
        case PrimitiveType::UINT8:   col.u8.reserve(n); break;
        case PrimitiveType::UINT16:  col.u16.reserve(n); break;
        case PrimitiveType::UINT32:  col.u32.reserve(n); break;
        case PrimitiveType::UINT64:  col.u64.reserve(n); break;
        case PrimitiveType::BOOL:    col.bools.reserve(n); break;
        default: break;
    }
}

/// CDR binary reader with alignment tracking.
class CdrReader {
public:
    CdrReader(const uint8_t* data, size_t size);

    bool is_little_endian() const { return little_endian_; }
    size_t offset() const { return pos_; }
    size_t remaining() const { return size_ - pos_; }

    void align(size_t n);

    bool     read_bool();
    int8_t   read_int8();
    uint8_t  read_uint8();
    int16_t  read_int16();
    uint16_t read_uint16();
    int32_t  read_int32();
    uint32_t read_uint32();
    int64_t  read_int64();
    uint64_t read_uint64();
    float    read_float32();
    double   read_float64();
    std::string read_string();

    uint32_t read_sequence_length();

    /// Skip N bytes (for skipping fields we don't need).
    void skip(size_t n);

private:
    const uint8_t* data_;
    size_t size_;
    size_t pos_ = 0;
    bool little_endian_ = true;
};

/// Pre-compiled plan for extracting fields from CDR messages.
struct FieldPlan {
    std::string dot_path;    // e.g. "twist.linear.x"
    size_t column_index;     // index into the columns vector
};

/// Build extraction plan: identifies which fields to extract and in what order.
/// The plan accounts for all fields in the message (must walk entire CDR stream)
/// but only stores target fields.
struct ExtractionPlan {
    struct Step {
        enum class Action {
            READ_SCALAR,      // read and store in column
            SKIP_SCALAR,      // read and discard
            READ_STRING,      // read string and store
            SKIP_STRING,      // read string and discard
            READ_ARRAY,       // read fixed array and store as pylist
            SKIP_ARRAY,       // skip fixed array
            READ_SEQUENCE,    // read dynamic sequence and store as pylist
            SKIP_SEQUENCE,    // skip dynamic sequence
            ENTER_NESTED,     // recurse into nested message
        };

        Action action;
        PrimitiveType type{};
        size_t column_index = 0;
        size_t array_size = 0;
        // For nested messages: pointer to sub-plan
        std::vector<Step> sub_steps;
        // For message sequences (READ_SEQUENCE with sub_steps):
        // metadata describing the columns that sub_steps produce.
        std::vector<std::string> sub_column_names;
        std::vector<PrimitiveType> sub_column_types;
        std::vector<bool> sub_column_is_list;
    };

    std::vector<Step> steps;
    std::vector<std::string> column_names;
    std::vector<PrimitiveType> column_types;
    std::vector<bool> column_is_list;
};

/// Build an extraction plan from the message layout.
/// If target_fields is empty, extract all fields.
ExtractionPlan build_extraction_plan(
    const std::string& root_type,
    const std::map<std::string, MessageLayout>& layouts,
    const std::vector<std::string>& target_fields);

/// Execute the extraction plan on a single CDR message.
void execute_plan(
    CdrReader& reader,
    const std::vector<ExtractionPlan::Step>& steps,
    std::vector<Column>& columns);

/// Build the final ColumnarResult from accumulated columns + timestamps.
ColumnarResult build_columnar_result(
    const std::vector<int64_t>& timestamps,
    const ExtractionPlan& plan,
    std::vector<Column>& columns);

}  // namespace baglab_mcap
