cmake_minimum_required(VERSION 3.22)
project(baglab_mcap_backend LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(pybind11 REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(LZ4 REQUIRED liblz4)
pkg_check_modules(ZSTD REQUIRED libzstd)

pybind11_add_module(_core
    src/cpp/bindings.cpp
    src/cpp/mcap_reader.cpp
    src/cpp/schema_parser.cpp
    src/cpp/cdr_deserializer.cpp
)

target_include_directories(_core PRIVATE
    src/cpp
    third_party/mcap_repo/cpp/mcap/include
)
target_include_directories(_core SYSTEM PRIVATE
    ${LZ4_INCLUDE_DIRS}
    ${ZSTD_INCLUDE_DIRS}
)
target_link_libraries(_core PRIVATE ${LZ4_LIBRARIES} ${ZSTD_LIBRARIES})

install(TARGETS _core LIBRARY DESTINATION baglab_mcap_backend)
