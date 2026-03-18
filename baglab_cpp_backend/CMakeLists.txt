cmake_minimum_required(VERSION 3.22)
project(baglab_cpp_backend LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---- Detect build mode: colcon (ament_cmake) or pip (scikit-build-core) ----
# SKBUILD_STATE is defined by scikit-build-core; when set, always use the pip path.
if(NOT DEFINED SKBUILD_STATE)
    find_package(ament_cmake QUIET)
endif()

if(ament_cmake_FOUND)
    # --- colcon build path ---
    find_package(ament_cmake_python REQUIRED)
    find_package(python_cmake_module REQUIRED)
    find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
    find_package(pybind11_vendor REQUIRED)
    find_package(pybind11 REQUIRED)
else()
    # --- pip install path (scikit-build-core) ---
    list(APPEND CMAKE_PREFIX_PATH $ENV{AMENT_PREFIX_PATH})
    find_package(pybind11 REQUIRED)
endif()

# ---- ROS 2 dependencies ----
find_package(rosbag2_cpp REQUIRED)
find_package(rosbag2_storage REQUIRED)
find_package(rcpputils REQUIRED)
find_package(rmw REQUIRED)
find_package(rcutils REQUIRED)
find_package(rosidl_runtime_c REQUIRED)
find_package(rosidl_runtime_cpp REQUIRED)
find_package(rosidl_typesupport_introspection_cpp REQUIRED)
find_package(rosidl_typesupport_cpp REQUIRED)

# ---- Build the extension module ----
pybind11_add_module(_core
    src/cpp/bindings.cpp
    src/cpp/reader.cpp
    src/cpp/deserializer.cpp
)

target_include_directories(_core PRIVATE src/cpp)

# Detect Jazzy+ API change: time_stamp → recv_timestamp
include(CheckCXXSourceCompiles)
set(CMAKE_REQUIRED_INCLUDES ${rosbag2_storage_INCLUDE_DIRS})
check_cxx_source_compiles("
#include <rosbag2_storage/serialized_bag_message.hpp>
int main() { rosbag2_storage::SerializedBagMessage m; (void)m.recv_timestamp; return 0; }
" HAS_RECV_TIMESTAMP)
if(HAS_RECV_TIMESTAMP)
    target_compile_definitions(_core PRIVATE BAGLAB_HAS_RECV_TIMESTAMP)
endif()

target_link_libraries(_core PRIVATE
    rosbag2_cpp::rosbag2_cpp
    rosbag2_storage::rosbag2_storage
    rcpputils::rcpputils
    rmw::rmw
    rcutils::rcutils
    rosidl_runtime_c::rosidl_runtime_c
    rosidl_runtime_cpp::rosidl_runtime_cpp
    rosidl_typesupport_introspection_cpp::rosidl_typesupport_introspection_cpp
    rosidl_typesupport_cpp::rosidl_typesupport_cpp
)

if(ament_cmake_FOUND)
    # colcon: install to ament Python site-packages
    ament_python_install_package(${PROJECT_NAME} PACKAGE_DIR src/baglab_cpp_backend)
    install(TARGETS _core LIBRARY DESTINATION "${PYTHON_INSTALL_DIR}/${PROJECT_NAME}")
    ament_package()
else()
    # pip: install alongside the Python package
    install(TARGETS _core LIBRARY DESTINATION baglab_cpp_backend)
endif()
