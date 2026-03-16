#!/bin/bash
# Generate test bag for baglab tests.
# Requires: test_rosbag_pkg built and sourced in a ROS2 workspace.
#
# Usage: ./generate_test_bag.sh [output_dir]

set -eu

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUTPUT_DIR="${1:-${SCRIPT_DIR}/test_bag}"

if [ -d "${OUTPUT_DIR}" ]; then
    echo "Test bag already exists at ${OUTPUT_DIR}, skipping."
    exit 0
fi

echo "Generating test bag at ${OUTPUT_DIR} ..."

ros2 bag record -s mcap -o "${OUTPUT_DIR}" /test/joint_state /test/twist &
RECORD_PID=$!
sleep 0.5

ros2 run test_rosbag test_publisher --ros-args -p duration:=5.0

kill -INT "${RECORD_PID}" 2>/dev/null
wait "${RECORD_PID}" 2>/dev/null || true

echo "Done. Saved to ${OUTPUT_DIR}/"
