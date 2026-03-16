#!/bin/bash
# Usage: ./record.sh [output_dir] [duration]
#   output_dir: output bag directory name (default: test_bag)
#   duration:   recording duration in seconds (default: 5.0)

set -eu

OUTPUT_DIR="${1:-test_bag}"
DURATION="${2:-5.0}"

echo "Recording test data to '${OUTPUT_DIR}' for ${DURATION}s ..."

ros2 bag record -s mcap -o "${OUTPUT_DIR}" /test/joint_state /test/twist &
RECORD_PID=$!
sleep 0.5

ros2 run test_rosbag test_publisher --ros-args -p duration:="${DURATION}"

kill -INT "${RECORD_PID}" 2>/dev/null
wait "${RECORD_PID}" 2>/dev/null || true

echo "Done. Saved to ${OUTPUT_DIR}/"
