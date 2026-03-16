"""Launch test publisher and record to mcap."""

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    EmitEvent,
    ExecuteProcess,
    RegisterEventHandler,
)
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    output_arg = DeclareLaunchArgument(
        "output", default_value="test_bag", description="Output bag directory name"
    )
    duration_arg = DeclareLaunchArgument(
        "duration", default_value="5.0", description="Duration in seconds"
    )

    test_publisher = Node(
        package="test_rosbag",
        executable="test_publisher",
        parameters=[{"duration": LaunchConfiguration("duration")}],
    )

    bag_record = ExecuteProcess(
        cmd=[
            "ros2",
            "bag",
            "record",
            "-s", "mcap",
            "-o", LaunchConfiguration("output"),
            "/test/joint_state",
            "/test/twist",
        ],
    )

    shutdown_on_exit = RegisterEventHandler(
        OnProcessExit(
            target_action=test_publisher,
            on_exit=[EmitEvent(event=Shutdown())],
        )
    )

    return LaunchDescription([
        output_arg,
        duration_arg,
        bag_record,
        test_publisher,
        shutdown_on_exit,
    ])
