"""Publish test data for baglab development.

Generates:
- /test/joint_state (sensor_msgs/JointState)
    Step response of a second-order underdamped system with noise.
    position: actual angle, velocity: angular velocity, effort: target command.
- /test/twist (geometry_msgs/TwistStamped)
    Sinusoidal tracking with noise. For testing nested field expansion.
"""

import math
import random

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped
from sensor_msgs.msg import JointState


class TestPublisher(Node):

    def __init__(self):
        super().__init__("test_publisher")

        self.declare_parameter("duration", 5.0)
        self.declare_parameter("rate", 100.0)

        self._duration = self.get_parameter("duration").value
        rate = self.get_parameter("rate").value

        self._joint_pub = self.create_publisher(JointState, "/test/joint_state", 10)
        self._twist_pub = self.create_publisher(TwistStamped, "/test/twist", 10)

        self._start_time = self.get_clock().now()
        self._prev_position = 0.0
        self._prev_t = 0.0
        self._timer = self.create_timer(1.0 / rate, self._timer_callback)

        self.get_logger().info(
            f"Publishing test data for {self._duration}s at {rate}Hz"
        )

    def _step_response(self, t: float) -> float:
        """Second-order underdamped step response.

        Step occurs at t=1.0s. wn=10, zeta=0.3.
        """
        if t < 1.0:
            return 0.0
        ts = t - 1.0
        wn = 10.0
        zeta = 0.3
        wd = wn * math.sqrt(1.0 - zeta**2)
        phi = math.acos(zeta)
        return 1.0 - (math.exp(-zeta * wn * ts) / math.sqrt(1.0 - zeta**2)) * math.sin(
            wd * ts + phi
        )

    def _timer_callback(self):
        now = self.get_clock().now()
        t = (now - self._start_time).nanoseconds * 1e-9

        if t > self._duration:
            self.get_logger().info("Test data generation complete")
            raise SystemExit

        stamp = now.to_msg()

        # --- JointState: step response ---
        target = 0.0 if t < 1.0 else 1.0
        actual = self._step_response(t) + random.gauss(0.0, 0.005)
        dt = t - self._prev_t if self._prev_t > 0.0 else 0.01
        velocity = (actual - self._prev_position) / dt
        self._prev_position = actual
        self._prev_t = t

        joint_msg = JointState()
        joint_msg.header.stamp = stamp
        joint_msg.header.frame_id = "motor"
        joint_msg.name = ["motor_joint"]
        joint_msg.position = [actual]
        joint_msg.velocity = [velocity]
        joint_msg.effort = [target]
        self._joint_pub.publish(joint_msg)

        # --- TwistStamped: sinusoidal tracking ---
        target_vx = math.sin(2.0 * math.pi * 0.5 * t)
        actual_vx = target_vx + random.gauss(0.0, 0.02)
        target_wz = math.cos(2.0 * math.pi * 0.3 * t)
        actual_wz = target_wz + random.gauss(0.0, 0.02)

        twist_msg = TwistStamped()
        twist_msg.header.stamp = stamp
        twist_msg.header.frame_id = "base_link"
        twist_msg.twist.linear.x = actual_vx
        twist_msg.twist.linear.y = 0.0
        twist_msg.twist.linear.z = 0.0
        twist_msg.twist.angular.x = 0.0
        twist_msg.twist.angular.y = 0.0
        twist_msg.twist.angular.z = actual_wz
        self._twist_pub.publish(twist_msg)


def main(args=None):
    rclpy.init(args=args)
    node = TestPublisher()
    try:
        rclpy.spin(node)
    except SystemExit:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
