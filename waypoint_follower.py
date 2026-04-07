#!/usr/bin/env python3
"""
waypoint_follower.py — Improved Pure Pursuit waypoint follower.
Fixes: wrapping search, larger search window, loop support, warmup TF wait.

USAGE:
    ros2 run autonomous_driving_pkg waypoint_follower --ros-args \
        -p csv_path:=/home/f1tenth_ws/waypoints.csv \
        -p speed:=0.5 -p lookahead:=0.8
"""
import csv
import math
from typing import List, Tuple, Optional

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.time import Time

from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
from tf2_ros import Buffer, TransformListener

from .pure_pursuit_controller import PurePursuitCore
from .euler_transform import quaternion_to_euler

Point2 = Tuple[float, float]


class WaypointFollower(Node):
    def __init__(self):
        super().__init__("waypoint_follower")

        self.declare_parameter("csv_path", "")
        self.declare_parameter("lookahead", 0.8)
        self.declare_parameter("wheelbase", 0.25)
        self.declare_parameter("speed", 0.5)
        self.declare_parameter("max_steer", 0.35)
        self.declare_parameter("N", 15)
        self.declare_parameter("global_frame", "map")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("odom_timeout", 0.25)
        self.declare_parameter("loop", True)

        csv_path = self.get_parameter("csv_path").value
        self.lookahead = self.get_parameter("lookahead").value
        self.wheelbase = self.get_parameter("wheelbase").value
        self.speed = self.get_parameter("speed").value
        self.max_steer = self.get_parameter("max_steer").value
        self.N = self.get_parameter("N").value
        self.global_frame = self.get_parameter("global_frame").value
        self.base_frame = self.get_parameter("base_frame").value
        self.odom_timeout = self.get_parameter("odom_timeout").value
        self.loop = self.get_parameter("loop").value

        if not csv_path:
            raise RuntimeError("csv_path required")

        self.window_index = 0
        self._latest_odom = None  # type: Optional[Odometry]
        self.last_odom_time = None
        self.log_counter = 0

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.route_xy = self._load_csv(csv_path)
        self.n = len(self.route_xy)

        if self.n < 2:
            raise RuntimeError("Need at least 2 waypoints, got %d" % self.n)

        self.controller_PP = PurePursuitCore(self.wheelbase, self.lookahead)

        self.odom_sub = self.create_subscription(Odometry, "/odom", self.on_odom, 10)
        self.cmd_pub = self.create_publisher(AckermannDriveStamped, "/ackermann_cmd", 10)

        # Warmup phase — wait for TF before starting 50Hz control
        self.warmup_timer = self.create_timer(1.0, self.warmup_check)
        self.control_timer = None

        self.get_logger().info(
            "Loaded %d waypoints | lookahead=%.2f speed=%.2f loop=%s"
            % (self.n, self.lookahead, self.speed, self.loop))
        self.get_logger().info("Waiting for TF and odom...")

    def _load_csv(self, path):
        # type: (str) -> List[Point2]
        pts = []
        with open(path, "r") as f:
            for row in csv.reader(f):
                if not row:
                    continue
                try:
                    pts.append((float(row[0]), float(row[1])))
                except (ValueError, IndexError):
                    continue
        return pts

    def on_odom(self, msg):
        # type: (Odometry) -> None
        self._latest_odom = msg
        self.last_odom_time = self.get_clock().now()

    def warmup_check(self):
        if self._latest_odom is None:
            self.get_logger().warn("Waiting for /odom...")
            self._publish_drive(0.0, 0.0)
            return
        try:
            self.tf_buffer.lookup_transform(
                self.global_frame, self.base_frame, Time(),
                timeout=Duration(seconds=2.0))
            self.get_logger().info("TF ready! Starting control loop at 50Hz.")
            self.warmup_timer.cancel()
            self.control_timer = self.create_timer(0.02, self.control_tick)
        except Exception as e:
            self.get_logger().warn("Waiting for TF %s->%s: %s"
                                   % (self.global_frame, self.base_frame, str(e)))
            self._publish_drive(0.0, 0.0)

    def _select_window(self, car_xy):
        # type: (Point2) -> List[Point2]
        """Find nearest waypoint ahead and return N points with wrap-around."""
        cx, cy = car_xy
        if not self.route_xy:
            return []

        # Search forward from current index (wraps around for closed tracks)
        search = max(30, self.N * 3)
        best_i = self.window_index
        best_d = float("inf")

        for off in range(search):
            if self.loop:
                i = (self.window_index + off) % self.n
            else:
                i = self.window_index + off
                if i >= self.n:
                    break
            d = math.hypot(self.route_xy[i][0] - cx, self.route_xy[i][1] - cy)
            if d < best_d:
                best_d = d
                best_i = i

        self.window_index = best_i

        # Build window of N points ahead
        window = []
        for j in range(self.N):
            if self.loop:
                idx = (best_i + j) % self.n
            else:
                idx = best_i + j
                if idx >= self.n:
                    break
            window.append(self.route_xy[idx])

        # Pad if we ran out of points (non-loop mode)
        if 0 < len(window) < self.N:
            window.extend([window[-1]] * (self.N - len(window)))

        return window

    def _map_to_base(self, pts, bx, by, yaw):
        # type: (List[Point2], float, float, float) -> List[Point2]
        cy = math.cos(yaw)
        sy = math.sin(yaw)
        out = []
        for px, py in pts:
            dx = px - bx
            dy = py - by
            x_b =  cy * dx + sy * dy
            y_b = -sy * dx + cy * dy
            out.append((x_b, y_b))
        return out

    def _publish_drive(self, speed, steer):
        # type: (float, float) -> None
        msg = AckermannDriveStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.drive.speed = float(speed)
        msg.drive.steering_angle = float(max(-self.max_steer, min(self.max_steer, steer)))
        self.cmd_pub.publish(msg)

    def control_tick(self):
        if self._latest_odom is None or not self.route_xy:
            self._publish_drive(0.0, 0.0)
            return

        if self.last_odom_time is None or \
           (self.get_clock().now() - self.last_odom_time) > Duration(seconds=self.odom_timeout):
            self._publish_drive(0.0, 0.0)
            return

        try:
            tf = self.tf_buffer.lookup_transform(
                self.global_frame, self.base_frame, Time(),
                timeout=Duration(seconds=0.1))
        except Exception as e:
            self._publish_drive(0.0, 0.0)
            return

        bx = tf.transform.translation.x
        by = tf.transform.translation.y
        q = tf.transform.rotation
        yaw, _, _ = quaternion_to_euler(q.x, q.y, q.z, q.w)

        window = self._select_window((bx, by))
        if len(window) < 2:
            self._publish_drive(0.0, 0.0)
            return

        window_base = self._map_to_base(window, bx, by, yaw)
        steer = self.controller_PP.compute(window_base)
        self._publish_drive(self.speed, steer)

        # Periodic logging (once per second at 50Hz)
        self.log_counter += 1
        if self.log_counter % 50 == 0:
            self.get_logger().info(
                "pos=(%.2f,%.2f) yaw=%.1f steer=%.1f wp=%d"
                % (bx, by, math.degrees(yaw), math.degrees(steer), self.window_index))


def main(args=None):
    rclpy.init(args=args)
    node = WaypointFollower()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
