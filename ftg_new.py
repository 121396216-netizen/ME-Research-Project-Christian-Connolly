#!/usr/bin/env python3
"""
follow_the_gap.py — Reactive Follow the Gap controller for F1TENTH.

No map, no localization, no waypoints needed. Uses only lidar.

Algorithm:
  1. Preprocess lidar: clip ranges, apply safety bubble around closest point
  2. Find the largest contiguous gap of free space
  3. Steer toward the best point in that gap (deepest or most centered)
  4. Scale speed based on steering angle

Publishes: /ackermann_cmd (AckermannDriveStamped)
Subscribes: /scan (LaserScan)

USAGE:
    ros2 run autonomous_driving_pkg follow_the_gap
    Or standalone:
    python3 follow_the_gap.py
"""
import math
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped


class FollowTheGap(Node):
    def __init__(self):
        super().__init__("follow_the_gap")

        self.declare_parameter("max_speed", 1.7)
        self.declare_parameter("min_speed", 1.2)
        self.declare_parameter("max_steer", 0.35)
        self.declare_parameter("bubble_radius", 0.3)
        self.declare_parameter("max_range_clip", 3.0)
        self.declare_parameter("min_gap_size", 10)
        self.declare_parameter("disparity_threshold", 0.5)
        self.declare_parameter("window_size", 5)
        self.declare_parameter("fov_deg", 180.0)

        self.max_speed = self.get_parameter("max_speed").value
        self.min_speed = self.get_parameter("min_speed").value
        self.max_steer = self.get_parameter("max_steer").value
        self.bubble_radius = self.get_parameter("bubble_radius").value
        self.max_range_clip = self.get_parameter("max_range_clip").value
        self.min_gap_size = self.get_parameter("min_gap_size").value
        self.disparity_threshold = self.get_parameter("disparity_threshold").value
        self.window_size = self.get_parameter("window_size").value
        self.fov_deg = self.get_parameter("fov_deg").value

        self.scan_sub = self.create_subscription(
            LaserScan, "/scan", self.scan_callback, 10)
        self.drive_pub = self.create_publisher(
            AckermannDriveStamped, "/ackermann_cmd", 10)

        self.get_logger().info(
            f"Follow the Gap started | speed=[{self.min_speed}-{self.max_speed}] "
            f"max_steer={math.degrees(self.max_steer):.0f}deg "
            f"bubble={self.bubble_radius}m fov={self.fov_deg}deg")

    def scan_callback(self, msg):
        ranges = np.array(msg.ranges, dtype=np.float32)
        n = len(ranges)

        if n == 0:
            self.publish_drive(0.0, 0.0)
            return

        fov_rad = math.radians(self.fov_deg)
        half_fov = fov_rad / 2.0
        angles = np.linspace(msg.angle_min, msg.angle_max, n)
        fov_mask = np.abs(angles) <= half_fov

        proc = np.copy(ranges)
        proc[~np.isfinite(proc)] = 0.0
        proc = np.clip(proc, 0.0, self.max_range_clip)
        proc[~fov_mask] = 0.0

        if self.window_size > 1:
            kernel = np.ones(self.window_size) / self.window_size
            proc = np.convolve(proc, kernel, mode='same')

        fov_ranges = np.copy(proc)
        fov_ranges[~fov_mask] = self.max_range_clip + 1
        fov_ranges[fov_ranges <= 0.01] = self.max_range_clip + 1
        closest_idx = np.argmin(fov_ranges)

        closest_range = proc[closest_idx]
        if closest_range > 0.01:
            bubble_angle = math.atan2(self.bubble_radius, closest_range)
            bubble_beams = int(bubble_angle / msg.angle_increment)
            lo = max(0, closest_idx - bubble_beams)
            hi = min(n - 1, closest_idx + bubble_beams)
            proc[lo:hi + 1] = 0.0

        for i in range(1, n):
            diff = abs(proc[i] - proc[i - 1])
            if diff > self.disparity_threshold:
                shorter_range = min(proc[i], proc[i - 1])
                if shorter_range > 0.01:
                    extend_angle = math.atan2(self.bubble_radius, shorter_range)
                    extend_beams = int(extend_angle / msg.angle_increment)
                    if proc[i] < proc[i - 1]:
                        end = min(n, i + extend_beams)
                        proc[i:end] = np.minimum(proc[i:end], shorter_range)
                    else:
                        start = max(0, i - extend_beams)
                        proc[start:i] = np.minimum(proc[start:i], shorter_range)

        threshold = 0.1
        open_mask = proc > threshold

        best_gap_start = 0
        best_gap_end = 0
        best_gap_len = 0

        gap_start = None
        for i in range(n):
            if open_mask[i]:
                if gap_start is None:
                    gap_start = i
            else:
                if gap_start is not None:
                    gap_len = i - gap_start
                    if gap_len > best_gap_len:
                        best_gap_len = gap_len
                        best_gap_start = gap_start
                        best_gap_end = i - 1
                    gap_start = None

        if gap_start is not None:
            gap_len = n - gap_start
            if gap_len > best_gap_len:
                best_gap_len = gap_len
                best_gap_start = gap_start
                best_gap_end = n - 1

        if best_gap_len < self.min_gap_size:
            self.get_logger().warn("No valid gap found! Stopping.")
            self.publish_drive(0.0, 0.0)
            return

        gap_ranges = proc[best_gap_start:best_gap_end + 1]
        best_local = np.argmax(gap_ranges)
        best_idx = best_gap_start + best_local

        best_angle = angles[best_idx]
        steer = np.clip(best_angle, -self.max_steer, self.max_steer)

        steer_ratio = abs(steer) / self.max_steer
        speed = self.max_speed - steer_ratio * (self.max_speed - self.min_speed)

        self.publish_drive(speed, steer)

    def publish_drive(self, speed, steer):
        msg = AckermannDriveStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"
        msg.drive.speed = float(speed)
        msg.drive.steering_angle = float(steer)
        self.drive_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = FollowTheGap()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
