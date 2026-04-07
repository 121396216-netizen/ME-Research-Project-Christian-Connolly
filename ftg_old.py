#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped

class ReactiveFollowGap(Node):
    def __init__(self):
        super().__init__('reactive_follow_gap_node')
        self.scan_subscription = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 25)
        self.drive_publisher = self.create_publisher(
            AckermannDriveStamped, '/ackermann_cmd', 25)

        # PID controller parameters for steering
        self.kp = 5.0  # Proportional gain
        self.ki = 0.9  # Integral gain
        self.kd = 0.0  # Derivative gain

        # PID control variables for steering
        self.integral = 0.0
        self.prev_error = 0.0

        # Additional parameters
        self.bubble_radius = 0.2  # Safety bubble radius in meters
        self.max_scan_distance = 3.0  # Maximum scan distance to consider in meters

    def preprocess_lidar(self, ranges):
        # Set high values to 0 and apply a moving average for smoothing
        proc_ranges = np.array(ranges)
        proc_ranges[proc_ranges > self.max_scan_distance] = 0
        window_size = 5
        proc_ranges = np.convolve(proc_ranges, np.ones(window_size)/window_size, mode='same')
        return proc_ranges

    def find_max_gap(self, free_space_ranges):
        # Identify continuous segments of non-zero values (navigable space)
        gaps = np.split(free_space_ranges, np.where(free_space_ranges == 0)[0])
        max_gap = max(gaps, key=len)
        start_i = np.where(free_space_ranges == max_gap[0])[0][0]
        end_i = start_i + len(max_gap) - 1
        print("gaps:",gaps)
        return start_i, end_i

    def find_best_point(self, start_i, end_i, ranges):
        # Select the middle point of the largest gap as the best point
        best_point_index = (start_i + end_i) // 2
        return best_point_index

    def pid_control_steering(self, error):
        self.integral += error
        derivative = error - self.prev_error
        output = (self.kp * error) + (self.ki * self.integral) + (self.kd * derivative)
        self.prev_error = error
        return output

    def scan_callback(self, data):
        ranges = np.array(data.ranges)
        print("raw:",*ranges)
        proc_ranges = self.preprocess_lidar(ranges)

        # Assuming the vehicle and LiDAR setup is such that indices directly ahead are at the center
        center_index = len(data.ranges) // 2

        start_i, end_i = self.find_max_gap(proc_ranges)
        best_point_index = self.find_best_point(start_i, end_i, proc_ranges)
        
        # Calculate steering error as the difference between the best point's position and the center of the LiDAR scan
        error = (best_point_index - center_index)
        
        # Apply PID control to compute the steering angle adjustment needed
        steering_correction = self.pid_control_steering(error)
        
        # Assuming a direct mapping of correction to steering angle for simplicity
        # Clamp the steering angle to reasonable bounds if necessary
        steering_angle = np.clip(steering_correction, -np.pi/6, np.pi/6)
        
        # Set a fixed speed or adjust based on conditions
        speed = 0.3  # Simple fixed speed for demonstration
        #print("raw:",proc_ranges)
        # Publish drive message
        drive_msg = AckermannDriveStamped()
        drive_msg.drive.steering_angle = steering_angle
        drive_msg.drive.speed = speed
        self.drive_publisher.publish(drive_msg)

def main(args=None):
    rclpy.init(args=args)
    node = ReactiveFollowGap()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()