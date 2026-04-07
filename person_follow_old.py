#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
from cv_bridge import CvBridge
import cv2
import torch

class PersonFollow(Node):
    def __init__(self):
        super().__init__('person_follow_node')
        #self.scan_subscription = self.create_subscription(
         #   LaserScan, '/scan', self.scan_callback, 25)
        self.drive_publisher = self.create_publisher(
            AckermannDriveStamped, '/ackermann_cmd', 10)

        self.get_logger().info("Person following starts now hehe")
        # PID controller parameters for steering
        self.kp = 2.0  # Proportional gain
        self.ki = 0.5  # Integral gain
        self.kd = 0.1  # Derivative gain

        # PID control variables for steering
        self.integral = 0.0
        self.prev_error = 0.0

        # YOLO model for person detection
        self.yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
        self.bridge = CvBridge()
        self.person_detected = False
        self.person_center_x = 0
        self.person_bbox_area = 0

        # Initialize TCP camera stream
        self.video_stream_url = "tcp://192.168.137.25:34808"
        self.cap = cv2.VideoCapture(self.video_stream_url)
        if not self.cap.isOpened():
            self.get_logger().error("Error: Could not open video stream")
            rclpy.shutdown()
            exit()

        # Timer to periodically call image processing
        self.timer = self.create_timer(0.1, self.image_callback)
        self.get_logger().info("PersonFollow node has been initialized")

        # Smoothing for steering and speed
        self.steering_angle_ema = 0.0
        self.speed_ema = 1.0
        self.ema_alpha = 0.2  # Smoothing factor
    '''
    def preprocess_lidar(self, ranges):
        self.get_logger().debug("Preprocessing LIDAR data")
        proc_ranges = np.array(ranges)
        proc_ranges[proc_ranges > 3.0] = 0
        window_size = 5
        proc_ranges = np.convolve(proc_ranges, np.ones(window_size)/window_size, mode='same')
        return proc_ranges

    def find_max_gap(self, free_space_ranges):
        self.get_logger().debug("Finding max gap in LIDAR data")
        gaps = np.split(free_space_ranges, np.where(free_space_ranges == 0)[0])
        max_gap = max(gaps, key=len)
        start_i = np.where(free_space_ranges == max_gap[0])[0][0]
        end_i = start_i + len(max_gap) - 1
        return start_i, end_i

    def find_best_point(self, start_i, end_i, ranges):
        best_point_index = (start_i + end_i) // 2
        self.get_logger().debug(f"Best point found at index {best_point_index}")
        return best_point_index
    '''
    def pid_control_steering(self, error):
        self.integral += error
        derivative = error - self.prev_error
        output = (self.kp * error) + (self.ki * self.integral) + (self.kd * derivative)
        self.prev_error = error
        self.get_logger().debug(f"PID control steering output: {output}")
        return output
    '''
    def scan_callback(self, data):
        self.get_logger().info("Received LIDAR scan data")
        ranges = np.array(data.ranges)
        proc_ranges = self.preprocess_lidar(ranges)
        center_index = len(data.ranges) // 2

        start_i, end_i = self.find_max_gap(proc_ranges)
        best_point_index = self.find_best_point(start_i, end_i, proc_ranges)
        
        error = (best_point_index - center_index)
        steering_correction = self.pid_control_steering(error)
        steering_angle = np.clip(steering_correction, -np.pi/6, np.pi/6)
        
        # Adjust speed based on person detection
        if self.person_detected:
            speed = self.calculate_speed()
        else:
            speed = 1.0

        # Smooth the steering angle and speed
        self.steering_angle_ema = self.ema_alpha * steering_angle + (1 - self.ema_alpha) * self.steering_angle_ema
        self.speed_ema = self.ema_alpha * speed + (1 - self.ema_alpha) * self.speed_ema

        self.get_logger().info(f"Steering angle: {self.steering_angle_ema}, Speed: {self.speed_ema}")
        drive_msg = AckermannDriveStamped()
        drive_msg.drive.steering_angle = self.steering_angle_ema
        drive_msg.drive.speed = self.speed_ema
        self.drive_publisher.publish(drive_msg)
    '''
    def image_callback(self):
        self.get_logger().info("Capturing image from TCP stream")
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().error("Error: Failed to capture image")
            return

        results = self.yolo_model(frame)
        self.process_detections(results)

    def process_detections(self, results):
        self.person_detected = False
        for *xyxy, conf, cls in results.xyxy[0]:
            if int(cls) == 0:  # Class 0 is 'person' 
                x1, y1, x2, y2 = map(int, xyxy)
                self.person_center_x = (x1 + x2) // 2
                self.person_bbox_area = (x2 - x1) * (y2 - y1)
                self.person_detected = True
                self.get_logger().info(f"Person detected at center x: {self.person_center_x}, bbox area: {self.person_bbox_area}")
                break

        #if self.person_detected:
        self.follow_person()

    def follow_person(self):
        frame_center_x = 320 
        error = self.person_center_x - frame_center_x
        self.get_logger().info(f"Following person, error: {error}")
        steering_correction = self.pid_control_steering(error / frame_center_x)
        steering_angle = np.clip(steering_correction, -np.pi/6, np.pi/6)
        
        # Calculate speed and ensure it's positive
        speed = self.calculate_speed()

        # Smooth the steering angle and speed
        self.steering_angle_ema = self.ema_alpha * steering_angle + (1 - self.ema_alpha) * self.steering_angle_ema
        self.speed_ema = self.ema_alpha * speed + (1 - self.ema_alpha) * self.speed_ema

        self.get_logger().info(f"Steering angle: {self.steering_angle_ema}, Speed: {self.speed_ema}")
        drive_msg = AckermannDriveStamped()
        drive_msg.drive.steering_angle = self.steering_angle_ema
        #drive_msg.drive.speed = self.speed_ema
        drive_msg.drive.speed = speed
        self.drive_publisher.publish(drive_msg)

    def calculate_speed(self):
        # Calculate speed based on the bounding box area
        desired_bbox_area = 30000
        max_speed = 2.0
        min_speed = 1.0

        if self.person_bbox_area >= desired_bbox_area:
            self.get_logger().info("Person is very close, setting minimum speed")
            return min_speed
        else:
            speed = max_speed - (self.person_bbox_area / desired_bbox_area) * (max_speed - min_speed)
            self.get_logger().info(f"Calculated speed: {speed}")
            return np.clip(speed, min_speed, max_speed)

def main(args=None):
    rclpy.init(args=args)
    node = PersonFollow()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
