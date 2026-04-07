#!/usr/bin/env python3
"""
Person follower node - runs natively on the RB5.

YOLO thread  : ~5-15 fps  -> updates target_angle (shared), runs on NPU via YoloNPU
LiDAR thread : ~20-50 Hz  -> steers toward target_angle, stops when close
"""

import os
import sys
import threading
import time
import math

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import LaserScan, CompressedImage
from ackermann_msgs.msg import AckermannDriveStamped

sys.path.insert(0, '/home')
from yolo_npu import YoloNPU

# ── YOLO paths ────────────────────────────────────────────────────────────────
YOLO_MODEL_PATH = os.environ.get('YOLO_MODEL_PATH', '/home/yolov5s.dlc')
SNPE_LIB_PATH   = '/home/2.25.0.240728/lib/aarch64-ubuntu-gcc9.4'
# ─────────────────────────────────────────────────────────────────────────────

CAMERA_HFOV_DEG   = 70.0
IMG_WIDTH         = 640
YOLO_CONF         = 0.3
YOLO_TIMEOUT_S    = 3.0
STOP_DIST_M       = 0.3
FOLLOW_SPEED      = 1.5
MAX_STEER_RAD     = 0.4
STEER_GAIN        = 1.2

BEST_EFFORT_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
)

HFOV_RAD   = math.radians(CAMERA_HFOV_DEG)
IMG_CENTER = IMG_WIDTH / 2.0


class PersonFollower(Node):
    def __init__(self):
        super().__init__('person_follower')

        self._lock = threading.Lock()
        self._target_angle = None
        self._last_detection_time = 0.0
        self._yolo_busy = False
        self._cmd_speed = 0.0
        self._cmd_steer = 0.0

        self.get_logger().info(f'Loading YOLOv5s on NPU from {YOLO_MODEL_PATH}...')
        self._detector = YoloNPU(model_path=YOLO_MODEL_PATH, conf_thresh=YOLO_CONF)
        self.get_logger().info('NPU ready.')

        self._img_sub = self.create_subscription(
            CompressedImage, '/camera/image/compressed',
            self._image_callback, BEST_EFFORT_QOS,
        )
        self._scan_sub = self.create_subscription(
            LaserScan, '/scan',
            self._scan_callback, BEST_EFFORT_QOS,
        )
        self._drive_pub = self.create_publisher(
            AckermannDriveStamped, '/ackermann_cmd', 10,
        )
        self.create_timer(0.02, self._timer_callback)
        self.get_logger().info('Person follower started. Waiting for camera + scan...')

    def _image_callback(self, msg):
        with self._lock:
            if self._yolo_busy:
                return
            self._yolo_busy = True
        np_arr = np.frombuffer(msg.data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None:
            with self._lock:
                self._yolo_busy = False
            return
        threading.Thread(target=self._run_yolo, args=(frame,), daemon=True).start()

    def _run_yolo(self, frame):
        try:
            detections = self._detector.detect(frame)
            persons = [d for d in detections if d['class_name'] == 'person']
            if not persons:
                self.get_logger().info('YOLO: no person detected', throttle_duration_sec=2.0)
                return
            self.get_logger().info(f'YOLO: {len(persons)} person(s) detected')
            # Track the largest bounding box
            best = max(persons, key=lambda d: (d['box'][2] - d['box'][0]) * (d['box'][3] - d['box'][1]))
            box = best['box']  # [x1, y1, x2, y2] in 640px space
            cx = (box[0] + box[2]) / 2.0
            offset = (cx - IMG_CENTER) / IMG_CENTER
            angle = -offset * (HFOV_RAD / 2.0)
            with self._lock:
                self._target_angle = angle
                self._last_detection_time = time.time()
        finally:
            with self._lock:
                self._yolo_busy = False

    def _scan_callback(self, msg):
        with self._lock:
            target = self._target_angle
            last_t = self._last_detection_time

        if target is None or (time.time() - last_t) > YOLO_TIMEOUT_S:
            with self._lock:
                self._cmd_speed = 0.0
                self._cmd_steer = 0.0
            return

        ranges = np.array(msg.ranges, dtype=np.float32)
        angles = np.linspace(msg.angle_min, msg.angle_max, len(ranges))
        mask = np.abs(angles - target) <= math.radians(15)
        valid = ranges[mask]
        valid = valid[np.isfinite(valid) & (valid > msg.range_min) & (valid < msg.range_max)]

        if valid.size == 0:
            with self._lock:
                self._cmd_speed = 0.5
                self._cmd_steer = self._clamp(target * STEER_GAIN)
            return

        min_dist = float(np.min(valid))
        if min_dist <= STOP_DIST_M:
            with self._lock:
                self._cmd_speed = 0.0
                self._cmd_steer = 0.0
        else:
            with self._lock:
                self._cmd_speed = min(FOLLOW_SPEED, (min_dist - STOP_DIST_M) * 1.5)
                self._cmd_steer = self._clamp(target * STEER_GAIN)

    def _clamp(self, angle):
        return max(-MAX_STEER_RAD, min(MAX_STEER_RAD, angle))

    def _timer_callback(self):
        with self._lock:
            speed, steer = self._cmd_speed, self._cmd_steer
        msg = AckermannDriveStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.drive.speed = float(speed)
        msg.drive.steering_angle = float(steer)
        self._drive_pub.publish(msg)

    def _publish(self, speed, steer):
        with self._lock:
            self._cmd_speed = speed
            self._cmd_steer = steer


def main():
    os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    os.environ.setdefault('MKL_NUM_THREADS', '1')
    # SNPE libs required by snpe_server subprocess
    existing = os.environ.get('LD_LIBRARY_PATH', '')
    os.environ['LD_LIBRARY_PATH'] = f'{SNPE_LIB_PATH}:{existing}' if existing else SNPE_LIB_PATH

    rclpy.init()
    node = PersonFollower()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._publish(0.0, 0.0)
        node._detector.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
