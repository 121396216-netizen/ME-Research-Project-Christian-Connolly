#!/usr/bin/env python3
"""
YOLOv5s NPU inference on Qualcomm RB5.

The SNPE SDK has no ARM64 Python bindings for on-device inference.
Inference runs by calling snpe-parallel-run via subprocess, using temp
files for input/output tensors — the same pattern as run_yolo_npu_c/run.py.

Standalone usage:
    python3 yolo_npu.py [--save-frames] [--skip N]

Import usage (for LM pipeline):
    from yolo_npu import YoloNPU
    detector = YoloNPU()
    detections = detector.detect(bgr_frame)   # list of dicts
"""

import argparse
import os
import signal as _signal
import subprocess
import threading
import time
import numpy as np
import cv2

SNPE_SERVER = '/home/snpe_server'
SNPE_LIB    = '/home/2.25.0.240728/lib/aarch64-ubuntu-gcc9.4'
MODEL_PATH  = '/home/yolov5s.dlc'

INPUT_SIZE  = 640
NUM_ANCHORS = 25200
NUM_CLASSES = 80

CLASS_NAMES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
    'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
    'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
    'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
    'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
    'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant',
    'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush',
]


# Native JPEG output resolution per camera index.
# qtiqmmfsrc outputs image/jpeg natively; jpegdec+videoconvert decode to BGR.
_CAMERA_NATIVE = {
    0: (640, 480),
    1: (640, 480),
}
_DEFAULT_NATIVE = (640, 480)


class GstCamera:
    """
    Reads JPEG frames from qtiqmmfsrc, decodes to BGR via jpegdec+videoconvert,
    then resizes to 640×640 for the detector.

    qtiqmmfsrc natively outputs image/jpeg at 640×480 (not NV12).
    Decoding is done in the GStreamer pipeline so Python receives raw BGR bytes.

    A background thread continuously drains the pipe so the kernel pipe
    buffer never stalls during snpe-parallel-run inference.
    read() always returns the most-recent complete BGR frame.
    """

    def __init__(self, camera_id=0, warmup_frames=30):
        native_w, native_h = _CAMERA_NATIVE.get(camera_id, _DEFAULT_NATIVE)
        self._native_w  = native_w
        self._native_h  = native_h
        self._warmup    = warmup_frames
        # BGR: 3 bytes per pixel
        self.frame_size = native_w * native_h * 3
        cmd = [
            'gst-launch-1.0', '-q',
            'qtiqmmfsrc', f'camera={camera_id}', '!',
            'jpegdec', '!',
            'videoconvert', '!',
            f'video/x-raw,format=BGR,width={native_w},height={native_h}', '!',
            'fdsink', 'fd=1', 'sync=false',
        ]
        self._proc   = subprocess.Popen(cmd, stdout=subprocess.PIPE)
        self._latest = None          # most-recent BGR frame (640×640)
        self._ok     = True          # False once the pipe closes
        self._lock   = threading.Lock()
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    def _reader(self):
        """Background thread: drain pipe continuously, keep latest BGR frame."""
        skipped = 0
        while True:
            raw = self._proc.stdout.read(self.frame_size)
            if len(raw) < self.frame_size:
                with self._lock:
                    self._ok = False
                break
            if skipped < self._warmup:
                skipped += 1
                continue
            # Reshape BGR bytes and resize to 640×640 for the detector
            bgr = np.frombuffer(raw, dtype=np.uint8).reshape(
                self._native_h, self._native_w, 3)
            bgr = cv2.resize(bgr, (INPUT_SIZE, INPUT_SIZE))
            with self._lock:
                self._latest = bgr

    def read(self, startup_timeout=60.0):
        # Wait for at least one frame (pipeline startup can take a few seconds).
        # If no frame arrives within startup_timeout seconds, the pipeline failed.
        deadline = time.time() + startup_timeout
        while True:
            with self._lock:
                if not self._ok:
                    return False, None
                if self._latest is not None:
                    return True, self._latest
            if time.time() > deadline:
                print(f'Camera timeout: no frame in {startup_timeout:.0f}s '
                      '— check QMMF is running and no other process holds the camera',
                      flush=True)
                return False, None
            time.sleep(0.005)

    def release(self):
        # SIGINT triggers gst-launch-1.0 EOS → graceful QMMF shutdown.
        # SIGTERM kills immediately and crashes the camera service.
        self._proc.send_signal(_signal.SIGINT)
        try:
            self._proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            self._proc.kill()
            self._proc.wait()


def _nms(boxes, scores, iou_threshold):
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[np.where(iou <= iou_threshold)[0] + 1]
    return keep


class YoloNPU:
    """YOLOv5s on the RB5 NPU via a persistent snpe_server subprocess."""

    def __init__(self, model_path=MODEL_PATH, conf_thresh=0.5,
                 iou_thresh=0.5, min_box_px=10):
        self.conf_thresh = conf_thresh
        self.iou_thresh  = iou_thresh
        self.min_box_px  = min_box_px

        env = os.environ.copy()
        existing = env.get('LD_LIBRARY_PATH', '')
        env['LD_LIBRARY_PATH'] = f'{SNPE_LIB}:{existing}' if existing else SNPE_LIB

        self._proc = subprocess.Popen(
            [SNPE_SERVER, model_path],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            env=env,
        )
        self._input_bytes  = INPUT_SIZE * INPUT_SIZE * 3 * 4   # float32
        self._output_bytes = NUM_ANCHORS * (5 + NUM_CLASSES) * 4

        # Block until server has finished Build() and is ready for frames.
        # Server writes a single 'R' byte to stdout as the readiness signal.
        print('Loading model on NPU…', flush=True)
        self._proc.stdout.read(1)

        # Discard the first inference — the NPU produces garbage output on its
        # first pass while JIT-compiling ops. Send a black frame and throw away
        # the result.
        dummy = np.zeros(self._input_bytes, dtype=np.uint8)
        def _write_dummy():
            self._proc.stdin.write(dummy.tobytes())
            self._proc.stdin.flush()
        t = threading.Thread(target=_write_dummy, daemon=True)
        t.start()
        self._proc.stdout.read(self._output_bytes)
        t.join()
        print('NPU ready.', flush=True)

    def close(self):
        self._proc.stdin.close()
        self._proc.wait()

    def detect(self, frame):
        """
        Run YOLOv5s on a BGR frame.

        Args:
            frame: numpy BGR image (any resolution).

        Returns:
            list of dicts, each with keys:
              class_id   (int)
              class_name (str)
              score      (float)
              box        (list[float]): [x1, y1, x2, y2] in 640×640 px space
        """
        # ── pre-process ───────────────────────────────────────────────────────
        img = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE))
        data = (img.astype(np.float32) / 255.0).tobytes()

        # Write frame to stdin in a thread — the frame (4.7 MB) is much larger
        # than the kernel pipe buffer (64 KB), so a blocking write on the same
        # thread as the stdout read would deadlock.
        def _write():
            self._proc.stdin.write(data)
            self._proc.stdin.flush()
        writer = threading.Thread(target=_write, daemon=True)
        writer.start()

        # ── read output tensor from stdout ────────────────────────────────────
        buf = self._proc.stdout.read(self._output_bytes)
        writer.join()
        if len(buf) < self._output_bytes:
            raise RuntimeError('snpe_server closed unexpectedly')
        raw = np.frombuffer(buf, dtype=np.float32).reshape(NUM_ANCHORS, 5 + NUM_CLASSES)
        print(f'  DBG input mean={img.mean():.2f}  '
              f'max_obj={raw[:,4].max():.4f}  '
              f'max_score={(raw[:,4]*raw[:,5:].max(axis=1)).max():.4f}', flush=True)

        # ── confidence filter ─────────────────────────────────────────────────
        obj_scores = raw[:, 4]
        mask = obj_scores > self.conf_thresh
        if not np.any(mask):
            return []

        raw    = raw[mask]
        scores = raw[:, 4]
        cx, cy, w, h = raw[:, 0], raw[:, 1], raw[:, 2], raw[:, 3]
        class_ids = raw[:, 5:].argmax(axis=1)

        # ── cx,cy,w,h → x1,y1,x2,y2 ──────────────────────────────────────────
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        boxes = np.stack([x1, y1, x2, y2], axis=1)
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, INPUT_SIZE)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, INPUT_SIZE)

        # ── NMS ───────────────────────────────────────────────────────────────
        keep      = _nms(boxes, scores, self.iou_thresh)
        boxes     = boxes[keep]
        scores    = scores[keep]
        class_ids = class_ids[keep]

        # ── build result list ─────────────────────────────────────────────────
        detections = []
        for box, score, cid in zip(boxes, scores, class_ids):
            if (box[2] - box[0]) >= self.min_box_px and (box[3] - box[1]) >= self.min_box_px:
                detections.append({
                    'class_id':   int(cid),
                    'class_name': CLASS_NAMES[int(cid)],
                    'score':      float(score),
                    'box':        box.tolist(),
                })
        return detections

    def draw(self, frame_640, detections):
        """Draw detection boxes onto a 640×640 BGR frame (in-place)."""
        for d in detections:
            x1, y1, x2, y2 = (int(v) for v in d['box'])
            cv2.rectangle(frame_640, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{d['class_name']} {d['score']:.2f}"
            lw, lh = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            ty = y1 - lh if y1 - lh > 0 else y1 + lh
            cv2.rectangle(frame_640, (x1, ty - lh), (x1 + lw, ty), (0, 255, 0), cv2.FILLED)
            cv2.putText(frame_640, label, (x1, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        return frame_640


# ── standalone test loop ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--camera',      type=int,   default=0)
    parser.add_argument('--model',       default=MODEL_PATH)
    parser.add_argument('--conf',        type=float, default=0.25)
    parser.add_argument('--skip',        type=int,   default=6,
                        help='run inference every N frames')
    parser.add_argument('--save-frames', action='store_true',
                        help='save annotated frame to /tmp/yolo_out.jpg')
    args = parser.parse_args()

    print(f'Loading model: {args.model}')
    detector = YoloNPU(model_path=args.model, conf_thresh=args.conf)
    print(f'Confidence threshold: {args.conf}')
    print('Opening camera…')
    cam = GstCamera(camera_id=args.camera)  # native NV12 → BGR 640×640
    print('Running (Ctrl-C to stop)…')

    frame_n = 0
    t_last  = time.time()

    try:
        while True:
            ret, frame = cam.read()
            if not ret:
                print('Camera read failed.')
                break

            frame_n += 1
            if frame_n % args.skip != 0:
                continue

            t0 = time.time()
            detections = detector.detect(frame)
            ms = (time.time() - t0) * 1000

            fps = args.skip / (time.time() - t_last)
            t_last = time.time()

            print(f'[frame {frame_n:6d}]  {len(detections)} det  '
                  f'infer={ms:.1f}ms  eff_fps={fps:.1f}')
            for d in detections:
                print(f'  {d["class_name"]:20s}  score={d["score"]:.2f}  '
                      f'box={[round(v) for v in d["box"]]}')

            if args.save_frames:
                detector.draw(frame, detections)
                cv2.imwrite('/tmp/yolo_out.jpg', frame)

    except KeyboardInterrupt:
        print('Stopped.')
    finally:
        detector.close()
        cam.release()


if __name__ == '__main__':
    main()
