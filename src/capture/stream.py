import cv2
import time
from datetime import datetime


class VideoStream:
    """
    Unified stream capture handler for webcam feeds, video files, and IP camera URLs.
    Handles reconnection attempts, frame timing, and resolution querying.
    """

    def __init__(self, source=0, target_fps=30, max_failed_reads=5):
        self.source = self._parse_source(source)
        self.target_fps = target_fps
        self.max_failed_reads = max_failed_reads

        self.cap = None
        self.frame_count = 0
        self.failed_reads = 0
        self.start_time = None
        self.previous_time = None
        self.fps = 0.0

        self.open()

    def _parse_source(self, source):
        if isinstance(source, str) and source.isdigit():
            return int(source)
        return source

    def open(self):
        if self.cap is not None:
            self.cap.release()

        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            print(f"[STREAM ERROR] Could not open video source: {self.source}")
            return False

        if isinstance(self.source, int):
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)

        self.start_time = time.time()
        self.previous_time = self.start_time
        return True

    def get_declared_fps(self):
        if not self.cap or not self.cap.isOpened():
            return float(self.target_fps)
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps != fps or fps <= 0:
            return float(self.target_fps)
        return float(fps)

    def read(self):
        if not self.cap or not self.cap.isOpened():
            return False, None

        ret, frame = self.cap.read()
        if not ret or frame is None:
            self.failed_reads += 1
            if self.failed_reads >= self.max_failed_reads:
                if isinstance(self.source, str) and not self.source.endswith(('.mp4', '.avi', '.mkv', '.mov')):
                    print("[STREAM WARNING] Stream read failed. Attempting reconnect...")
                    time.sleep(1.0)
                    if self.open():
                        self.failed_reads = 0
                        return self.read()
            return False, None

        self.failed_reads = 0
        self.frame_count += 1

        current_time = time.time()
        elapsed = current_time - (self.previous_time or current_time)
        if elapsed > 0:
            self.fps = 1.0 / elapsed
        self.previous_time = current_time

        return True, frame

    def release(self):
        if self.cap:
            self.cap.release()
            self.cap = None
        print(f"[STREAM] Released stream source '{self.source}'. Total frames read: {self.frame_count}")
