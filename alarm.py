import time
from pathlib import Path

import cv2
import numpy as np

try:
    from playsound import playsound
except ImportError:  # pragma: no cover
    playsound = None


class Alarm:
    """Plays an alarm sound in the background with a cooldown."""

    def __init__(self, sound_path=None, cooldown_sec=3):
        base_dir = Path(__file__).resolve().parent
        if sound_path is None:
            candidates = [base_dir / "alarm.mp3", Path.cwd() / "alarm.mp3"]
            sound_path = next((str(p) for p in candidates if p.exists()), str(base_dir / "alarm.mp3"))
        self.sound_path = str(Path(sound_path).expanduser())
        self.cooldown_sec = cooldown_sec
        self.last_played = 0

    def trigger(self):
        now = time.time()
        if now - self.last_played < self.cooldown_sec:
            return

        try:
            if playsound is None:
                print("[Alarm Warning] playsound is not installed; audio output unavailable.")
                return
            playsound(self.sound_path, block=False)
        except Exception as exc:
            print(f"[Alarm Warning] Could not play sound: {exc}")
        self.last_played = now


def get_pose_bbox_px(pose_landmarks, frame_width, frame_height, padding=20):
    """Compute a pixel-space bounding box around all pose landmarks."""
    xs = [lm.x for lm in pose_landmarks]
    ys = [lm.y for lm in pose_landmarks]

    x1 = int(min(xs) * frame_width) - padding
    y1 = int(min(ys) * frame_height) - padding
    x2 = int(max(xs) * frame_width) + padding
    y2 = int(max(ys) * frame_height) + padding

    return max(x1, 0), max(y1, 0), min(x2, frame_width), min(y2, frame_height)


def draw_flag_overlay(frame, bbox, student_id, color=(0, 0, 255), thickness=3):
    """Draw a red cross and label over the flagged student's bounding box."""
    x1, y1, x2, y2 = bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    cv2.line(frame, (x1, y1), (x2, y2), color, thickness)
    cv2.line(frame, (x2, y1), (x1, y2), color, thickness)
    cv2.putText(frame, f"Student #{student_id} - FLAGGED", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


def spotlight_bbox(frame, bbox, dim_factor=0.3):
    """Dim the whole frame except the given bounding box region."""
    x1, y1, x2, y2 = bbox
    dimmed = (frame.astype(np.float32) * dim_factor).astype(np.uint8)
    output = dimmed.copy()
    output[y1:y2, x1:x2] = frame[y1:y2, x1:x2]
    return output


class DummyLandmark:
    """Mock MediaPipe landmark object for standalone testing."""

    def __init__(self, x, y):
        self.x = x
        self.y = y


def main():
    alarm = Alarm(sound_path="alarm.mp3", cooldown_sec=3)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Pipeline running... Press 'f' to toggle flag/alert state, 'q' to quit.")
    flagged = False

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            h, w, _ = frame.shape
            mock_landmarks = [
                DummyLandmark(0.3, 0.2),
                DummyLandmark(0.7, 0.2),
                DummyLandmark(0.3, 0.8),
                DummyLandmark(0.7, 0.8),
            ]

            if flagged:
                alarm.trigger()
                bbox = get_pose_bbox_px(mock_landmarks, w, h)
                frame = spotlight_bbox(frame, bbox, dim_factor=0.35)
                draw_flag_overlay(frame, bbox, student_id=1)

            cv2.imshow("Smart Monitoring System", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('f'):
                flagged = not flagged
                print(f"Flagged state set to: {flagged}")
            elif key == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()