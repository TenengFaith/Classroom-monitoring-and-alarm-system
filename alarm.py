import time
import cv2
import numpy as np
from playsound3 import playsound

class Alarm:
    """Plays an alarm sound in the background, with a cooldown

    so it doesn't overlap itself into an unlistenable mess.
    """

    def __init__(self, sound_path="alarm.mp3", cooldown_sec=3):
        self.sound_path = sound_path
        self.cooldown_sec = cooldown_sec
        self.last_played = 0

    def trigger(self):
        now = time.time()
        if now - self.last_played >= self.cooldown_sec:
            try:
                playsound(self.sound_path, block=False)  # Non-blocking execution
            except Exception as e:
                print(f"[Alarm Warning] Could not play sound: {e}")
            self.last_played = now


def get_pose_bbox_px(pose_landmarks, frame_width, frame_height, padding=20):
    """Computes a pixel-space bounding box around all pose landmarks,

    with a little padding so the box isn't drawn skin-tight.
    """
    xs = [lm.x for lm in pose_landmarks]
    ys = [lm.y for lm in pose_landmarks]

    x1 = int(min(xs) * frame_width) - padding
    y1 = int(min(ys) * frame_height) - padding
    x2 = int(max(xs) * frame_width) + padding
    y2 = int(max(ys) * frame_height) + padding

    return max(x1, 0), max(y1, 0), min(x2, frame_width), min(y2, frame_height)


def draw_flag_overlay(frame, bbox, student_id, color=(0, 0, 255), thickness=3):
    """Draws a red cross and label over the flagged student's bounding box."""
    x1, y1, x2, y2 = bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    cv2.line(frame, (x1, y1), (x2, y2), color, thickness)
    cv2.line(frame, (x2, y1), (x1, y2), color, thickness)
    cv2.putText(
        frame,
        f"Student #{student_id} - FLAGGED",
        (x1, y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
    )


def spotlight_bbox(frame, bbox, dim_factor=0.3):
    """Dims the whole frame except the given bounding box region."""
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
    # Initialize the non-blocking Alarm (expects 'alarm.mp3' in current directory)
    alarm = Alarm(sound_path="alarm.mp3", cooldown_sec=3)

    # Open default video camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Pipeline running... Press 'f' to toggle flag/alert state, 'q' to quit.")
    flagged = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape

        # Simulated landmarks (representing normalized pose detection output)
        mock_landmarks = [
            DummyLandmark(0.3, 0.2),
            DummyLandmark(0.7, 0.2),
            DummyLandmark(0.3, 0.8),
            DummyLandmark(0.7, 0.8),
        ]

        if flagged:
            # 1. Trigger non-blocking audio alarm
            alarm.trigger()

            # 2. Get bounding box coordinates from landmarks
            bbox = get_pose_bbox_px(mock_landmarks, w, h)

            # 3. Apply background dimming/spotlight effect
            frame = spotlight_bbox(frame, bbox, dim_factor=0.35)

            # 4. Render red "X" overlay and text label
            draw_flag_overlay(frame, bbox, student_id=1)

        cv2.imshow("Smart Monitoring System", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('f'):
            flagged = not flagged
            print(f"Flagged state set to: {flagged}")
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()