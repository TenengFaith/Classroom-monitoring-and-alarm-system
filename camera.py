import time
from datetime import datetime

import cv2


# configuration
DEFAULT_CAMERA_ID = 0
DEFAULT_FRAME_WIDTH = 1280
DEFAULT_FRAME_HEIGHT = 720
DEFAULT_TARGET_FPS = 30
MAX_FAILED_READS = 5


def open_camera(camera_id=DEFAULT_CAMERA_ID,
               frame_width=DEFAULT_FRAME_WIDTH,
               frame_height=DEFAULT_FRAME_HEIGHT,
               target_fps=DEFAULT_TARGET_FPS):
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print("Cannot open camera.")
        return None

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
    cap.set(cv2.CAP_PROP_FPS, target_fps)
    return cap


def run_camera_loop(camera_id=DEFAULT_CAMERA_ID,
                   frame_width=DEFAULT_FRAME_WIDTH,
                   frame_height=DEFAULT_FRAME_HEIGHT,
                   target_fps=DEFAULT_TARGET_FPS,
                   max_failed_reads=MAX_FAILED_READS,
                   window_name="Classroom Capture - Track 1"):
    cap = open_camera(camera_id, frame_width, frame_height, target_fps)
    if cap is None:
        raise RuntimeError("Could not open camera")

    print("Camera connected")

    frame_counts = 0
    failed_reads = 0
    previous_time = time.time()
    fps = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                failed_reads += 1
                print(f"WARNING: Failed to receive frame ({failed_reads}/{max_failed_reads})")

                if failed_reads >= max_failed_reads:
                    print("Camera connection may be lost. Attempting to reconnect...")
                    cap.release()
                    time.sleep(2)
                    cap = open_camera(camera_id, frame_width, frame_height, target_fps)
                    if cap is None:
                        print("ERROR: Reconnection failed.")
                        break
                    print("Camera reconnected.")
                    failed_reads = 0
                continue

            failed_reads = 0
            frame_counts += 1

            current_time = time.time()
            elapsed = current_time - previous_time
            if elapsed > 0:
                fps = 1 / elapsed
            previous_time = current_time

            height, width = frame.shape[:2]
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            cv2.putText(frame, f"FPS: {fps:.1f}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"Resolution: {width}*{height}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"Frames: {frame_counts}", (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"Time: {timestamp}", (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

            cv2.imshow(window_name, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    print("\nCapture stopped.")
    print(f"Total frames received: {frame_counts}")


if __name__ == "__main__":
    run_camera_loop()