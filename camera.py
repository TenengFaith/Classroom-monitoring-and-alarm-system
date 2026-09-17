import cv2
import time
from datetime import datetime
import mediapipe as mp

# configuration
camera_Id = 0
frame_widths = 1280
frame_hieght = 720
Target_FPS = 30
max_failed_reads = 5

# open camera
def open_camera():
    cap = cv2.VideoCapture(camera_Id)

    if not cap.isOpened():
        print("cant get camera")
        return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_widths)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_hieght)
    cap.set(cv2.CAP_PROP_FPS, Target_FPS)
    return cap

# start camera
cap = open_camera()
if cap is None:
    print("error, could not open camera")
    exit()

print("camera connected")

# monitoring variables
frame_counts = 0
failed_reads = 0
start_time = time.time()
previous_time = start_time
fps = 0

# Main capture loop
while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        failed_reads += 1
        print(f"WARNING: Failed to recieve frame ({failed_reads}/{max_failed_reads})")

        if failed_reads >= max_failed_reads:
            print("Camera connection may be lost.")
            print("Attempting to reconnect...")

            cap.release()
            time.sleep(2)

            cap = open_camera()

            if cap is None:
                print("ERROR: reconnection failed.")
                break

            print("Camera reconnected.")
            failed_reads = 0
        continue

    # valid frame
    failed_reads = 0
    frame_counts += 1

    # FPS calculation
    current_time = time.time()
    elapsed = current_time - previous_time
    if elapsed > 0:
        fps = 1 / elapsed
    previous_time = current_time

    # getting actual resolution
    height, width = frame.shape[:2]
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # displaying information on frame
    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8,
        (0, 255, 0), 2
    )
    cv2.putText(
        frame, 
        f"Resolution: {width}*{height}",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8,
        (0, 255, 0), 2
    )
    cv2.putText(
        frame,  # <-- Added missing 'frame' argument
        f"Frames: {frame_counts}",
        (20, 105),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8,
        (0, 255, 0), 2
    )
    cv2.putText(
        frame,  # <-- Added missing 'frame' argument
        f"Time: {timestamp}",
        (20, 140),
        cv2.FONT_HERSHEY_SIMPLEX, 0.65,
        (0, 255, 0), 2
    )

    # display frame
    cv2.imshow("Classroom Capture - Track 1", frame)

    # exit (Fixed typo: waitKey with a capital K)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
print("\nCapture stopped.")
print(f"Total frames recieved: {frame_counts}")