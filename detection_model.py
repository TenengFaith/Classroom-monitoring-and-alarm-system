import cv2
import os
import time
from ultralytics import YOLO

class ClassroomDetectionPipeline:
    def __init__(self, yolo_model_path="yolov8n.pt", confidence_threshold=0.35):
        self.confidence_threshold = confidence_threshold
        self.model = YOLO(yolo_model_path)

    def process_frame(self, frame_bgr, timestamp_ms):
        results = self.model(frame_bgr, conf=self.confidence_threshold, verbose=False)[0]

        structured_output = {
            "timestamp_ms": timestamp_ms,
            "poses": [],
            "faces": []
        }

        h, w, _ = frame_bgr.shape
        detection_idx = 0

        for box in results.boxes:
            cls_id = int(box.cls[0])
            
            if cls_id == 0:
                confidence = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                bbox = {
                    "x_min": x1 / w,
                    "y_min": y1 / h,
                    "x_max": x2 / w,
                    "y_max": y2 / h,
                    "confidence": confidence
                }

                structured_output["poses"].append({
                    "pose_id": detection_idx,
                    "bbox": bbox,
                    "landmarks": []
                })
                
                detection_idx += 1

        return structured_output

    def close(self):
        pass


if __name__ == "__main__":
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    VIDEO_PATH = os.path.join(SCRIPT_DIR, "sample-video.mp4")
    MODEL_PATH = os.path.join(SCRIPT_DIR, "yolov8n.pt")

    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        print(f"Error: Could not open video file {VIDEO_PATH}")
        exit()

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps is None:
        fps = 30.0

    frame_duration_ms = int(1000 / fps)
    pipeline = ClassroomDetectionPipeline(yolo_model_path=MODEL_PATH)

    frame_counter = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Finished processing all video frames.")
            break

        timestamp_ms = int(frame_counter * frame_duration_ms)
        frame_counter += 1

        detections = pipeline.process_frame(frame, timestamp_ms)

        h, w, _ = frame.shape
        for person in detections["poses"]:
            bbox = person["bbox"]
            x1, y1 = int(bbox["x_min"] * w), int(bbox["y_min"] * h)
            x2, y2 = int(bbox["x_max"] * w), int(bbox["y_max"] * h)
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame, 
                f"Person {bbox['confidence']:.2f}", 
                (x1, max(y1 - 10, 15)), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                (0, 255, 0), 
                2
            )

        print(f"Frame {frame_counter} | Timestamp: {timestamp_ms}ms | Persons Detected: {len(detections['poses'])}")

        cv2.imshow("Classroom YOLO Detection (Track 2 Test)", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    pipeline.close()
    cv2.destroyAllWindows()