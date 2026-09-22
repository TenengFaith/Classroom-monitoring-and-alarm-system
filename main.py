import os
import sys
import time
import cv2

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Track 2 Detection Pipeline
from detection_model import ClassroomDetectionPipeline

# Track 3 Tracking & Suspicion Modules
from track3.adapter import adapt
from track3.trackers.centroid_tracker import CentroidTracker
from track3.trackers.suspicion_tracker import StudentSuspicionTracker

# Track 4 System Components
from alarm import Alarm
from incident_logger import IncidentLogger
from overlay import draw_flag_overlay, apply_spotlight, draw_dashboard


def main():
    video_path = sys.argv[1] if len(sys.argv) > 1 else "sample-video.mp4"
    if not os.path.exists(video_path):
        # Fallback search in test_video directory
        video_path = os.path.join("test_video", os.path.basename(video_path))

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video source: {video_path}")
        return

    fps_input = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_duration_ms = int(1000 / fps_input)

    # Initialize Track 2 Pipeline
    pipeline = ClassroomDetectionPipeline(
        yolo_model_path="yolov8n.pt",
        pose_model_path="pose_landmarker_lite.task",
        face_model_path="face_landmarker.task"
    )

    # Initialize Track 3 Trackers
    centroid_tracker = CentroidTracker(max_distance=0.15, max_missed_frames=20)
    suspicion_tracker = StudentSuspicionTracker(threshold=0.6, required_frames=10)

    # Initialize Track 4 Alarm & Logger
    alarm = Alarm(sound_path="alarm.mp3", cooldown_sec=2.0)
    logger = IncidentLogger(csv_filepath="classroom_alerts.csv", debounce_sec=5.0)

    total_frames = 0
    t0 = time.time()
    fps = 0.0

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None:
                print("\n[INFO] End of video stream.")
                break

            total_frames += 1
            timestamp_ms = int(total_frames * frame_duration_ms)

            h_full, w_full = frame.shape[:2]

            # 1. Run Track 2 Detection
            raw_detections = pipeline.process_frame(frame, timestamp_ms)

            # 2. Run Track 3 Adapter & Centroid Matching
            records = adapt(raw_detections, full_w=w_full, full_h=h_full)
            centroids = [r["centroid"] for r in records if "centroid" in r]
            assignments = centroid_tracker.update(centroids)

            flagged_bboxes = []
            has_alarm_this_frame = False

            # 3. Update Suspicion & Track 4 System per Student
            for det_idx, student_id in assignments.items():
                if det_idx >= len(records):
                    continue

                rec = records[det_idx]
                yaw = rec.get("head_yaw_deg")
                lean = rec.get("torso_lean_deg")
                obj_near = rec.get("object_near_hand", False)

                score, should_alert = suspicion_tracker.update(
                    student_id=student_id,
                    head_yaw_deg=yaw,
                    torso_lean_deg=lean,
                    object_near_hand=obj_near
                )

                # Get pixel coordinates for bounding box
                raw_bbox = rec.get("bbox")
                if isinstance(raw_bbox, dict):
                    pixel_bbox = [
                        int(raw_bbox["x_min"] * w_full),
                        int(raw_bbox["y_min"] * h_full),
                        int(raw_bbox["x_max"] * w_full),
                        int(raw_bbox["y_max"] * h_full)
                    ]
                else:
                    cx, cy = rec["centroid"]
                    pixel_bbox = [
                        int((cx - 0.08) * w_full),
                        int((cy - 0.15) * h_full),
                        int((cx + 0.08) * w_full),
                        int((cy + 0.15) * h_full)
                    ]

                if should_alert:
                    has_alarm_this_frame = True
                    flagged_bboxes.append(pixel_bbox)

                    # Trigger Track 4 Sound Alarm & CSV Logging
                    alarm.trigger()
                    logger.log_incident(student_id, score, yaw, lean, obj_near)

                draw_flag_overlay(frame, pixel_bbox, student_id, score, is_flagged=should_alert)

            # 4. Calculate FPS and render overlays
            elapsed = time.time() - t0
            fps = total_frames / max(elapsed, 1e-6)

            if flagged_bboxes:
                frame = apply_spotlight(frame, flagged_bboxes, dim_factor=0.3)

            draw_dashboard(frame, fps, len(assignments), has_alarm_this_frame)

            cv2.imshow("Smart Classroom Monitoring System", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        pipeline.close()
        cv2.destroyAllWindows()
        print(f"\nFinal Summary: Processed {total_frames} frames in {time.time()-t0:.1f}s ({fps:.1f} FPS)")


if __name__ == "__main__":
    main()