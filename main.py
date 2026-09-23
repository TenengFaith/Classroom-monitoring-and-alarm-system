#!/usr/bin/env python3
"""
Classroom Monitoring & Alarm System
Main entry point for real-time cheating and suspicious behavior detection.
"""

import os
import sys
import time
import argparse
import cv2

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.capture import VideoStream
from src.detection import ClassroomDetectionPipeline
from src.tracking import adapt, CentroidTracker, StudentSuspicionTracker
from src.alert import Alarm, IncidentLogger
from src.ui import draw_flag_overlay, apply_spotlight, draw_dashboard


def parse_args():
    parser = argparse.ArgumentParser(
        description="Smart Classroom Monitoring & Cheating Detection System"
    )
    parser.add_argument(
        "--source", type=str, default="test_video/sample-video.mp4",
        help="Input video source: file path, camera index (e.g. 0), or IP URL."
    )
    parser.add_argument(
        "--yolo-model", type=str, default=os.path.join(PROJECT_ROOT, "models", "yolov8n.pt"),
        help="Path to YOLOv8 model file."
    )
    parser.add_argument(
        "--pose-model", type=str, default=os.path.join(PROJECT_ROOT, "models", "pose_landmarker_lite.task"),
        help="Path to MediaPipe Pose Landmarker task file."
    )
    parser.add_argument(
        "--face-model", type=str, default=os.path.join(PROJECT_ROOT, "models", "face_landmarker.task"),
        help="Path to MediaPipe Face Landmarker task file."
    )
    parser.add_argument(
        "--alarm-sound", type=str, default=os.path.join(PROJECT_ROOT, "assets", "alarm.mp3"),
        help="Path to alarm MP3 sound file."
    )
    parser.add_argument(
        "--output-csv", type=str, default="classroom_alerts.csv",
        help="Path to output CSV alert log file."
    )
    parser.add_argument(
        "--no-alarm", action="store_true",
        help="Disable audio alarm playback."
    )
    parser.add_argument(
        "--no-display", action="store_true",
        help="Run without displaying OpenCV window (headless mode)."
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # 1. Resolve video source fallback
    video_source = args.source
    if not video_source.isdigit() and not video_source.startswith("http") and not os.path.exists(video_source):
        fallback = os.path.join(PROJECT_ROOT, "test_video", os.path.basename(video_source))
        if os.path.exists(fallback):
            video_source = fallback

    print(f"[SYSTEM] Initializing stream source: {video_source}")
    stream = VideoStream(source=video_source)

    # 2. Initialize Detection Pipeline
    pipeline = ClassroomDetectionPipeline(
        yolo_model_path=args.yolo_model,
        pose_model_path=args.pose_model,
        face_model_path=args.face_model
    )

    # 3. Initialize Tracking Components
    centroid_tracker = CentroidTracker(max_distance=0.15, max_missed_frames=20)
    suspicion_tracker = StudentSuspicionTracker(threshold=0.6, required_frames=10)

    # 4. Initialize Alerting & Logging
    alarm = Alarm(sound_path=args.alarm_sound, cooldown_sec=2.0) if not args.no_alarm else None
    logger = IncidentLogger(csv_filepath=args.output_csv, debounce_sec=5.0)

    total_frames = 0
    start_time = time.time()
    declared_fps = stream.get_declared_fps()
    frame_duration_ms = int(1000 / declared_fps)

    print("[SYSTEM] Classroom Monitoring Pipeline Active. Press 'q' in video window to exit.\n")

    try:
        while True:
            ret, frame = stream.read()
            if not ret or frame is None:
                print("\n[INFO] End of stream or stream disconnected.")
                break

            total_frames += 1
            timestamp_ms = int(total_frames * frame_duration_ms)
            h_full, w_full = frame.shape[:2]

            # Step 1: Detect persons & objects via YOLOv8 + MediaPipe
            raw_detections = pipeline.process_frame(frame, timestamp_ms)

            # Step 2: Convert to normalized records & update centroid tracking
            records = adapt(raw_detections, full_w=w_full, full_h=h_full)
            centroids = [r["centroid"] for r in records if "centroid" in r]
            assignments = centroid_tracker.update(centroids)

            flagged_bboxes = []
            has_alarm_this_frame = False

            # Step 3: Evaluate student behavior & update suspicion scores
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

                    if alarm:
                        alarm.trigger()
                    logger.log_incident(student_id, score, yaw, lean, obj_near)

                draw_flag_overlay(frame, pixel_bbox, student_id, score, is_flagged=should_alert)

            # Step 4: Render UI dashboard & spotlight
            elapsed = time.time() - start_time
            fps = total_frames / max(elapsed, 1e-6)

            if flagged_bboxes:
                frame = apply_spotlight(frame, flagged_bboxes, dim_factor=0.3)

            draw_dashboard(frame, fps, len(assignments), has_alarm_this_frame)

            if not args.no_display:
                cv2.imshow("Classroom Monitoring System", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    finally:
        stream.release()
        pipeline.close()
        if not args.no_display:
            cv2.destroyAllWindows()

        elapsed_total = time.time() - start_time
        print(f"\n[SUMMARY] Processed {total_frames} frames in {elapsed_total:.1f}s ({total_frames / max(elapsed_total, 1e-6):.1f} FPS)")


if __name__ == "__main__":
    main()