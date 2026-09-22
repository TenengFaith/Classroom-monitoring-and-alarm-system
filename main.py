import os
import sys
import time
import csv
import cv2

# Import Core Pipeline and Track 3
from detection_model import ClassroomDetectionPipeline
from track3.adapter import adapt
from track3.trackers.centroid_tracker import CentroidTracker
from track3.trackers.suspicion_tracker import StudentSuspicionTracker

# Import Track 4 Modules directly from Root
from alarm import AlarmManager
from overlay import draw_dashboard_hud, draw_student_overlay

ALERT_SCORE_THRESHOLD = 0.60  # Trigger threshold for flagging and audio alarm


class IncidentLogger:
    """Logs flagged suspicious events directly to classroom_alerts.csv."""
    def __init__(self, log_filename="classroom_alerts.csv"):
        self.log_filename = log_filename
        self._ensure_header()

    def _ensure_header(self):
        if not os.path.exists(self.log_filename):
            with open(self.log_filename, mode="w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Timestamp_MS", "Student_ID", "Suspicion_Score", "Status"])

    def log_event(self, timestamp_ms, student_id, score):
        with open(self.log_filename, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([timestamp_ms, student_id, f"{score:.2f}", "FLAGGED_ALERT"])


def resolve_source(argv):
    if len(argv) < 2:
        return 0, False
    arg = argv[1]
    if arg.isdigit():
        return int(arg), False
    return arg, True


def main():
    source, is_file = resolve_source(sys.argv)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[System Error] Could not open video source {source!r}")
        return

    fps_declared = cap.get(cv2.CAP_PROP_FPS)
    if not fps_declared or fps_declared != fps_declared or fps_declared <= 0:
        fps_declared = 30.0

    # Initialize Detection Pipeline & Track 3 Trackers
    pipeline = ClassroomDetectionPipeline()
    tracker = CentroidTracker(max_distance=0.15, max_missed_frames=20)
    suspicion = StudentSuspicionTracker(threshold=ALERT_SCORE_THRESHOLD, required_frames=3)

    # Initialize Alarm Manager & Logger
    alarm_manager = AlarmManager(cooldown_sec=1.5)
    logger = IncidentLogger(log_filename="classroom_alerts.csv")

    frame_idx = 0
    processed = 0
    start_time = time.time()
    seen_ids = set()

    print("[System Active] Monitoring classroom for suspicious behavior >= 0.60...")

    try:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                print("[System] Stream completed.")
                break

            if is_file:
                timestamp_ms = int((frame_idx / fps_declared) * 1000)
            else:
                timestamp_ms = int((time.time() - start_time) * 1000)
            frame_idx += 1

            # 1. Feature Detection Pipeline & Pose Adapter
            raw_detections = pipeline.process_frame(frame, timestamp_ms)
            records = adapt(raw_detections)

            # 2. Track 3: Assign Unique IDs via Centroid Tracker
            centroids = [r["centroid"] for r in records if r["centroid"] is not None]
            assignments = tracker.update(centroids)

            idx_to_record = {}
            ci = 0
            for r in records:
                if r["centroid"] is not None:
                    idx_to_record[ci] = r
                    ci += 1

            flagged_students_this_frame = False

            # 3. Track 3 Suspicion Calculation & Flagging
            for det_idx, student_id in assignments.items():
                seen_ids.add(student_id)
                rec = idx_to_record[det_idx]

                # Update suspicion tracker for student ID
                score, should_alert = suspicion.update(
                    student_id=student_id,
                    head_yaw_deg=rec["head_yaw_deg"],
                    torso_lean_deg=rec["torso_lean_deg"],
                    object_near_hand=rec["object_near_hand"],
                )

                # Check if student crosses threshold (0.60 or higher)
                is_flagged = should_alert or (score >= ALERT_SCORE_THRESHOLD)

                if is_flagged:
                    flagged_students_this_frame = True
                    logger.log_event(timestamp_ms, student_id, score)

                # Draw student overlay (Red if flagged, Orange/Green otherwise)
                draw_student_overlay(
                    frame=frame,
                    bbox_norm=rec["bbox"],
                    student_id=student_id,
                    score=score,
                    alert=is_flagged,
                )

            # 4. Trigger Non-blocking Alarm Sound if any student is flagged
            if flagged_students_this_frame:
                alarm_manager.trigger()

            # 5. Render HUD Dashboard Banner
            processed += 1
            elapsed = max(time.time() - start_time, 1e-6)
            fps = processed / elapsed

            draw_dashboard_hud(
                frame=frame,
                fps=fps,
                active_count=len(assignments),
                total_ids=len(seen_ids),
                has_alert=flagged_students_this_frame,
            )

            cv2.imshow("Smart Classroom Monitoring & Alert System", frame)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break

    finally:
        cap.release()
        pipeline.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()