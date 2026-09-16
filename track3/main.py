"""
Track 3 & Track 4 Integrated Pipeline
"""

import os
import sys
import time
import cv2

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from detection_model import ClassroomDetectionPipeline
from track3.adapter import adapt
from track3.trackers.centroid_tracker import CentroidTracker
from track3.trackers.suspicion_tracker import StudentSuspicionTracker

# Track 4 Imports
from track4 import Alarm, AlertLogger, draw_flag_overlay, draw_dashboard_hud, spotlight_bbox


def draw_tracked(frame, bbox_norm, student_id, score=None, alert=False):
    h, w = frame.shape[:2]
    x1 = int(bbox_norm["x_min"] * w)
    y1 = int(bbox_norm["y_min"] * h)
    x2 = int(bbox_norm["x_max"] * w)
    y2 = int(bbox_norm["y_max"] * h)
    
    # Standard Track 3 box for normal students
    if not alert:
        color = (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"Student #{student_id}"
        if score is not None:
            label += f" score={score:.2f}"
        cv2.putText(frame, label, (x1, max(y1 - 8, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    else:
        # Track 4 Flag Overlay (Red Cross Box + FLAGGED label)
        draw_flag_overlay(frame, (x1, y1, x2, y2), student_id, score)


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
        print(f"Could not open source: {source!r}")
        return

    fps_declared = cap.get(cv2.CAP_PROP_FPS)
    if not fps_declared or fps_declared != fps_declared or fps_declared <= 0:
        fps_declared = 30.0

    pipeline = ClassroomDetectionPipeline()
    tracker = CentroidTracker(max_distance=0.15, max_missed_frames=20)
    suspicion = StudentSuspicionTracker(threshold=0.6, required_frames=15)

    # Initialize Track 4 Components
    alarm = Alarm(sound_path="alarm.mp3", cooldown_sec=3)
    logger = AlertLogger(log_filepath="classroom_alerts.csv")

    frame_idx = 0
    processed = 0
    start = time.time()
    seen_ids = set()

    try:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                print("End of stream.")
                break

            if is_file:
                timestamp_ms = int((frame_idx / fps_declared) * 1000)
            else:
                timestamp_ms = int((time.time() - start) * 1000)
            frame_idx += 1

            raw = pipeline.process_frame(frame, timestamp_ms)
            records = adapt(raw)

            centroids = [r["centroid"] for r in records if r["centroid"] is not None]
            assignments = tracker.update(centroids)

            idx_to_record = {}
            ci = 0
            for r in records:
                if r["centroid"] is not None:
                    idx_to_record[ci] = r
                    ci += 1

            # Track live scores for Track 4 Dashboard HUD
            active_student_scores = {}

            for det_idx, student_id in assignments.items():
                seen_ids.add(student_id)
                rec = idx_to_record[det_idx]
                score, should_alert = suspicion.update(
                    student_id=student_id,
                    head_yaw_deg=rec["head_yaw_deg"],
                    torso_lean_deg=rec["torso_lean_deg"],
                    object_near_hand=rec["object_near_hand"],
                )
                
                active_student_scores[student_id] = score

                # --- TRACK 4 ALERT HANDLING ---
                if should_alert:
                    # 1. Trigger background non-blocking alarm audio
                    alarm.trigger()
                    # 2. Log alert timestamp and details to CSV
                    logger.log_alert(student_id, score)

                # 3. Draw tracked student box or flagged red cross overlay
                draw_tracked(frame, rec["bbox"], student_id,
                             score=score, alert=should_alert)

            processed += 1
            elapsed = max(time.time() - start, 1e-6)
            fps = processed / elapsed

            # --- TRACK 4 HUD OVERLAY ---
            draw_dashboard_hud(frame, active_student_scores, fps=fps, total_seen=len(seen_ids))

            cv2.imshow("Classroom Monitoring Dashboard (Track 3 + Track 4)", frame)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break
    finally:
        cap.release()
        pipeline.close()
        cv2.destroyAllWindows()
        elapsed = max(time.time() - start, 1e-6)
        print("")
        print("Summary:")
        print(f"  Frames processed  : {processed}")
        print(f"  Wall time (s)     : {elapsed:.1f}")
        print(f"  Avg FPS           : {processed / elapsed:.1f}")
        print(f"  Unique student IDs: {len(seen_ids)}")
        print(f"  IDs               : {sorted(seen_ids)}")


if __name__ == "__main__":
    main()