import os
import cv2
import csv
from datetime import datetime


class IncidentLogger:
    """
    Logs flagged exam cheating incidents to CSV and saves annotated frame snapshots.
    Enforces a 5-second per-student pause after an incident is flagged to prevent duplicate logs.
    """
    def __init__(self, csv_filepath="classroom_alerts.csv", output_dir="flagged_incidents", cooldown_sec=5.0):
        self.csv_filepath = csv_filepath
        self.output_dir = output_dir
        self.cooldown_sec = cooldown_sec
        self.last_flagged_time = {}  # student_id -> float (timestamp)

        os.makedirs(self.output_dir, exist_ok=True)

        # Initialize CSV header if file doesn't exist
        if not os.path.exists(self.csv_filepath):
            with open(self.csv_filepath, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp",
                    "student_id",
                    "suspicion_score",
                    "cheat_reason",
                    "head_pitch_deg",
                    "head_yaw_deg",
                    "mouth_open_ratio",
                    "torso_lean_deg",
                    "phone_detected",
                    "snapshot_path"
                ])

    def can_flag_student(self, student_id):
        """Checks if 5-second cooldown period has elapsed for this student."""
        now_ts = datetime.now().timestamp()
        if student_id in self.last_flagged_time:
            if now_ts - self.last_flagged_time[student_id] < self.cooldown_sec:
                return False
        return True

    def log_and_save_snapshot(self, frame, student_id, score, cheat_reason, pixel_bbox, metrics=None):
        """
        Pauses alerts for 5 seconds for student_id, draws annotation on frame,
        saves snapshot image file, and records incident in CSV log.
        """
        now = datetime.now()
        now_ts = now.timestamp()

        # Enforce 5-second tracking pause / cooldown
        if student_id in self.last_flagged_time:
            if now_ts - self.last_flagged_time[student_id] < self.cooldown_sec:
                return False, None

        self.last_flagged_time[student_id] = now_ts
        str_time = now.strftime("%Y-%m-%d %H:%M:%S")
        file_ts = now.strftime("%Y%m%d_%H%M%S_%f")[:19]

        metrics = metrics or {}
        pitch = metrics.get("head_pitch_deg")
        yaw = metrics.get("head_yaw_deg")
        mouth = metrics.get("mouth_open_ratio")
        lean = metrics.get("torso_lean_deg")
        phone = metrics.get("object_near_hand", False)

        # Create annotated snapshot copy of frame
        annotated_frame = frame.copy()
        x1, y1, x2, y2 = pixel_bbox

        # Highlight flagged student in red with incident details
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)

        badge_text = f"FLAGGED Student #{student_id}: {cheat_reason} ({score:.2f})"
        cv2.rectangle(annotated_frame, (x1, max(0, y1 - 30)), (x1 + 450, max(30, y1)), (0, 0, 255), -1)
        cv2.putText(annotated_frame, badge_text, (x1 + 5, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        # Stamp timestamp at bottom
        cv2.putText(annotated_frame, f"Time: {str_time}", (20, annotated_frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Save snapshot file
        filename = f"incident_student_{student_id}_{file_ts}.jpg"
        snapshot_path = os.path.join(self.output_dir, filename)
        cv2.imwrite(snapshot_path, annotated_frame)

        # Write to CSV log
        with open(self.csv_filepath, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                str_time,
                student_id,
                round(float(score), 3),
                cheat_reason,
                round(float(pitch), 1) if pitch is not None else "N/A",
                round(float(yaw), 1) if yaw is not None else "N/A",
                round(float(mouth), 3) if mouth is not None else "N/A",
                round(float(lean), 1) if lean is not None else "N/A",
                bool(phone),
                snapshot_path
            ])

        print(f"[INCIDENT FLAGGED] Student #{student_id} | Reason: '{cheat_reason}' | Snapshot saved: {snapshot_path} (5s pause active)")
        return True, snapshot_path
