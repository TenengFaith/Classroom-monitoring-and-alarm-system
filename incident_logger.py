import os
import csv
from datetime import datetime


class IncidentLogger:
    """
    Logs flagged suspicious incidents to CSV with debouncing per student.
    """
    def __init__(self, csv_filepath="classroom_alerts.csv", debounce_sec=5.0):
        self.csv_filepath = csv_filepath
        self.debounce_sec = debounce_sec
        self.last_logged = {}  # student_id -> timestamp

        # Create file with headers if it doesn't exist
        if not os.path.exists(self.csv_filepath):
            with open(self.csv_filepath, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "student_id", "suspicion_score", "head_yaw_deg", "torso_lean_deg", "object_near_hand"])

    def log_incident(self, student_id, score, head_yaw, torso_lean, object_near):
        """Logs incident if debounce window has passed for this student_id."""
        now = datetime.now()
        now_ts = now.timestamp()

        if student_id in self.last_logged:
            if now_ts - self.last_logged[student_id] < self.debounce_sec:
                return False

        self.last_logged[student_id] = now_ts
        str_time = now.strftime("%Y-%m-%d %H:%M:%S")

        with open(self.csv_filepath, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                str_time,
                student_id,
                round(float(score), 3),
                round(float(head_yaw), 1) if head_yaw is not None else "N/A",
                round(float(torso_lean), 1) if torso_lean is not None else "N/A",
                bool(object_near)
            ])

        print(f"[CSV LOGGED] Incident saved for Student #{student_id} at {str_time}")
        return True