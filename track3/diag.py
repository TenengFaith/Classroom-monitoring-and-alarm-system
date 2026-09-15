
import os
import sys
import time

import cv2

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from detection_model import ClassroomDetectionPipeline
from track3.adapter import adapt
from track3.trackers.centroid_tracker import CentroidTracker


def main():
    source = sys.argv[1] if len(sys.argv) > 1 else 0
    if isinstance(source, str) and source.isdigit():
        source = int(source)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Could not open: {source}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    pipeline = ClassroomDetectionPipeline()
    tracker = CentroidTracker(max_distance=0.15, max_missed_frames=20)

    frame_idx = 0
    last_id_count = 0

    try:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break

            ts = int((frame_idx / fps) * 1000)
            frame_idx += 1

            raw = pipeline.process_frame(frame, ts)
            records = adapt(raw)
            centroids = [r["centroid"] for r in records if r["centroid"] is not None]
            assignments = tracker.update(centroids)

            live_ids = len(tracker.tracked)
            total_ids = tracker.next_id
            if total_ids != last_id_count or frame_idx % 30 == 0:
                print(f"frame={frame_idx:5d}  detections={len(centroids)}  "
                      f"live_ids={live_ids}  total_ids_ever={total_ids}")
                last_id_count = total_ids
    finally:
        cap.release()
        pipeline.close()
        print(f"\nFinal: total IDs ever created = {tracker.next_id}")


if __name__ == "__main__":
    main()