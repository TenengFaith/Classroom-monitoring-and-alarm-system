"""Track 3 end-to-end smoke test with video-file support."""

import argparse
import sys
import time
import traceback
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from detection_model import ClassroomDetectionPipeline
    from track3.adapter import adapt
    from track3.trackers.centroid_tracker import CentroidTracker
    from track3.trackers.suspicion_tracker import StudentSuspicionTracker
except ImportError:  # pragma: no cover
    from ..detection_model import ClassroomDetectionPipeline
    from .adapter import adapt
    from .trackers.centroid_tracker import CentroidTracker
    from .trackers.suspicion_tracker import StudentSuspicionTracker


def draw_tracked(frame, bbox_norm, student_id, score=None, alert=False):
    h, w = frame.shape[:2]
    x1 = int(bbox_norm["x_min"] * w)
    y1 = int(bbox_norm["y_min"] * h)
    x2 = int(bbox_norm["x_max"] * w)
    y2 = int(bbox_norm["y_max"] * h)
    color = (0, 0, 255) if alert else (0, 255, 0)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    label = f"Student #{student_id}"
    if score is not None:
        label += f"  score={score:.2f}"
    cv2.putText(frame, label, (x1, max(y1 - 8, 20)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


def resolve_source(argv):
    if len(argv) < 2:
        return 0, False
    arg = argv[1]
    if arg.isdigit():
        return int(arg), False
    return arg, True


def parse_args():
    parser = argparse.ArgumentParser(description="Track 3 classroom detection smoke test")
    parser.add_argument("source", nargs="?", default=None, help="Video file path or camera index")
    parser.add_argument("--video", dest="video_path", help="Path to a recorded video file")
    parser.add_argument("--camera", type=int, default=0, help="Camera index to use when no video file is provided")
    parser.add_argument("--limit-frames", type=int, default=0, help="Stop after N frames for reproducible testing")
    return parser.parse_args()


def main():
    args = parse_args()
    source = args.video_path if args.video_path else args.source
    if source is None:
        source, is_file = resolve_source(sys.argv)
    elif source.isdigit():
        source = int(source)
        is_file = False
    else:
        is_file = True

    if source is None:
        source = args.camera
        is_file = False

    print(f"Opening source: {source!r} | file_mode={is_file}")
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

            if args.limit_frames and processed >= args.limit_frames:
                print(f"Frame limit reached ({args.limit_frames}).")
                break

            if is_file:
                timestamp_ms = int((frame_idx / fps_declared) * 1000)
            else:
                timestamp_ms = int((time.time() - start) * 1000)
            frame_idx += 1

            try:
                raw = pipeline.process_frame(frame, timestamp_ms)
                print(f"frame={frame_idx:5d} detection_count={len(raw.get('poses', []))} faces={len(raw.get('faces', []))}")

                records = adapt(raw)
                print(f"frame={frame_idx:5d} adapted_records={len(records)}")

                centroids = [r["centroid"] for r in records if r["centroid"] is not None]
                assignments = tracker.update(centroids)
                print(f"frame={frame_idx:5d} tracker_assignments={assignments}")

                idx_to_record = {}
                ci = 0
                for r in records:
                    if r["centroid"] is not None:
                        idx_to_record[ci] = r
                        ci += 1

                for det_idx, student_id in assignments.items():
                    seen_ids.add(student_id)
                    rec = idx_to_record[det_idx]
                    score, should_alert = suspicion.update(
                        student_id=student_id,
                        head_yaw_deg=rec["head_yaw_deg"],
                        torso_lean_deg=rec["torso_lean_deg"],
                        object_near_hand=rec["object_near_hand"],
                    )
                    print(f"frame={frame_idx:5d} student_id={student_id} score={score:.2f} alert={should_alert}")
                    draw_tracked(frame, rec["bbox"], student_id, score=score, alert=should_alert)
            except Exception:
                print(f"ERROR on frame {frame_idx}: ")
                traceback.print_exc()

            processed += 1
            elapsed = max(time.time() - start, 1e-6)
            fps = processed / elapsed
            cv2.putText(frame, f"FPS: {fps:.1f}  IDs seen: {len(seen_ids)}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

            cv2.imshow("Track 3 - Stable IDs", frame)
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
