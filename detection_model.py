import cv2
import os
import time
import numpy as np
import mediapipe as mp
from ultralytics import YOLO


class ClassroomDetectionPipeline:

    def __init__(self,
                 yolo_model_path="yolov8n.pt",
                 pose_model_path="pose_landmarker_lite.task",
                 face_model_path="face_landmarker.task",
                 confidence_threshold=0.15,  # Lowered threshold for small objects
                 crop_padding=0.05,
                 yolo_input_width=1280,      # Increased input resolution
                 crop_size_px=256,
                 landmark_every_n_frames=3):
        self.confidence_threshold = confidence_threshold
        self.crop_padding = crop_padding
        self.yolo_input_width = yolo_input_width
        self.crop_size_px = crop_size_px
        self.landmark_every_n_frames = landmark_every_n_frames

        self.yolo = YOLO(yolo_model_path)

        BaseOptions = mp.tasks.BaseOptions
        PoseLandmarker = mp.tasks.vision.PoseLandmarker
        PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
        FaceLandmarker = mp.tasks.vision.FaceLandmarker
        FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
        RunningMode = mp.tasks.vision.RunningMode

        pose_options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=pose_model_path),
            running_mode=RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.pose_detector = PoseLandmarker.create_from_options(pose_options)

        face_options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=face_model_path),
            running_mode=RunningMode.VIDEO,
            num_faces=1,
            output_face_blendshapes=False,
            min_face_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.face_detector = FaceLandmarker.create_from_options(face_options)

        self._frame_count = 0

    def _crop_with_padding(self, frame, x1, y1, x2, y2):
        h, w = frame.shape[:2]
        pw = int((x2 - x1) * self.crop_padding)
        ph = int((y2 - y1) * self.crop_padding)
        cx1 = max(x1 - pw, 0)
        cy1 = max(y1 - ph, 0)
        cx2 = min(x2 + pw, w)
        cy2 = min(y2 + ph, h)
        if cx2 <= cx1 or cy2 <= cy1:
            return None, (0, 0)
        return frame[cy1:cy2, cx1:cx2], (cx1, cy1)

    def _run_landmarks(self, crop, crop_ts):
        h, w = crop.shape[:2]
        if w > self.crop_size_px:
            scale = self.crop_size_px / w
            crop = cv2.resize(crop,
                              (self.crop_size_px, int(h * scale)),
                              interpolation=cv2.INTER_AREA)

        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        pose_out = []
        face_out = []

        try:
            pr = self.pose_detector.detect_for_video(mp_image, crop_ts)
            if pr and pr.pose_landmarks:
                for lm in pr.pose_landmarks[0]:
                    pose_out.append({
                        "x": lm.x, "y": lm.y, "z": lm.z,
                        "visibility": getattr(lm, "visibility", 1.0),
                    })
        except Exception:
            pass

        try:
            fr = self.face_detector.detect_for_video(mp_image, crop_ts)
            if fr and fr.face_landmarks:
                for lm in fr.face_landmarks[0]:
                    face_out.append({"x": lm.x, "y": lm.y, "z": lm.z})
        except Exception:
            pass

        return pose_out, face_out

    def check_phone_posture(self, landmarks, crop_offset, crop_size, frame_shape):
        """
        Detects if head is bent forward and hands (wrists 15, 16) are close together.
        Returns a bounding box around the hands if true.
        """
        if len(landmarks) < 17:
            return None

        # MediaPipe Keypoints: 0=Nose, 11=L_Shoulder, 12=R_Shoulder, 15=L_Wrist, 16=R_Wrist
        nose = landmarks[0]
        l_sh = landmarks[11]
        r_sh = landmarks[12]
        l_wrist = landmarks[15]
        r_wrist = landmarks[16]

        shoulder_y = (l_sh['y'] + r_sh['y']) / 2.0

        # Check if head is lowered towards chest
        head_bent = nose['y'] > (shoulder_y - 0.05)

        # Check wrist proximity (normalized to crop)
        wrist_dist = np.sqrt((l_wrist['x'] - r_wrist['x'])**2 + (l_wrist['y'] - r_wrist['y'])**2)
        hands_together = wrist_dist < 0.25

        if head_bent and hands_together:
            cx1, cy1 = crop_offset
            cw, ch = crop_size
            fw, fh = frame_shape[1], frame_shape[0]

            # Calculate hand bounding box in absolute pixels
            hx1 = int(cx1 + min(l_wrist['x'], r_wrist['x']) * cw - 20)
            hy1 = int(cy1 + min(l_wrist['y'], r_wrist['y']) * ch - 20)
            hx2 = int(cx1 + max(l_wrist['x'], r_wrist['x']) * cw + 20)
            hy2 = int(cy1 + max(l_wrist['y'], r_wrist['y']) * ch + 20)

            # Normalize bounding box format
            return {
                "x_min": max(0, hx1) / fw,
                "y_min": max(0, hy1) / fh,
                "x_max": min(fw, hx2) / fw,
                "y_max": min(fh, hy2) / fh,
                "confidence": 0.85,
                "heuristic": True
            }
        return None

    def process_frame(self, frame_bgr, timestamp_ms):
        self._frame_count += 1
        do_landmarks = (self._frame_count % self.landmark_every_n_frames == 0)

        h_full, w_full = frame_bgr.shape[:2]

        if w_full > self.yolo_input_width:
            scale = self.yolo_input_width / w_full
            yolo_input = cv2.resize(
                frame_bgr,
                (self.yolo_input_width, int(h_full * scale)),
                interpolation=cv2.INTER_AREA,
            )
        else:
            yolo_input = frame_bgr

        # Lowered confidence to catch small/occluded phone boxes
        results = self.yolo(yolo_input, conf=self.confidence_threshold,
                            classes=[0, 67], verbose=False)[0]

        yolo_h, yolo_w = yolo_input.shape[:2]

        structured = {
            "timestamp_ms": timestamp_ms,
            "poses": [],
            "faces": [],
            "objects": [],
        }

        person_idx = 0
        object_idx = 0

        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            x1 = x1 * w_full / yolo_w
            x2 = x2 * w_full / yolo_w
            y1 = y1 * h_full / yolo_h
            y2 = y2 * h_full / yolo_h

            bbox_norm = {
                "x_min": x1 / w_full,
                "y_min": y1 / h_full,
                "x_max": x2 / w_full,
                "y_max": y2 / h_full,
                "confidence": conf,
            }

            if cls_id == 0:
                crop, (cx1, cy1) = self._crop_with_padding(
                    frame_bgr, int(x1), int(y1), int(x2), int(y2))
                if crop is None or crop.size == 0:
                    continue

                crop_h, crop_w = crop.shape[:2]
                crop_ts = timestamp_ms + person_idx

                if do_landmarks:
                    pose_lm, face_lm = self._run_landmarks(crop, crop_ts)
                else:
                    pose_lm, face_lm = [], []

                # Fallback check for bent-head phone usage
                phone_box = self.check_phone_posture(
                    pose_lm, (cx1, cy1), (crop_w, crop_h), frame_bgr.shape
                )
                if phone_box:
                    structured["objects"].append({
                        "object_id": object_idx,
                        "type": "cell phone",
                        "bbox": phone_box
                    })
                    object_idx += 1

                structured["poses"].append({
                    "pose_id": person_idx,
                    "bbox": bbox_norm,
                    "landmarks": pose_lm,
                    "face_landmarks": face_lm,
                    "crop_offset": (cx1, cy1),
                    "crop_size": (crop_w, crop_h),
                })

                if face_lm:
                    structured["faces"].append({
                        "face_id": person_idx,
                        "landmarks": face_lm,
                        "blendshapes": {},
                    })

                person_idx += 1

            elif cls_id == 67:
                structured["objects"].append({
                    "object_id": object_idx,
                    "type": "cell phone",
                    "bbox": bbox_norm,
                })
                object_idx += 1

        return structured

    def close(self):
        self.pose_detector.close()
        self.face_detector.close()


if __name__ == "__main__":
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    VIDEO_PATH = os.path.join(SCRIPT_DIR, "test_video/sample-video.mp4")
    YOLO_PATH = os.path.join(SCRIPT_DIR, "yolov8n.pt")
    POSE_PATH = os.path.join(SCRIPT_DIR, "pose_landmarker_lite.task")
    FACE_PATH = os.path.join(SCRIPT_DIR, "face_landmarker.task")

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Could not open {VIDEO_PATH}")
        exit()

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_duration_ms = int(1000 / fps)

    pipeline = ClassroomDetectionPipeline(
        yolo_model_path=YOLO_PATH,
        pose_model_path=POSE_PATH,
        face_model_path=FACE_PATH,
    )

    frame_counter = 0
    t0 = time.time()
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            ts = int(frame_counter * frame_duration_ms)
            frame_counter += 1

            det = pipeline.process_frame(frame, ts)

            fh, fw = frame.shape[:2]

            # Draw Persons (Green)
            for person in det["poses"]:
                b = person["bbox"]
                x1 = int(b["x_min"] * fw)
                y1 = int(b["y_min"] * fh)
                x2 = int(b["x_max"] * fw)
                y2 = int(b["y_max"] * fh)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw Cell Phones (Purple)
            for obj in det["objects"]:
                b = obj["bbox"]
                x1 = int(b["x_min"] * fw)
                y1 = int(b["y_min"] * fh)
                x2 = int(b["x_max"] * fw)
                y2 = int(b["y_max"] * fh)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
                label = "Phone (Posture)" if b.get("heuristic") else f"Phone {b['confidence']:.2f}"
                cv2.putText(frame, label, (x1, max(y1 - 10, 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

            cv2.imshow("Classroom Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        pipeline.close()
        cv2.destroyAllWindows()