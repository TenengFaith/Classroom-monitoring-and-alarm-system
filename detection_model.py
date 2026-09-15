import cv2
import mediapipe as mp
import time

class ClassroomDetectionPipeline:
    def __init__(self, pose_model_path="pose_landmarker_full.task", face_model_path="face_landmarker.task"):
        BaseOptions = mp.tasks.BaseOptions
        
        PoseLandmarker = mp.tasks.vision.PoseLandmarker
        PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
        
        FaceLandmarker = mp.tasks.vision.FaceLandmarker
        FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
        
        RunningMode = mp.tasks.vision.RunningMode

        pose_options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=pose_model_path),
            running_mode=RunningMode.VIDEO,
            num_poses=10,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.pose_detector = PoseLandmarker.create_from_options(pose_options)

        face_options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=face_model_path),
            running_mode=RunningMode.VIDEO,
            num_faces=10,
            output_face_blendshapes=True,
            min_face_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.face_detector = FaceLandmarker.create_from_options(face_options)

    def process_frame(self, frame_bgr, timestamp_ms):
        rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        pose_result = self.pose_detector.detect_for_video(mp_image, timestamp_ms)
        face_result = self.face_detector.detect_for_video(mp_image, timestamp_ms)

        structured_output = {
            "timestamp_ms": timestamp_ms,
            "poses": [],
            "faces": []
        }

        if pose_result and pose_result.pose_landmarks:
            for pose_idx, landmarks in enumerate(pose_result.pose_landmarks):
                landmarks_list = [
                    {"x": lm.x, "y": lm.y, "z": lm.z, "visibility": lm.visibility}
                    for lm in landmarks
                ]
                
                xs = [lm.x for lm in landmarks]
                ys = [lm.y for lm in landmarks]
                bbox = {
                    "x_min": min(xs),
                    "y_min": min(ys),
                    "x_max": max(xs),
                    "y_max": max(ys)
                }

                structured_output["poses"].append({
                    "pose_id": pose_idx,
                    "bbox": bbox,
                    "landmarks": landmarks_list
                })

        if face_result and face_result.face_landmarks:
            for face_idx, landmarks in enumerate(face_result.face_landmarks):
                landmarks_list = [
                    {"x": lm.x, "y": lm.y, "z": lm.z}
                    for lm in landmarks
                ]

                blendshapes_dict = {}
                if face_result.face_blendshapes and face_idx < len(face_result.face_blendshapes):
                    blendshapes_dict = {
                        category.category_name: category.score
                        for category in face_result.face_blendshapes[face_idx]
                    }

                structured_output["faces"].append({
                    "face_id": face_idx,
                    "landmarks": landmarks_list,
                    "blendshapes": blendshapes_dict
                })

        return structured_output

    def close(self):
        self.pose_detector.close()
        self.face_detector.close()


if __name__ == "__main__":
    VIDEO_PATH = "sample-video.mp4"

    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        print(f"Error: Could not open video file {VIDEO_PATH}")
        exit()

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps is None:
        fps = 30.0

    frame_duration_ms = int(1000 / fps)
    
    pipeline = ClassroomDetectionPipeline()

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
        for pose in detections["poses"]:
            bbox = pose["bbox"]
            x1, y1 = int(bbox["x_min"] * w), int(bbox["y_min"] * h)
            x2, y2 = int(bbox["x_max"] * w), int(bbox["y_max"] * h)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        print(f"Frame {frame_counter} | Timestamp: {timestamp_ms}ms | Poses: {len(detections['poses'])} | Faces: {len(detections['faces'])}")

        cv2.imshow("Classroom Video Processing (Track 2 Test)", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    pipeline.close()
    cv2.destroyAllWindows()