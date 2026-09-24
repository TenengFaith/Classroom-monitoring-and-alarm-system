# Exam Hall Cheating Detection System - Future Improvement Recommendations

This report outlines key technical architecture recommendations and roadmap enhancements to improve system accuracy, robustness, and performance in real exam hall environments.

---

## 1. Computer Vision & Model Enhancements

### A. Custom Fine-Tuned Object Detection Model
- **Current Limitation**: Pre-trained COCO YOLOv8n only detects general objects (`cell phone`, `person`).
- **Recommendation**: Fine-tune YOLOv8 / YOLOv11 on a dedicated **Exam Cheating Dataset**:
  - Class labels: `cheat_sheet`, `micro_earbud`, `smartwatch`, `phone_under_desk`, `paper_passing`.
  - Enables detecting small unauthorized materials hidden under desk surfaces.

### B. 3D Head Pose Estimation via Perspective-n-Point (PnP)
- **Current Limitation**: Heuristic 2D landmark ratios for head pitch and yaw.
- **Recommendation**: Implement OpenCV `cv2.solvePnP` using 3D canonical face mesh keypoints:
  - Computes exact 3D Euler angles (**Pitch**, **Yaw**, **Roll**) in physical degrees.
  - Precisely measures downwards gaze angle when students look directly into their lap or under the desk.

---

## 2. Tracking & Spatial Mapping

### A. Advanced Multi-Object Tracking (ByteTrack / DeepSORT)
- **Current Limitation**: Centroid tracker can lose identity during heavy occlusions or when a student leans behind a desk.
- **Recommendation**: Implement **ByteTrack** or **DeepSORT** with Re-Identification (ReID) feature embeddings:
  - Retains 100% consistent student IDs even during long occlusions or rapid movements.

### B. Classroom Desk Grid Spatial Mapping
- **Current Limitation**: Students are tagged with dynamic integer IDs (`ID #1`, `ID #2`).
- **Recommendation**: Map the camera perspective matrix to a 2D classroom seating plan (e.g. `Row 2 - Desk 4`):
  - Automatically associates flagged incidents with specific seat numbers on the exam seating chart.

---

## 3. Multi-Modal Behavior & Temporal Analysis

### A. Temporal Action Recognition (3D CNNs / Transformers)
- **Current Limitation**: Frame-by-frame static landmark analysis.
- **Recommendation**: Use 16-frame temporal sequence models (e.g., VideoMAE or Action Recognition Transformers):
  - Recognizes dynamic cheating actions over time (e.g., reaching into pockets, passing notes under desks, repetitive glancing at peer's paper).

### B. Directional Audio Anomaly Detection
- **Current Limitation**: Vision-only monitoring.
- **Recommendation**: Integrate audio input with sound classification (e.g. YAMNet):
  - Flags acoustic anomalies like whispering, page rustling, or covert speech during quiet exam conditions.

---

## 4. Hardware & Production Deployment

### A. TensorRT / ONNX Runtime GPU Optimization
- **Recommendation**: Export PyTorch YOLO models to **ONNX / TensorRT FP16**:
  - Increases processing speeds from ~25-30 FPS to **100+ FPS** per stream, allowing 4-8 IP camera streams to run on a single edge GPU (e.g. NVIDIA Jetson or RTX GPU).
