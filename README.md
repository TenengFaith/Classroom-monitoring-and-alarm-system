# Smart Classroom Monitoring & Alarm System

A real-time vision-based classroom monitoring and automated alarm system for identifying suspicious student behaviors and potential cheating incidents (e.g., turning head excessively, leaning towards peers, using mobile phones).

## Project Architecture

```
Classroom-monitoring-and-alarm-system/
├── assets/
│   └── alarm.mp3                   # Audio alert sound file
├── models/
│   ├── face_landmarker.task        # MediaPipe Face Landmarker model
│   ├── pose_landmarker_lite.task   # MediaPipe Pose Landmarker model
│   ├── pose_landmarker_full.task   # MediaPipe Pose Landmarker (optional high quality)
│   └── yolov8n.pt                  # YOLOv8 object & person detection model
├── src/
│   ├── capture/                    # Video stream capture (webcam, file, IP camera)
│   ├── detection/                  # YOLOv8 + MediaPipe detection pipeline
│   ├── tracking/                   # Centroid tracker & suspicion scoring algorithm
│   ├── alert/                      # Audio alarm trigger & CSV incident logger
│   └── ui/                         # Visual overlays, HUD dashboard & spotlight renderer
├── test_video/
│   └── sample-video.mp4            # Sample test video
├── main.py                         # Unified CLI entry point
├── README.md                       # Project documentation
└── requirements.txt                # Python dependencies
```

## Setup Instructions

1. **Activate virtual environment**:
   ```bash
   python -m venv venv
   venv\Scripts\activate      # Windows
   # source venv/bin/activate # macOS / Linux
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Models & Assets**:
   Model files (`.task`, `.pt`) are stored in the `models/` directory, and sound assets are stored in `assets/`.

## Running the System

Run real-time monitoring on a video file or camera feed using `main.py`:

- **Run with sample video file**:
  ```bash
  python main.py --source test_video/sample-video.mp4
  ```

- **Run with default webcam**:
  ```bash
  python main.py --source 0
  ```

- **Run with IP / Phone Camera URL**:
  ```bash
  python main.py --source "http://192.168.1.100:8080/video"
  ```

- **Headless mode (no OpenCV display window)**:
  ```bash
  python main.py --source test_video/sample-video.mp4 --no-display
  ```

## CLI Options

| Argument | Description | Default |
|---|---|---|
| `--source` | Path to video file, camera index (e.g., 0), or IP camera URL | `test_video/sample-video.mp4` |
| `--yolo-model` | Path to YOLOv8 model weights | `models/yolov8n.pt` |
| `--pose-model` | Path to MediaPipe Pose model | `models/pose_landmarker_lite.task` |
| `--face-model` | Path to MediaPipe Face model | `models/face_landmarker.task` |
| `--alarm-sound` | Path to MP3 alarm sound file | `assets/alarm.mp3` |
| `--output-csv` | Output file path for CSV incident log | `classroom_alerts.csv` |
| `--no-alarm` | Disable audio sound alarm playback | Disabled |
| `--no-display` | Run in headless mode without window | Disabled |
