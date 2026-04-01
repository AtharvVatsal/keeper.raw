# ONNX Models

This directory contains the ONNX models used by keeper.raw. These files are **not included in the Git repository** due to their size.

## Required Models

| Model | File | Size | Purpose |
|---|---|---|---|
| YOLOv8-nano Face | `yolov8n-face.onnx` | ~12 MB | Face detection |
| MediaPipe Face Mesh | `face_landmark.onnx` | ~3 MB | Facial landmark detection (blink/EAR) |

## How to Obtain

### Option A: Download from Releases (recommended)

Download the model files from the [latest release](https://github.com/AtharvVatsal/keeper.raw/releases/latest).

### Option B: Export Yourself

#### YOLOv8-nano Face
```bash
pip install ultralytics
python -c "
from ultralytics import YOLO
model = YOLO('yolov8n-face.pt')
model.export(format='onnx', imgsz=640, simplify=True)
"
```

#### MediaPipe Face Mesh

Download the TFLite model from MediaPipe and convert to ONNX using `tf2onnx`. See the [MediaPipe documentation](https://developers.google.com/mediapipe/solutions/vision/face_landmarker) for details.

## License

- YOLOv8: [AGPL-3.0](https://github.com/ultralytics/ultralytics/blob/main/LICENSE) (model weights may have different terms for inference-only use)
- MediaPipe: [Apache-2.0](https://github.com/google/mediapipe/blob/master/LICENSE)