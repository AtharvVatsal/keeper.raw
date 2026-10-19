# ONNX Models

This directory contains the ONNX models used by keeper.raw. These files are **not included in the Git repository** due to their size.

## Required Models

| Model | File | Size | Purpose | Licence |
|---|---|---|---|---|
| YOLOv8-nano Face | `yolov8n-face.onnx` | ~12 MB | Face detection + 5 face keypoints | AGPL-3.0 ⚠️ (to be replaced before public release) |
| MediaPipe FaceMesh V2 | `face_landmarks_detector.onnx` | ~5 MB | 478 facial landmarks | Apache-2.0 |
| MediaPipe Blendshape V2 | `face_blendshapes.onnx` | ~2 MB | 52 expression scores, incl. `eyeBlinkLeft` / `eyeBlinkRight` (blink detection) | Apache-2.0 |

`mobilenetv2-12.onnx` is only used by `crates/keeper-vision/tests/onnx_smoke_test.rs` (an ONNX Runtime smoke test) and is optional.

## How to Obtain

### Option A: Download from Releases (recommended)

Download the model files from the [latest release](https://github.com/AtharvVatsal/keeper.raw/releases/latest).

### Option B: Build them yourself

#### MediaPipe FaceMesh V2 + Blendshape V2

Both come from Google's official Face Landmarker bundle
(`https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task`),
converted from TensorFlow Lite to ONNX with `tf2onnx`. The script does everything and
installs its tools into a throwaway virtual environment (PowerShell, from the repo root):

```powershell
python -m venv target\mediapipe-venv
target\mediapipe-venv\Scripts\python -m pip install tensorflow-cpu tf2onnx
target\mediapipe-venv\Scripts\python models\convert_mediapipe.py
```

The conversion was checked against the original TFLite models: landmark outputs agree to
within 0.0004 px and blendshape scores to within 0.000002.

#### YOLOv8-nano Face
```bash
pip install ultralytics
python -c "
from ultralytics import YOLO
model = YOLO('yolov8n-face.pt')
model.export(format='onnx', imgsz=640, simplify=True)
"
```

## Licences

- **MediaPipe FaceMesh V2 and Blendshape V2** — Apache License 2.0, per Google's model cards
  ([FaceMesh V2](https://storage.googleapis.com/mediapipe-assets/Model%20Card%20MediaPipe%20Face%20Mesh%20V2.pdf),
  [Blendshape V2](https://storage.googleapis.com/mediapipe-assets/Model%20Card%20Blendshape%20V2.pdf)).
  Redistribution requires including the [Apache-2.0 licence text](https://www.apache.org/licenses/LICENSE-2.0).
- **YOLOv8** — [AGPL-3.0](https://github.com/ultralytics/ultralytics/blob/main/LICENSE). Not compatible with
  shipping inside an MIT-licensed installer; scheduled to be replaced by YuNet (MIT) before the first public release.
