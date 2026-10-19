"""
Rebuild keeper.raw's MediaPipe ONNX models from Google's official release.

Produces, in this folder:
  face_landmarks_detector.onnx  (MediaPipe FaceMesh V2, 478 landmarks)
  face_blendshapes.onnx         (MediaPipe Blendshape V2, incl. eyeBlinkLeft/Right)

Both models are licensed under Apache-2.0 (see models/readme.md).

Usage (PowerShell, from the repo root) — uses a throwaway virtual environment
so nothing is installed into your system Python:

    python -m venv target\mediapipe-venv
    target\mediapipe-venv\Scripts\python -m pip install tensorflow-cpu tf2onnx
    target\mediapipe-venv\Scripts\python models\convert_mediapipe.py
"""

import hashlib
import os
import subprocess
import sys
import tempfile
import urllib.request
import zipfile

BUNDLE_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)
# SHA-256 of the bundle the committed pipeline was validated against
# (downloaded 2026-10-19). "latest" may change; a mismatch is reported, not fatal.
VALIDATED_BUNDLE_SHA256 = "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff"
MODELS = ["face_landmarks_detector", "face_blendshapes"]
OUT_DIR = os.path.dirname(os.path.abspath(__file__))


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    with tempfile.TemporaryDirectory() as tmp:
        bundle = os.path.join(tmp, "face_landmarker.task")
        print(f"Downloading {BUNDLE_URL}")
        urllib.request.urlretrieve(BUNDLE_URL, bundle)
        digest = sha256(bundle)
        if digest != VALIDATED_BUNDLE_SHA256:
            print(f"WARNING: bundle SHA-256 {digest} differs from the validated one;"
                  " re-run the blink validation before shipping.")

        # The .task file is a zip containing the individual .tflite models.
        with zipfile.ZipFile(bundle) as z:
            z.extractall(tmp)

        for name in MODELS:
            src = os.path.join(tmp, f"{name}.tflite")
            dst = os.path.join(OUT_DIR, f"{name}.onnx")
            print(f"Converting {name}.tflite -> {dst}")
            subprocess.run(
                [sys.executable, "-m", "tf2onnx.convert", "--tflite", src,
                 "--output", dst, "--opset", "17"],
                check=True,
            )
            print(f"  sha256 {sha256(dst)}")


if __name__ == "__main__":
    main()
