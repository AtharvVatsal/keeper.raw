from ultralytics import YOLO

model = YOLO(r"A:\TestRAWs\yolov8n-face.pt")

model.export(
    format="onnx",
    imgsz=640,
    simplify=True,
    opset=17,
)
print("Done! Look for yolov8n-face.onnx in the same directory as the .pt file")
