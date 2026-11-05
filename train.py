from ultralytics import YOLO

YOLO("yolov8n.pt").train(
    data="dataset/data.yaml",
    imgsz=640,
    epochs=100,
    batch=4,
    workers=0,
    device=0,
    mosaic=0.0,
    amp=False,
    plots=False,
    project="runs/detect",
    name="train",
    exist_ok=True,
)
