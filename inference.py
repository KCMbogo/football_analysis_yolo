from ultralytics import YOLO

model = YOLO('yolo_weights/yolov8n.pt')
video_path = "videos/match.mp4"
results = model.predict(source=video_path)
print(results)
