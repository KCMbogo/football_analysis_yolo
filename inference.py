from ultralytics import YOLO
import cv2, torch

device = "cuda" if torch.cuda.is_available else "cpu"

cap = cv2.VideoCapture("videos/input_videos/match.mp4")
model = YOLO('models/best.pt').to(device)

while True:
    success, img = cap.read()
    if not success:
        break
    
    results = model.predict(source=img, stream=True)
    
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(img=img, pt1=(x1, y1), pt2=(x2, y2), color=(0, 255, 0), thickness=1)

    cv2.imshow(winname="Image", mat=img)
    if cv2.waitKey(1) == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()