"""Simple script to test YOLO model inference on a video."""

from ultralytics import YOLO

model = YOLO('model/best.pt')

results = model.predict("input_videos/footbal_match.mp4", save=True)
print(results[0])
print('=======================================================')
for box in results[0].boxes:
    print(box)
