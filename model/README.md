# Model Directory

This directory should contain your YOLO model file(s).

## Required Model File

- **File name**: `best.pt` (or update the path in `main.py`)
- **Format**: PyTorch model file (.pt)
- **Type**: YOLOv8 model trained for soccer player/ball detection

## Getting a Model

### Option 1: Use Pre-trained YOLOv8 Model

You can use a pre-trained YOLOv8 model and fine-tune it:

```python
from ultralytics import YOLO

# Download pre-trained model
model = YOLO('yolov8n.pt')  # or yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt

# Fine-tune on your soccer dataset
model.train(data='path/to/your/dataset.yaml', epochs=100, imgsz=640)
```

### Option 2: Train Your Own Model

Use the `Finetuning_yolo_model.ipynb` notebook in the root directory to train a custom model on your soccer video dataset.

### Option 3: Download a Soccer-Specific Model

If you have a trained model from a previous training session, place it here as `best.pt`.

## Model Classes

The model should detect:
- `player` - Soccer players
- `goalkeeper` - Goalkeepers (will be converted to players)
- `referee` - Referees
- `ball` - Soccer ball

## File Structure

```
model/
├── best.pt                    # Your trained YOLO model
├── track_stubs.pkl          # Cached tracking data (auto-generated)
└── camera_movement_stub.pkl # Cached camera movement data (auto-generated)
```

Note: The `.pkl` files are automatically generated during processing and can be deleted to force recalculation.
