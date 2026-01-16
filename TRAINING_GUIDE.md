# 🎓 Model Training Guide

Train a custom YOLOv8 model for better soccer player detection accuracy.

## Overview

Training a custom YOLOv8 model on soccer-specific data will significantly improve tracking accuracy compared to using the general-purpose pre-trained model. A custom model will:

- ✅ Only detect soccer players, referees, and balls (not audience)
- ✅ Better accuracy for soccer-specific scenarios
- ✅ Reduced false positives
- ✅ Better handling of different camera angles and lighting

## Quick Start

### Option 1: Use the Training Script (Recommended)

```bash
# Install dependencies
pip install ultralytics roboflow

# Train with default settings (downloads dataset automatically)
python train_model.py

# Or with custom settings
python train_model.py --model-size s --epochs 150 --batch 8
```

### Option 2: Use the Jupyter Notebook

1. Open `Finetuning_yolo_model.ipynb`
2. Run all cells
3. The model will be saved to `model/best.pt`

### Option 3: Use Ultralytics CLI

```bash
# Download dataset first (or use your own)
python -c "from roboflow import Roboflow; rf = Roboflow(api_key='i3ATEZUr7bnksWuK2Izn'); project = rf.workspace('roboflow-jvuqo').project('football-players-detection-3zvbc'); dataset = project.version(1).download('yolov8')"

# Train
yolo task=detect mode=train model=yolov8n.pt data=football-players-detection-1/data.yaml epochs=100 imgsz=640
```

## Training Parameters

### Model Size

Choose based on your hardware and accuracy needs:

- **n (nano)**: Fastest, smallest, least accurate (~6MB)
- **s (small)**: Good balance (~22MB)
- **m (medium)**: Better accuracy (~52MB)
- **l (large)**: High accuracy (~87MB)
- **x (xlarge)**: Best accuracy, slowest (~136MB)

**Recommendation**: Start with `n` or `s` for faster training, then try larger models if needed.

### Epochs

- **50-100**: Quick training, may underfit
- **100-200**: Good balance (recommended)
- **200+**: Better accuracy but diminishing returns

### Batch Size

Adjust based on your GPU memory:

- **GPU with 8GB+**: batch=16-32
- **GPU with 4-8GB**: batch=8-16
- **GPU with <4GB or CPU**: batch=4-8

### Image Size

- **640**: Standard, good balance
- **1280**: Higher accuracy, slower training (use if you have high-res video)

## Dataset Options

### Option 1: Roboflow Dataset (Easiest)

The training script automatically downloads a pre-labeled soccer dataset from Roboflow:

```python
python train_model.py  # Downloads dataset automatically
```

### Option 2: Your Own Dataset

Prepare your dataset in YOLO format:

```
your_dataset/
├── data.yaml
├── train/
│   ├── images/
│   │   ├── img1.jpg
│   │   └── ...
│   └── labels/
│       ├── img1.txt
│       └── ...
├── val/
│   ├── images/
│   └── labels/
└── test/ (optional)
    ├── images/
    └── labels/
```

**data.yaml** should look like:

```yaml
path: /path/to/your_dataset
train: train/images
val: val/images
test: test/images  # optional

names:
  0: player
  1: goalkeeper
  2: referee
  3: ball
```

Then train:

```bash
python train_model.py --dataset path/to/your_dataset
```

## Training on Different Platforms

### Google Colab

1. Upload `Finetuning_yolo_model.ipynb` to Colab
2. Run all cells
3. Download the trained model from `runs/detect/train/weights/best.pt`

### Local Machine (GPU)

```bash
# Ensure CUDA is available
python -c "import torch; print(torch.cuda.is_available())"

# Train
python train_model.py --model-size s --epochs 100
```

### Local Machine (CPU)

Training on CPU is much slower but works:

```bash
python train_model.py --model-size n --epochs 50 --batch 4
```

## Monitoring Training

During training, you'll see:

- **Loss curves**: Should decrease over time
- **mAP (mean Average Precision)**: Higher is better (aim for >0.5)
- **Precision/Recall**: Balance between false positives and false negatives

Results are saved to:
- `soccer_tracker/train/weights/best.pt` - Best model
- `soccer_tracker/train/results.png` - Training plots
- `soccer_tracker/train/confusion_matrix.png` - Confusion matrix

## Using the Trained Model

After training, the model is automatically copied to `model/best.pt`. You can use it directly:

```python
# In main.py, it will automatically use model/best.pt
python main.py
```

Or specify the path:

```python
tracker = Tracker("model/best.pt")
```

## Troubleshooting

### Out of Memory Error

- Reduce batch size: `--batch 4` or `--batch 8`
- Use smaller model: `--model-size n`
- Reduce image size: `--imgsz 416`

### Poor Training Results

- Increase epochs: `--epochs 200`
- Use larger model: `--model-size m` or `--model-size l`
- Check dataset quality and labeling
- Increase image size: `--imgsz 1280`

### Dataset Download Fails

- Check internet connection
- Verify Roboflow API key is valid
- Use your own dataset instead

### Model Not Improving

- Check if dataset is diverse enough
- Verify labels are correct
- Try data augmentation (enabled by default)
- Increase training time (more epochs)

## Best Practices

1. **Start Small**: Begin with `yolov8n.pt` and 50-100 epochs to test
2. **Validate Dataset**: Check that labels are correct before training
3. **Monitor Training**: Watch loss curves to detect overfitting
4. **Use Validation Set**: Ensure you have a good validation split
5. **Save Checkpoints**: Training automatically saves best model
6. **Test on Your Video**: After training, test on your actual video

## Expected Results

With a well-trained custom model:

- **mAP@0.5**: Should be >0.6 for good results
- **Precision**: >0.7 (fewer false positives)
- **Recall**: >0.6 (catches most players)

## Next Steps

After training:

1. Test the model: `python main.py`
2. If accuracy is still poor:
   - Train for more epochs
   - Use a larger model
   - Improve your dataset
3. Fine-tune field boundaries in `main.py` if needed
4. Adjust filtering parameters if needed

## Additional Resources

- [Ultralytics YOLOv8 Documentation](https://docs.ultralytics.com/)
- [YOLOv8 Training Guide](https://docs.ultralytics.com/modes/train/)
- [Roboflow Dataset](https://roboflow.com/)
