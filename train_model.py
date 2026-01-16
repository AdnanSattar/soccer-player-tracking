"""Train YOLOv8 model on soccer player detection dataset."""

import os
import sys
from pathlib import Path


def train_model(
    dataset_path=None,
    model_size="n",
    epochs=100,
    imgsz=640,
    batch=16,
    output_dir="model",
    project_name="soccer_tracker",
    device=None,  # None = auto-detect, "cuda" = GPU, "cpu" = CPU
):
    """Train a YOLOv8 model on soccer player detection data.

    Args:
        dataset_path: Path to dataset directory (should contain data.yaml)
                     If None, will download from Roboflow
        model_size: Model size - 'n' (nano), 's' (small), 'm' (medium), 'l' (large), 'x' (xlarge)
        epochs: Number of training epochs
        imgsz: Image size for training
        batch: Batch size
        output_dir: Directory to save the trained model
        project_name: Project name for training runs
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        print("Error: ultralytics package not found.")
        print("Please install it with: pip install ultralytics")
        sys.exit(1)

    # Download dataset if not provided
    if dataset_path is None:
        print("=" * 60)
        print("Downloading dataset from Roboflow...")
        print("=" * 60)
        try:
            from roboflow import Roboflow

            rf = Roboflow(api_key="i3ATEZUr7bnksWuK2Izn")
            project = rf.workspace("roboflow-jvuqo").project("football-players-detection-3zvbc")
            version = project.version(1)
            dataset = version.download("yolov8")
            dataset_path = dataset.location
            print(f"✓ Dataset downloaded to: {dataset_path}")
        except ImportError:
            print("Error: roboflow package not found.")
            print("Please install it with: pip install roboflow")
            print("\nOr provide a dataset_path argument with your own dataset.")
            sys.exit(1)
        except Exception as e:
            print(f"Error downloading dataset: {e}")
            print("\nYou can manually download the dataset and provide the path:")
            print("  train_model(dataset_path='path/to/dataset')")
            sys.exit(1)

    # Verify dataset structure
    data_yaml = os.path.join(dataset_path, "data.yaml")
    if not os.path.exists(data_yaml):
        print(f"Error: data.yaml not found in {dataset_path}")
        print("Please ensure the dataset directory contains a data.yaml file.")
        sys.exit(1)

    print("=" * 60)
    print("YOLOv8 Model Training")
    print("=" * 60)
    print(f"Dataset: {dataset_path}")
    print(f"Model: yolov8{model_size}.pt")
    print(f"Epochs: {epochs}")
    print(f"Image size: {imgsz}")
    print(f"Batch size: {batch}")
    print("=" * 60)

    # Initialize model
    model_name = f"yolov8{model_size}.pt"
    print(f"\nInitializing model: {model_name}")
    model = YOLO(model_name)

    # Train the model
    print("\nStarting training...")
    print("This may take a while depending on your hardware and dataset size.")
    print("=" * 60)

    # Check for GPU availability
    import torch

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cuda" and torch.cuda.is_available():
        print(f"\n✓ GPU detected: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        print(f"  Using device: {device}")
    elif device == "cuda" and not torch.cuda.is_available():
        print("\n⚠ GPU requested but not available, falling back to CPU")
        device = "cpu"
    else:
        print("\n⚠ Training on CPU (will be slower)")
        print("  Consider using a GPU for faster training")
        print(f"  Using device: {device}")

    try:
        results = model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device=device,  # Use GPU if available
            project=project_name,
            name="train",
            save=True,
            plots=True,
        )

        print("\n" + "=" * 60)
        print("Training completed!")
        print("=" * 60)

        # Find the best model
        best_model_path = os.path.join(project_name, "train", "weights", "best.pt")
        if os.path.exists(best_model_path):
            # Copy to output directory
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, "best.pt")

            import shutil

            shutil.copy2(best_model_path, output_path)
            print(f"✓ Best model saved to: {output_path}")
            print(f"\nYou can now use this model in main.py:")
            print(f'  tracker = Tracker("{output_path}")')
        else:
            print(f"⚠ Best model not found at expected location: {best_model_path}")
            print("Check the training output directory for the model file.")

        print("\nTraining metrics:")
        print(f"  Results saved to: {os.path.join(project_name, 'train')}")
        print(f"  Check results.png for training plots")

    except Exception as e:
        print(f"\n✗ Error during training: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train YOLOv8 model for soccer player detection")
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to dataset directory (should contain data.yaml). If not provided, will download from Roboflow.",
    )
    parser.add_argument(
        "--model-size",
        choices=["n", "s", "m", "l", "x"],
        default="n",
        help="Model size: n (nano, fastest), s (small), m (medium), l (large), x (xlarge, most accurate)",
    )
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of training epochs (default: 100)"
    )
    parser.add_argument(
        "--imgsz", type=int, default=640, help="Image size for training (default: 640)"
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size (default: 16, reduce if you run out of memory)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="model",
        help="Directory to save the trained model (default: model)",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="soccer_tracker",
        help="Project name for training runs (default: soccer_tracker)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "cpu"],
        help="Device to use: 'cuda' for GPU, 'cpu' for CPU, or omit for auto-detect",
    )

    args = parser.parse_args()

    train_model(
        dataset_path=args.dataset,
        model_size=args.model_size,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        output_dir=args.output_dir,
        project_name=args.project,
        device=args.device,
    )
