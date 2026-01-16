"""Download a pre-trained YOLOv8 model for testing."""

import os
import sys


def download_model(model_size="n", output_dir="model"):
    """Download a pre-trained YOLOv8 model.

    Args:
        model_size: Model size - 'n' (nano), 's' (small), 'm' (medium), 'l' (large), 'x' (xlarge)
        output_dir: Directory to save the model
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        print("Error: ultralytics package not found.")
        print("Please install it with: pip install ultralytics")
        sys.exit(1)

    model_name = f"yolov8{model_size}.pt"
    output_path = os.path.join(output_dir, "best.pt")

    print(f"Downloading {model_name}...")
    print("Note: This is a general-purpose YOLOv8 model.")
    print("For best results, you should fine-tune it on soccer-specific data.")
    print("=" * 60)

    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Download the model
        model = YOLO(model_name)

        # Save it as best.pt
        import shutil

        model_path = model.ckpt_path if hasattr(model, "ckpt_path") else None

        # The model is automatically downloaded by YOLO, we just need to copy it
        # YOLO downloads to a cache directory, so we'll export it
        model.export(format="torchscript")  # This ensures the model is available

        # Actually, YOLO() constructor downloads the model automatically
        # We can just use it directly, but let's save a copy
        print(f"\nModel downloaded successfully!")
        print(f"Using model: {model_name}")
        print(f"\nTo use this model, update main.py:")
        print(f'  model_path = "model/{model_name}"')
        print("\nOr rename it to best.pt:")
        print(f"  Rename {model_name} to model/best.pt")

        # Try to find where YOLO stored the model
        import torch.hub

        cache_dir = torch.hub.get_dir()
        possible_paths = [
            os.path.join(cache_dir, "hub", "checkpoints", model_name),
            os.path.join(os.path.expanduser("~"), ".ultralytics", model_name),
        ]

        for path in possible_paths:
            if os.path.exists(path):
                print(f"\nFound model at: {path}")
                print(f"Copying to: {output_path}")
                shutil.copy2(path, output_path)
                print(f"✓ Model saved to {output_path}")
                return

        print("\n⚠ Could not automatically copy the model.")
        print("The model has been downloaded by YOLO and will be used automatically.")
        print("You may need to manually copy it or use the model name directly.")

    except Exception as e:
        print(f"Error downloading model: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download a pre-trained YOLOv8 model")
    parser.add_argument(
        "--size",
        choices=["n", "s", "m", "l", "x"],
        default="n",
        help="Model size: n (nano, fastest), s (small), m (medium), l (large), x (xlarge, most accurate)",
    )
    parser.add_argument(
        "--output-dir", default="model", help="Output directory for the model (default: model)"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("YOLOv8 Model Downloader")
    print("=" * 60)
    print(f"Model size: {args.size}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 60)

    download_model(args.size, args.output_dir)
