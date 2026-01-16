# ⚡ Quick Start Guide

Get up and running with Soccer Player Tracking in minutes!

## Prerequisites

- Python 3.8 or higher
- UV package manager

## Installation Steps

### 1. Install UV (if not already installed)

```bash
pip install uv
```

### 2. Set Up Virtual Environment

**Windows:**

```bash
setup.bat
```

**Linux/Mac:**

```bash
chmod +x setup.sh
./setup.sh
```

**Or manually:**

```bash
# Create virtual environment
uv venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# Install dependencies
uv pip install -e .
```

### 3. Verify Installation

```bash
python verify_setup.py
```

You should see all tests passing.

### 4. Prepare Your Data

1. Place your input video in `input_videos/` directory
2. **Get a YOLO model**:
   - **Quick start**: Download a pre-trained model:

     ```bash
     python scripts/download_model.py
     ```

   - **Best results**: Train your own using `Finetuning_yolo_model.ipynb`
   - Or place your trained model as `model/best.pt`
3. Update paths in `main.py` if needed:
   - Video path: `input_videos/your_video.mp4`
   - Model path: `model/best.pt`

### 5. Run the Tracker

```bash
python main.py
```

The output video will be saved to `output_videos/output_video.mp4`

## Project Structure

```
Soccer-Player-Tracking/
├── src/soccer_tracker/     # Main package
│   ├── tracking/           # Detection and tracking modules
│   ├── analysis/           # Analysis modules (team assignment, etc.)
│   └── utils/              # Utility functions
├── main.py                 # Main entry point
├── pyproject.toml          # Project configuration
├── scripts/               # Utility scripts
├── model/                  # YOLO model files
├── input_videos/          # Input video directory
└── output_videos/         # Output video directory
```

## Troubleshooting

### Import Errors

If you get import errors, make sure:

1. Virtual environment is activated
2. Package is installed: `uv pip install -e .`
3. You're running from the project root directory

### Missing Dependencies

If dependencies are missing:

```bash
uv pip install -e .
```

### Model File Not Found

Make sure your YOLO model file (`.pt`) is in the `model/` directory and the path in `main.py` is correct.

## Next Steps

- Read `README.md` for detailed documentation
- Check `MIGRATION.md` if migrating from old structure
- Customize court dimensions in `ViewTransformer` if needed
- Adjust tracking parameters in respective modules
