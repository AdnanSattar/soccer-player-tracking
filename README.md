# Soccer Player Tracking

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![UV](https://img.shields.io/badge/package%20manager-uv-orange)](https://github.com/astral-sh/uv)

Real-time soccer player tracking system using **YOLOv8** and **ByteTrack** algorithm. Tracks players, ball possession, speed, distance, and team assignments with camera movement compensation.

## Features

- Real-time Tracking**: Multi-object tracking of players, referees, and ball
- Team Assignment**: Automatic team identification based on jersey colors
- Ball Possession**: Tracks which player has the ball
- Analytics**: Calculates player speed and distance traveled
- Field Transformation**: Converts camera view to top-down field coordinates
- Camera Compensation**: Accounts for camera movement in tracking
- Field Filtering**: Excludes audience members for better accuracy

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Training Custom Models](#training-custom-models)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## Installation

### Prerequisites

- Python 3.8 or higher
- [UV](https://github.com/astral-sh/uv) package manager (recommended) or pip

### Setup

1. **Clone the repository**:

   ```bash
   git clone https://github.com/your-username/soccer-player-tracking.git
   cd soccer-player-tracking
   ```

2. **Install UV** (if not already installed):

   ```bash
   pip install uv
   ```

3. **Create virtual environment and install dependencies**:

   ```bash
   uv venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   uv pip install -e .
   ```

   Or with development dependencies:

   ```bash
   uv pip install -e ".[dev]"
   ```

## Quick Start

1. **Place your video** in `input_videos/` directory

2. **Get a model** (choose one):
   - **Option A**: Train custom model (best accuracy):

     ```bash
     pip install roboflow
     python train_model.py
     ```

   - **Option B**: Use pre-trained model (quick start):

     ```bash
     python scripts/download_model.py
     ```

3. **Update video path** in `main.py`:

   ```python
   video_path = "input_videos/your_video.mp4"
   ```

4. **Run tracking**:

   ```bash
   python main.py
   ```

5. **Find output** in `output_videos/output_video.mp4`

## Usage

### Basic Usage

```python
from soccer_tracker.tracking import Tracker
from soccer_tracker.utils import read_video, save_video

# Read video
video_frames = read_video("input_videos/match.mp4")

# Initialize tracker
tracker = Tracker("model/best.pt")

# Track objects
tracks = tracker.get_object_tracks(video_frames)

# Process and save
# ... (see main.py for full example)
```

### Command Line

```bash
# Basic tracking
python main.py

# With custom model
python main.py --model custom_model.pt
```

## Training Custom Models

For best accuracy, train a custom YOLOv8 model on soccer-specific data:

```bash
# Install training dependencies
pip install roboflow

# Train with default settings
python train_model.py

# Train on GPU (faster)
python train_model.py --device cuda --model-size s --epochs 150

# Train with custom dataset
python train_model.py --dataset path/to/your/dataset
```

See [TRAINING_GUIDE.md](TRAINING_GUIDE.md) for detailed instructions.

## Configuration

### Field Boundaries

Update field boundaries in `src/soccer_tracker/analysis/view_transformer.py`:

```python
self.pixel_vertices = np.array([
    [489, 200],   # Top-left
    [236, 813],   # Bottom-left
    [2193, 813],  # Bottom-right
    [1960, 180]   # Top-right
])
```

### Filtering Settings

Adjust in `main.py` to exclude audience:

```python
tracker = Tracker(
    model_path,
    field_boundaries=field_boundaries,
    min_bbox_area=1000,  # Increase to exclude smaller detections
    max_bbox_area=40000
)
```

See [FILTERING_GUIDE.md](FILTERING_GUIDE.md) for more details.

## Project Structure

```
soccer-player-tracking/
├── src/
│   └── soccer_tracker/          # Main package
│       ├── tracking/             # Detection and tracking
│       │   ├── tracker.py
│       │   └── camera_movement_estimator.py
│       ├── analysis/             # Analysis modules
│       │   ├── team_assigner.py
│       │   ├── player_ball_assigner.py
│       │   ├── speed_and_distance_estimator.py
│       │   └── view_transformer.py
│       └── utils/                # Utilities
│           ├── bbox_utils.py
│           └── video_utils.py
├── main.py                       # Main entry point
├── train_model.py                # Model training script
├── scripts/                      # Utility scripts
├── pyproject.toml                # Project configuration
├── README.md
├── TRAINING_GUIDE.md
├── FILTERING_GUIDE.md
└── CONTRIBUTING.md
```

## Dependencies

- **ultralytics**: YOLOv8 model inference
- **supervision**: ByteTrack tracking algorithm
- **opencv-python**: Video processing
- **numpy**: Numerical operations
- **pandas**: Data manipulation
- **scikit-learn**: KMeans clustering for team assignment

## Example Output

The system generates annotated videos with:

- Player tracking with team colors
- Ball possession indicators
- Speed and distance metrics
- Team ball control statistics
- Camera movement visualization

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [YOLOv8](https://github.com/ultralytics/ultralytics) by Ultralytics
- [ByteTrack](https://github.com/ifzhang/ByteTrack) tracking algorithm
- [Supervision](https://github.com/roboflow/supervision) library
- [Roboflow](https://roboflow.com/) for dataset

## 📧 Support

For issues, questions, or contributions, please open an issue on GitHub.

## Additional Documentation

- [Quick Start Guide](QUICKSTART.md) - Get started in minutes
- [Training Guide](TRAINING_GUIDE.md) - Train custom models
- [Filtering Guide](FILTERING_GUIDE.md) - Improve accuracy
- [Contributing Guide](CONTRIBUTING.md) - How to contribute
- [GitHub Setup](GITHUB_SETUP.md) - Repository setup instructions

## Roadmap

- [ ] Real-time video stream processing
- [ ] Multi-camera support
- [ ] Advanced analytics dashboard
- [ ] Export tracking data to JSON/CSV
- [ ] Web interface
- [ ] API endpoints

## ⭐ Star History

If you find this project useful, please consider giving it a star!

---

**Made with ⚽ for soccer analytics**
