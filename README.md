# Strawberry Detection & Depth Estimation

A Python tool for detecting strawberries in images with segmentation masks and accurate depth (distance) estimation.

## Features

- 🍓 **Strawberry Detection** - YOLOv8-based detection and instance segmentation
- 📏 **Depth Estimation** - Metric depth using Depth-Anything-V2 (in meters)
- 📊 **Rich Output** - Bounding boxes, masks, depth statistics per strawberry
- 🐍 **Python API** - Use as a library in your code
- 💻 **CLI Tool** - Run from command line

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd strawberry_detector

# Install dependencies
pip install -r requirements.txt

# Models will be downloaded automatically on first run
```

## Usage

### Python API

```python
from strawberry_detector import detect, StrawberryDetector

# Quick detection (one-liner)
result = detect("path/to/image.jpg")
print(f"Found {result['detections_count']} strawberries")

# For processing multiple images
detector = StrawberryDetector()
for image_path in image_paths:
    result = detector.detect(image_path)
    for det in result['detections']:
        print(f"Strawberry {det['id']}: {det['depth']['mean_cm']:.1f} cm away")
```

### Command Line

```bash
# Basic usage
python -m strawberry_detector --image photo.jpg --output results.json

# With visualization
python -m strawberry_detector --image photo.jpg --output results.json --visualize

# From URL
python -m strawberry_detector --image "https://example.com/strawberries.jpg" --output results.json
```

## Output Format

The output JSON contains:

```json
{
  "image_path": "image.jpg",
  "image_size": {"width": 1920, "height": 1080},
  "detections_count": 5,
  "detections": [
    {
      "id": 0,
      "class_name": "ripe",
      "confidence": 0.95,
      "bbox": {"x1": 100, "y1": 200, "x2": 300, "y2": 400, ...},
      "mask": {"polygon": [...], "area": 15000},
      "depth": {
        "mean_meters": 0.78,
        "mean_cm": 78.0,
        "min_meters": 0.70,
        "max_meters": 0.85
      }
    }
  ],
  "statistics": {
    "total_ripe": 3,
    "total_unripe": 2,
    "closest_distance_meters": 0.70
  }
}
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA (optional, for GPU acceleration)

## Models

| Model | Source |
|-------|--------|
| Depth-Anything-V2 | [HuggingFace](https://huggingface.co/Depth-Anything/Depth-Anything-V2-Metric-Hypersim-Large) |
| YOLOv8 Strawberry | Custom trained |

Models are downloaded automatically on first run (~1.3 GB total).

## License

MIT License
