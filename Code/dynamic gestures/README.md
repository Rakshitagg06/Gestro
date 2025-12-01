# Dynamic Gestures Recognition System

A real-time dynamic gesture recognition system using hand detection, tracking, and LSTM-based sequence classification. This module is built on top of the HaGRID dataset and supports custom dynamic gesture training.

## Overview

This system provides:
- **Hand detection** using ONNX models
- **Hand tracking** using Observation-Centric SORT (OC-SORT) with Kalman filtering
- **Static gesture classification** for crop-based hand gesture recognition
- **Dynamic gesture recognition** using LSTM models trained on hand movement sequences
- **Action recognition** for events like swipes, zoom, drag-and-drop, and clicks

## Project Structure

```
dynamic gestures/
├── ocsort/                     # Observation-Centric Sorting for tracking
│   ├── __init__.py
│   ├── kalmanfilter.py         # Kalman filter implementation
│   ├── kalmanboxtracker.py     # Kalman box tracker
│   └── association.py          # Association of boxes with trackers
├── utils/                      # Utility modules
│   ├── __init__.py
│   ├── action_controller.py    # Action controller for dynamic gestures
│   ├── box_utils_numpy.py      # Box utilities for numpy operations
│   ├── enums.py                # Enums for gestures and actions
│   ├── hand.py                 # Hand class for gesture recognition
│   └── drawer.py               # Debug drawer utilities
├── models/                     # Pre-trained ONNX models
│   ├── hand_detector.onnx      # Hand detection model
│   └── crops_classifier.onnx   # Gesture classification model
├── custom_models/              # Custom trained dynamic gesture models
│   ├── dynamic_gesture_model_left_hand.pth
│   ├── dynamic_gesture_model_right_hand.pth
│   ├── dynamic_class_names_left_hand.txt
│   └── dynamic_class_names_right_hand.txt
├── dynamic_dataset/            # Dataset for custom dynamic gestures
│   ├── left/                   # Left hand gesture sequences
│   │   ├── acc/
│   │   └── ret/
│   └── right/                  # Right hand gesture sequences
│       ├── sl/
│       └── sr/
├── license/
│   └── LICENSE-2.0.txt
├── collect_data.py             # Data collection for custom gestures
├── train_model.py              # Train custom LSTM models
├── predict.py                  # Standalone prediction with custom models
├── onnx_models.py              # ONNX model wrappers
├── main_controller.py          # Main tracking and recognition controller
├── run_demo.py                 # Demo script for real-time recognition
├── requirements.txt            # Python dependencies
└── pyproject.toml              # Project configuration
```

## Installation

### Prerequisites
- Python 3.9+
- Webcam for real-time demos

### Setup

```bash
# Clone the repository (if not already done)
cd "dynamic gestures"

# Create virtual environment
conda create -n dynamic_gestures python=3.9 -y
conda activate dynamic_gestures

# Install requirements
pip install -r requirements.txt
```

### Dependencies

```
filterpy==1.4.5
numpy==1.23.5
onnx==1.13.0
onnxruntime==1.13.1
opencv-contrib-python==4.6.0.66
torch>=1.10.0  # For LSTM training
```

## Usage

### 1. Run Demo (Pre-trained Models)

Run the main demo with pre-trained models for hand detection and gesture classification:

```bash
python run_demo.py --detector models/hand_detector.onnx --classifier models/crops_classifier.onnx --debug
```

**Arguments:**
| Argument | Description | Default |
|----------|-------------|---------|
| `--detector` | Path to hand detector model | `models/hand_detector.onnx` |
| `--classifier` | Path to crops classifier model | `models/crops_classifier.onnx` |
| `--debug` | Enable debug mode (shows bounding boxes and labels) | `False` |

### 2. Collect Custom Gesture Data

Record sequences of hand movements for custom dynamic gestures:

```bash
python collect_data.py --hand left --gesture swipe_left
python collect_data.py --hand right --gesture swipe_right
```

**Arguments:**
| Argument | Description | Required |
|----------|-------------|----------|
| `--hand` | Hand to track (`left` or `right`) | Yes |
| `--gesture` | Name of the gesture to record | Yes |

**Controls:**
- Press `s` to start recording a sequence
- Press `q` to quit
- Each sequence records 30 frames of hand positions

Data is saved to `dynamic_dataset/{hand}/{gesture}/` as `.npy` files.

### 3. Train Custom Dynamic Gesture Model

Train an LSTM model on your collected gesture data:

```bash
# Train for left hand
python train_model.py --hand left

# Train for right hand
python train_model.py --hand right
```

**Training Details:**
- Model: LSTM with 64 hidden units
- Input: 2D hand center coordinates per frame
- Epochs: 50
- Batch size: 16
- Optimizer: Adam (lr=0.001)

Models are saved to `custom_models/` directory.

### 4. Real-time Prediction with Custom Models

Run predictions using your trained LSTM models:

```bash
python predict.py --detector models/hand_detector.onnx --classifier models/crops_classifier.onnx
```

## Supported Dynamic Gestures

The system supports 6 groups of dynamic gestures:

| Gesture Group | Actions |
|--------------|---------|
| **Zoom** | Zoom In, Zoom Out (two-hand pinch gestures) |
| **Drag and Drop** | Drag, Drop (fist gestures with movement) |
| **Fast Swipe Up/Down** | Quick vertical swipes |
| **Click** | Single and double clicks (finger tap gestures) |
| **Swipe Left/Right** | Horizontal swipe gestures (3 variations) |
| **Swipe Up/Down** | Vertical swipe gestures (3 variations) |

## Pipeline Architecture

1. **Hand Detection**: ONNX-based detector locates hand bounding boxes
2. **Hand Tracking**: OC-SORT with Kalman filtering maintains hand identity across frames
3. **Gesture Classification**: Crops classifier identifies static hand poses
4. **Sequence Analysis**: Action controller analyzes movement patterns over time
5. **Event Generation**: Recognized actions trigger corresponding events

## Main Controller API

```python
from main_controller import MainController

# Initialize controller
controller = MainController(
    detection_model="models/hand_detector.onnx",
    classification_model="models/crops_classifier.onnx",
    max_age=30,          # Maximum frames to keep unmatched tracks
    min_hits=3,          # Minimum detections before track is confirmed
    iou_threshold=0.3,   # IoU threshold for track association
    maxlen=30,           # Deque length for action recognition
    min_frames=20        # Minimum frames for action detection
)

# Process frame
bboxes, ids, labels = controller(frame)
```

## Model Files

### Required Pre-trained Models
- `models/hand_detector.onnx` - SSD-based hand detector (320×240 input)
- `models/crops_classifier.onnx` - Gesture classifier (128×128 input)

### Custom Model Outputs
- `custom_models/dynamic_gesture_model_{hand}_hand.pth` - PyTorch LSTM model
- `custom_models/dynamic_class_names_{hand}_hand.txt` - Class name mapping

## License

This work is licensed under the [Apache License, Version 2.0](./license/LICENSE-2.0.txt).

## Citation

If you use this code, please cite the HaGRID dataset:

```bibtex
@misc{nuzhdin2024hagridv21mimagesstatic,
      title={HaGRIDv2: 1M Images for Static and Dynamic Hand Gesture Recognition},
      author={Anton Nuzhdin and Alexander Nagaev and Alexander Sautin and Alexander Kapitanov and Karina Kvanchiani},
      year={2024},
      eprint={2412.01508},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2412.01508},
}
```

## References

- [HaGRID GitHub Repository](https://github.com/hukenovs/hagrid)
- [OC-SORT Paper](https://arxiv.org/abs/2203.14360)
