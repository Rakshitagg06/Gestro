# Qualcomm MediaPipe Hand Tracking & Gesture Classification

A TensorFlow Lite implementation of MediaPipe Hand Tracking optimized for Qualcomm platforms, integrated with a custom gesture classification model for real-time hand gesture recognition.

## Overview

This module provides:
- **Hand Detection**: TFLite-based palm detection model
- **Hand Landmark Detection**: 21-point hand landmark estimation
- **Gesture Classification**: Custom TFLite classifier for gesture recognition
- **Real-time Pipeline**: Webcam-based real-time inference

The implementation uses MediaPipe's hand detection pipeline adapted for TensorFlow Lite inference, making it suitable for deployment on edge devices including Qualcomm AI Development Kit (QIDK).

### Hand Control Mapping

> 🎮 **Gesture Controls**:
> - **Left Hand**: Used for **directional controls** (Left / Right navigation)
> - **Right Hand**: Used for **speed controls** (Accelerate / Decelerate)

## Project Structure

```
qualcomm_medipipe/
├── isolated_hand_tracking (Copy)/     # Core hand tracking module
│   ├── app.py                         # MediaPipe hand app implementation
│   ├── demo.py                        # Standalone demo script
│   ├── tflite_inference.py            # TFLite wrapper for inference
│   ├── models/                        # TFLite models
│   │   ├── hand_detector.tflite       # Palm detection model
│   │   ├── hand_landmarker.tflite     # Hand landmark model
│   │   └── anchors_palm.npy           # Anchor boxes for detection
│   └── qai_utils/                     # Qualcomm AI utilities
│       ├── __init__.py
│       ├── base_app.py                # Base MediaPipe application
│       ├── box_utils.py               # Bounding box utilities
│       ├── draw_utils.py              # Drawing utilities
│       ├── generic_base_app.py        # Generic base class
│       ├── image_utils.py             # Image processing utilities
│       └── mediapipe_utils.py         # MediaPipe-specific utilities
├── predict.py                         # Main prediction script with gesture classification
├── your_model.tflite                  # Gesture classification model
├── label_map.npy                      # Label mapping for gestures
└── requirements.txt                   # Python dependencies
```

## Installation

### Prerequisites
- Python 3.8+
- Webcam for real-time demos
- TensorFlow 2.10+ or tflite-runtime

### Setup

```bash
cd qualcomm_medipipe

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install requirements
pip install -r requirements.txt
```

### Dependencies

```
numpy>=1.21.0
opencv-python>=4.5.0
Pillow>=8.0.0
torch>=1.10.0
torchvision>=0.11.0
tensorflow>=2.10.0
```

Alternatively, for lighter deployment:
```bash
pip install tflite-runtime
```

## Usage

### Run Real-time Gesture Prediction

The main script combines hand detection, landmark extraction, and gesture classification:

```bash
python predict.py
```

**Controls:**
- Press `ESC` or `q` to quit

**Display Features:**
- Hand landmarks with skeleton connections
- Color-coded hands (Blue for Right, Red for Left)
- Gesture classification label with confidence
- Hand count indicator

### Required Model Files

1. **Hand Detection Models** (in `isolated_hand_tracking (Copy)/models/`):
   - `hand_detector.tflite` - Palm detection model
   - `hand_landmarker.tflite` - 21-point landmark model
   - `anchors_palm.npy` - Pre-computed anchor boxes

2. **Gesture Classification** (in root directory):
   - `your_model.tflite` - Custom gesture classifier
   - `label_map.npy` - Gesture label mapping

## Pipeline Architecture

### 1. Hand Detection Stage
- Input: RGB image resized to detector input dimensions
- Model: Palm detection TFLite model
- Output: Bounding boxes around detected hands

### 2. Landmark Detection Stage
- Input: Cropped and rotated hand regions
- Model: Hand landmarker TFLite model
- Output: 21 3D landmarks per hand + handedness score

### 3. Gesture Classification Stage
- Input: 126-dimensional feature vector (63 features × 2 hands)
- Model: Custom TFLite classifier
- Output: Gesture class prediction

### Feature Vector Format

The 126-dimensional feature vector is structured as:
```
[Left Hand (63 features)] + [Right Hand (63 features)]
```

Each hand's 63 features consist of:
- 21 landmarks × 3 coordinates (x, y, z normalized)

## Core Components

### TFLiteWrapper (`tflite_inference.py`)

Generic wrapper for TFLite model inference:

```python
from tflite_inference import TFLiteWrapper

# Load model
model = TFLiteWrapper("path/to/model.tflite")

# Get input specifications
input_spec = model.get_input_spec()  # Returns (H, W)

# Run inference
outputs = model(input_tensor)
```

### FixedMediaPipeHandApp (`predict.py`)

Extended MediaPipe hand application with fixed landmark detection:

```python
from predict import build_hand_pipeline

# Build the pipeline
hand_app = build_hand_pipeline()

# Run inference on an image
outputs = hand_app.predict_landmarks_from_image([pil_image], raw_output=True)

# Unpack outputs
boxes, keypoints, rois, landmarks, is_right_hand = outputs
```

### Key Functions

| Function | Description |
|----------|-------------|
| `build_hand_pipeline()` | Initialize hand detection + landmark pipeline |
| `format_feature_vector()` | Convert landmarks to 126-dim classification input |
| `normalize_landmarks()` | Normalize coordinates to [0, 1] range |
| `interpret_handedness()` | Determine if hand is left or right |
| `predict_real_time()` | Main real-time prediction loop |

## Hand Landmark Indices

The 21 hand landmarks follow MediaPipe's convention:

| Index | Landmark | Index | Landmark |
|-------|----------|-------|----------|
| 0 | Wrist | 11 | Middle PIP |
| 1 | Thumb CMC | 12 | Middle TIP |
| 2 | Thumb MCP | 13 | Ring MCP |
| 3 | Thumb IP | 14 | Ring PIP |
| 4 | Thumb TIP | 15 | Ring DIP |
| 5 | Index MCP | 16 | Ring TIP |
| 6 | Index PIP | 17 | Pinky MCP |
| 7 | Index DIP | 18 | Pinky PIP |
| 8 | Index TIP | 19 | Pinky DIP |
| 9 | Middle MCP | 20 | Pinky TIP |
| 10 | Middle DIP | | |

## Skeleton Connections

```python
HAND_LANDMARK_CONNECTIONS = [
    # Thumb
    (0, 1), (1, 2), (2, 3), (3, 4),
    # Index finger
    (0, 5), (5, 6), (6, 7), (7, 8),
    # Middle finger
    (0, 9), (9, 10), (10, 11), (11, 12),
    # Ring finger
    (0, 13), (13, 14), (14, 15), (15, 16),
    # Pinky
    (0, 17), (17, 18), (18, 19), (19, 20),
    # Palm connections
    (5, 9), (9, 13), (13, 17),
]
```

## Configuration Options

Adjustable parameters in `predict.py`:

```python
# Detection thresholds
min_detector_hand_box_score = 0.75  # Minimum palm detection confidence
min_landmark_score = 0.5            # Minimum landmark detection confidence
nms_iou_threshold = 0.3             # NMS IoU threshold

# Camera settings
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
```

## Model Requirements

### Hand Detector
- Input: NHWC format, normalized to [-1, 1]
- Output: Bounding boxes + keypoints for palm anchors

### Hand Landmarker
- Input: 224×224 cropped hand region
- Output: 
  - Scores (confidence)
  - Handedness (left/right)
  - 21 landmarks with x, y, z coordinates

### Gesture Classifier
- Input: 126-dimensional float32 vector
- Output: Softmax probabilities over gesture classes

## Troubleshooting

### Common Issues

1. **Model not found error**
   ```
   Ensure all .tflite models are in the correct directories:
   - isolated_hand_tracking (Copy)/models/ for hand models
   - Root directory for gesture classifier
   ```

2. **TFLite import error**
   ```bash
   # Try installing tflite-runtime instead
   pip install tflite-runtime
   ```

3. **Slow inference**
   - Reduce input resolution
   - Skip frames for classification (currently every 2nd frame)
   - Use GPU delegate if available

4. **No hands detected**
   - Check lighting conditions
   - Adjust `min_detector_hand_box_score` threshold
   - Ensure full hand is visible in frame

## Deployment Notes

For Qualcomm AI Development Kit (QIDK) deployment:
1. Convert models to QNN format using `qairt-converter`
2. Quantize for NPU acceleration
3. Use SNPE runtime for optimal performance

See the `quantization_and_benchmarking` folder for detailed deployment instructions.

## References

- [MediaPipe Hands](https://google.github.io/mediapipe/solutions/hands.html)
- [TensorFlow Lite](https://www.tensorflow.org/lite)
- [Qualcomm AI Hub](https://aihub.qualcomm.com/)
