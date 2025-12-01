# Gesture Classification Code Guide

> ⚠️ **Important**: This module uses **Google's MediaPipe library** for hand landmark detection. MediaPipe provides the 21-point hand skeleton that serves as input features for the gesture classification model.

This folder contains everything needed to capture hand-gesture data, train a classifier on Mediapipe landmarks, export the model to TensorFlow Lite, and benchmark it on recorded datasets or live hardware. The sections below walk through every script and explain how they fit into the pipeline.

### Hand Control Mapping

> 🎮 **Gesture Controls**:
> - **Left Hand**: Used for **speed controls** (Accelerate / Decelerate navigation)
> - **Right Hand**: Used for **directional controls** (Right / Left)


## Directory Layout
- `classification_model/`
	- `gestures/` – raw gesture images grouped by class (e.g., `Acc+Left`, `Stop`).
	- `collect_data.py` – webcam landmark recorder that writes labeled CSV rows.
	- `augment_images.py` – heavy-duty image augmentation utility for `gestures/`.
	- `extract_landmarks.py` – converts gesture images into train/test CSVs.
	- `train_model.py` – trains a dense neural net and saves `gesture_model.h5` plus datasets.
	- `tfllite_conevrtor.py` – exports the trained Keras model to `your_model.tflite`.
	- `benchmarking/` – benchmarking + evaluation helpers:
		- `benchmark_inference.py` – full inference-time + accuracy benchmarker.
		- `benchmark_database.py` – persistent CSV log of benchmark runs.
		- `image_folder_inference.py` – Mediapipe + classifier evaluation on saved images.
		- `image_inference_results.csv`, `image_inference_stats.csv`, `image_inference_summary.json` – sample outputs.
	- Artifacts: `gestures_train.csv`, `gestures_test.csv`, `training_dataset.csv`, `testing_dataset.csv`, `label_map.npy`, `gesture_model.h5`, `your_model.tflite`.

## Environment & Dependencies
- Python 3.9+ (tested on Linux).
- Core packages: `opencv-python`, `mediapipe`, `numpy`, `pandas`, `scikit-learn`, `tensorflow` (and/or `tflite-runtime`), `matplotlib`, `seaborn`, `tqdm`.
- Optional for benchmarking: `tensorflow-gpu`, `tflite-runtime`, Edge TPU/NNAPI delegates, `tqdm` (already listed).

Install everything via pip:
```bash
pip install opencv-python mediapipe numpy pandas scikit-learn tensorflow matplotlib seaborn tqdm
```

## End-to-End Workflow
1. **Capture labeled CSVs** – use `collect_data.py` to record live webcam gestures as Mediapipe landmarks.
2. **(Optional) Augment images** – expand the `gestures/` folders with `augment_images.py` to balance classes.
3. **Extract landmarks from images** – run `extract_landmarks.py` to build `gestures_train.csv` and `gestures_test.csv` from the curated image folders.
4. **Train the classifier** – execute `train_model.py` to split data, train a dense network, and save `gesture_model.h5` plus `label_map.npy`.
5. **Export to TensorFlow Lite** – convert the `.h5` to `.tflite` with `tfllite_conevrtor.py` for deployment.
6. **Benchmark & validate** – choose between:
	 - `benchmark_inference.py` for large-scale timing/accuracy sweeps on CSV data.
	 - `image_folder_inference.py` for end-to-end Mediapipe + classifier validation on stored images.
	 - Use `benchmark_database.py` to keep a running history of benchmark runs.

## Script Reference

### `collect_data.py`
- Opens the default webcam, runs Mediapipe Hands (up to two hands), and saves 126 landmark values (Left + Right) along with the gesture label.
- Skips low-confidence detections (`score < 0.7`), ignores ambiguous two-right/left-hand frames, and enforces consistent ordering (Left first).
- CLI prompts for `gesture_label`, `sample_limit` (use `-1` for unlimited), and the output CSV path; change `FRAME_SKIP` to control downsampling.
- Overlay text on the preview window shows recording status and sample counts; press `Esc` to stop.

### `augment_images.py`
- Class `GestureImageAugmenter` applies rotation, translation, zoom, brightness/contrast shifts, Gaussian noise/blur, color jitter, shadows, and perspective tweaks.
- Runs in-place within each gesture subfolder: original images stay untouched; augmented versions receive `_aug_<idx>_<ops>.jpg` suffixes.
- Key arguments:
	- `--input_dir ./gestures` – root folder with class subdirectories.
	- `--augmentations_per_image 5` – how many random augmentation combos per original.
	- `--seed 42` – reproducible randomness.
	- `--stats_only` – report original/augmented counts without creating images.
- Use `python augment_images.py --input_dir gestures --augmentations_per_image 8` to bulk augment.

### `extract_landmarks.py`
- Recursively walks every subfolder under `gestures/`, loads each image, and runs Mediapipe Hands in static-image mode.
- Uses the same landmark extraction logic as the live recorder and filters frames lacking one Left + Right pair.
- Randomly shuffles collected samples per gesture, splits into train/test via `split_ratio` (default 0.7), and writes CSV rows `[label, 126 features...]`.
- Run with defaults: `python extract_landmarks.py` (produces `gestures_train.csv` and `gestures_test.csv`). Pass custom paths via function arguments if importing.

### `train_model.py`
- Reads `gestures_train.csv`, builds `label_map` (string label → class index), and stratifies an internal 70/30 train-test split using `train_test_split`.
- Saves the split as `training_dataset.csv` & `testing_dataset.csv` so downstream scripts can inspect the exact samples used.
- Network: `Dense(128, relu) → Dropout(0.3) → Dense(64, relu) → Dropout(0.2) → Dense(num_classes, softmax)` trained for 20 epochs with Adam + categorical cross-entropy.
- Persists `gesture_model.h5` and `label_map.npy`; prints a ✅ confirmation on success.
- Run: `python train_model.py`. To train on a different CSV, edit `DATA_FILE` at the top or refactor into a function call.

### `tfllite_conevrtor.py`
- Minimal converter that loads `gesture_model.h5`, runs `tf.lite.TFLiteConverter.from_keras_model`, and writes `your_model.tflite`.
- Uncomment `converter.optimizations = [tf.lite.Optimize.DEFAULT]` to enable post-training quantization heuristics.
- Run: `python tfllite_conevrtor.py` once the `.h5` exists.

## Benchmarking & Evaluation Suite (`classification_model/benchmarking/`)

### `benchmark_inference.py`
- Full-featured runner for CSV-based latency and accuracy benchmarking on CPU/GPU/NPU.
- Detects available devices, configures TensorFlow accordingly, warms up the model, then times every sample.
- Outputs:
	- Markdown report summarizing per-device stats.
	- PNG plots (distribution, per-gesture violin/heatmap, device comparison).
	- Detailed CSV with per-sample timings and predictions.
	- JSON summary with overall/per-gesture metrics.
	- Entries appended to `benchmark_complete_data.csv` via `BenchmarkDatabase`.
- Usage examples:
```bash
python benchmarking/benchmark_inference.py --model gesture_model.h5 --device cpu
python benchmarking/benchmark_inference.py --model your_model.tflite --device all --dataset gestures_train.csv
```

### `benchmark_database.py`
- Provides the `BenchmarkDatabase` helper used by the main benchmark script.
- Ensures `benchmark_complete_data.csv` exists with rich metadata columns (device info, latency stats, accuracy, FPS).
- Includes helper methods to append new runs, fetch summaries, and print recents.

### `image_folder_inference.py`
- Re-runs the entire Mediapipe → classifier pipeline on stored images to validate end-to-end accuracy.
- Accepts Keras or TFLite models, checks confidence thresholds, and tracks per-image inference time.
- Generates three artifacts by default: `image_inference_results.csv` (per image), `image_inference_stats.csv` (aggregate latency), and `image_inference_summary.json` (accuracy counts per gesture).
- Typical run:
```bash
python benchmarking/image_folder_inference.py \
	--model your_model.tflite \
	--label-map label_map.npy \
	--image-dir gestures \
	--min-confidence 0.6
```

## Generated Artifacts & Their Roles
- `gestures_train.csv` / `gestures_test.csv` – master datasets derived from images.
- `training_dataset.csv` / `testing_dataset.csv` – actual train/test splits used during the most recent `train_model.py` run.
- `gesture_model.h5` – Keras model for experimentation and re-training.
- `label_map.npy` – mapping between gesture names and numeric class IDs (required by inference/benchmark scripts).
- `your_model.tflite` – deployment-ready classifier for edge devices.
- Benchmark outputs (`*.md`, `*.json`, `*.csv`, `*.png`) – saved under `classification_model/benchmarking/` for reproducibility.

## Tips & Troubleshooting
- **Mediapipe detections** – ensure good lighting and keep both hands fully visible; low-confidence or duplicate-handed frames are discarded.
- **Dataset balance** – use `augment_images.py` before `extract_landmarks.py` to avoid class imbalance during training.
- **TensorFlow Lite delegation** – when targeting Edge TPU/NNAPI, install the correct runtime and update `benchmark_inference.py` to locate shared libraries if needed.
- **Benchmark history** – use `python benchmarking/benchmark_database.py` directly to print summaries of past runs.
- **GPU memory errors** – `benchmark_inference.py` enables memory growth, but you can further limit batch sizes or switch to CPU via `--device cpu`.

This README should give you enough context to modify or extend any part of the pipeline without hunting through the individual scripts. Happy experimenting!