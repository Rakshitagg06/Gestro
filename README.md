# 🏎️ Racing Car Game - GESTRO-> Gesture Controlled Android Game

A gesture-controlled racing game for Android that uses real-time hand gesture recognition powered by MediaPipe and Qualcomm Neural Processing SDK (SNPE/QNN). Control your car using intuitive hand gestures captured through your device's camera!

<p align="center">
  <img src="https://img.shields.io/badge/Platform-Android-green?style=for-the-badge&logo=android" alt="Android"/>
  <img src="https://img.shields.io/badge/Language-Kotlin-purple?style=for-the-badge&logo=kotlin" alt="Kotlin"/>
  <img src="https://img.shields.io/badge/ML-TensorFlow%20Lite-orange?style=for-the-badge&logo=tensorflow" alt="TensorFlow Lite"/>
  <img src="https://img.shields.io/badge/Accelerator-Qualcomm%20QNN-red?style=for-the-badge" alt="Qualcomm QNN"/>
</p>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Architecture](#-architecture)
- [Hand Gesture Controls](#-hand-gesture-controls)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Requirements](#-requirements)
- [Installation](#-installation)
- [Usage](#-usage)
- [Performance Benchmarks](#-performance-benchmarks)
- [Model Details](#-model-details)
- [Troubleshooting](#-troubleshooting)
- [License](#-license)

---

## 🎯 Overview

Racing Car Game is an innovative Android application that combines computer vision with mobile gaming. Instead of traditional touch controls, players use **hand gestures** to steer and control the speed of their racing car. The app leverages:

- **MediaPipe Hand Landmarker** for real-time 21-point hand skeleton detection
- **Custom TFLite Gesture Classifier** trained on gesture landmarks
- **Qualcomm QNN/HTP acceleration** for optimized inference on Snapdragon devices

This creates an immersive, hands-free gaming experience while demonstrating edge AI capabilities on mobile devices.

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🖐️ **Real-time Hand Tracking** | Detects and tracks both hands simultaneously using MediaPipe |
| 🎮 **Gesture-based Controls** | Intuitive steering and speed control via hand gestures |
| ⚡ **Hardware Acceleration** | Qualcomm QNN/HTP support for low-latency inference |
| 📱 **Native Android** | Built with Kotlin for optimal performance |
| 🔧 **Configurable Settings** | Adjustable game and gesture sensitivity settings |
| 📊 **Multi-delegate Support** | CPU, GPU, and NNAPI delegate options |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Camera Input                              │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Hand Detection Pipeline                       │
│  ┌─────────────────┐    ┌─────────────────┐    ┌──────────────┐ │
│  │  Palm Detector  │ → │ Hand Landmarker │ → │ 21 Landmarks │  │
│  │   (TFLite)      │    │    (TFLite)     │    │  per Hand    │  │
│  └─────────────────┘    └─────────────────┘    └──────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Gesture Classification                         │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │  Input: 126-dim feature vector (63 per hand × 2 hands)      ││
│  │  Model: Dense Neural Network (TFLite/DLC)                   ││
│  │  Output: Gesture Class (Accelerate, Brake, Left, Right...)  ││
│  └─────────────────────────────────────────────────────────────┘│
│                    ↓ Qualcomm QNN/HTP Acceleration ↓            │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                        Game Engine                               │
│  ┌─────────────────┐    ┌─────────────────┐    ┌──────────────┐ │
│  │  GameView       │ ← │ Gesture Mapper  │ ← │ Car Controls │  │
│  │  (Canvas)       │    │                 │    │              │  │
│  └─────────────────┘    └─────────────────┘    └──────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🎮 Hand Gesture Controls

### Control Mapping

| Hand | Function | Gestures |
|------|----------|----------|
| **Left Hand** 👈 | **Speed Control** | Accelerate / Decelerate / Stop |
| **Right Hand** 👉 | **Steering** | Turn Left / Turn Right / Straight |

### Supported Gestures

```
┌────────────────────────────────────────────────┐
│           LEFT HAND (Speed)                    │
├────────────────────────────────────────────────┤
│  🖐️ Open Palm    → Accelerate                  │
│  ✊ Closed Fist   → Brake/Decelerate            │
│  ✋ Stop Sign     → Stop                        │
└────────────────────────────────────────────────┘

┌────────────────────────────────────────────────┐
│           RIGHT HAND (Direction)               │
├────────────────────────────────────────────────┤
│  👈 Point Left   → Steer Left                  │
│  👉 Point Right  → Steer Right                 │
│  🖐️ Open Palm    → Go Straight                 │
└────────────────────────────────────────────────┘
```

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| **Platform** | Android (API 24+) |
| **Language** | Kotlin |
| **Build System** | Gradle (Kotlin DSL) |
| **Hand Detection** | MediaPipe Hand Landmarker |
| **ML Inference** | TensorFlow Lite |
| **HW Acceleration** | Qualcomm QNN (HTP/DSP/GPU) |
| **Camera** | CameraX API |
| **Graphics** | Android Canvas / Custom GameView |

---

## 📁 Project Structure

```
Racingcargame/
├── app/
│   ├── src/main/
│   │   ├── java/com/example/racingcargame/
│   │   │   ├── MainActivity.kt           # Main activity & game lifecycle
│   │   │   ├── ConfigActivity.kt         # Settings & configuration
│   │   │   ├── GameView.kt               # Custom game rendering view
│   │   │   ├── CameraUtils.kt            # CameraX setup & image capture
│   │   │   ├── HandDetectorProcessor.kt  # Palm detection preprocessing
│   │   │   ├── HandLandmarkProcessor.kt  # Landmark extraction
│   │   │   ├── GestureClassifier.kt      # TFLite gesture classification
│   │   │   ├── QnnHandPipeline.kt        # QNN-accelerated pipeline
│   │   │   └── QnnHandUtils.kt           # QNN utility functions
│   │   │
│   │   ├── assets/
│   │   │   ├── hand_landmarker.task      # MediaPipe hand task bundle
│   │   │   ├── mediapipe_hand-handdetector.tflite
│   │   │   ├── mediapipe_hand-handlandmarkdetector.tflite
│   │   │   ├── gesture.tflite            # Gesture classification model
│   │   │   ├── gesture_model.dlc         # Qualcomm DLC model
│   │   │   ├── anchors.bin               # Detection anchor boxes
│   │   │   └── labels.txt                # Gesture class labels
│   │   │
│   │   ├── jniLibs/arm64-v8a/            # Qualcomm QNN native libraries
│   │   │   ├── libQnnHtp.so
│   │   │   ├── libQnnHtpPrepare.so
│   │   │   ├── libQnnHtpV75Skel.so
│   │   │   ├── libQnnHtpV75Stub.so
│   │   │   └── libQnnSystem.so
│   │   │
│   │   ├── res/
│   │   │   ├── layout/
│   │   │   │   ├── activity_main.xml
│   │   │   │   └── activity_config.xml
│   │   │   ├── drawable/                 # Game assets & icons
│   │   │   └── values/                   # Themes, strings, colors
│   │   │
│   │   └── AndroidManifest.xml
│   │
│   ├── build.gradle.kts                  # App-level build config
│   └── proguard-rules.pro
│
├── build.gradle.kts                      # Project-level build config
├── settings.gradle.kts
├── gradle.properties
└── README.md
```

---

## 📋 Requirements

### Hardware
- Android device with ARM64 processor (arm64-v8a)
- Front or rear camera
- **Recommended**: Qualcomm Snapdragon chipset for QNN acceleration

### Software
- Android 7.0 (API 24) or higher
- Android Studio Arctic Fox or newer (for development)

### Tested Devices & Performance

| Device | Chipset | MediaPipe Delegate | FPS | Accuracy |
|--------|---------|-------------------|-----|----------|
| Nothing A015 | Snapdragon 7+ Gen 2 | GPU + NNAPI | **28 FPS** | 84.1% |
| OnePlus LE2111 | Snapdragon 888 | GPU + CPU | **16 FPS** | 84.1% |
| Xiaomi 23122PCD1I | Snapdragon 8 Gen 2 | GPU + NNAPI | **15 FPS** | 84.1% |
| realme RMX3997 | Dimensity 7050 | GPU + CPU | **14 FPS** | 84.7% |
| vivo V2109 | Snapdragon 778G | GPU + NNAPI | **11 FPS** | 84.1% |
| Samsung SM-F127G | Exynos 850 | GPU + GPU | **6 FPS** | 84.7% |

---

## 🚀 Installation

### From Source

1. **Clone the repository**
   ```bash
   git clone https://github.com/Embedded-Systems-Workshop/final-codes-15_dominator069.git
   cd final-codes-15_dominator069/Code/Racingcargame
   ```

2. **Open in Android Studio**
   - Launch Android Studio
   - Select "Open an existing project"
   - Navigate to the `Racingcargame` folder

3. **Sync Gradle**
   - Android Studio will automatically sync dependencies
   - Ensure all dependencies are resolved

4. **Build & Run**
   ```bash
   ./gradlew assembleDebug
   # or use Android Studio's Run button
   ```

5. **Install on device**
   ```bash
   adb install app/build/outputs/apk/debug/app-debug.apk
   ```

### Pre-built APK
Download the latest APK from the [Releases](../../releases) page.

---

## 🎮 Usage

### Getting Started

1. **Launch the app** on your Android device
2. **Grant camera permissions** when prompted
3. **Position your hands** in front of the camera
4. **Use gestures** to control the car:
   - Left hand controls speed (accelerate/brake)
   - Right hand controls steering (left/right)

### Configuration

Access the **Settings** menu to configure:
- Gesture sensitivity
- Camera selection (front/back)
- Delegate selection (CPU/GPU/NNAPI)
- Game difficulty

### Tips for Best Performance

- ✅ Ensure good lighting conditions
- ✅ Keep hands within camera frame
- ✅ Maintain consistent hand distance from camera
- ✅ Use contrasting background for better detection
- ✅ Enable GPU/NNAPI delegate for faster inference

---

## 📊 Performance Benchmarks

### Inference Pipeline Latency

| Stage | Mean Time | Description |
|-------|-----------|-------------|
| Hand Detection | ~15-30ms | Palm detection + ROI extraction |
| Landmark Estimation | ~10-20ms | 21-point hand skeleton |
| Gesture Classification | ~2-5ms | Neural network inference |
| **Total Pipeline** | **~30-60ms** | End-to-end latency |

### Qualcomm SNPE/QNN Acceleration

| Runtime | Precision | Mean Latency | Accuracy |
|---------|-----------|--------------|----------|
| CPU | FP32 | 92.86ms | 99.54% |
| GPU | FP32 | 67.11ms | 99.54% |
| GPU | FP16 | 65.64ms | 99.54% |
| HTP/NPU | INT8 | **35.57ms** | 99.54% |
| HTP/NPU | W8A16 | **31.16ms** | 99.54% |

> 💡 **Note**: HTP (Hexagon Tensor Processor) provides up to **3x speedup** compared to CPU inference while maintaining accuracy.

---

## 🧠 Model Details

### Gesture Classification Model

| Property | Value |
|----------|-------|
| Input Shape | (1, 126) - 63 landmarks × 2 hands |
| Output | Softmax over gesture classes |
| Architecture | Dense(128) → Dropout(0.3) → Dense(64) → Dropout(0.2) → Dense(N) |
| Training Data | Custom gesture dataset with augmentation |
| Accuracy | 99.54% on test set |

### Model Formats

| Format | File | Use Case |
|--------|------|----------|
| TensorFlow Lite | `gesture.tflite` | CPU/GPU inference |
| Qualcomm DLC | `gesture_model.dlc` | QNN/HTP acceleration |
| MediaPipe Task | `hand_landmarker.task` | Hand detection bundle |

---

## 🔧 Troubleshooting

### Common Issues

| Problem | Solution |
|---------|----------|
| Camera not working | Check camera permissions in Settings |
| Low FPS | Switch to GPU/NNAPI delegate |
| Hand not detected | Improve lighting, check hand position |
| App crashes on launch | Ensure device meets minimum requirements |
| Gestures not recognized | Recalibrate hand distance from camera |

### Debug Mode

Enable debug overlay to visualize:
- Hand landmarks
- Bounding boxes
- Gesture predictions
- FPS counter

---

## 🔗 Related Modules

This game is part of a larger gesture recognition project:

| Module | Description |
|--------|-------------|
| [`classification_model/`](../classification_model/) | Gesture classifier training pipeline |
| [`qualcomm_medipipe/`](../qualcomm_medipipe/) | TFLite hand tracking for Qualcomm |
| [`dynamic gestures/`](../dynamic%20gestures/) | LSTM-based dynamic gesture recognition |
| [`quantization_and_benchmarking/`](../quantization_and_benchmarking/) | Model optimization & SNPE conversion |


---