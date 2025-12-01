"""Run Mediapipe + classifier inference on gesture image folders.

This script walks through an image dataset arranged as:
    gestures/
        gesture_name_1/
            *.jpg, *.png, ...
        gesture_name_2/
            ...

For each image it:
  1. Runs Mediapipe Hands to extract 3D landmarks.
  2. Feeds the landmarks into the trained classifier (Keras .h5/.keras or TFLite .tflite).
  3. Compares predictions against the folder label and records metrics.

A detailed CSV plus console summary are produced.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

# Ensure we can import helper utilities from the project root
CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from extract_landmarks import extract_hand_landmarks  # noqa: E402

DEFAULT_MODEL_PATH = ROOT_DIR / "your_model.tflite"
DEFAULT_LABEL_MAP = ROOT_DIR / "label_map.npy"
DEFAULT_IMAGE_DIR = ROOT_DIR / "gestures"
DEFAULT_OUTPUT_CSV = CURRENT_DIR / "image_inference_results.csv"
DEFAULT_SUMMARY_JSON = CURRENT_DIR / "image_inference_summary.json"
DEFAULT_STATS_CSV = CURRENT_DIR / "image_inference_stats.csv"

mp_hands = mp.solutions.hands


@dataclass
class ImageResult:
    gesture: str
    image_path: str
    predicted: Optional[str]
    confidence: Optional[float]
    correct: Optional[bool]
    status: str
    inference_ms: Optional[float]


class ImageFolderInference:
    """Run Mediapipe + classifier inference over gesture image folders."""

    def __init__(
        self,
        model_path: Path,
        label_map_path: Path,
        image_root: Path,
        min_confidence: float = 0.5,
        output_csv: Optional[Path] = None,
        summary_json: Optional[Path] = None,
        stats_csv: Optional[Path] = None,
    ) -> None:
        self.model_path = model_path
        self.label_map_path = label_map_path
        self.image_root = image_root
        self.min_confidence = min_confidence
        self.output_csv = output_csv
        self.summary_json = summary_json
        self.stats_csv = stats_csv

        self.label_map = self._load_label_map(label_map_path)
        self.inv_label_map = {v: k for k, v in self.label_map.items()}
        self.model_type = self._detect_model_type()
        self._load_model()

        self.results: List[ImageResult] = []
        self.summary: Dict[str, Dict[str, float]] = {
            "total_images": 0,
            "processed": 0,
            "skipped": 0,
            "correct": 0,
            "per_gesture": {},
        }

    @staticmethod
    def _load_label_map(path: Path) -> Dict[str, int]:
        data = np.load(path, allow_pickle=True)
        if isinstance(data, np.lib.npyio.NpzFile):
            raise ValueError("Expected .npy label map, got npz archive")
        label_map = data.item()
        return {str(k): int(v) for k, v in label_map.items()}

    def _detect_model_type(self) -> str:
        suffix = self.model_path.suffix.lower()
        if suffix in {".h5", ".keras"}:
            return "keras"
        if suffix == ".tflite":
            return "tflite"
        raise ValueError(f"Unsupported model format: {self.model_path}")

    def _load_model(self) -> None:
        if self.model_type == "keras":
            self.model = tf.keras.models.load_model(self.model_path)
        else:
            self.interpreter = tf.lite.Interpreter(model_path=str(self.model_path))
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()

    def _predict(self, features: np.ndarray) -> Tuple[str, float, np.ndarray]:
        features = features.reshape(1, -1).astype(np.float32)
        if self.model_type == "keras":
            probs = self.model.predict(features, verbose=0)[0]
        else:
            self.interpreter.set_tensor(self.input_details[0]["index"], features)
            self.interpreter.invoke()
            probs = self.interpreter.get_tensor(self.output_details[0]["index"])[0]
            # Convert logits to probabilities if needed
            if probs.ndim == 1 and not np.isclose(probs.sum(), 1.0, atol=1e-3):
                probs = tf.nn.softmax(probs).numpy()

        return (
            self.inv_label_map[int(np.argmax(probs))],
            float(np.max(probs)),
            probs,
        )

    def run(self) -> Dict[str, Dict[str, float]]:
        if not self.image_root.exists():
            raise FileNotFoundError(f"Image root not found: {self.image_root}")

        gesture_dirs = [d for d in sorted(self.image_root.iterdir()) if d.is_dir()]
        if not gesture_dirs:
            raise ValueError(f"No gesture subfolders found in {self.image_root}")

        with mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as hands:
            for gesture_dir in gesture_dirs:
                gesture_name = gesture_dir.name
                images = sorted(
                    [p for p in gesture_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]
                )
                if not images:
                    continue

                gesture_stats = self.summary["per_gesture"].setdefault(
                    gesture_name,
                    {"images": 0, "processed": 0, "correct": 0, "skipped": 0},
                )

                for image_path in images:
                    self.summary["total_images"] += 1
                    gesture_stats["images"] += 1

                    image = cv2.imread(str(image_path))
                    if image is None:
                        self._record_result(gesture_name, image_path, None, None, None, "load-failed", None)
                        self.summary["skipped"] += 1
                        gesture_stats["skipped"] += 1
                        continue

                    pipeline_start = time.perf_counter()
                    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    mp_result = hands.process(rgb)
                    landmarks = extract_hand_landmarks(mp_result)
                    if not landmarks:
                        total_time_ms = (time.perf_counter() - pipeline_start) * 1000.0
                        self._record_result(gesture_name, image_path, None, None, None, "no-landmarks", total_time_ms)
                        self.summary["skipped"] += 1
                        gesture_stats["skipped"] += 1
                        continue

                    pred_label, confidence, _ = self._predict(np.array(landmarks))
                    total_time_ms = (time.perf_counter() - pipeline_start) * 1000.0
                    if confidence < self.min_confidence:
                        status = "low-confidence"
                        correct = False
                        predicted = None
                    else:
                        predicted = pred_label
                        correct = predicted == gesture_name
                        status = "ok"
                        if correct:
                            self.summary["correct"] += 1
                            gesture_stats["correct"] += 1

                    self.summary["processed"] += 1
                    gesture_stats["processed"] += 1

                    self._record_result(
                        gesture_name,
                        image_path,
                        predicted,
                        confidence,
                        correct,
                        status,
                        total_time_ms,
                    )

        if self.output_csv:
            self._write_csv(self.output_csv)
        if self.summary_json:
            self._write_summary(self.summary_json)
        if self.stats_csv:
            self._write_stats_csv(self.stats_csv)

        return self.summary

    def _record_result(
        self,
        gesture: str,
        image_path: Path,
        predicted: Optional[str],
        confidence: Optional[float],
        correct: Optional[bool],
        status: str,
        inference_ms: Optional[float],
    ) -> None:
        self.results.append(
            ImageResult(
                gesture=gesture,
                image_path=str(image_path),
                predicted=predicted,
                confidence=confidence,
                correct=correct,
                status=status,
                inference_ms=inference_ms,
            )
        )

    def _write_csv(self, csv_path: Path) -> None:
        csv_path.parent.mkdir(exist_ok=True)
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "gesture",
                "image_path",
                "predicted",
                "confidence",
                "correct",
                "status",
                "inference_ms",
            ])
            for result in self.results:
                writer.writerow([
                    result.gesture,
                    result.image_path,
                    result.predicted or "",
                    "" if result.confidence is None else f"{result.confidence:.4f}",
                    "" if result.correct is None else int(result.correct),
                    result.status,
                    "" if result.inference_ms is None else f"{result.inference_ms:.3f}",
                ])
        print(f"CSV results saved to: {csv_path}")

    def _write_summary(self, json_path: Path) -> None:
        json_path.parent.mkdir(exist_ok=True)
        with open(json_path, "w") as f:
            json.dump(self.summary, f, indent=2)
        print(f"Summary JSON saved to: {json_path}")

    def _write_stats_csv(self, csv_path: Path) -> None:
        times_by_gesture: Dict[str, List[float]] = {}
        for result in self.results:
            if result.inference_ms is None:
                continue
            times_by_gesture.setdefault(result.gesture, []).append(result.inference_ms)

        all_times: List[float] = [t for times in times_by_gesture.values() for t in times]
        csv_path.parent.mkdir(exist_ok=True)
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "gesture",
                "samples",
                "mean_inference_ms",
                "median_inference_ms",
                "min_inference_ms",
                "max_inference_ms",
            ])

            def _write_row(label: str, values: List[float]) -> None:
                if not values:
                    writer.writerow([label, 0, "", "", "", ""])
                    return
                writer.writerow([
                    label,
                    len(values),
                    f"{mean(values):.3f}",
                    f"{median(values):.3f}",
                    f"{min(values):.3f}",
                    f"{max(values):.3f}",
                ])

            _write_row("OVERALL", all_times)
            for gesture in sorted(times_by_gesture.keys()):
                _write_row(gesture, times_by_gesture[gesture])

        print(f"Stats CSV saved to: {csv_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Mediapipe + classifier inference on gesture image folders",
    )
    parser.add_argument(
        "--model",
        default=str(DEFAULT_MODEL_PATH),
        help="Path to classifier model (.h5/.keras/.tflite)",
    )
    parser.add_argument(
        "--label-map",
        default=str(DEFAULT_LABEL_MAP),
        help="Path to label_map.npy",
    )
    parser.add_argument(
        "--image-dir",
        default=str(DEFAULT_IMAGE_DIR),
        help="Root directory containing gesture subfolders",
    )
    parser.add_argument(
        "--output-csv",
        default=str(DEFAULT_OUTPUT_CSV),
        help="Where to store per-image results CSV",
    )
    parser.add_argument(
        "--summary-json",
        default=str(DEFAULT_SUMMARY_JSON),
        help="Where to store aggregated summary JSON",
    )
    parser.add_argument(
        "--stats-csv",
        default=str(DEFAULT_STATS_CSV),
        help="Where to store aggregate inference-time statistics CSV",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.5,
        help="Minimum confidence required to accept a prediction",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    runner = ImageFolderInference(
        model_path=Path(args.model).expanduser(),
        label_map_path=Path(args.label_map).expanduser(),
        image_root=Path(args.image_dir).expanduser(),
        min_confidence=args.min_confidence,
        output_csv=Path(args.output_csv).expanduser(),
        summary_json=Path(args.summary_json).expanduser(),
        stats_csv=Path(args.stats_csv).expanduser(),
    )

    summary = runner.run()

    total = summary["total_images"]
    processed = summary["processed"]
    skipped = summary["skipped"]
    correct = summary["correct"]
    accuracy = (correct / processed * 100.0) if processed else 0.0

    print("\n" + "=" * 80)
    print("IMAGE FOLDER INFERENCE SUMMARY")
    print("=" * 80)
    print(f"Total images   : {total}")
    print(f"Processed      : {processed}")
    print(f"Skipped        : {skipped}")
    print(f"Correct        : {correct}")
    print(f"Accuracy       : {accuracy:.2f}%")
    print("=" * 80)

    print("\nPer-gesture breakdown:")
    for gesture, stats in summary["per_gesture"].items():
        gesture_accuracy = (
            stats["correct"] / stats["processed"] * 100.0 if stats["processed"] else 0.0
        )
        print(
            f"  - {gesture}: images={stats['images']}, processed={stats['processed']}, "
            f"correct={stats['correct']}, skipped={stats['skipped']}, accuracy={gesture_accuracy:.2f}%"
        )


if __name__ == "__main__":
    main()
