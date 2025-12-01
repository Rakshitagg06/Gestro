import os
import sys
import cv2
import numpy as np
import torch
from PIL import Image
from pathlib import Path

# Try to import TFLite
try:
    import tensorflow.lite as tflite
except ImportError:
    try:
        import tflite_runtime.interpreter as tflite
    except ImportError:
        raise ImportError("Please install 'tensorflow' or 'tflite_runtime' to run these models.")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ISOLATED_HAND_DIR = os.path.join(BASE_DIR, "isolated_hand_tracking (Copy)")

if ISOLATED_HAND_DIR not in sys.path:
    sys.path.insert(0, ISOLATED_HAND_DIR)

from app import MediaPipeHandApp
from tflite_inference import TFLiteWrapper
from qai_utils.box_utils import compute_box_affine_crop_resize_matrix
from qai_utils.image_utils import (
    apply_affine_to_coordinates,
    apply_batched_affines_to_frame,
    numpy_image_to_torch,
)

# --- CONFIGURATION ---
MODEL_DIR = Path(ISOLATED_HAND_DIR) / "models"
DETECTOR_PATH = MODEL_DIR / "hand_detector.tflite"
LANDMARKER_PATH = MODEL_DIR / "hand_landmarker.tflite"
ANCHORS_PATH = MODEL_DIR / "anchors_palm.npy"

# Hand landmark connections for drawing skeleton
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


class FixedMediaPipeHandApp(MediaPipeHandApp):
    """Wrapper to fix the batched_selected_landmarks issue in qai_utils"""

    def _run_landmark_detector(self, NHWC_int_numpy_frames, batched_roi_4corners):
        """
        Fixed version that properly handles empty ROI batches, multiple hands,
        and computes is_right_hand.
        """
        batched_selected_landmarks = []
        batched_is_right_hand = []

        # For each input image...
        for batch_idx, roi_4corners in enumerate(batched_roi_4corners):
            if roi_4corners.nelement() == 0:
                # Append empty tensor for batches with no ROI
                batched_selected_landmarks.append(torch.Tensor())
                batched_is_right_hand.append([])
                continue

            affines = compute_box_affine_crop_resize_matrix(
                roi_4corners[:, :3], self.landmark_input_dims
            )

            # Create input images by applying the affine transforms.
            keypoint_net_inputs = numpy_image_to_torch(
                apply_batched_affines_to_frame(
                    NHWC_int_numpy_frames[batch_idx], affines, self.landmark_input_dims
                )
            )

            num_hands = keypoint_net_inputs.shape[0]
            
            # Run inference for EACH hand separately (TFLite model has batch size 1)
            all_ld_scores = []
            all_handedness = []
            all_landmarks = []
            
            for hand_idx in range(num_hands):
                single_hand_input = keypoint_net_inputs[hand_idx:hand_idx+1]
                outputs = self.landmark_detector(single_hand_input)
                
                # Output 0: scores (Shape: [1])
                # Output 1: lr (Shape: [1])  
                # Output 2: landmarks (Shape: [1, 21, 3])
                all_ld_scores.append(outputs[0])
                all_handedness.append(outputs[1])
                all_landmarks.append(outputs[2])
            
            # Concatenate results from all hands
            ld_scores = torch.cat(all_ld_scores, dim=0).squeeze()
            handedness = torch.cat(all_handedness, dim=0).squeeze()
            landmarks = torch.cat(all_landmarks, dim=0)

            # Handle Single-Hand Case (squeeze might make it 0-dim):
            if ld_scores.ndim == 0:
                ld_scores = ld_scores.unsqueeze(0)
            if handedness.ndim == 0:
                handedness = handedness.unsqueeze(0)

            # If landmarks got squeezed too aggressively
            if landmarks.ndim == 2:  # (21, 3) -> (1, 21, 3)
                landmarks = landmarks.unsqueeze(0)

            # Convert [0-1] ranged values of landmarks to integer pixel space.
            landmarks[:, :, 0] *= self.landmark_input_dims[0]
            landmarks[:, :, 1] *= self.landmark_input_dims[1]

            selected_landmarks = []
            is_right_hand_list = []

            for ld_batch_idx in range(landmarks.shape[0]):
                if ld_scores[ld_batch_idx] >= self.min_landmark_score:
                    inverted_affine = torch.from_numpy(
                        cv2.invertAffineTransform(affines[ld_batch_idx])
                    ).float()
                    landmarks[ld_batch_idx][:, :2] = apply_affine_to_coordinates(
                        landmarks[ld_batch_idx][:, :2], inverted_affine
                    )

                    selected_landmarks.append(landmarks[ld_batch_idx])
                    is_right_hand_list.append(handedness[ld_batch_idx].item() > 0.5)

            batched_selected_landmarks.append(
                torch.stack(selected_landmarks, dim=0) if selected_landmarks else torch.Tensor()
            )
            batched_is_right_hand.append(is_right_hand_list)

        return (batched_selected_landmarks, batched_is_right_hand)


def build_hand_pipeline():
    """Build the hand detection and landmark pipeline."""
    # Load Anchors
    if not ANCHORS_PATH.exists():
        raise FileNotFoundError(f"Missing {ANCHORS_PATH}")
    anchors_np = np.load(ANCHORS_PATH)
    anchors = torch.from_numpy(anchors_np)
    print(f"✅ Loaded Anchors: {anchors.shape}")

    # Load TFLite Models via Wrapper
    if not DETECTOR_PATH.exists() or not LANDMARKER_PATH.exists():
        raise FileNotFoundError(f"Missing .tflite models in {MODEL_DIR}")

    print(f"Loading Detector: {DETECTOR_PATH.name}")
    detector = TFLiteWrapper(DETECTOR_PATH)

    print(f"Loading Landmarker: {LANDMARKER_PATH.name}")
    landmarker = TFLiteWrapper(LANDMARKER_PATH)

    # Initialize App
    app = FixedMediaPipeHandApp(
        hand_detector=detector,
        hand_landmark_detector=landmarker,
        anchors=anchors,
        hand_detector_input_spec=detector.get_input_spec(),
        landmark_detector_input_spec=landmarker.get_input_spec(),
    )
    print("✅ Hand Pipeline Initialized")
    return app


# Build hand pipeline
hand_app = build_hand_pipeline()

# Load TFLite classifier model and allocate tensors
CLASSIFIER_MODEL_PATH = os.path.join(BASE_DIR, "your_model.tflite")
LABEL_MAP_PATH = os.path.join(BASE_DIR, "label_map.npy")

# Check if classifier model and label map exist
if not os.path.exists(CLASSIFIER_MODEL_PATH):
    print(f"⚠️ Warning: Classifier model not found at {CLASSIFIER_MODEL_PATH}")
    print("   Please provide 'your_model.tflite' in the project directory.")
    classifier_available = False
else:
    interpreter = tflite.Interpreter(model_path=CLASSIFIER_MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    classifier_available = True
    print("✅ Classifier Model Loaded")

if os.path.exists(LABEL_MAP_PATH):
    label_map = np.load(LABEL_MAP_PATH, allow_pickle=True).item()
    inv_label_map = {v: k for k, v in label_map.items()}
    print("✅ Label Map Loaded")
else:
    print(f"⚠️ Warning: Label map not found at {LABEL_MAP_PATH}")
    inv_label_map = {}


def interpret_handedness(is_right_hand_list, idx):
    """Interpret handedness from the is_right_hand list."""
    if is_right_hand_list is None or idx >= len(is_right_hand_list):
        return None
    return "Right" if is_right_hand_list[idx] else "Left"


def infer_from_geometry(coords):
    """Infer handedness from landmark geometry."""
    wrist = coords[0]
    index_mcp = coords[5]
    pinky_mcp = coords[17]
    cross = (index_mcp[0] - wrist[0]) * (pinky_mcp[1] - wrist[1]) - (
        index_mcp[1] - wrist[1]
    ) * (pinky_mcp[0] - wrist[0])
    return "Right" if cross < 0 else "Left"


def normalize_landmarks(coords, frame_shape):
    """Normalize landmark coordinates to [0, 1] range."""
    height, width = frame_shape[:2]
    normalized = coords.copy()
    if width > 0:
        normalized[:, 0] = np.clip(normalized[:, 0] / width, 0.0, 1.0)
    if height > 0:
        normalized[:, 1] = np.clip(normalized[:, 1] / height, 0.0, 1.0)
    return normalized


def format_feature_vector(landmarks_tensor, is_right_hand_list, frame_shape):
    """Format landmarks into a 126-dimensional feature vector."""
    feature_vec = np.zeros(126, dtype=np.float32)
    if landmarks_tensor is None or landmarks_tensor.nelement() == 0:
        return feature_vec

    landmarks_np = landmarks_tensor.detach().cpu().numpy()

    slots = {"Left": np.zeros(63, dtype=np.float32), "Right": np.zeros(63, dtype=np.float32)}
    filled = {"Left": False, "Right": False}

    max_hands = min(len(landmarks_np), 2)
    for idx in range(max_hands):
        coords = landmarks_np[idx]
        label = interpret_handedness(is_right_hand_list, idx)
        if label not in slots:
            label = infer_from_geometry(coords)

        target_label = label if not filled[label] else ("Left" if not filled["Left"] else "Right")
        normalized = normalize_landmarks(coords, frame_shape)
        slots[target_label] = normalized.astype(np.float32).flatten()
        filled[target_label] = True

    feature_vec[:63] = slots["Left"]
    feature_vec[63:] = slots["Right"]
    return feature_vec


def predict_real_time():
    """Run real-time gesture prediction."""
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("❌ Could not open webcam.")
        return

    frame_count = 0
    last_label = ""

    print("\n🎥 Starting Real-time Prediction... Press ESC or 'q' to quit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert to PIL Image for the pipeline
        pil_frame = Image.fromarray(rgb_frame)

        # Run Pipeline (Detection + Landmarks)
        outputs = hand_app.predict_landmarks_from_image([pil_frame], raw_output=True)

        # Unpack outputs
        boxes, kps, rois, landmarks, is_right_hand = outputs

        # Classify (Every 2nd Frame)
        frame_count += 1
        if frame_count % 2 == 0 and classifier_available:
            if not landmarks or landmarks[0].nelement() == 0:
                features = np.zeros(126, dtype=np.float32)
            else:
                features = format_feature_vector(
                    landmarks[0],
                    is_right_hand[0] if is_right_hand else None,
                    frame.shape
                )

            features = features.reshape(1, -1).astype(np.float32)
            interpreter.set_tensor(input_details[0]['index'], features)
            interpreter.invoke()
            pred = interpreter.get_tensor(output_details[0]['index'])
            idx = int(np.argmax(pred))
            confidence = float(pred[0][idx]) if pred.ndim > 1 else float(pred[idx])
            # Use label map if available, otherwise show class index
            if inv_label_map:
                last_label = inv_label_map.get(idx, f"Class {idx}")
            else:
                last_label = f"Class {idx} ({confidence:.2f})"

        # Draw hand landmarks with skeleton
        if landmarks and landmarks[0].nelement() > 0:
            lm_np = landmarks[0].detach().cpu().numpy()
            irh_list = is_right_hand[0] if is_right_hand else []
            
            for hand_idx, hand_lm in enumerate(lm_np):
                # Determine color based on handedness (Right=Blue, Left=Red)
                is_right = irh_list[hand_idx] if hand_idx < len(irh_list) else True
                base_color = (255, 100, 100) if is_right else (100, 100, 255)  # BGR
                connection_color = (255, 150, 150) if is_right else (150, 150, 255)
                
                # Draw connections (skeleton lines)
                for start_idx, end_idx in HAND_LANDMARK_CONNECTIONS:
                    pt1 = (int(hand_lm[start_idx][0]), int(hand_lm[start_idx][1]))
                    pt2 = (int(hand_lm[end_idx][0]), int(hand_lm[end_idx][1]))
                    cv2.line(frame, pt1, pt2, connection_color, 2, cv2.LINE_AA)
                
                # Draw landmark points
                for i, point in enumerate(hand_lm):
                    x, y = int(point[0]), int(point[1])
                    # Fingertips (4, 8, 12, 16, 20) get larger circles
                    if i in [4, 8, 12, 16, 20]:
                        cv2.circle(frame, (x, y), 6, base_color, -1, cv2.LINE_AA)
                        cv2.circle(frame, (x, y), 6, (255, 255, 255), 1, cv2.LINE_AA)
                    # Wrist (0) gets medium circle
                    elif i == 0:
                        cv2.circle(frame, (x, y), 5, base_color, -1, cv2.LINE_AA)
                        cv2.circle(frame, (x, y), 5, (255, 255, 255), 1, cv2.LINE_AA)
                    else:
                        cv2.circle(frame, (x, y), 4, base_color, -1, cv2.LINE_AA)

        # Draw label with background for better visibility
        if last_label:
            text_size = cv2.getTextSize(last_label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
            cv2.rectangle(frame, (45, 15), (60 + text_size[0], 55), (0, 0, 0), -1)
            cv2.putText(frame, last_label, (50, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
        
        # Show hand count
        hand_count = len(landmarks[0]) if landmarks and landmarks[0].nelement() > 0 else 0
        cv2.putText(frame, f"Hands: {hand_count}", (50, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1, cv2.LINE_AA)

        cv2.imshow("Gesture Prediction", frame)

        key = cv2.waitKey(1)
        if key == 27 or key == ord('q'):  # ESC or 'q'
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    predict_real_time()