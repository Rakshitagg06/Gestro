import cv2
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from app import MediaPipeHandApp
from tflite_inference import TFLiteWrapper

# --- CONFIGURATION ---
MODEL_DIR = Path("models")
DETECTOR_PATH = MODEL_DIR / "hand_detector.tflite"
LANDMARKER_PATH = MODEL_DIR / "hand_landmarker.tflite"
ANCHORS_PATH = MODEL_DIR / "anchors_palm.npy"

class FixedMediaPipeHandApp(MediaPipeHandApp):
    """Wrapper to fix the batched_selected_landmarks issue in qai_utils"""
    
    def _run_landmark_detector(self, NHWC_int_numpy_frames, batched_roi_4corners):
        """
        Fixed version that properly handles empty ROI batches and computes is_right_hand.
        """
        import cv2
        from qai_utils.image_utils import (
            apply_batched_affines_to_frame,
            numpy_image_to_torch,
            apply_affine_to_coordinates,
        )
        from qai_utils.box_utils import compute_box_affine_crop_resize_matrix
        
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

            # --- INFERENCE (Run Once) ---
            outputs = self.landmark_detector(keypoint_net_inputs)

            # --- STRICT MAPPING (Verified) ---
            # Output 0: scores (Shape: [N, 1])
            # Output 1: lr (Shape: [N, 1])
            # Output 2: landmarks (Shape: [N, 21, 3])
            
            ld_scores = outputs[0]
            handedness = outputs[1]
            landmarks = outputs[2]

            # Ensure shapes are correct (remove extra batch dimensions if any)
            ld_scores = ld_scores.squeeze()
            handedness = handedness.squeeze()
            
            # Handle Single-Hand Case:
            # If batch size is 1, squeeze() might make it a scalar (0-d tensor).
            # We need it to be at least 1-d for the loop below to work.
            if ld_scores.ndim == 0:
                ld_scores = ld_scores.unsqueeze(0)
            if handedness.ndim == 0:
                handedness = handedness.unsqueeze(0)
            
            # If landmarks got squeezed too aggressively (rare, but safe to check)
            if landmarks.ndim == 2: # (21, 3) -> (1, 21, 3)
                landmarks = landmarks.unsqueeze(0)

            # Convert [0-1] ranged values of landmarks to integer pixel space.
            landmarks[:, :, 0] *= self.landmark_input_dims[0]
            landmarks[:, :, 1] *= self.landmark_input_dims[1]

            # 1 landmark is predicted for each ROI of each input image.
            # For each region of interest & associated predicted landmarks...
            all_landmarks = []
            is_right_hand_list = []
            
            for ld_batch_idx in range(landmarks.shape[0]):
                # Exclude landmarks that don't meet the appropriate score threshold.
                if ld_scores[ld_batch_idx] >= self.min_landmark_score:
                    # Apply the inverse of affine transform used above to the landmark coordinates.
                    inverted_affine = torch.from_numpy(
                        cv2.invertAffineTransform(affines[ld_batch_idx])
                    ).float()
                    landmarks[ld_batch_idx][:, :2] = apply_affine_to_coordinates(
                        landmarks[ld_batch_idx][:, :2], inverted_affine
                    )

                    # Add the predicted landmarks to our list.
                    all_landmarks.append(landmarks[ld_batch_idx])
                    
                    # Determine if it's a right hand (handedness > 0.5 means right hand)
                    is_right_hand_list.append(handedness[ld_batch_idx].item() > 0.5)

            # Add this batch of landmarks to the output list.
            batched_selected_landmarks.append(
                torch.stack(all_landmarks, dim=0) if all_landmarks else torch.Tensor()
            )
            batched_is_right_hand.append(is_right_hand_list)

        return (batched_selected_landmarks, batched_is_right_hand)

def main():
    print("--- Initializing Isolated Hand Tracking ---")
    
    # 1. Load Anchors
    if not ANCHORS_PATH.exists():
        raise FileNotFoundError(f"Missing {ANCHORS_PATH}")
    anchors_np = np.load(ANCHORS_PATH)
    anchors = torch.from_numpy(anchors_np)
    print(f"✅ Loaded Anchors: {anchors.shape}")

    # 2. Load TFLite Models via Wrapper
    if not DETECTOR_PATH.exists() or not LANDMARKER_PATH.exists():
        raise FileNotFoundError(f"Missing .tflite models in {MODEL_DIR}")
        
    print(f"Loading Detector: {DETECTOR_PATH.name}")
    detector = TFLiteWrapper(DETECTOR_PATH)
    
    print(f"Loading Landmarker: {LANDMARKER_PATH.name}")
    landmarker = TFLiteWrapper(LANDMARKER_PATH)

    # 3. Initialize App
    app = FixedMediaPipeHandApp(
        hand_detector=detector,
        hand_landmark_detector=landmarker,
        anchors=anchors,
        hand_detector_input_spec=detector.get_input_spec(),
        landmark_detector_input_spec=landmarker.get_input_spec(),
    )
    print("✅ App Initialized")

    # 4. Open Webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not open webcam.")
        return

    print("\nStarting Loop... Press 'q' to quit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Mirror the frame horizontally to fix mirrored camera output
        frame = cv2.flip(frame, 1)
        
        # Convert Frame (BGR -> RGB)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # --- FIX: Convert NumPy Array to PIL Image ---
        pil_frame = Image.fromarray(frame_rgb)
        
        # Inference
        # Returns list of numpy arrays (processed frames)
        processed_frames = app.predict_landmarks_from_image([pil_frame])
        
        # Display
        out_frame = processed_frames[0]
        
        # Convert back to BGR for OpenCV
        out_frame_bgr = cv2.cvtColor(out_frame, cv2.COLOR_RGB2BGR)
        
        cv2.imshow('Isolated Hand Tracking', out_frame_bgr)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()