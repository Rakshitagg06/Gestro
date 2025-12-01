# app.py (Final Correct Version)
from __future__ import annotations
from collections.abc import Callable
import numpy as np
import torch
from PIL.Image import Image

# --- LOCAL ISOLATED IMPORTS ---
from qai_utils.base_app import MediaPipeApp
from qai_utils import box_utils
from qai_utils import draw_utils
from qai_utils import image_utils

# --- CONSTANTS ---
DETECT_DSCALE = 2.5
DETECT_DXY = 0.5
DETECT_SCORE_SLIPPING_THRESHOLD = 100
WRIST_CENTER_KEYPOINT_INDEX = 0
MIDDLE_FINDER_KEYPOINT_INDEX = 2
ROTATION_VECTOR_OFFSET_RADS = np.pi / 2

HAND_LANDMARK_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4), (5, 6), (6, 7), (7, 8),
    (9, 10), (10, 11), (11, 12), (13, 14), (14, 15), (15, 16),
    (17, 18), (18, 19), (19, 20), (0, 5), (5, 9), (9, 13),
    (13, 17), (0, 17),
]

class MediaPipeHandApp(MediaPipeApp):
    def __init__(
        self,
        hand_detector: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
        hand_landmark_detector: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
        anchors: torch.Tensor, 
        hand_detector_input_spec: tuple[int, int],
        landmark_detector_input_spec: tuple[int, int],
        min_detector_hand_box_score: float = 0.75,
        nms_iou_threshold: float = 0.3,
        min_landmark_score: float = 0.5,
    ):
        super().__init__(
            hand_detector,
            anchors,
            hand_landmark_detector,
            hand_detector_input_spec,
            landmark_detector_input_spec,
            WRIST_CENTER_KEYPOINT_INDEX,
            MIDDLE_FINDER_KEYPOINT_INDEX,
            ROTATION_VECTOR_OFFSET_RADS,
            DETECT_DXY,
            DETECT_DSCALE,
            min_detector_hand_box_score,
            DETECT_SCORE_SLIPPING_THRESHOLD,
            nms_iou_threshold,
            min_landmark_score,
            HAND_LANDMARK_CONNECTIONS,
        )

    # --- CRITICAL FIX: Overridden to handle Tuple unpacking manually ---
    def predict_landmarks_from_image(
        self,
        pixel_values_or_image: torch.Tensor | np.ndarray | Image | list[Image],
        raw_output: bool = False,
    ):
        # 1. Convert Input
        NHWC_int_numpy_frames, NCHW_fp32_torch_frames = image_utils.app_to_net_image_inputs(
            pixel_values_or_image
        )

        # 2. Run Model Logic
        # The base run_model returns: (boxes, keypoints, rois, (landmarks, handedness))
        outputs = self.run_model(NCHW_fp32_torch_frames)

        batched_selected_boxes = outputs[0]
        batched_selected_keypoints = outputs[1]
        batched_roi_4corners = outputs[2]
        
        # 3. Unpack the nested tuple
        remainder = outputs[3:]
        
        # Check if we got a nested tuple [(landmarks, handedness)]
        if len(remainder) == 1 and isinstance(remainder[0], tuple):
            batched_selected_landmarks = remainder[0][0]
            batched_is_right_hand = remainder[0][1]
        else:
            # Fallback
            batched_selected_landmarks = remainder[0]
            batched_is_right_hand = remainder[1]

        if raw_output:
            return (
                batched_selected_boxes,
                batched_selected_keypoints,
                batched_roi_4corners,
                batched_selected_landmarks,
                batched_is_right_hand,
            )

        # 4. Draw Predictions (Passing arguments explicitly)
        self._draw_predictions(
            NHWC_int_numpy_frames,
            batched_selected_boxes,
            batched_selected_keypoints,
            batched_roi_4corners,
            batched_selected_landmarks,
            batched_is_right_hand, 
        )

        return NHWC_int_numpy_frames

    def _draw_predictions(
        self,
        NHWC_int_numpy_frames: list[np.ndarray],
        batched_selected_boxes: list[torch.Tensor],
        batched_selected_keypoints: list[torch.Tensor],
        batched_roi_4corners: list[torch.Tensor],
        batched_selected_landmarks: list[torch.Tensor],
        batched_is_right_hand: list[list[bool]],
    ):
        for batch_idx in range(len(NHWC_int_numpy_frames)):
            image = NHWC_int_numpy_frames[batch_idx]
            ld = batched_selected_landmarks[batch_idx]
            box = batched_selected_boxes[batch_idx]
            kp = batched_selected_keypoints[batch_idx]
            roi_4corners = batched_roi_4corners[batch_idx]
            irh = batched_is_right_hand[batch_idx]

            # Skip drawing boxes/roi for cleaner look - just draw landmarks
            if ld.nelement() != 0 and len(irh) != 0:
                self._draw_landmarks(image, ld, irh)

    def _draw_landmarks(
        self,
        NHWC_int_numpy_frame: np.ndarray,
        landmarks: torch.Tensor,
        is_right_hand: list[bool],
    ):
        import cv2
        
        for ldm, irh in zip(landmarks, is_right_hand):
            points = ldm[:, :2].cpu().numpy() if isinstance(ldm, torch.Tensor) else ldm[:, :2]
            
            # Colors: Right hand = Blue tones, Left hand = Red tones (RGB format)
            if irh:
                connection_color = (100, 150, 255)  # Light blue
                point_color = (50, 100, 255)        # Blue
                tip_color = (0, 80, 255)            # Darker blue
            else:
                connection_color = (255, 150, 100)  # Light red/orange
                point_color = (255, 100, 50)        # Red/orange
                tip_color = (255, 80, 0)            # Darker orange
            
            # Draw connections first (so points are on top)
            if self.landmark_connections:
                for start_idx, end_idx in self.landmark_connections:
                    pt1 = (int(points[start_idx][0]), int(points[start_idx][1]))
                    pt2 = (int(points[end_idx][0]), int(points[end_idx][1]))
                    cv2.line(NHWC_int_numpy_frame, pt1, pt2, connection_color, 2, cv2.LINE_AA)
            
            # Draw landmark points
            fingertips = [4, 8, 12, 16, 20]
            for i, point in enumerate(points):
                x, y = int(point[0]), int(point[1])
                if i in fingertips:
                    # Fingertips: larger circles with white outline
                    cv2.circle(NHWC_int_numpy_frame, (x, y), 6, tip_color, -1, cv2.LINE_AA)
                    cv2.circle(NHWC_int_numpy_frame, (x, y), 6, (255, 255, 255), 1, cv2.LINE_AA)
                elif i == 0:
                    # Wrist: medium circle with white outline
                    cv2.circle(NHWC_int_numpy_frame, (x, y), 5, point_color, -1, cv2.LINE_AA)
                    cv2.circle(NHWC_int_numpy_frame, (x, y), 5, (255, 255, 255), 1, cv2.LINE_AA)
                else:
                    # Other joints: smaller circles
                    cv2.circle(NHWC_int_numpy_frame, (x, y), 3, point_color, -1, cv2.LINE_AA)