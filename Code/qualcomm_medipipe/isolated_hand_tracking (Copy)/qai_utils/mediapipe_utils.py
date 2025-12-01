# ---------------------------------------------------------------------
# Isolated mediapipe_utils.py
# ---------------------------------------------------------------------

import torch

# REMOVED: MediaPipePyTorchAsRoot (Not needed for local inference)
# REMOVED: trace_mediapipe (Not needed for local inference)

def decode_preds_from_anchors(
    box_coords: torch.Tensor, img_size: tuple[int, int], anchors: torch.Tensor
):
    """
    Decode predictions using the provided anchors.
    
    CRITICAL: This logic must match the model training exactly.
    """
    assert box_coords.shape[-1] == anchors.shape[-1] == 2
    assert box_coords.shape[-3] == anchors.shape[-3]

    w_size, h_size = img_size
    anchors_x, anchors_y, anchors_w, anchors_h = (
        anchors[..., 0, 0],
        anchors[..., 0, 1],
        anchors[..., 1, 0],
        anchors[..., 1, 1],
    )
    expanded_anchors_shape = [*list(anchors_w.shape), 1]

    # Determine real center X and Y, as well as real pixel W and H
    box_coords[..., 0, 0] = (
        box_coords[..., 0, 0] / w_size * anchors_w + anchors_x
    )  # x_center
    box_coords[..., 0, 1] = (
        box_coords[..., 0, 1] / h_size * anchors_h + anchors_y
    )  # y_center
    box_coords[..., 1, 0] = box_coords[..., 1, 0] / w_size * anchors_w  # w
    box_coords[..., 1, 1] = box_coords[..., 1, 1] / h_size * anchors_h  # h

    # Get X and Y values of keypoints
    box_coords[..., 2:, 0] = box_coords[..., 2:, 0] / w_size * anchors_w.view(
        expanded_anchors_shape
    ) + anchors_x.view(expanded_anchors_shape)
    box_coords[..., 2:, 1] = box_coords[..., 2:, 1] / h_size * anchors_h.view(
        expanded_anchors_shape
    ) + anchors_y.view(expanded_anchors_shape)