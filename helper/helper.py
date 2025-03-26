import os


on_cpu = os.environ.get("USE_GPU") != False and os.environ.get("USE_GPU") != 'True' and os.environ.get("USE_GPU") != 'True'

if on_cpu:
    import numpy as cp
else:
    import cupy as cp

def calc_conv_layer_out_dim(image_dim: tuple, padding: int, kernel_size: int, stride: int):
    image_width, image_height = image_dim
    out_width = int((image_width + 2 * padding - kernel_size) / stride + 1)
    out_height = int((image_height + 2 * padding - kernel_size) / stride + 1)
    return out_width, out_height


def compute_iou_for_anchors(pred_box: cp.ndarray, anchors: cp.ndarray):
    """
    Computes IoU between a single predicted box and multiple anchor boxes.

    Args:
        pred_box (np.ndarray): Array of shape (2,) containing [width, height] for the predicted box
        anchors (np.ndarray): Array of shape (N, 2) containing [width, height] for each anchor

    Returns:
        np.ndarray: IoU values for each anchor (shape: (N,)).
    """
    intersect_w = cp.minimum(anchors[:, 0], pred_box[0])
    intersect_h = cp.minimum(anchors[:, 1], pred_box[1])
    inter_area = intersect_w * intersect_h

    area_anchor = anchors[:, 0] * anchors[:, 1]
    area_pred = pred_box[0] * pred_box[1]

    # Union area
    union_area = area_anchor + area_pred - inter_area

    # Compute IoU
    iou = inter_area / union_area
    return iou
