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
        pred_box (cp.ndarray): Array of shape (2,) containing [width, height] for the predicted box
        anchors (cp.ndarray): Array of shape (N, 2) containing [width, height] for each anchor

    Returns:
        cp.ndarray: IoU values for each anchor (shape: (N,)).
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



def create_orthogonal_matrix(shape: tuple, gain: float = 1.0):
    """
    Creates an orthogonal matrix of the given shape

    Args:
        shape (tuple): shape of the matrix to be created
        gain (float): gain to be applied to the matrix

    Returns:
        cp.ndarray
        """

    # reshape to 2D matrix (out_channels, in_channels * kernel_h * kernel_w)
    flat_shape = (shape[0], int(cp.prod(shape[1:])))
    a = cp.random.randn(*flat_shape)
    
    u, _, vh = cp.linalg.svd(a, full_matrices=False)
    
    # Choose orthogonal matrix based on shape
    if flat_shape[0] >= flat_shape[1]:
        W = u  # orthogonal columns
    else:
        W = vh  # orthogonal rows
    
    return W.reshape(shape).astype(cp.float32) * gain

