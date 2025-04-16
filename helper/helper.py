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


def create_orthogonal_4d_matrix(shape: tuple, gain: float = 1.0):
    """
    Creates a 4D orthogonal matrix for convolutional layers.

    This function assumes the weight shape is:
        (filter_size, filter_size, input_channels, num_filters)
    and returns a weight tensor of the same shape such that each filter (the last dimension)
    is orthogonal with respect to its flattened version.

    Args:
        shape (tuple): (filter_size, filter_size, input_channels, num_filters)
        gain (float): Gain factor to scale the weights.

    Returns:
        cp.ndarray: Weight tensor of shape (filter_size, filter_size, input_channels, num_filters)
    """
    filter_height, filter_width, in_channels, num_filters = shape

    # We want each filter to be a row, so flatten each filter to a 1D vector.
    # Create a 2D matrix of shape (num_filters, filter_size * filter_size * input_channels)
    flat_shape = (num_filters, filter_height * filter_width * in_channels)
    a = cp.random.randn(*flat_shape)

    # Compute the SVD of the flattened matrix
    u, _, vh = cp.linalg.svd(a, full_matrices=False)

    # Select the matrix with the correct shape.
    # If num_filters >= (filter_height * filter_width * in_channels), then u is (num_filters, num_filters) and we need only the first (filter_height*filter_width*in_channels) columns.
    # If not, vh is (filter_height*filter_width*in_channels, filter_height*filter_width*in_channels). We'll choose the one with shape flat_shape.
    if flat_shape[0] >= flat_shape[1]:
        W_flat = u  # shape: (num_filters, num_filters)
        # Truncate to the first (filter_height*filter_width*in_channels) columns if needed
        W_flat = W_flat[:, :flat_shape[1]]
    else:
        W_flat = vh  # shape: (filter_height*filter_width*in_channels, filter_height*filter_width*in_channels)
        # Truncate to the first num_filters rows if needed
        W_flat = W_flat[:flat_shape[0], :]

    # Now, W_flat has shape (num_filters, filter_height*filter_width*in_channels).
    # Reshape it back to (num_filters, filter_height, filter_width, in_channels)
    W = W_flat.reshape((num_filters, filter_height, filter_width, in_channels))
    # Transpose to get the final shape (filter_height, filter_width, in_channels, num_filters)
    W = W.transpose(1, 2, 3, 0)
    return (W.astype(cp.float32) * gain)

def create_orthogonal_3d_matrix(shape: tuple, gain: float = 1.0):
    """
    Creates a 3D orthogonal matrix for depthwise convolution layers.

    Args:
        shape (tuple): (filter_height, filter_width, input_channels)
        gain (float): Gain factor to scale the weights.

    Returns:
        cp.ndarray: Orthogonal weight tensor of shape (filter_height, filter_width, input_channels)
    """
    filter_height, filter_width, input_channels = shape

    # Flatten the 3D shape into 2D (input_channels, filter_height * filter_width)
    flat_shape = (input_channels, filter_height * filter_width)
    a = cp.random.randn(*flat_shape).astype(cp.float32)

    # Compute the SVD of the flattened matrix
    u, s, vh = cp.linalg.svd(a, full_matrices=False)

    # Select the matrix with orthogonal rows or columns based on shape
    if flat_shape[0] <= flat_shape[1]:
        # Case: input_channels <= filter_height * filter_width
        # Use rows of vh (orthonormal rows)
        W_flat = vh[:flat_shape[0], :]  # shape (input_channels, filter_height * filter_width)
    else:
        # Case: input_channels > filter_height * filter_width
        # Use columns of u (orthonormal columns)
        W_flat = u[:, :flat_shape[1]]  # shape (input_channels, filter_height * filter_width)

    # Reshape back to 3D (input_channels, filter_height, filter_width)
    W = W_flat.reshape(input_channels, filter_height, filter_width)

    # Transpose to (filter_height, filter_width, input_channels)
    W = W.transpose(1, 2, 0)

    return (W * gain).astype(cp.float32)


def create_orthogonal_2d_matrix(shape: tuple, gain: float = 1.0):
    """
    Creates a 2D orthogonal matrix for a fully connected (fc) layer.

    Args:
        shape (tuple): Shape of the weight matrix, e.g. (input_dim, output_dim) or (output_dim, input_dim).
        gain (float): Scaling factor applied to the orthogonal matrix (e.g., gain = sqrt(2) for ReLU).

    Returns:
        cp.ndarray: An orthogonal matrix of the specified shape, scaled by gain.

    Explanation:
        - If shape = (n, m) and n >= m, then the columns of the resulting matrix will be orthonormal,
          i.e. W.T @ W ≈ I.
        - If n < m, then the rows will be orthonormal, i.e. W @ W.T ≈ I.
    """
    # Create a random matrix of the desired shape
    A = cp.random.randn(*shape).astype(cp.float32)

    # Compute the SVD of A
    u, s, vh = cp.linalg.svd(A, full_matrices=False)

    # Depending on the shape, choose u or vh to enforce orthogonality
    if shape[0] >= shape[1]:
        # u has shape (n, n); take the first m columns so that W has shape (n, m)
        W = u[:, :shape[1]]
    else:
        # vh has shape (m, m); take the first n rows so that W has shape (n, m)
        W = vh[:shape[0], :]

    return gain * W


def adadelta_update(param, grad, E_g, E_delta, rho, eps):
    # Update the running average of squared gradients.
    E_g[:] = rho * E_g + (1 - rho) * (grad ** 2)
    # Compute the parameter update.
    update = - cp.sqrt(E_delta + eps) / cp.sqrt(E_g + eps) * grad
    # Apply the update.
    param += update
    # Update the running average of squared updates.
    E_delta[:] = rho * E_delta + (1 - rho) * (update ** 2)
    
def rmsprop_update(param, grad, cache, decay_rate, learning_rate, eps):
    """
    RMSProp update rule.

    Args:
        param (cp.ndarray): Parameter to be updated.
        grad (cp.ndarray): Gradient for the parameter.
        cache (cp.ndarray): Running average of squared gradients.
        decay_rate (float): Decay rate for the running average (typically around 0.9).
        learning_rate (float): Learning rate for the update.
        eps (float): Small epsilon to avoid division by zero.

    Returns:
        Updated cache (cp.ndarray). The parameter `param` is updated in-place.
    """
    # Update running average of squared gradients.
    cache[:] = decay_rate * cache + (1 - decay_rate) * (grad ** 2)
    # Compute the RMSProp update.
    update = - learning_rate * grad / cp.sqrt(cache + eps)
    param += update
    return cache