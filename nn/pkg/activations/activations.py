import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(sigmoid_x):
    return sigmoid_x * (1 - sigmoid_x)

# numerically stable softmax
def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)

def softmax_derivative(softmax_output, grad_soft):
    """
    Given a softmax output vector s (shape: (C,)) and upstream gradient grad_soft (shape: (C,)),
    compute the gradient with respect to the softmax input z.
    The Jacobian of softmax is: J = diag(s) - s s^T.
    So, grad_z = J dot grad_soft.
    """
    # We assume softmax_output and grad_soft are 1D arrays of length C.
    s = softmax_output.reshape(-1, 1)  # (C, 1)
    J = np.diagflat(s) - np.dot(s, s.T)  # (C, C)
    return np.dot(J, grad_soft)

def vectorized_softmax_derivative(s: np.ndarray, grad_soft: np.ndarray):
    """
    Compute the derivative of the softmax function in a vectorized way.

    Args:
      s (np.ndarray): Softmax outputs, shape (..., C).
      grad_soft (np.ndarray): Upstream gradients for the softmax outputs, shape (..., C).

    Returns:
      np.ndarray: The gradient with respect to the softmax input, of shape (..., C).
    """
    dot = np.sum(s * grad_soft, axis=-1, keepdims=True)
    # Jacobian-vector product in a vectorized manner.
    return s * (grad_soft - dot)
