import os

if os.environ.get("USE_GPU") != False and os.environ.get("USE_GPU") != 'True':
    import numpy as cp
else:
    import cupy as cp

# numerically stable sigmoid
def sigmoid(x):
    return 0.5 * (1 + cp.tanh(x / 2))

def sigmoid_derivative(sigmoid_x):
    return sigmoid_x * (1 - sigmoid_x)

# numerically stable softmax
def softmax(x):
    e_x = cp.exp(x - cp.max(x, axis=-1, keepdims=True))
    return e_x / cp.sum(e_x, axis=-1, keepdims=True)

def softmax_derivative(softmax_output, grad_soft):
    """
    Given a softmax output vector s (shape: (C,)) and upstream gradient grad_soft (shape: (C,)),
    compute the gradient with respect to the softmax input z.
    The Jacobian of softmax is: J = diag(s) - s s^T.
    So, grad_z = J dot grad_soft.
    """
    # We assume softmax_output and grad_soft are 1D arrays of length C.
    s = softmax_output.reshape(-1, 1)  # (C, 1)
    J = cp.diagflat(s) - cp.dot(s, s.T)  # (C, C)
    return cp.dot(J, grad_soft)

def vectorized_softmax_derivative(s: cp.ndarray, grad_soft: cp.ndarray):
    """
    Compute the derivative of the softmax function in a vectorized way.

    Args:
      s (cp.ndarray): Softmax outputs, shape (..., C).
      grad_soft (cp.ndarray): Upstream gradients for the softmax outputs, shape (..., C).

    Returns:
      cp.ndarray: The gradient with respect to the softmax input, of shape (..., C).
    """
    dot = cp.sum(s * grad_soft, axis=-1, keepdims=True)
    # Jacobian-vector product in a vectorized manner.
    return s * (grad_soft - dot)


# leaky relu
def relu(x, alpha=0.01):
    return cp.maximum(alpha * x, x)

def relu_derivative(x, alpha=0.01):
    return cp.where(x > 0, 1, alpha)

# Gaussian Error Linear Unit
def gelu(x):
    return x * sigmoid(1.702 * x)

def gelu_derivative(vector):
    s = sigmoid(1.702 * vector)
    return s + 1.702 * vector * s * (1 - s)
