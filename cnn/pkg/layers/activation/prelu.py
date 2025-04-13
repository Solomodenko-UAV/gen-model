import os


on_cpu = os.environ.get("USE_GPU") != False and os.environ.get("USE_GPU") != 'True'

if on_cpu:
    import numpy as cp
else:
    import cupy as cp


class PReLU:
    def __init__(self, alpha=0.25):
        # alpha is a learnable parameter; it can be a scalar or a vector (e.g. per channel)
        self.alpha = cp.array(alpha, dtype=cp.float32)
        self.grad_alpha = 0.0  # will accumulate gradients for alpha
        self.cache = None      # to store the input x for backward pass

    def forward(self, x):
        """
        Forward pass for PReLU activation.
        For each element:
            f(x) = x if x > 0
                 = alpha * x if x <= 0
        """
        self.cache = x  # store input for use in backward pass
        return cp.where(x > 0, x, self.alpha * x)

    def backward(self, d_out, learning_rate=0.01):
        """
        Backward pass for PReLU activation.

        Args:
            d_out (cp.ndarray): Upstream gradient of the loss with respect to the output.

        Returns:
            dx (cp.ndarray): Gradient of the loss with respect to the input.
        """
        x = self.cache if self.cache is not None else cp.zeros_like(d_out)
        # derivative of PReLU with respect to x is 1 for x > 0, and alpha for x <= 0.
        dx = d_out * cp.where(x > 0, 1, self.alpha)
        # derivative with respect to alpha: only nonzero where x <= 0.
        # d_alpha = d_out * x (only when x <= 0)
        grad_alpha = cp.sum(d_out * cp.where(x > 0, 0, x))
        
        self.alpha -= grad_alpha * learning_rate
        
        return dx
