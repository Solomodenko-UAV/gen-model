import numpy as np


class ReLU:
    def forward(self, X: np.ndarray):
        """
        forward pass of the ReLU layer

        Args:
            X (np.ndarray): input to the ReLU layer - matrix of shape (m, height, width, num_filters), represents a batch of m images

        Returns:
            A (np.ndarray): output of the ReLU layer - matrix of shape (m, height, width, num_filters), represents a batch of m images
        """
        return np.maximum(0, X)

    def backward(self, dA: np.ndarray):
        """
        backward pass of the ReLU layer

        Args:
            dA (np.ndarray): gradient of the cost with respect to the output of the ReLU layer - matrix of shape (m, height, width, num_filters), represents a batch of m images

        Returns:
            dX (np.ndarray): gradient of the cost with respect to the input of the ReLU layer - matrix of shape (m, height, width, num_filters), represents a batch of m images
        """
        return np.where(dA > 0, dA, 0)