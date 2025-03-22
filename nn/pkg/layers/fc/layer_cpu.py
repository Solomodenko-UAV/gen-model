import numpy as np


class FullyConnected:
    def __init__(self,
                 input_size: int,
                 output_size: int,
                 l2_lambda=0.0001,
                 clip_value=5.0
                 ):
        """
        creates a fully connected layer

        Args:
            input_size (int): number of input neurons
            output_size (int): number of output neurons
        """
        std = np.sqrt(2 / input_size).astype(np.float64)  # He init for ReLU
        self.weights = np.random.randn(input_size, output_size).astype(np.float64) * std
        self.biases = np.zeros((1, output_size), dtype=np.float64)
        self.clip_value = clip_value
        self.l2_lambda = l2_lambda

    def feed_forward(self, X: np.ndarray):
        """
        forward pass of the fully connected layer

        Args:
            X (np.ndarray): input to the fully connected layer - matrix of shape (m, input_size), represents a batch of m images

        Returns:
            A (np.ndarray): output of the fully connected layer - matrix of shape (m, output_size), represents a batch of m images
        """

        self.cache = X

        return np.dot(X, self.weights) + self.biases

    def feed_backward(self, dZ: np.ndarray, learning_rate: float):
        """
        backward pass of the fully connected

        Args:
            dZ (np.ndarray): gradient of the cost with respect to the output of the fully connected layer - matrix of shape (m, output_size), represents a batch of m images
            learning_rate (float): learning rate to be used for updating the weights and biases

        Returns: 
            dX(np.ndarray): gradient of the cost with respect to the input of the fully connected layer - matrix of shape (m, input_size), represents a batch of m images
        """
        X = self.cache
        m = X.shape[0]

        dW = np.dot(X.T, dZ) / m
        db = np.sum(dZ, axis=0, keepdims=True) / m

        dX = np.dot(dZ, self.weights.T)

        dW = np.clip(dW, -self.clip_value, self.clip_value)
        db = np.clip(db, -self.clip_value, self.clip_value)

        self.weights -= learning_rate * (dW + self.l2_lambda * self.weights)  # L2 regularization
        self.biases -= db * learning_rate

        return dX

    def get_params(self, params: dict, key: str):
        params[f'{key}_weights'] = self.weights
        params[f'{key}_biases'] = self.biases
        params[f'{key}_l2_lambda'] = self.l2_lambda
        params[f'{key}_clip_value'] = self.clip_value

        return params

    def set_params(self, params: dict, key: str):
        self.weights = params[f'{key}_weights']
        self.biases = params[f'{key}_biases']
        self.l2_lambda = params[f'{key}_l2_lambda']
        self.clip_value = params[f'{key}_clip_value']
