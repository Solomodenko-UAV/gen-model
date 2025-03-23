# import numpy as np
import cupy as cp
from nn.pkg.activations import activations


class YoloOutput:
    """
    basically just FullyConnected layer, but with activation functions applied to only the specific part of output
    """

    def __init__(self,
                 input_size: int,
                 S: int,
                 B: int,
                 C: int,
                 l2_lambda=0.0001,
                 clip_value=5.0):
        """
        creates a yolo output layer

        Args:
            input_size (int): number of input neurons
            S (int): number of grid cells
            B (int): number of bounding boxes per grid cell
            C (int): number of classes
        """
        out_dim = S * S * (C + B * 5)

        self.S = S
        self.B = B
        self.C = C
        self.out_per_cell = C + B * 5

        std = cp.sqrt(2.0 / (input_size + out_dim)).astype(cp.float32)  # Xavier for softmax
        self.weights = cp.random.randn(input_size, out_dim).astype(cp.float32) * std
        self.biases = cp.zeros((1, out_dim), dtype=cp.float32)
        self.cache = {}

        self.l2_lambda = l2_lambda
        self.clip_value = clip_value

    def feed_forward(self, X: cp.ndarray):
        """
        forward pass of the yolo output layer

        Args:
            X (np.ndarray): input to the yolo output layer - matrix of shape (m, input_size), represents a batch of m images

        Returns:
            A (np.ndarray): output of the yolo output layer - matrix of shape (m, S, S, out_per_cell), represents a batch of m images
        """

        self.cache['X'] = X

        m = X.shape[0]
        A = cp.dot(X, self.weights) + self.biases

        A = A.reshape(m, self.S, self.S, self.out_per_cell)

        # iterate over predicted bounding boxes in every grid cell
        for b in range(self.B):
            x_coordinate_idx = b * 5  # relative center x coordinate value index
            y_coordinate_idx = b * 5 + 1  # relative center y coordinate value index
            confidence_idx = b * 5 + 4  # confidence value index

            x_sigmoid = activations.sigmoid(A[:, :, :, x_coordinate_idx])
            y_sigmoid = activations.sigmoid(A[:, :, :, y_coordinate_idx])

            self.cache[f'x_sigmoid_{b}'] = x_sigmoid
            self.cache[f'y_sigmoid_{b}'] = y_sigmoid

            grid_x = cp.arange(self.S).reshape(1, self.S, 1)
            grid_y = cp.arange(self.S).reshape(1, 1, self.S)

            # apply sigmoid and add grid offset, then normalize by grid size
            A[:, :, :, x_coordinate_idx] = (x_sigmoid + grid_x) / self.S
            A[:, :, :, y_coordinate_idx] = (y_sigmoid + grid_y) / self.S

            A[:, :, :, confidence_idx] = activations.sigmoid(A[:, :, :, confidence_idx])
            # leave weight and height (indices b*5+2 and b*5+3) as is

        A[:, :, :, self.B * 5:] = activations.softmax(A[:, :, :, self.B * 5:])  # apply softmax to class scores
        self.cache['Z'] = A  # after activation

        return A

    def feed_backward(self, dZ: cp.ndarray, learning_rate: float):
        """
        backward pass of the yolo output layer

        Args:
            dZ (np.ndarray): gradient of the cost with respect to the output of the yolo output layer - matrix of shape (m, S, S, out_per_cell), represents a batch of m images
            learning_rate (float): learning rate to be used for updating the weights and biases

        Returns:
            dX(np.ndarray): gradient of the cost with respect to the input of the yolo output layer - matrix of shape (m, input_size), represents a batch of m images
        """

        m = dZ.shape[0]

        X = self.cache['X']  # (m, input_size)
        Z = self.cache['Z']  # (m, S, S, out_per_cell)

        # pre-activation output
        dX_pre = cp.asarray(dZ)

        for b in range(self.B):
            x_coordinate_idx = b * 5
            y_coordinate_idx = b * 5 + 1
            confidence_idx = b * 5 + 4

            sigmoid_x = self.cache[f'x_sigmoid_{b}']
            sigmoid_y = self.cache[f'y_sigmoid_{b}']

            dX_pre[:, :, :, x_coordinate_idx] *= (sigmoid_x * (1 - sigmoid_x)) / self.S
            dX_pre[:, :, :, y_coordinate_idx] *= (sigmoid_y * (1 - sigmoid_y)) / self.S

            confidence_value = Z[:, :, :, confidence_idx]
            dX_pre[:, :, :, confidence_idx] *= activations.sigmoid_derivative(confidence_value) + 1e-6
            # leave weight and height (indices b*5+2 and b*5+3) as is

        softmax_output = Z[:, :, :, self.B * 5:]
        grad_soft = dX_pre[:, :, :, self.B * 5:]
        dX_pre[:, :, :, self.B * 5:] = activations.vectorized_softmax_derivative(softmax_output, grad_soft)

        dX_pre = dX_pre.reshape(m, self.S * self.S * self.out_per_cell)

        dW = cp.dot(X.T, dX_pre) / m  # shape (input_size, out_dim)
        db = cp.sum(dX_pre, axis=0, keepdims=True) / m  # shape (1, out_dim)

        dX = cp.dot(dX_pre, self.weights.T)  # shape (m, input_size)

        dW = cp.clip(dW, -self.clip_value, self.clip_value)
        db = cp.clip(db, -self.clip_value, self.clip_value)

        self.weights -= learning_rate * (dW + self.l2_lambda * self.weights)  # L2 regularization
        self.biases -= db * learning_rate

        return dX

    def get_params(self, params: dict, key: str):
        params[f'{key}_weights'] = cp.asnumpy(self.weights)
        params[f'{key}_biases'] = cp.asnumpy(self.biases)
        params[f'{key}_l2_lambda'] = cp.asnumpy(self.l2_lambda)
        params[f'{key}_clip_value'] = cp.asnumpy(self.clip_value)
        params[f'{key}_S'] = cp.asnumpy(self.S)
        params[f'{key}_B'] = cp.asnumpy(self.B)
        params[f'{key}_C'] = cp.asnumpy(self.C)
        params[f'{key}_out_per_cell'] = cp.asnumpy(self.out_per_cell)

    def set_params(self, params: dict, key: str):
        self.weights = cp.array(params[f'{key}_weights'])
        self.biases = cp.array(params[f'{key}_biases'])
        self.l2_lambda = params[f'{key}_l2_lambda'].item()
        self.clip_value = params[f'{key}_clip_value'].item()
        self.S = params[f'{key}_S'].item()
        self.B = params[f'{key}_B'].item()
        self.C = params[f'{key}_C'].item()
        self.out_per_cell = params[f'{key}_out_per_cell'].item()
