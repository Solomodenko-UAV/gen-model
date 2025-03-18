import numpy as np

from nn.pkg.activations import activations


class YoloOutput:
    """
    basically just FullyConnected layer, but with activation functions applied to only the specific part of output
    """

    def __init__(self, input_size: int, S: int, B: int, C: int):
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

        std = np.sqrt(2.0 / (input_size + out_dim)).astype(np.float64)  # Xavier for softmax
        self.weights = np.random.randn(input_size, out_dim).astype(np.float64) * std
        self.biases = np.zeros((1, out_dim), dtype=np.float64)
        self.cache = {}

    def feed_forward(self, X: np.ndarray):
        """
        forward pass of the yolo output layer

        Args:
            X (np.ndarray): input to the yolo output layer - matrix of shape (m, input_size), represents a batch of m images
            
        Returns:
            A (np.ndarray): output of the yolo output layer - matrix of shape (m, S, S, out_per_cell), represents a batch of m images
        """

        self.cache['X'] = X

        m = X.shape[0]
        A = np.dot(X, self.weights) + self.biases
        self.cache['A'] = np.copy(A)  # before any activation

        A = A.reshape(m, self.S, self.S, self.out_per_cell)

        # iterate over predicted bounding boxes in every grid cell
        for b in range(self.B):
            x_coordinate_idx = b * 5  # relative center x coordinate value index
            y_coordinate_idx = b * 5 + 1  # relative center y coordinate value index
            confidence_idx = b * 5 + 4  # confidence value index

            A[:, :, :, x_coordinate_idx] = activations.sigmoid(A[:, :, :, x_coordinate_idx])
            A[:, :, :, y_coordinate_idx] = activations.sigmoid(A[:, :, :, y_coordinate_idx])
            A[:, :, :, confidence_idx] = activations.sigmoid(A[:, :, :, confidence_idx])
            # leave weight and height (indices b*5+2 and b*5+3) as is

        A[:, :, :, self.B * 5:] = activations.softmax(A[:, :, :, self.B * 5:])  # apply softmax to class scores
        self.cache['Z'] = np.copy(A)  # after activation

        return A

    def feed_backward(self, dZ: np.ndarray, learning_rate: float):
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
        A = self.cache['A']  # (m, S, S, out_per_cell)
        Z = self.cache['Z']  # (m, S, S, out_per_cell)

        # pre-activation output
        dX_pre = np.copy(dZ)

        for b in range(self.B):
            x_coordinate_idx = b * 5
            y_coordinate_idx = b * 5 + 1
            confidence_idx = b * 5 + 4

            xs = Z[:, :, :, x_coordinate_idx]
            ys = Z[:, :, :, y_coordinate_idx]
            cs = Z[:, :, :, confidence_idx]

            dX_pre[:, :, :, x_coordinate_idx] *= activations.sigmoid_derivative(xs)
            dX_pre[:, :, :, y_coordinate_idx] *= activations.sigmoid_derivative(ys)
            dX_pre[:, :, :, confidence_idx] *= activations.sigmoid_derivative(cs)
            # leave weight and height (indices b*5+2 and b*5+3) as is

        softmax_output = Z[:, :, :, self.B * 5:]
        grad_soft = dX_pre[:, :, :, self.B * 5:]
        dX_pre[:, :, :, self.B * 5:] = activations.vectorized_softmax_derivative(softmax_output, grad_soft)

        dX_pre = dX_pre.reshape(m, self.S * self.S * self.out_per_cell)

        dW = np.dot(X.T, dX_pre) / m # shape (input_size, out_dim)
        db = np.sum(dX_pre, axis=0, keepdims=True) / m # shape (1, out_dim)
        
        dX = np.dot(dX_pre, self.weights.T) # shape (m, input_size)
        
        self.weights -= dW * learning_rate
        self.biases -= db * learning_rate
        
        return dX
