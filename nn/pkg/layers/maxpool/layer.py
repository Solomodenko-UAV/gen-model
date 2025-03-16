import numpy as np


class MaxPool:
    def __init__(self, pool_size: int, stride: int):
        """
        creates a max pooling layer

        Args:
            pool_size (int): size of the pooling window
            stride (int): stride of the pooling window
        """
        if pool_size < 1:
            raise ValueError("pool size should be greater than 0")
        if stride < 1:
            raise ValueError("stride should be greater than 0")

        self.pool_size = pool_size
        self.stride = stride

    def pool_forward(self, X: np.ndarray):
        """
        forward pass of the pooling layer

        Args:
            X (np.ndarray): input to the pooling layer - matrix of shape (m, input_height, input_width, num_input_channels), represents a batch of m images

        Returns:
            A (np.ndarray): output of the pooling layer - matrix of shape (m, output_height, output_width, input_num_filters), represents a batch of m images
        """

        (m, input_height, input_width, num_input_channels) = X.shape

        output_height = int((input_height - self.pool_size) / self.stride) + 1
        output_width = int((input_width - self.pool_size) / self.stride) + 1

        output_channels = num_input_channels
        A = np.zeros((m, output_height, output_width, output_channels))

        for i in range(m):
            for h in range(output_height):
                for w in range(output_width):
                    for c in range(output_channels):
                        vert_start = h * self.stride
                        vert_end = vert_start + self.pool_size
                        horiz_start = w * self.stride
                        horiz_end = horiz_start + self.pool_size

                        A[i, h, w, c] = np.max(X[i, vert_start:vert_end, horiz_start:horiz_end, c])

        self.cache = X

        return A

    def create_mask_from_window(self, X: np.ndarray):
        """
        creates a mask from the input matrix X

        Args:
            X (np.ndarray): input matrix of shape (pool_size, pool_size)

        Returns:
            np.ndarray: mask of the same shape as X, contains a True at the position corresponding to the max value of X
        """
        return X == np.max(X)

    def pool_backward(self, dZ: np.ndarray):
        """
        backward pass of the pooling

        Args:
            dZ (np.ndarray): gradient of the cost with respect to the output of the pooling layer - matrix of shape (m, input_height, input_width, num_input_channels)

        Returns:
            dX (np.ndarray): gradient of the cost with respect to the input of the pooling layer - matrix of shape (m, input_height, input_width, num_input_channels)
        """
        X = self.cache
        (m, input_height, input_width, num_input_channels) = dZ.shape

        dX = np.zeros(X.shape)

        for i in range(m):
            for h in range(input_height):
                vert_start = h * self.stride
                vert_end = vert_start + self.pool_size

                for w in range(input_width):
                    horiz_start = w * self.stride
                    horiz_end = horiz_start + self.pool_size

                    for c in range(num_input_channels):
                        X_slice = X[i, vert_start:vert_end, horiz_start:horiz_end, c]
                        mask = self.create_mask_from_window(X_slice)
                        dX[i, vert_start:vert_end, horiz_start:horiz_end, c] += mask * dZ[i, h, w, c]

        return dX
