import cupy as cp


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

    def pool_forward(self, X: cp.ndarray):
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
        A = cp.zeros((m, output_height, output_width, output_channels))

        for i in range(m):
            for h in range(output_height):
                for w in range(output_width):
                    for c in range(output_channels):
                        vert_start = h * self.stride
                        vert_end = vert_start + self.pool_size
                        horiz_start = w * self.stride
                        horiz_end = horiz_start + self.pool_size

                        A[i, h, w, c] = cp.max(X[i, vert_start:vert_end, horiz_start:horiz_end, c])

        self.cache = X

        return A

    def as_strided(x, shape, strides):
        return cp.ndarray(shape, dtype=x.dtype, memptr=x.data, strides=strides)

    # TODO: find out what is this
    def pool_forward_vectorized(self, X: cp.ndarray):
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

        shape = (m, output_height, output_width, self.pool_size, self.pool_size, num_input_channels)
        strides = (X.strides[0],
                   X.strides[1] * self.stride,
                   X.strides[2] * self.stride,
                   X.strides[1],
                   X.strides[2],
                   X.strides[3])

        X_windows = self.as_strided(X, shape=shape, strides=strides)

        A = cp.max(X_windows, axis=(3, 4))

        self.cache = (X, X_windows)
        return A

    def pool_backward_vectorized(self, dZ: cp.ndarray):
        """
        Vectorized backward pass of the pooling layer.
        Assumes non-overlapping pooling (stride == pool_size).

        Args:
            dA (cp.ndarray): Upstream gradient of shape (m, out_h, out_w, C)

        Returns:
            dX (cp.ndarray): Gradient w.r.t. input X of shape (m, H, W, C)
        """
        X, X_windows = self.cache  # X_windows has shape (m, out_h, out_w, pool_size, pool_size, C)
        (m, H, W, C) = X.shape
        pool_size = self.pool_size
        stride = self.stride

        out_h = (H - pool_size) // stride + 1
        out_w = (W - pool_size) // stride + 1

        # compute mask for each window: shape (m, out_h, out_w, pool_size, pool_size, C)
        max_vals = cp.max(X_windows, axis=(3, 4), keepdims=True)
        mask = (X_windows == max_vals)

        # same shape as the windows.
        dA_expanded = dZ[:, :, :, None, None, :]  # shape (m, out_h, out_w, 1, 1, C)

        # gradient contribution for each window.
        dX_windows = mask * dA_expanded  # shape (m, out_h, out_w, pool_size, pool_size, C)

        dX = cp.zeros_like(X)

        # Scatter the gradients from each window back to dX.
        # Since pooling is non-overlapping, each window maps to a unique region in dX.
        for i in range(out_h):
            for j in range(out_w):
                vert_start = i * stride
                vert_end = vert_start + pool_size
                horiz_start = j * stride
                horiz_end = horiz_start + pool_size
                # Add the gradients from the window to the corresponding region of dX.
                dX[:, vert_start:vert_end, horiz_start:horiz_end, :] += dX_windows[:, i, j, :, :, :]

        return dX

    def create_mask_from_window(self, X: cp.ndarray):
        """
        creates a mask from the input matrix X

        Args:
            X (np.ndarray): input matrix of shape (pool_size, pool_size)

        Returns:
            np.ndarray: mask of the same shape as X, contains a True at the position corresponding to the max value of X
        """
        return X == cp.max(X)

    def pool_backward(self, dZ: cp.ndarray):
        """
        backward pass of the pooling

        Args:
            dZ (np.ndarray): gradient of the cost with respect to the output of the pooling layer - matrix of shape (m, input_height, input_width, num_input_channels)

        Returns:
            dX (np.ndarray): gradient of the cost with respect to the input of the pooling layer - matrix of shape (m, input_height, input_width, num_input_channels)
        """
        X = self.cache
        (m, input_height, input_width, num_input_channels) = dZ.shape

        dX = cp.zeros(X.shape)

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

    def get_params(self, params: dict, key: str):
        params[f'{key}_pool_size'] = self.pool_size
        params[f'{key}_stride'] = self.stride

        return params

    def set_params(self, params: dict, key: str):
        self.pool_size = params[f'{key}_pool_size']
        self.stride = params[f'{key}_stride']
