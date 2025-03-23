import cupy as cp


class Convolution:

    def __init__(self,
                 input_channels: int,
                 filter_size: int,
                 num_filters: int,
                 stride: int,
                 padding: int,
                 l2_lambda: float = 0.0001,
                 clip_value: float = 5.0,
                 momentum=0.8,
                 ):
        """
        creates a convolution layer

        Args:
            input_channels (int): number of filters in the previous layer
            filter_size (int): filter size, kernel will be square with this size
            num_filters (int): number of channels in the output
            stride (int): stride of the convolution
            padding (int): padding to be added to the input

        Raises:
            ValueError: if input_channels, stride, filter_size or num_filters are less than 1            
        """

        if input_channels < 1:
            raise ValueError("input channels should be greater than 0")

        if stride < 1:
            raise ValueError("stride should be greater than 0")

        if filter_size < 1:
            raise ValueError("filter size should be greater than 0")

        if num_filters < 1:
            raise ValueError("number of filters should be greater than 0")

        std = cp.sqrt(2 / (input_channels * filter_size**2)).astype(cp.float32)  # Xavier for ReLU
        self.filters = cp.random.randn(filter_size, filter_size, input_channels, num_filters).astype(cp.float32) * std
        self.biases = cp.zeros((1, 1, 1, num_filters)).astype(cp.float32)

        self.stride = stride
        self.padding = padding

        self.l2_lambda = l2_lambda
        self.clip_value = clip_value
        self.momentum = momentum
        self.eps = 1e-5
        self.gamma = cp.ones((1, 1, 1, num_filters)).astype(cp.float32)  # activations remain unaffected at the start
        self.beta = cp.zeros((1, 1, 1, num_filters)).astype(cp.float32)  # no initial shift

        self.running_mean = cp.zeros((1, 1, 1, self.filters.shape[3]))  # initial activations are assumed to be centered around zero
        self.running_variance = cp.ones((1, 1, 1, self.filters.shape[3]))  # variance shouldn't start at 0 (to avoid division by zero).

    def zero_pad(self, X: cp.ndarray):
        """
        add zero padding to the input

        Args:
            X (cp.ndarray): image to be padded - matrix of shape (m, height, width, num_filters), represents a batch of m images

        Returns:
            cp.ndarray: padded input
        """

        return cp.pad(X, ((0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0)), 'constant', constant_values=0)

    def convolve_single_step(self, X: cp.ndarray):
        """
        single step of the convolution

        Args:
            X (cp.ndarray): input to be convolved - matrix of shape (filter_size, filter_size, num_filters)
            filter_idx (int): index of the filter to be used

        Returns:
            float: scalar value of the convolution
        """
        X_col = X.flatten()  # (filter_size * filter_size * num_filters, 1)
        filters = self.filters.reshape(-1, self.filters.shape[3])  # (filter_size * filter_size * num_filters, num_filters)

        return cp.dot(X_col, filters) + self.biases.reshape(-1)  # (1, num_filters)

    def convolve_forward_vectorized(self, X: cp.ndarray):
        """
            convolves the input with the predefined filters

        Args:
            X (cp.ndarray): output activations of the previous layer - matrix of shape (m, height, width, input_channels)

        Returns:
            Z (cp.ndarray): output of the convolution - matrix of shape (m, output_height, output_width, num_filters)
        """

        (m, prev_height, prev_width, _) = X.shape

        filter_size = self.filters.shape[0]
        num_filters = self.filters.shape[3]

        output_height = int((prev_height - filter_size + 2 * self.padding) / self.stride) + 1  # just a formula to calculate the output height
        output_width = int((prev_width - filter_size + 2 * self.padding) / self.stride) + 1  # just a formula to calculate the output width

        Z = cp.zeros((m, output_height, output_width, num_filters))
        X_padded = self.zero_pad(X)

        for i in range(m):
            x = X_padded[i]

            for h in range(output_height):
                vert_start = h * self.stride
                vert_end = vert_start + filter_size

                for w in range(output_width):
                    horiz_start = w * self.stride
                    horiz_end = horiz_start + filter_size

                    part_to_convolve = x[vert_start:vert_end, horiz_start:horiz_end, :]

                    Z[i, h, w, :] = self.convolve_single_step(part_to_convolve)

        Z, batch_cache = self._normalize_forward(Z)

        self.cache = (X, batch_cache)

        return Z

    def convolve_backward(self, dZ: cp.ndarray, learning_rate: float):
        """
        backward propagation for a convolution function

        Args:
            dZ (cp.ndarray): gradient of the cost with respect to the output of the conv layer (Z), matrix shape (m, output_height, output_width, num_filters)
            learning_rate (float): learning rate for the optimization

        Returns:
            dX (cp.ndarray): gradient of the input (X), matrix shape (m, height, width, input_channels)
        """

        X = self.cache[0]
        (m, output_height, output_width, num_filters) = dZ.shape

        filter_size = self.filters.shape[0]

        dX = cp.zeros(X.shape)
        dW = cp.zeros(self.filters.shape)
        db = cp.zeros(self.biases.shape)

        X_padded = self.zero_pad(X)
        dX_padded = self.zero_pad(dX)

        dZ, dgamma, dbeta = self._normalize_backward(dZ)

        for i in range(m):
            x_padded = X_padded[i]
            dx_padded = dX_padded[i]

            for h in range(output_height):
                vert_start = h * self.stride
                vert_end = vert_start + filter_size

                for w in range(output_width):
                    horiz_start = w * self.stride
                    horiz_end = horiz_start + filter_size

                    for c in range(num_filters):
                        dx_padded[vert_start:vert_end, horiz_start:horiz_end, :] += self.filters[:, :, :, c] * dZ[i, h, w, c]
                        dW[:, :, :, c] += x_padded[vert_start:vert_end, horiz_start:horiz_end, :] * dZ[i, h, w, c]
                        db[:, :, :, c] += dZ[i, h, w, c]

            # remove padding if necessary
            if self.padding == 0:
                dX[i, :, :, :] = dx_padded
            else:
                dX[i, :, :, :] = dx_padded[self.padding:-self.padding, self.padding:-self.padding, :]

        dW = cp.clip(dW, -self.clip_value, self.clip_value)
        db = cp.clip(db, -self.clip_value, self.clip_value)

        self.filters -= learning_rate * (dW + self.l2_lambda * self.filters)  # L2 regularization
        self.biases -= db * learning_rate

        self.gamma -= learning_rate * dgamma
        self.beta -= learning_rate * dbeta

        return dX

    def convolve_backward_vectorized(self, dZ: cp.ndarray, learning_rate: float):
        """
        backward propagation for a convolution function

        Args:
            dZ (np.ndarray): gradient of the cost with respect to the output of the conv layer (Z), matrix shape (m, output_height, output_width, num_filters)
            learning_rate (float): learning rate for the optimization

        Returns:
            dX (np.ndarray): gradient of the input (X), matrix shape (m, height, width, input_channels)
        """

        X = self.cache[0]
        _, _, _, input_channels = X.shape
        filter_height, filter_width, _, output_filters = self.filters.shape

        dZ, dgamma, dbeta = self._normalize_backward(dZ)

        X_cols = _im2col(X, filter_height, filter_width, self.padding, self.stride)  # shape: (filter_height*filter_width*C_in, N*out_height*out_width)

        dZ_reshaped = dZ.transpose(0, 3, 1, 2).reshape(output_filters, -1)  # (C_out, N*out_height*out_width)
        dW_cols = cp.dot(X_cols, dZ_reshaped.T)  # (filter_height*filter_width*C_in, C_out)
        dW = dW_cols.reshape(filter_height, filter_width, input_channels, output_filters)

        filters_reshaped = self.filters.reshape(-1, output_filters)  # (filter_height*filter_width*C_in, C_out)
        dX_cols = cp.dot(filters_reshaped, dZ_reshaped)  # (filter_height*filter_width*C_in, N*out_height*out_width)
        dX = _col2im(dX_cols, X.shape, filter_height, filter_width, self.padding, self.stride)

        db = cp.sum(dZ, axis=(0, 1, 2), keepdims=True)

        dW = cp.clip(dW, -self.clip_value, self.clip_value)
        db = cp.clip(db, -self.clip_value, self.clip_value)

        self.filters -= learning_rate * (dW + self.l2_lambda * self.filters)  # L2 regularization
        self.biases -= db * learning_rate

        self.gamma -= learning_rate * dgamma
        self.beta -= learning_rate * dbeta

        return dX

    def _normalize_forward(self, Z: cp.ndarray):
        """
        batch normalization
        Args:
            X (cp.ndarray): input to the convolution layer - matrix of shape (m, height, width, num_filters)
        Returns:
            cp.ndarray: normalized input
        """
        mean = cp.mean(Z, axis=(0, 1, 2), keepdims=True)
        variance = cp.var(Z, axis=(0, 1, 2), keepdims=True)
        Z_norm = (Z - mean) / cp.sqrt(variance + self.eps)
        out = self.gamma * Z_norm + self.beta
        self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mean
        self.running_variance = self.momentum * self.running_variance + (1 - self.momentum) * variance
        cache = (Z, Z_norm, mean, variance)
        return out, cache

    def _normalize_backward(self, dZ):
        (Z, Z_norm, mean, variance) = self.cache[1]
        m = Z.shape[0]
        # Gradients scale (gamma) and shift (beta)
        dgamma = cp.sum(dZ * Z_norm, axis=(0, 1, 2), keepdims=True)
        dbeta = cp.sum(dZ, axis=(0, 1, 2), keepdims=True)
        dZ_norm = dZ * self.gamma
        # Gradient of variance
        dvar = cp.sum(dZ_norm * (Z - mean) * -0.5 * cp.power(variance + self.eps, -1.5), axis=0, keepdims=True)
        # Gradient of mean
        dmean = cp.sum(dZ_norm * -1 / cp.sqrt(variance + self.eps), axis=0, keepdims=True) + dvar * cp.mean(-2 * (Z - mean), axis=0, keepdims=True)
        dX = dZ_norm / cp.sqrt(variance + self.eps) + dvar * 2 * (Z - mean) / m + dmean / m
        return dX, dgamma, dbeta

    def get_params(self, params: dict, key: str):
        params[f'{key}_filters'] = self.filters
        params[f'{key}_biases'] = self.biases
        params[f'{key}_stride'] = self.stride
        params[f'{key}_padding'] = self.padding
        params[f'{key}_l2_lambda'] = self.l2_lambda
        params[f'{key}_clip_value'] = self.clip_value
        params[f'{key}_momentum'] = self.momentum
        params[f'{key}_gamma'] = self.gamma
        params[f'{key}_beta'] = self.beta
        params[f'{key}_eps'] = self.eps
        params[f'{key}_running_mean'] = self.running_mean
        params[f'{key}_running_variance'] = self.running_variance

    def set_params(self, params: dict, key: str):
        self.filters = params[f'{key}_filters']
        self.biases = params[f'{key}_biases']
        self.stride = params[f'{key}_stride'].item()
        self.padding = params[f'{key}_padding'].item()
        self.l2_lambda = params[f'{key}_l2_lambda'].item()
        self.clip_value = params[f'{key}_clip_value'].item()
        self.momentum = params[f'{key}_momentum'].item()
        self.gamma = params[f'{key}_gamma']
        self.beta = params[f'{key}_beta']
        self.eps = params[f'{key}_eps'].item()
        self.running_mean = params[f'{key}_running_mean']
        self.running_variance = params[f'{key}_running_variance']

# TODO make out what's happening here
def _im2col(X, filter_height, filter_width, padding, stride):
    """
    Rearranges image blocks into columns.
    Args:
        X (cp.ndarray): Input images of shape (N, H, W, C)
        filter_height (int): Height of the filter.
        filter_width (int): Width of the filter.
        padding (int): Amount of zero-padding around each image.
        stride (int): Stride of the convolution.
    Returns:
        cols (cp.ndarray): 2D array of shape 
            (filter_height * filter_width * C, N * out_height * out_width)
    """
    N, H, W, C = X.shape
    H_padded, W_padded = H + 2 * padding, W + 2 * padding
    X_padded = cp.pad(X, ((0, 0), (padding, padding), (padding, padding), (0, 0)), mode='constant', constant_values=0)
    out_height = (H_padded - filter_height) // stride + 1
    out_width = (W_padded - filter_width) // stride + 1
    # Compute indices for im2col
    i0 = cp.repeat(cp.arange(filter_height), filter_width)
    i0 = cp.tile(i0, C)
    i1 = stride * cp.repeat(cp.arange(out_height), out_width)
    j0 = cp.tile(cp.arange(filter_width), filter_height * C)
    j1 = stride * cp.tile(cp.arange(out_width), out_height)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)
    k = cp.repeat(cp.arange(C), filter_height * filter_width).reshape(-1, 1)
    # Use advanced indexing to extract the patches
    cols = X_padded[:, i, j, k]  # shape: (N, filter_height*filter_width*C, out_height*out_width)
    cols = cols.transpose(1, 2, 0).reshape(filter_height * filter_width * C, -1)
    
    return cols


def _col2im(cols, X_shape, filter_height, filter_width, padding, stride):
    """
    Inverse of im2col. Reconstructs the image from column representation.
    Args:
        cols (cp.ndarray): 2D array of shape 
            (filter_height * filter_width * C, N * out_height * out_width)
        X_shape (tuple): Shape of the original input X, (N, H, W, C)
        filter_height (int): Height of the filter.
        filter_width (int): Width of the filter.
        padding (int): Padding used in the forward pass.
        stride (int): Stride used in the forward pass.
    Returns:
        X_reconstructed (cp.ndarray): Reconstructed images of shape (N, H, W, C)
    """
    N, H, W, C = X_shape
    H_padded, W_padded = H + 2 * padding, W + 2 * padding
    out_height = (H_padded - filter_height) // stride + 1
    out_width = (W_padded - filter_width) // stride + 1
    X_padded = cp.zeros((N, H_padded, W_padded, C), dtype=cols.dtype)
    # Compute indices for im2col (same as in your _im2col)
    i0 = cp.repeat(cp.arange(filter_height), filter_width)
    i0 = cp.tile(i0, C)
    i1 = stride * cp.repeat(cp.arange(out_height), out_width)
    j0 = cp.tile(cp.arange(filter_width), filter_height * C)
    j1 = stride * cp.tile(cp.arange(out_width), out_height)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)        # shape: (F, out_height*out_width)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)        # shape: (F, out_height*out_width)
    k = cp.repeat(cp.arange(C), filter_height * filter_width).reshape(-1, 1)  # shape: (F, 1)
    # Tile k so that it matches the shape of i and j:
    k = cp.tile(k, (1, out_height * out_width))      # shape: (F, out_height*out_width)
    # Reshape cols to (N, F, out_height*out_width)
    cols_reshaped = cols.reshape(filter_height * filter_width * C, out_height * out_width, N)
    cols_reshaped = cols_reshaped.transpose(2, 0, 1)
    # Accumulate the columns back into the image
    for n in range(N):
        cp.add.at(X_padded[n], (i, j, k), cols_reshaped[n])
    if padding == 0:
        return X_padded
    return X_padded[:, padding:-padding, padding:-padding, :]
