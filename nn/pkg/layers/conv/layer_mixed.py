import numpy as np


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

        std = np.sqrt(2 / (input_channels * filter_size**2)).astype(np.float32)  # Xavier for ReLU
        self.filters = np.random.randn(filter_size, filter_size, input_channels, num_filters).astype(np.float32) * std
        self.biases = np.zeros((1, 1, 1, num_filters)).astype(np.float32)

        self.stride = stride
        self.padding = padding

        self.l2_lambda = l2_lambda
        self.clip_value = clip_value
        self.momentum = momentum
        self.eps = 1e-5
        self.gamma = np.ones((1, 1, 1, num_filters)).astype(np.float32)  # activations remain unaffected at the start
        self.beta = np.zeros((1, 1, 1, num_filters)).astype(np.float32)  # no initial shift

        self.running_mean = np.zeros((1, 1, 1, self.filters.shape[3]))  # initial activations are assumed to be centered around zero
        self.running_variance = np.ones((1, 1, 1, self.filters.shape[3]))  # variance shouldn't start at 0 (to avoid division by zero).

    def zero_pad(self, X: np.ndarray):
        """
        add zero padding to the input

        Args:
            X (np.ndarray): image to be padded - matrix of shape (m, height, width, num_filters), represents a batch of m images

        Returns:
            np.ndarray: padded input
        """

        return np.pad(X, ((0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0)), 'constant', constant_values=0)

    def convolve_single_step(self, X: np.ndarray):
        """
        single step of the convolution

        Args:
            X (np.ndarray): input to be convolved - matrix of shape (filter_size, filter_size, num_filters)
            filter_idx (int): index of the filter to be used

        Returns:
            float: scalar value of the convolution
        """
        X_col = X.flatten()  # (filter_size * filter_size * num_filters, 1)
        filters = self.filters.reshape(-1, self.filters.shape[3])  # (filter_size * filter_size * num_filters, num_filters)

        return np.dot(X_col, filters) + self.biases.reshape(-1)

    def convolve_forward(self, X: np.ndarray):
        """
            convolves the input with the predefined filters

        Args:
            X (np.ndarray): output activations of the previous layer - matrix of shape (m, height, width, input_channels)

        Returns:
            Z (np.ndarray): output of the convolution - matrix of shape (m, output_height, output_width, num_filters)
        """

        (m, prev_height, prev_width, _) = X.shape

        filter_size = self.filters.shape[0]
        num_filters = self.filters.shape[3]

        output_height = int((prev_height - filter_size + 2 * self.padding) / self.stride) + 1  # just a formula to calculate the output height
        output_width = int((prev_width - filter_size + 2 * self.padding) / self.stride) + 1  # just a formula to calculate the output width

        Z = np.zeros((m, output_height, output_width, num_filters))
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

    def convolve_backward(self, dZ: np.ndarray, learning_rate: float):
        """
        backward propagation for a convolution function

        Args:
            dZ (np.ndarray): gradient of the cost with respect to the output of the conv layer (Z), matrix shape (m, output_height, output_width, num_filters)
            learning_rate (float): learning rate for the optimization

        Returns:
            dX (np.ndarray): gradient of the input (X), matrix shape (m, height, width, input_channels)
        """

        X = self.cache[0]
        N, H, W, input_channels = X.shape
        filter_height, filter_width, _, output_filters = self.filters.shape
        (_, output_height, output_width, _) = dZ.shape

        dX = np.zeros(X.shape)
        dW = np.zeros(self.filters.shape)
        db = np.zeros(self.biases.shape)

        dZ, dgamma, dbeta = self._normalize_backward(dZ)
        
        X_cols = _im2col(X, filter_height, filter_width, self.padding, self.stride)  # shape: (filter_height*filter_width*C_in, N*out_height*out_width)

                
        dZ_reshaped = dZ.transpose(0, 3, 1, 2).reshape(output_filters, -1) # (C_out, N*out_height*out_width)
        dW_cols = np.dot(X_cols, dZ_reshaped.T) # (filter_height*filter_width*C_in, C_out)
        dW = dW_cols.reshape(filter_height, filter_width, input_channels, output_filters)
        
        filters_reshaped = self.filters.reshape(-1, output_filters)  # (filter_height*filter_width*C_in, C_out)
        dX_cols = np.dot(filters_reshaped, dZ_reshaped)  # (filter_height*filter_width*C_in, N*out_height*out_width)
        dX = _col2im(dX_cols, X.shape, filter_height, filter_width, self.padding, self.stride)
        
        db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)

        dW = np.clip(dW, -self.clip_value, self.clip_value)
        db = np.clip(db, -self.clip_value, self.clip_value)

        self.filters -= learning_rate * (dW + self.l2_lambda * self.filters)  # L2 regularization
        self.biases -= db * learning_rate

        self.gamma -= learning_rate * dgamma
        self.beta -= learning_rate * dbeta

        return dX

    def _normalize_forward(self, Z: np.ndarray):
        """
        batch normalization

        Args:
            X (np.ndarray): input to the convolution layer - matrix of shape (m, height, width, num_filters)

        Returns:
            np.ndarray: normalized input
        """

        mean = np.mean(Z, axis=(0, 1, 2), keepdims=True)
        variance = np.var(Z, axis=(0, 1, 2), keepdims=True)

        Z_norm = (Z - mean) / np.sqrt(variance + self.eps)

        out = self.gamma * Z_norm + self.beta

        self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mean
        self.running_variance = self.momentum * self.running_variance + (1 - self.momentum) * variance

        cache = (Z, Z_norm, mean, variance)

        return out, cache

    def _normalize_backward(self, dZ):
        (Z, Z_norm, mean, variance) = self.cache[1]
        m = Z.shape[0]

        # Gradients scale (gamma) and shift (beta)
        dgamma = np.sum(dZ * Z_norm, axis=(0, 1, 2), keepdims=True)
        dbeta = np.sum(dZ, axis=(0, 1, 2), keepdims=True)

        dZ_norm = dZ * self.gamma

        # Gradient of variance
        dvar = np.sum(dZ_norm * (Z - mean) * -0.5 * np.power(variance + self.eps, -1.5), axis=0, keepdims=True)

        # Gradient of mean
        dmean = np.sum(dZ_norm * -1 / np.sqrt(variance + self.eps), axis=0, keepdims=True) + dvar * np.mean(-2 * (Z - mean), axis=0, keepdims=True)

        dX = dZ_norm / np.sqrt(variance + self.eps) + dvar * 2 * (Z - mean) / m + dmean / m

        return dX, dgamma, dbeta

 #TODO make out what's happening here
def _im2col(X, filter_height, filter_width, padding, stride):
    """
    Rearranges image blocks into columns.

    Args:
        X (np.ndarray): Input images of shape (N, H, W, C)
        filter_height (int): Height of the filter.
        filter_width (int): Width of the filter.
        padding (int): Amount of zero-padding around each image.
        stride (int): Stride of the convolution.

    Returns:
        cols (np.ndarray): 2D array of shape 
            (filter_height * filter_width * C, N * out_height * out_width)
    """
    N, H, W, C = X.shape
    H_padded, W_padded = H + 2 * padding, W + 2 * padding
    X_padded = np.pad(X, ((0, 0), (padding, padding), (padding, padding), (0, 0)), mode='constant', constant_values=0)

    out_height = (H_padded - filter_height) // stride + 1
    out_width = (W_padded - filter_width) // stride + 1

    # Compute indices for im2col
    i0 = np.repeat(np.arange(filter_height), filter_width)
    i0 = np.tile(i0, C)
    i1 = stride * np.repeat(np.arange(out_height), out_width)
    j0 = np.tile(np.arange(filter_width), filter_height * C)
    j1 = stride * np.tile(np.arange(out_width), out_height)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)
    k = np.repeat(np.arange(C), filter_height * filter_width).reshape(-1, 1)

    # Use advanced indexing to extract the patches
    cols = X_padded[:, i, j, k]  # shape: (N, filter_height*filter_width*C, out_height*out_width)
    cols = cols.transpose(1, 2, 0).reshape(filter_height * filter_width * C, -1)
    return cols


def _col2im(cols, X_shape, filter_height, filter_width, padding, stride):
    """
    Inverse of im2col. Reconstructs the image from column representation.

    Args:
        cols (np.ndarray): 2D array of shape 
            (filter_height * filter_width * C, N * out_height * out_width)
        X_shape (tuple): Shape of the original input X, (N, H, W, C)
        filter_height (int): Height of the filter.
        filter_width (int): Width of the filter.
        padding (int): Padding used in the forward pass.
        stride (int): Stride used in the forward pass.

    Returns:
        X_reconstructed (np.ndarray): Reconstructed images of shape (N, H, W, C)
    """
    N, H, W, C = X_shape
    H_padded, W_padded = H + 2 * padding, W + 2 * padding
    out_height = (H_padded - filter_height) // stride + 1
    out_width = (W_padded - filter_width) // stride + 1

    X_padded = np.zeros((N, H_padded, W_padded, C), dtype=cols.dtype)

    # Compute indices for im2col (same as in your _im2col)
    i0 = np.repeat(np.arange(filter_height), filter_width)
    i0 = np.tile(i0, C)
    i1 = stride * np.repeat(np.arange(out_height), out_width)
    j0 = np.tile(np.arange(filter_width), filter_height * C)
    j1 = stride * np.tile(np.arange(out_width), out_height)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)        # shape: (F, out_height*out_width)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)        # shape: (F, out_height*out_width)
    k = np.repeat(np.arange(C), filter_height * filter_width).reshape(-1, 1)  # shape: (F, 1)
    # Tile k so that it matches the shape of i and j:
    k = np.tile(k, (1, out_height * out_width))      # shape: (F, out_height*out_width)

    # Reshape cols to (N, F, out_height*out_width)
    cols_reshaped = cols.reshape(filter_height * filter_width * C, out_height * out_width, N)
    cols_reshaped = cols_reshaped.transpose(2, 0, 1)

    # Accumulate the columns back into the image
    for n in range(N):
        np.add.at(X_padded[n], (i, j, k), cols_reshaped[n])

    if padding == 0:
        return X_padded
    return X_padded[:, padding:-padding, padding:-padding, :]

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
        self.stride = params[f'{key}_stride']
        self.padding = params[f'{key}_padding']
        self.l2_lambda = params[f'{key}_l2_lambda']
        self.clip_value = params[f'{key}_clip_value']
        self.momentum = params[f'{key}_momentum']
        self.gamma = params[f'{key}_gamma']
        self.beta = params[f'{key}_beta']
        self.eps = params[f'{key}_eps']
        self.running_mean = params[f'{key}_running_mean']
        self.running_variance = params[f'{key}_running_variance']
