import helper.helper as helper
import os

on_cpu = os.environ.get("USE_GPU") != False and os.environ.get("USE_GPU") != 'True'

if on_cpu:
    import numpy as cp
else:
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
        self.filters = helper.create_orthogonal_matrix((filter_size, filter_size, input_channels, num_filters), std)
        self.biases = cp.full((1, 1, 1, num_filters), 0.).astype(cp.float32)

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
        
        self.weight_norms = []

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

    def convolve_forward(self, X: cp.ndarray):
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

    def convolve_forward_vectorized(self, X: cp.ndarray):
        """
        Convolves the input with the predefined filters using a vectorized approach.

        Args:
            X (cp.ndarray): Activations from the previous layer of shape (m, height, width, input_channels).

        Returns:
            Z (cp.ndarray): Convolution output of shape (m, output_height, output_width, num_filters).
        """
        (m, prev_height, prev_width, input_channels) = X.shape
        filter_size = self.filters.shape[0]
        num_filters = self.filters.shape[3]

        output_height = int((prev_height - filter_size + 2 * self.padding) / self.stride) + 1
        output_width = int((prev_width - filter_size + 2 * self.padding) / self.stride) + 1

        X_padded = self.zero_pad(X)

        # with this shape we will have matrix with each kernel window already extracted
        shape = (m, output_height, output_width, filter_size, filter_size, input_channels)
        strides = (X_padded.strides[0],
                   X_padded.strides[1] * self.stride,
                   X_padded.strides[2] * self.stride,
                   X_padded.strides[1],
                   X_padded.strides[2],
                   X_padded.strides[3])
        X_windows = cp.lib.stride_tricks.as_strided(X_padded, shape=shape, strides=strides)

        # reshape the windows into columns (each row is a flattened patch). Has shape (m * output_height * output_width, filter_size * filter_size * input_channels)
        X_col = X_windows.reshape(m * output_height * output_width, -1)

        # reshape filters to columns. Has shape (filter_size * filter_size * input_channels, num_filters)
        filters_col = self.filters.reshape(-1, num_filters)

        # perform the convolution
        conv_out = cp.dot(X_col, filters_col)  # (m * output_height * output_width, num_filters)

        # reshape the result back to the output dimensions
        conv_out = conv_out.reshape(m, output_height, output_width, num_filters)
        conv_out += self.biases

        conv_out, batch_cache = self._normalize_forward(conv_out)
        self.cache = (X, batch_cache)

        return conv_out

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

        # dW = cp.clip(dW, -self.clip_value, self.clip_value)
        # db = cp.clip(db, -self.clip_value, self.clip_value)

        self.filters -= learning_rate * (dW + self.l2_lambda * self.filters)  # L2 regularization
        self.biases -= db * learning_rate

        self.gamma -= learning_rate * dgamma
        self.beta -= learning_rate * dbeta

        return dX

    def convolve_backward_vectorized(self, dZ: cp.ndarray, learning_rate: float):
        """
        backward propagation for a convolution function

        Args:
            dZ (cp.ndarray): gradient of the cost with respect to the output of the conv layer (Z), matrix shape (m, output_height, output_width, num_filters)
            learning_rate (float): learning rate for the optimization

        Returns:
            dX (cp.ndarray): gradient of the input (X), matrix shape (m, height, width, input_channels)
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

        self.filters -= learning_rate * (dW + self.l2_lambda * self.filters)  # L2 regularization
        self.biases -= db * learning_rate

        self.gamma -= learning_rate * dgamma
        self.beta -= learning_rate * dbeta
        
        self.weight_norms.append(cp.linalg.norm(self.filters))

        return dX, dW

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

        dZ_norm = dZ_norm / cp.sqrt(variance + self.eps) + dvar * 2 * (Z - mean) / m + dmean / m

        return dZ_norm, dgamma, dbeta

    def get_params(self, params: dict, key: str):
        params[f'{key}_filters'] = self.filters if on_cpu else cp.asnumpy(self.filters)
        params[f'{key}_biases'] = self.biases if on_cpu else cp.asnumpy(self.biases)
        params[f'{key}_stride'] = self.stride if on_cpu else cp.asnumpy(self.stride)
        params[f'{key}_padding'] = self.padding if on_cpu else cp.asnumpy(self.padding)
        params[f'{key}_l2_lambda'] = self.l2_lambda if on_cpu else cp.asnumpy(self.l2_lambda)
        params[f'{key}_clip_value'] = self.clip_value if on_cpu else cp.asnumpy(self.clip_value)
        params[f'{key}_momentum'] = self.momentum if on_cpu else cp.asnumpy(self.momentum)
        params[f'{key}_gamma'] = self.gamma if on_cpu else cp.asnumpy(self.gamma)
        params[f'{key}_beta'] = self.beta if on_cpu else cp.asnumpy(self.beta)
        params[f'{key}_eps'] = self.eps if on_cpu else cp.asnumpy(self.eps)
        params[f'{key}_running_mean'] = self.running_mean if on_cpu else cp.asnumpy(self.running_mean)
        params[f'{key}_running_variance'] = self.running_variance if on_cpu else cp.asnumpy(self.running_variance)

    def set_params(self, params: dict, key: str):
        self.filters = cp.array(params[f'{key}_filters'])
        self.biases = cp.array(params[f'{key}_biases'])
        self.stride = params[f'{key}_stride'].item()
        self.padding = params[f'{key}_padding'].item()
        self.l2_lambda = params[f'{key}_l2_lambda'].item()
        self.clip_value = params[f'{key}_clip_value'].item()
        self.momentum = params[f'{key}_momentum'].item()
        self.gamma = cp.array(params[f'{key}_gamma'])
        self.beta = cp.array(params[f'{key}_beta'])
        self.eps = params[f'{key}_eps'].item()
        self.running_mean = cp.array(params[f'{key}_running_mean'])
        self.running_variance = cp.array(params[f'{key}_running_variance'])


def _im2col(X, filter_height, filter_width, padding, stride):
    """
    convert batch of images into the single matrix

    Args:
        X (cp.ndarray): Input images of shape (m, height, width, num_channels)
        filter_height (int): Height of the filter.
        filter_width (int): Width of the filter.
        padding (int): Amount of zero-padding around each image.
        stride (int): Stride of the convolution.
    Returns:
        cols (cp.ndarray): 2D array of shape 
            (filter_height * filter_width * num_channels, m * out_height * out_width)
    """
    _, height, weight, num_channels = X.shape

    height_padded, width_padded = height + 2 * padding, weight + 2 * padding
    X_padded = cp.pad(X, ((0, 0), (padding, padding), (padding, padding), (0, 0)), mode='constant', constant_values=0)

    # how many times the filter can "fit" in the padded image vertically and horizontally respectively
    out_height = (height_padded - filter_height) // stride + 1
    out_width = (width_padded - filter_width) // stride + 1

    # row indices within a filter
    row_indices = cp.repeat(cp.arange(filter_height), filter_width)  # [0, 0, 0, 1, 1, 1, 2, 2, 2, ...]
    # extend indices to all channels
    row_indices = cp.tile(row_indices, num_channels)  # [0, 0, 0, 1, 1, 1, 2, 2, 2, ...] * num_channels

    # row offsets for each patch
    row_offsets = stride * cp.repeat(cp.arange(out_height), out_width)

    # column offsets for each patch
    column_indices = cp.tile(cp.arange(filter_width), filter_height * num_channels)

    # row offsets for each patch
    column_offsets = stride * cp.tile(cp.arange(out_width), out_height)

    # index in X_padded for every patch
    rows = row_indices.reshape(-1, 1) + row_offsets.reshape(1, -1)
    columns = column_indices.reshape(-1, 1) + column_offsets.reshape(1, -1)

    # filter indices for every element in the filter patch
    filter_indices = cp.repeat(cp.arange(num_channels), filter_height * filter_width).reshape(-1, 1)

    # extract the patches
    patches = X_padded[:, rows, columns, filter_indices]  # shape: (m, filter_height*filter_width*num_channels, out_height*out_width)

    # rearranging the extracted patches into Columns
    cols = patches.transpose(1, 2, 0).reshape(filter_height * filter_width * num_channels, -1)

    return cols


def _col2im(cols, X_shape, filter_height, filter_width, padding, stride):
    """
    Inverse of im2col. Reconstructs the image from column representation.
    Args:
        cols (cp.ndarray): 2D array of shape 
            (filter_height * filter_width * C, N * out_height * out_width)
        X_shape (tuple): Shape of the original input X, (m, height, width, num_channels)
        filter_height (int): Height of the filter.
        filter_width (int): Width of the filter.
        padding (int): Padding used in the forward pass.
        stride (int): Stride used in the forward pass.
    Returns:
        X_reconstructed (cp.ndarray): Reconstructed images of shape (m, height, width, num_channels)
    """
    (m, height, width, num_filters) = X_shape
    height_padded, width_padded = height + 2 * padding, width + 2 * padding

    out_height = (height_padded - filter_height) // stride + 1
    out_width = (width_padded - filter_width) // stride + 1

    X_padded = cp.zeros((m, height_padded, width_padded, num_filters), dtype=cols.dtype)

    # Compute indices for im2col (basically the same as in the _im2col)
    row_indices = cp.repeat(cp.arange(filter_height), filter_width)
    row_indices = cp.tile(row_indices, num_filters)

    row_offsets = stride * cp.repeat(cp.arange(out_height), out_width)

    column_indices = cp.tile(cp.arange(filter_width), filter_height * num_filters)

    column_offsets = stride * cp.tile(cp.arange(out_width), out_height)

    rows = row_indices.reshape(-1, 1) + row_offsets.reshape(1, -1)
    columns = column_indices.reshape(-1, 1) + column_offsets.reshape(1, -1)

    filter_indices = cp.repeat(cp.arange(num_filters), filter_height * filter_width).reshape(-1, 1)

    # tile to match the shape of rows and columns
    filter_indices = cp.tile(filter_indices, (1, out_height * out_width))

    cols_reshaped = cols.reshape(filter_height * filter_width * num_filters, out_height * out_width, m)
    cols_reshaped = cols_reshaped.transpose(2, 0, 1)

    # Accumulate the columns back into the image
    for n in range(m):
        cp.add.at(X_padded[n], (rows, columns, filter_indices), cols_reshaped[n])

    if padding == 0:
        return X_padded

    return X_padded[:, padding:-padding, padding:-padding, :]
