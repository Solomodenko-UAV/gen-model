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

        std = np.sqrt(2 / (input_channels * filter_size**2)).astype(np.float64)  # Xavier for ReLU
        self.filters = np.random.randn(filter_size, filter_size, input_channels, num_filters).astype(np.float64) * std
        self.biases = np.zeros((1, 1, 1, num_filters)).astype(np.float64)

        self.stride = stride
        self.padding = padding

        self.l2_lambda = l2_lambda
        self.clip_value = clip_value
        self.momentum = momentum
        self.eps = 1e-5
        self.gamma = np.ones((1, 1, 1, num_filters)).astype(np.float64)  # activations remain unaffected at the start
        self.beta = np.zeros((1, 1, 1, num_filters)).astype(np.float64) # no initial shift

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

    def convolve_single_step(self, X: np.ndarray, filter_idx: int):
        """
        single step of the convolution

        Args:
            X (np.ndarray): input to be convolved - matrix of shape (filter_size, filter_size, num_filters)
            filter_idx (int): index of the filter to be used

        Returns:
            float: scalar value of the convolution
        """
        return np.sum(X * self.filters[:, :, :, filter_idx]) + self.biases[0, 0, 0, filter_idx]

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
                    for c in range(num_filters):
                        Z[i, h, w, c] = self.convolve_single_step(part_to_convolve, c)

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

        (X, batch_cache) = self.cache
        (m, output_height, output_width, num_filters) = dZ.shape

        filter_size = self.filters.shape[0]

        dX = np.zeros(X.shape)
        dW = np.zeros(self.filters.shape)
        db = np.zeros(self.biases.shape)

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
