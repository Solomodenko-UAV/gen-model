import os
import numpy as np
import helper.helper as helper

on_cpu = os.environ.get("USE_GPU") != False and os.environ.get("USE_GPU") != 'True'

if on_cpu:
    import numpy as cp
else:
    import cupy as cp


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
        std = cp.sqrt(2 / input_size).astype(cp.float32)  # He init for ReLU
        self.weights = helper.create_orthogonal_2d_matrix((input_size, output_size), std)
        self.biases = cp.full((1, output_size), 0., dtype=cp.float32)
        self.clip_value = clip_value
        self.l2_lambda = l2_lambda

        self.weight_norms = []

        self.eps = 1e-7
        
        # Adadelta
        self.rho = 0.95
        self.E_g_weights = cp.zeros_like(self.weights)
        self.E_delta_weights = cp.zeros_like(self.weights)
        self.E_g_biases = cp.zeros_like(self.biases)
        self.E_delta_biases = cp.zeros_like(self.biases)

        #   RMSProp
        # self.decay_rate = 0.9
        # self.cache_weights = cp.zeros_like(self.weights)
        # self.cache_biases = cp.zeros_like(self.biases)

    def feed_forward(self, X: np.ndarray):
        """
        forward pass of the fully connected layer

        Args:
            X (np.ndarray): input to the fully connected layer - matrix of shape (m, input_size), represents a batch of m images

        Returns:
            A (cp.ndarray): output of the fully connected layer - matrix of shape (m, output_size), represents a batch of m images
        """

        if on_cpu:
            X_gpu = X.copy()
        else:
            X_gpu = cp.asarray(X)

        self.cache = X_gpu

        return cp.dot(X_gpu, self.weights) + self.biases

    def feed_backward(self, dZ: cp.ndarray, learning_rate: float):
        """
        backward pass of the fully connected

        Args:
            dZ (cp.ndarray): gradient of the cost with respect to the output of the fully connected layer - matrix of shape (m, output_size), represents a batch of m images
            learning_rate (float): learning rate to be used for updating the weights and biases

        Returns: 
            dX(cp.ndarray): gradient of the cost with respect to the input of the fully connected layer - matrix of shape (m, input_size), represents a batch of m images
        """
        X = self.cache
        m = X.shape[0]

        dW = cp.dot(X.T, dZ) / m
        db = cp.sum(dZ, axis=0, keepdims=True) / m

        dX = cp.dot(dZ, self.weights.T)

        # self.weights -= learning_rate * (dW + self.l2_lambda * self.weights)  # L2 regularization
        # self.biases -= db * learning_rate
        grad_weights = dW + self.l2_lambda * self.weights

        # Adadelta
        helper.adadelta_update(self.weights, grad_weights, self.E_g_weights, self.E_delta_weights, self.rho, self.eps)
        helper.adadelta_update(self.biases, db, self.E_g_biases, self.E_delta_biases, self.rho, self.eps)

        # RMSProp
        # self.cache_weights = helper.rmsprop_update(self.weights, grad_weights, self.cache_weights,
        #                                            decay_rate=self.decay_rate, learning_rate=learning_rate, eps=self.eps)
        # self.cache_biases = helper.rmsprop_update(self.biases, db, self.cache_biases,
        #                                           decay_rate=self.decay_rate, learning_rate=learning_rate, eps=self.eps)

        self.weight_norms.append(cp.linalg.norm(self.weights))

        return dX, dW, db

    def get_params(self, params: dict, key: str):
        params[f'{key}_weights'] = self.weights if on_cpu else cp.asnumpy(self.weights)
        params[f'{key}_biases'] = self.biases if on_cpu else cp.asnumpy(self.biases)
        params[f'{key}_l2_lambda'] = self.l2_lambda if on_cpu else cp.asnumpy(self.l2_lambda)
        params[f'{key}_clip_value'] = self.clip_value if on_cpu else cp.asnumpy(self.clip_value)

        return params

    def set_params(self, params: dict, key: str):
        self.weights = cp.array(params[f'{key}_weights'])
        self.biases = cp.array(params[f'{key}_biases'])
        self.l2_lambda = params[f'{key}_l2_lambda'].item()
        self.clip_value = params[f'{key}_clip_value'].item()
