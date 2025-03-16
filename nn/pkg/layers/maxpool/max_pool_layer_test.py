import unittest
import numpy as np

from nn.pkg.layers.maxpool.layer import MaxPool


class TestMaxPoolLayer(unittest.TestCase):
    def test_pool_forward(self):
        pool = MaxPool(
            pool_size=2,
            stride=2
        )

        X = np.array(
            [[[[2], [3], [1], [9]],
              [[4], [7], [3], [5]],
              [[8], [2], [2], [2]],
              [[1], [3], [4], [5]]]]
        )

        X_pooled = pool.pool_forward(X)

        X_pooled_expected = np.array(
            [[[[7], [9]],
              [[8], [5]]]]
        )

        self.assertTrue(np.allclose(X_pooled, X_pooled_expected), "internal values are different")

    def test_pool_backward(self):
        pool = MaxPool(
            pool_size=2,
            stride=2
        )

        X = np.array(
            [[[[2], [3], [1], [9]],
              [[4], [7], [3], [5]],
              [[8], [2], [2], [2]],
              [[1], [3], [4], [5]]]]
        )

        X_pooled = pool.pool_forward(X)
        
        
        dX_actual = pool.pool_backward(X_pooled)   
        
        print(dX_actual)     
        dX_expected = np.array(
            [[[[0], [0], [0], [9]],
              [[0], [7], [0], [0]],
              [[8], [0], [0], [0]],
              [[0], [0], [0], [5]]]]
        )
        
        self.assertTrue(np.allclose(dX_actual, dX_expected), "internal values are different")
