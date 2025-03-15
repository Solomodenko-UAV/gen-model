import unittest
import numpy as np

from nn.pkg.layers.conv.layer import Convolution


class TestConvLayer(unittest.TestCase):
    def __init(self, *args, **kwargs):
        super(TestConvLayer, self).__init__(*args, **kwargs)

    def test_zero_pad(self):
        conv = Convolution(input_channels=1, filter_size=1, num_filters=1, stride=1, padding=1)

        X = np.array(
            [[[[1], [2], [3]],
              [[4], [5], [6]],
              [[7], [8], [9]]]]
        )

        X_pad_expected = np.array(
            [[[[0], [0], [0], [0], [0]],
              [[0], [1], [2], [3], [0]],
              [[0], [4], [5], [6], [0]],
              [[0], [7], [8], [9], [0]],
              [[0], [0], [0], [0], [0]]]]
        )

        X_pad_actual = conv.zero_pad(X)
        self.assertTrue(np.array_equal(X_pad_actual, X_pad_expected), "internal values are different")

    def test_convolve_single_step(self):
        np.random.seed(3)
        conv = Convolution(input_channels=3, filter_size=5, num_filters=3, stride=1, padding=0)
        X = np.random.randn(5, 5, 3)

        X_convolved = conv.convolve_single_step(X, 0)

        expected_value = np.float64(2.704508937548934)
        self.assertTrue(np.isclose(X_convolved, expected_value), f"Wrong value. Expected {expected_value}, got {X_convolved}")

    def test_convolve_forward(self):
        X = np.array(
            [[[[10], [10], [10], [0], [0], [0]],
              [[10], [10], [10], [0], [0], [0]],
              [[10], [10], [10], [0], [0], [0]],
              [[0], [0], [0], [10], [10], [10]],
              [[0], [0], [0], [10], [10], [10]],
              [[0], [0], [0], [10], [10], [10]]]]
        )

        filter = np.array(
            [[[[1]], [[1]], [[1]]],
             [[[0]], [[0]], [[0]]],
             [[[-1]], [[-1]], [[-1]]]]
        )

        conv = Convolution(input_channels=1, filter_size=3, num_filters=1, stride=1, padding=0)
        conv.filters = filter
        conv.biases = np.zeros((1, 1, 1, 3))

        X_convolved, _ = conv.convolve(X)
        X_convolved_expected = np.array(
            [[[[0], [0], [0], [0]],
              [[30], [10], [-10], [-30]],
              [[30], [10], [-10], [-30]],
              [[0], [0], [0], [0]]]]
        )

        self.assertTrue(np.array_equal(X_convolved, X_convolved_expected), "internal values are different")

    def test_convolve_forward_3_filters(self):
        X = np.array(
            [[[[1], [2], [3], [4], [5]],
              [[6], [7], [8], [9], [10]],
              [[11], [12], [13], [14], [15]],
              [[16], [17], [18], [19], [20]],
              [[21], [22], [23], [24], [25]]]]
        )

        filters = np.array([
            [
                [[1, 0, 1/9]], [[0, -1, 1/9]], [[-1, 0, 1/9]]
            ],
            [
                [[1, -1, 1/9]], [[0, 5, 1/9]], [[-1, -1, 1/9]]
            ],
            [
                [[1, 0, 1/9]], [[0, -1, 1/9]], [[-1, 0, 1/9]]
            ]
        ])

        conv = Convolution(input_channels=1, filter_size=3, num_filters=3, stride=1, padding=0)
        conv.filters = filters
        conv.biases = np.array([0.1, 0.2, 0.3]).reshape(1, 1, 1, 3)

        X_convolved, _ = conv.convolve(X)
        X_convolved_expected = np.array(
            [
                [
                    [
                        [-5.9,  7.2,  7.3],
                        [-5.9,  8.2,  8.3],
                        [-5.9,  9.2,  9.3]
                    ],

                    [
                        [-5.9, 12.2, 12.3],
                        [-5.9, 13.2, 13.3],
                        [-5.9, 14.2, 14.3]
                    ],

                    [
                        [-5.9, 17.2, 17.3],
                        [-5.9, 18.2, 18.3],
                        [-5.9, 19.2, 19.3]
                    ]
                ]
            ]
        )

        self.assertTrue(np.allclose(X_convolved, X_convolved_expected), "internal values are different")


if __name__ == '__main__':
    unittest.main()
