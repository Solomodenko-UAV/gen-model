import unittest
import numpy as np

from nn.pkg.layers.yolo_output.layer import YoloOutput

class TestYoloOutputLayer(unittest.TestCase):
    
    def test_feed_forward_output_shape(self):
        # Parameters: input_size, grid dimension S, B bounding boxes, C classes.
        input_size = 128
        S = 4
        B = 2
        C = 3
        m = 8  # batch size

        yolo = YoloOutput(input_size, S, B, C)
        X = np.random.randn(m, input_size)
        output = yolo.feed_forward(X)
        
        expected_shape = (m, S, S, B * 5 + C)
        self.assertEqual(output.shape, expected_shape,
                         f"Expected output shape {expected_shape}, got {output.shape}")
    
    def test_feed_forward_class_scores_softmax(self):
        # Test that the class scores in each cell sum to 1.
        input_size = 128
        S = 4
        B = 2
        C = 3
        m = 4

        yolo = YoloOutput(input_size, S, B, C)
        X = np.random.randn(m, input_size)
        output = yolo.feed_forward(X)
        
        # Class scores are located after the B*5 bounding box values.
        class_scores = output[:, :, :, B * 5:]  # shape (m, S, S, C)
        # Sum along the class axis for each grid cell.
        sums = np.sum(class_scores, axis=-1)
        np.testing.assert_allclose(sums, np.ones_like(sums), atol=1e-5,
                                   err_msg="Class scores do not sum to 1 in each grid cell")
    
    def test_feed_forward_activation_ranges(self):
        # Test that the bounding box center coordinates (x, y) and confidence scores
        # (which use sigmoid activation) are in the range [0, 1].
        input_size = 128
        S = 4
        B = 2
        C = 3
        m = 4

        yolo = YoloOutput(input_size, S, B, C)
        X = np.random.randn(m, input_size)
        output = yolo.feed_forward(X)
        
        # For each bounding box, x is at index b*5, y is at index b*5+1, confidence at b*5+4.
        for b in range(B):
            x_coords = output[:, :, :, b * 5]
            y_coords = output[:, :, :, b * 5 + 1]
            confs    = output[:, :, :, b * 5 + 4]
            self.assertTrue(np.all((x_coords >= 0) & (x_coords <= 1)),
                            f"Bounding box x values for box {b} not in [0, 1]")
            self.assertTrue(np.all((y_coords >= 0) & (y_coords <= 1)),
                            f"Bounding box y values for box {b} not in [0, 1]")
            self.assertTrue(np.all((confs >= 0) & (confs <= 1)),
                            f"Confidence values for box {b} not in [0, 1]")
    
    def test_feed_backward_gradient_shape(self):
        # Test that the backward pass returns a gradient with the shape (m, input_size).
        input_size = 128
        S = 4
        B = 2
        C = 3
        m = 8

        yolo = YoloOutput(input_size, S, B, C)
        X = np.random.randn(m, input_size)
        _ = yolo.feed_forward(X)
        
        # Create a dummy gradient of the same shape as the layer's output.
        dZ = np.random.randn(m, S, S, B * 5 + C)
        dX = yolo.feed_backward(dZ, learning_rate=0.01)
        
        expected_shape = (m, input_size)
        self.assertEqual(dX.shape, expected_shape,
                         f"Expected backward gradient shape {expected_shape}, got {dX.shape}")

if __name__ == '__main__':
    unittest.main()
