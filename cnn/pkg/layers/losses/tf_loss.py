import tensorflow as tf
import numpy as np
from keras.api.losses import Loss


def calculate_iou(boxes1: tf.Tensor, boxes2: tf.Tensor, batch_size, S):
    """
    Calculate IoU between boxes using TensorFlow operations

    Args:
        boxes1: tensor of shape (batch_size, S, S, 4) [x, y, w, h]
        boxes2: tensor of shape (batch_size, S, S, 4) [x, y, w, h]
        batch_size: batch size
        S: grid size

    Returns:
        tensor of shape (batch_size, S, S) containing IoU values
    """
    # Reshape to simplify operations
    flat_shape = tf.constant([-1, 4], dtype=tf.int32)
    boxes1_flat = tf.reshape(boxes1, flat_shape)
    boxes2_flat = tf.reshape(boxes2, flat_shape)

    # Extract components using tf.slice
    b1x = tf.slice(boxes1_flat, [0, 0], [-1, 1])
    b1y = tf.slice(boxes1_flat, [0, 1], [-1, 1])
    b1w = tf.slice(boxes1_flat, [0, 2], [-1, 1])
    b1h = tf.slice(boxes1_flat, [0, 3], [-1, 1])

    b2x = tf.slice(boxes2_flat, [0, 0], [-1, 1])
    b2y = tf.slice(boxes2_flat, [0, 1], [-1, 1])
    b2w = tf.slice(boxes2_flat, [0, 2], [-1, 1])
    b2h = tf.slice(boxes2_flat, [0, 3], [-1, 1])

    # Convert to corner format
    b1x1 = b1x - b1w/2
    b1y1 = b1y - b1h/2
    b1x2 = b1x + b1w/2
    b1y2 = b1y + b1h/2

    b2x1 = b2x - b2w/2
    b2y1 = b2y - b2h/2
    b2x2 = b2x + b2w/2
    b2y2 = b2y + b2h/2

    # Calculate intersection area
    x_left = tf.maximum(b1x1, b2x1)
    y_top = tf.maximum(b1y1, b2y1)
    x_right = tf.minimum(b1x2, b2x2)
    y_bottom = tf.minimum(b1y2, b2y2)

    width = tf.maximum(x_right - x_left, 0)
    height = tf.maximum(y_bottom - y_top, 0)
    intersection_area = tf.multiply(width, height)

    # Calculate areas of each box
    box1_area = tf.multiply(b1w, b1h)
    box2_area = tf.multiply(b2w, b2h)

    # Calculate union area
    union_area = box1_area + box2_area - intersection_area

    # Calculate IoU
    iou = tf.clip_by_value(intersection_area / (union_area + 1e-7), 0.0, 1.0)
    iou = tf.reshape(iou, [batch_size, S, S])

    return iou

class YOLOLoss(Loss):
    def __init__(self, S, B, C, lambda_coord=10.0, lambda_noobj=0.5, focal_gamma=2.0, focal_alpha=0.25):
        """
        YOLO Loss function

        Args:
            S: grid size
            B: number of bounding boxes per grid cell
            C: number of classes
            lambda_coord: weight for coordinate loss
            lambda_noobj: weight for no-object confidence loss
            focal_gamma: focal loss gamma parameter
            focal_alpha: focal loss alpha parameter
        """
        super(YOLOLoss, self).__init__()
        self.S = S
        self.B = B
        self.C = C
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha
        self.epsilon = 1e-6

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor):
        """
        Calculate YOLO loss using TensorFlow operations

        Args:
            y_true: true values - tensor of shape (batch_size, S, S, B*5+C)
            y_pred: predicted values - tensor of shape (batch_size, S, S, B*5+C)

        Returns:
            loss: scalar tensor
        """
        batch_size = tf.shape(y_pred)[0]

        # Pre-allocate tensors for results
        all_box_losses = []
        all_obj_losses = []
        all_noobj_losses = []
        all_obj_masks = []

        # Use a Python loop to calculate losses for each bounding box (B is a small constant)
        for b in range(self.B):  # This is a Python loop over a small constant, not a tensor
            box_idx = b * 5
            box_loss, obj_loss, noobj_loss, obj_mask = self._compute_box_loss(
                y_true, y_pred, box_idx, batch_size
            )
            all_box_losses.append(box_loss)
            all_obj_losses.append(obj_loss)
            all_noobj_losses.append(noobj_loss)
            all_obj_masks.append(obj_mask)

        # Sum the losses
        total_box_loss = tf.add_n(all_box_losses)
        total_obj_loss = tf.add_n(all_obj_losses)
        total_noobj_loss = tf.add_n(all_noobj_losses)

        # Calculate cells with objects using tf.concat and tf.reduce_max
        stacked_obj_masks = tf.concat(all_obj_masks, axis=-1)
        cells_with_obj = tf.reduce_max(stacked_obj_masks, axis=-1, keepdims=True)

        # Class prediction loss
        class_idx = self.B * 5
        pred_class = tf.slice(y_pred, [0, 0, 0, class_idx], [-1, -1, -1, self.C])
        true_class = tf.slice(y_true, [0, 0, 0, class_idx], [-1, -1, -1, self.C])

        class_loss_per_cell = -tf.reduce_sum(
            tf.multiply(true_class, tf.math.log(pred_class + self.epsilon)),
            axis=-1, keepdims=True
        )
        class_loss = tf.reduce_sum(tf.multiply(cells_with_obj, class_loss_per_cell))

        # Calculate total loss
        total_loss = (total_box_loss + total_obj_loss + total_noobj_loss + class_loss) / tf.cast(batch_size, tf.float32)  # type: ignore
        total_loss += 0.15 * self._negative_coordinate_penalty(y_pred)  
        
        return total_loss

    def _compute_box_loss(self, y_true: tf.Tensor, y_pred: tf.Tensor, box_idx: int, batch_size: int):
        """
        Compute loss for a single box index
        """
        # Extract boxes and confidence
        pred_box = y_pred[..., box_idx:box_idx + 4]
        true_box = y_true[..., box_idx:box_idx + 4]
        pred_conf = y_pred[..., box_idx + 4:box_idx + 5]
        true_conf = y_true[..., box_idx + 4:box_idx + 5]

        # Create masks
        obj_mask = true_conf
        noobj_mask = 1.0 - obj_mask

        # Calculate CIoU
        iou_values = calculate_iou(pred_box, true_box, batch_size, self.S)
        iou_values = tf.expand_dims(iou_values, -1)

        # Box coordinate loss (only for cells with objects)
        box_loss = self.lambda_coord * tf.reduce_sum(tf.multiply(obj_mask, (1.0 - iou_values)))

        # Object confidence loss with focal loss
        pt = tf.where(tf.equal(obj_mask, 1.0), pred_conf, 1 - pred_conf)
        focal_weight = self.focal_alpha * tf.pow(1.0 - pt, self.focal_gamma)
        obj_loss = tf.reduce_sum(obj_mask * focal_weight * tf.square(pred_conf - iou_values))
        
        noobj_loss = self.lambda_noobj * tf.reduce_sum(noobj_mask * tf.pow(pred_conf, self.focal_gamma) * tf.square(pred_conf))
        
        return box_loss, obj_loss, noobj_loss, obj_mask

    def _negative_coordinate_penalty(self, y_pred: tf.Tensor):
        """
        Calculate a penalty for negative coordinates
        
        Args:
            y_pred: Predicted tensor of shape (batch_size, S, S, B*5+C)
        
        Returns:
            Tensor representing the coordinate negativity penalty
        """
        box_predictions = y_pred[..., :self.B*5]
        
        # Reshape to separate individual box components
        boxes = tf.reshape(box_predictions, [-1, self.B, 5])
        
        x = boxes[..., 0]  # x-center
        y = boxes[..., 1]  # y-center
        w = boxes[..., 2]  # width
        h = boxes[..., 3]  # height
        
        x1 = x - w/2
        y1 = y - h/2
        x2 = x + w/2
        y2 = y + h/2
        
        x1_penalty = tf.maximum(0.0, -x1)
        y1_penalty = tf.maximum(0.0, -y1)
        x2_penalty = tf.maximum(0.0, -x2)
        y2_penalty = tf.maximum(0.0, -y2)
        
        coordinate_penalty = (
            tf.reduce_sum(x1_penalty**2) + 
            tf.reduce_sum(y1_penalty**2) + 
            tf.reduce_sum(x2_penalty**2) + 
            tf.reduce_sum(y2_penalty**2)
        )
        
        return coordinate_penalty