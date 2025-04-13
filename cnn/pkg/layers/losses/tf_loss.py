import tensorflow as tf
import numpy as np
from keras.api.losses import Loss


def calculate_iou(boxes1: tf.Tensor, boxes2: tf.Tensor, batch_size, S):
    """
    Calculate IoU between boxes using TensorFlow operations

    Args:
        boxes1: tensor of shape (batch_size, S, S, B, 4) [x, y, w, h]
        boxes2: tensor of shape (batch_size, S, S, B, 4) [x, y, w, h]
        batch_size: batch size
        S: grid size

    Returns:
        tensor of shape (batch_size, S, S) containing IoU values
    """
    b1 = tf.split(boxes1, 4, axis=-1)  # Each shape [batch_size, S, S, B, 1]
    b2 = tf.split(boxes2, 4, axis=-1)  # Each shape [batch_size, S, S, B, 1]

    b1x = b1[0]  # x-center
    b1y = b1[1]  # y-center
    b1w = b1[2]  # width
    b1h = b1[3]  # height

    b2x = b2[0]  # x-center
    b2y = b2[1]  # y-center
    b2w = b2[2]  # width
    b2h = b2[3]  # height

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
    iou = tf.clip_by_value(intersection_area / (union_area + 1e-7), 0.0, 1.0)  # (batch_size, S, S, B, 1)
    iou = tf.squeeze(iou, axis=-1)  # Remove last dimension to get (batch_size, S, S, B)

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

        pred_boxes = tf.reshape(y_pred[..., :self.B*5], [-1, self.S, self.S, self.B, 5])
        true_boxes = tf.reshape(y_true[..., :self.B*5], [-1, self.S, self.S, self.B, 5])

        pred_conf = pred_boxes[..., 4]
        true_conf = true_boxes[..., 4]
        obj_mask = true_conf  # (batch_size, S, S, B)
        noobj_mask = 1.0 - obj_mask

        # Calculate IoU between predicted and true boxes for each B
        pred_coords = pred_boxes[..., :4]
        true_coords = true_boxes[..., :4]

        iou_values = calculate_iou(pred_coords, true_coords, batch_size, self.S)  # type: ignore

        # box coordinate loss (only for cells with objects)
        boxes_losses = self.lambda_coord * tf.reduce_sum(tf.multiply(obj_mask, (1.0 - iou_values)))

        # focal weights for object loss
        pt = obj_mask * pred_conf + (1 - obj_mask) * (1 - pred_conf)  # probability for focal weighting
        focal_weight = self.focal_alpha * tf.pow(1.0 - pt, self.focal_gamma)
        obj_loss = tf.reduce_sum(focal_weight * obj_mask * tf.square(pred_conf - iou_values))

        # no-object confidence loss with focal loss
        noobj_loss = self.lambda_noobj * tf.reduce_sum(
            noobj_mask * tf.pow(pred_conf, self.focal_gamma) * tf.square(pred_conf)
        )

        boxes_total_loss = tf.add_n([boxes_losses, obj_loss, noobj_loss])

        # compute class loss
        # Calculate cells with objects using tf.concat and tf.reduce_max
        cells_with_obj = tf.reduce_max(obj_mask, axis=3, keepdims=True)  # (batch_size, S, S, 1)

        # Class prediction loss
        class_idx = self.B * 5
        pred_class = y_pred[..., class_idx:class_idx+self.C]
        true_class = y_true[..., class_idx:class_idx+self.C]

        class_loss_per_cell = -tf.reduce_sum(
            tf.multiply(true_class, tf.math.log(pred_class + self.epsilon)),
            axis=-1, keepdims=True
        )
        class_loss = tf.reduce_sum(tf.multiply(cells_with_obj, class_loss_per_cell))

        negative_coords_loss = self._negative_coordinate_penalty(y_pred) * 0.15

        total_loss = (boxes_total_loss + class_loss + negative_coords_loss) / tf.cast(batch_size, tf.float32)  # type: ignore

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
