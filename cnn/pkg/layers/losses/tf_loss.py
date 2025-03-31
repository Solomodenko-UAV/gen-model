import tensorflow as tf
import numpy as np
from tensorflow.python.keras.losses import Loss


def calculate_iou(boxes1, boxes2):
    """
    Calculate IoU between boxes

    Args:
        boxes1: tensor of shape (batch_size, S, S, 4) [x, y, w, h]
        boxes2: tensor of shape (batch_size, S, S, 4) [x, y, w, h]

    Returns:
        tensor of shape (batch_size, S, S, 1) containing IoU values
    """
    # Convert from center x, center y, width, height to xmin, ymin, xmax, ymax
    boxes1_mins = boxes1[..., :2] - boxes1[..., 2:4] / 2.0
    boxes1_maxes = boxes1[..., :2] + boxes1[..., 2:4] / 2.0

    boxes2_mins = boxes2[..., :2] - boxes2[..., 2:4] / 2.0
    boxes2_maxes = boxes2[..., :2] + boxes2[..., 2:4] / 2.0

    # Calculate intersections
    intersect_mins = tf.maximum(boxes1_mins, boxes2_mins)
    intersect_maxes = tf.minimum(boxes1_maxes, boxes2_maxes)
    intersect_wh = tf.maximum(intersect_maxes - intersect_mins, 0.0)

    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    boxes1_area = boxes1[..., 2] * boxes1[..., 3]
    boxes2_area = boxes2[..., 2] * boxes2[..., 3]

    union_area = boxes1_area + boxes2_area - intersect_area

    return tf.clip_by_value(intersect_area / (union_area + 1e-6), 0.0, 1.0)


def calculate_ciou(boxes1, boxes2):
    """
    Calculate CIoU (Complete IoU) between boxes

    Args:
        boxes1: tensor of shape (batch_size, S, S, 4) [x, y, w, h]
        boxes2: tensor of shape (batch_size, S, S, 4) [x, y, w, h]

    Returns:
        tensor of shape (batch_size, S, S, 1) containing CIoU values
    """
    # Calculate IoU
    iou = calculate_iou(boxes1, boxes2)

    # Convert to corners format for distance calculation
    boxes1_mins = boxes1[..., :2] - boxes1[..., 2:4] / 2.0
    boxes1_maxes = boxes1[..., :2] + boxes1[..., 2:4] / 2.0

    boxes2_mins = boxes2[..., :2] - boxes2[..., 2:4] / 2.0
    boxes2_maxes = boxes2[..., :2] + boxes2[..., 2:4] / 2.0

    # Calculate the diagonal distance of the smallest enclosing box
    enclose_mins = tf.minimum(boxes1_mins, boxes2_mins)
    enclose_maxes = tf.maximum(boxes1_maxes, boxes2_maxes)
    enclose_wh = tf.maximum(enclose_maxes - enclose_mins, 0.0)

    # Calculate the squared diagonal distance
    c_squared = enclose_wh[..., 0] ** 2 + enclose_wh[..., 1] ** 2

    # Calculate the center distance squared
    center_dist_squared = tf.reduce_sum(tf.square(boxes1[..., :2] - boxes2[..., :2]), axis=-1)

    # Calculate the aspect ratio consistency term
    v = (4 / (np.pi ** 2)) * tf.square(
        tf.atan(boxes1[..., 2] / (boxes1[..., 3] + 1e-6)) -
        tf.atan(boxes2[..., 2] / (boxes2[..., 3] + 1e-6))
    )

    # Calculate the trade-off parameter
    alpha = v / (1 - iou + v + 1e-6)

    # Calculate CIoU
    ciou = iou - (center_dist_squared / (c_squared + 1e-6) + alpha * v)

    return ciou


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

    def call(self, y_true, y_pred):
        """
        Calculate YOLO loss

        Args:
            y_true: true values - tensor of shape (batch_size, S, S, B*5+C)
            y_pred: predicted values - tensor of shape (batch_size, S, S, B*5+C)

        Returns:
            loss: scalar tensor
        """
        batch_size = tf.shape(y_pred)[0]

        # Initialize losses
        box_loss = 0
        obj_loss = 0
        noobj_loss = 0
        class_loss = 0

        for b in range(self.B):
            # Indices for this bounding box
            box_idx = b * 5
            x_idx, y_idx, w_idx, h_idx, conf_idx = [box_idx + i for i in range(5)]

            # Extract object existence masks
            obj_mask = y_true[..., conf_idx:conf_idx+1]
            noobj_mask = 1.0 - obj_mask

            # Extract predicted and target boxes
            pred_boxes = y_pred[..., [x_idx, y_idx, w_idx, h_idx]]
            true_boxes = y_true[..., [x_idx, y_idx, w_idx, h_idx]]

            # Calculate CIoU
            ciou = calculate_ciou(pred_boxes, true_boxes)
            ciou = tf.expand_dims(ciou, -1)

            # Coordinate loss for cells with objects
            box_loss += self.lambda_coord * tf.reduce_sum(obj_mask * (1.0 - ciou))

            # Confidence loss for cells with objects (using quality focal loss)
            pred_conf = y_pred[..., conf_idx:conf_idx+1]

            # Basic focal loss component
            basic_focal = -((1.0 - pred_conf) ** self.focal_gamma) * tf.math.log(pred_conf + self.epsilon)

            # Quality focal loss component
            quality = ciou  # Use CIoU as quality
            quality_diff = tf.abs(quality - pred_conf)
            quality_focal = -quality * (quality_diff ** self.focal_gamma) * tf.math.log(pred_conf + self.epsilon)

            # Combined confidence loss for objects
            combined_obj_loss = (1 - self.focal_alpha) * basic_focal + self.focal_alpha * quality_focal
            obj_loss += tf.reduce_sum(obj_mask * combined_obj_loss)

            # No-object confidence loss with focal loss
            noobj_focal_loss = -((pred_conf ** self.focal_gamma) * tf.math.log(1.0 - pred_conf + self.epsilon))
            noobj_loss += self.lambda_noobj * tf.reduce_sum(noobj_mask * noobj_focal_loss)

        # Class prediction loss (only for cells with objects)
        class_idx = self.B * 5
        pred_class = y_pred[..., class_idx:]
        true_class = y_true[..., class_idx:]

        # Calculate which cells have objects (for any box)
        cells_with_obj = tf.reduce_max(tf.reshape(
            y_true[..., 4:class_idx:5], [-1, self.S, self.S, self.B]
        ), axis=-1, keepdims=True)

        # Cross-entropy loss for classification
        class_loss_per_cell = -tf.reduce_sum(true_class * tf.math.log(pred_class + self.epsilon), axis=-1, keepdims=True)
        class_loss += tf.reduce_sum(cells_with_obj * class_loss_per_cell)

        # Calculate total loss
        total_loss = (box_loss + obj_loss + noobj_loss + class_loss) / tf.cast(batch_size, tf.float32)

        return total_loss
