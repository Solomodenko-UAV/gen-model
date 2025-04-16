import glob
import os
import numpy as np
import tensorflow as tf
from metadata import visDrone


class DataPreProcessor(tf.Module):
    def __init__(self, images_folder: str, annotations_folder: str, target_images_shape: tuple):
        super().__init__()
        self.images_folder = images_folder
        self.annotations_folder = annotations_folder
        self.downscale_shape = target_images_shape

        self.factors_H = tf.convert_to_tensor([])
        self.factors_W = tf.convert_to_tensor([])

    def create_dataset(self, target_images_shape: tuple, S: int, B: int, C: int, anchors: tf.Tensor, batch_size: int = 32):
        tf.print("Creating dataset...")

        def process_image_and_annotation(image_path, annotation_path):
            orig_image, processed_image = self.prepare_single_image(image_path, target_images_shape)

            orig_shape = tf.shape(orig_image)
            orig_height = tf.cast(orig_shape[0], tf.float32)
            orig_width = tf.cast(orig_shape[1], tf.float32)
            target_height, target_width = target_images_shape
            factor_H = orig_height / tf.cast(target_height, tf.float32)
            factor_W = orig_width / tf.cast(target_width, tf.float32)

            annotations = extract_visDrone_annotations_from_file(annotation_path)
            annotations = tf.cast(annotations, tf.float32)

            downscaled_annotations = _downscale_annotations(annotations, factor_H, factor_W)
            cooked_annotations = _cook_annotations(downscaled_annotations, self.downscale_shape, S, B, C, anchors)
            return processed_image, cooked_annotations

        image_files = tf.io.gfile.glob(os.path.join(self.images_folder, '*.jpg'))
        annotation_paths = [
            os.path.join(self.annotations_folder, os.path.splitext(os.path.basename(f))[0] + '.txt')
            for f in image_files
        ]

        valid_pairs = [(img, ann) for img, ann in zip(image_files, annotation_paths) if os.path.exists(ann)]
        if not valid_pairs:
            raise ValueError("No valid image-annotation pairs found")

        image_paths, annotation_paths = zip(*valid_pairs)
        dataset = tf.data.Dataset.from_tensor_slices((list(image_paths), list(annotation_paths)))

        dataset = dataset.map(lambda img, ann: process_image_and_annotation(img, ann),
                              num_parallel_calls=tf.data.AUTOTUNE)

        dataset = dataset.shuffle(buffer_size=len(valid_pairs), reshuffle_each_iteration=True)
        dataset = dataset.batch(batch_size, drop_remainder=False)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset

    def find_out_anchors(self, grid_cell_size: tuple, B: int):
        """
        Find anchor boxes using k-means clustering in a thread-safe and TensorFlow-compatible way.

        Args:
            grid_cell_size (tuple): The size of each grid cell (height, width).
            B (int): The number of anchor boxes to generate.

        Returns:
            tf.Tensor: A tensor of shape (B, 2) containing the anchor box dimensions (width, height).
        """
        tf.print("Finding out anchors...")

        # Extract annotations
        annotations_list = self.extract_visDrone_annotations()
        if self.factors_H.shape[0] == 0:
            self.find_out_factors_H_W()

        if annotations_list.shape[0] != self.factors_H.shape[0]:  # if number of annotation butches does not match number of images
            raise ValueError("Number of annotations does not match the number of images")

        # Downscale annotations using tf.map_fn
        def downscale_fn(idx):
            return _downscale_annotations(
                annotations_list[idx].to_tensor(),
                self.factors_H[idx],
                self.factors_W[idx]
            )

        downscaled_annotations = tf.map_fn(
            downscale_fn,
            tf.range(annotations_list.shape[0]),
            fn_output_signature=tf.TensorSpec(shape=(None, 6), dtype=tf.float32)
        )

        boxes = downscaled_annotations[:, :, visDrone.width_idx:visDrone.height_idx + 1]  # (m, n, 6)

        boxes = tf.concat([boxes[i] for i in range(boxes.shape[0])], axis=0)  # (m * n, 6)

        # Normalize box dimensions by grid cell size
        boxes = tf.stack([
            boxes[:, 0] / grid_cell_size[1],  # Normalize width
            boxes[:, 1] / grid_cell_size[0]  # Normalize height
        ], axis=1)

        default_anchors = _kmeans(boxes, k=B, max_iter=300)

        return default_anchors

    def prepare_single_image(self, image_path: tf.Tensor, target_size: tuple):
        orig_image = self.load_single_image(image_path)

        downscaled_image = tf.image.resize(orig_image, target_size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        downscaled_image = tf.image.convert_image_dtype(downscaled_image, dtype=tf.float32)
        downscaled_image = tf.image.per_image_standardization(downscaled_image)

        return orig_image, downscaled_image

    def extract_visDrone_annotations(self):
        """
        Extract annotations from the annotation folder.

        Returns:
            annotations_list (tf.Tensor): a tensor of shape (n, 6) where n is the number of annotations.
        """
        annotation_files = tf.io.gfile.glob(os.path.join(self.annotations_folder, '*.txt'))

        dataset = tf.data.Dataset.from_tensor_slices(annotation_files)

        dataset = dataset.map(
            lambda file_path: extract_visDrone_annotations_from_file(file_path),
            num_parallel_calls=tf.data.AUTOTUNE
        )

        annotations_list = tf.ragged.stack([annotations for annotations in dataset])

        return annotations_list

    def find_out_factors_H_W(self):
        files = tf.io.gfile.glob(os.path.join(self.images_folder, '*.jpg'))
        num_files = len(files)
        self.factors_H = tf.TensorArray(dtype=tf.float32, size=num_files, dynamic_size=False)
        self.factors_W = tf.TensorArray(dtype=tf.float32, size=num_files, dynamic_size=False)

        for idx, file in enumerate(files):
            image_path = os.path.join(self.images_folder, file)
            orig_image = self.load_single_image(image_path)
            orig_image = tf.cast(orig_image, tf.uint8)
            factor_H = tf.cast(tf.shape(orig_image)[0], tf.float32) / tf.cast(self.downscale_shape[0], tf.float32)
            factor_W = tf.cast(tf.shape(orig_image)[1], tf.float32) / tf.cast(self.downscale_shape[1], tf.float32)
            self.factors_H = self.factors_H.write(idx, factor_H)
            self.factors_W = self.factors_W.write(idx, factor_W)

        # Convert TensorArrays to tensors for further use
        self.factors_H = self.factors_H.stack()
        self.factors_W = self.factors_W.stack()

    def load_single_image(self, image_path):
        file_content = tf.io.read_file(image_path)
        return tf.image.decode_jpeg(file_content, channels=3)

    def debug_annotations_preprocessing(self, annotation_path: str, image_path: str, target_images_shape: tuple, S: int, B: int, C: int):
        orig_image, _ = self.prepare_single_image(image_path, target_images_shape)

        orig_shape = tf.shape(orig_image)
        orig_height = tf.cast(orig_shape[0], tf.float32)
        orig_width = tf.cast(orig_shape[1], tf.float32)
        target_height, target_width = target_images_shape
        factor_H = orig_height / tf.cast(target_height, tf.float32)
        factor_W = orig_width / tf.cast(target_width, tf.float32)

        annotations = extract_visDrone_annotations_from_file(annotation_path)
        annotations = tf.cast(annotations, tf.float32)

        downscaled_annotations = _downscale_annotations(annotations, factor_H, factor_W)
        grid_cell_size = (target_width / S, target_height / S)
        anchors = self.find_out_anchors(grid_cell_size, B)

        cooked_annotations = _cook_annotations(downscaled_annotations, self.downscale_shape, S, B, C, anchors)
        uncooked_annotations = _uncook_annotations(cooked_annotations, self.downscale_shape, S, B, C, anchors)

        digested_annotations = _downscale_annotations(uncooked_annotations, 1 / factor_H, 1 / factor_W)

        # digested_annotations = tf.cast(digested_annotations, tf.uint8)

        return orig_image, digested_annotations, annotations


def extract_visDrone_annotations_from_file(file_path):
    """
    Read an annotation file and extract annotations as a tf.Tensor.
    Each line is expected to be a comma-separated string of at least 6 numbers.
    A validity flag is appended (1 if valid, 0 if not), then filtered out.

    Returns:
        annotations(tf.Tensor) - a tensor of shape (m, 6) where m is the number of valid annotations.
    """
    file_content = tf.io.read_file(file_path)
    lines = tf.strings.split(file_content, '\n')
    lines = tf.boolean_mask(lines, tf.strings.length(lines) > 0)

    def _parse_line(line):
        values = tf.strings.split(line, ',')
        values = tf.strings.to_number(values[:6], out_type=tf.float32)
        valid = tf.cond(
            tf.logical_or(tf.equal(values[visDrone.object_existence_idx], 0),
                          tf.greater(values[visDrone.category_idx], 10)),
            lambda: tf.constant(0., tf.float32),
            lambda: tf.constant(1., tf.float32)
        )
        return tf.concat([values, [valid]], axis=0)

    annotations = tf.map_fn(_parse_line, lines, fn_output_signature=tf.TensorSpec(shape=(7,), dtype=tf.float32))
    mask = annotations[:, -1] == 1.
    annotations = tf.boolean_mask(annotations[:, :-1], mask)

    return annotations


def _downscale_annotations(ann, factor_H, factor_W):
    """
    Downscale annotation values for x, y, width, height.

    Args:
        ann (tf.Tensor): a tensor of shape (n, 6) where n is the number of annotations.
        factor_H (tf.float32): factor_H 
        factor_W (tf.float32): factor_W

    Returns:
        ann (tf.Tensor): a tensor of shape (n, 6) where n is the number of annotations.
    """

    top_left_x_scaled = ann[..., visDrone.top_left_x_idx] / factor_W
    top_left_y_scaled = ann[..., visDrone.top_left_y_idx] / factor_H
    width_scaled = ann[..., visDrone.width_idx] / factor_W
    height_scaled = ann[..., visDrone.height_idx] / factor_H

    ann_scaled = tf.stack([
        top_left_x_scaled,
        top_left_y_scaled,
        width_scaled,
        height_scaled,
        ann[..., 4],
        ann[..., 5]
    ], axis=-1)

    return ann_scaled


def _iou(boxes1, boxes2):
    """
    Compute Intersection over Union (IoU) between two sets of boxes.

    Args:
    boxes1 (tf.Tensor): First set of boxes with shape (num_boxes1, 2)
    boxes2 (tf.Tensor): Second set of boxes with shape (num_boxes2, 2)

    Returns:
    tf.Tensor: IoU scores with shape (num_boxes1, num_boxes2)
    """
    areas1 = boxes1[:, 0] * boxes1[:, 1]
    areas2 = boxes2[:, 0] * boxes2[:, 1]

    intersections = tf.minimum(
        tf.expand_dims(areas1, 1),
        tf.expand_dims(areas2, 0)
    )

    unions = tf.expand_dims(areas1, 1) + tf.expand_dims(areas2, 0) - intersections

    iou = intersections / (unions + 1e-8)
    return iou


def _cook_annotations(annotations, model_input_img_shape, S, B, C, anchors):
    """
    Convert annotations into the network target tensor.

    Args:
        annotations (tf.Tensor): A tensor of shape (m, 6).
        model_input_img_shape (tuple): (height, width).
        S (int): Number of grid cells along one dimension.
        B (int): Number of anchors per grid cell.
        C (int): Number of classes.
        anchors (tf.Tensor): A tensor of shape (B, 2) containing anchor box dimensions.

    Returns:
        tf.Tensor: A tensor of shape (S, S, B * 5 + C) representing the target tensor.
    """
    m = tf.shape(annotations)[0]

    X_center = annotations[:, visDrone.top_left_x_idx] + annotations[:, visDrone.width_idx] / 2
    Y_center = annotations[:, visDrone.top_left_y_idx] + annotations[:, visDrone.height_idx] / 2
    cell_height = tf.cast(model_input_img_shape[0], tf.float32) / tf.cast(S, tf.float32)
    cell_width = tf.cast(model_input_img_shape[1], tf.float32) / tf.cast(S, tf.float32)
    cells_x = tf.cast(X_center / cell_width, tf.int32)
    cells_y = tf.cast(Y_center / cell_height, tf.int32)
    X_cells_offset = (X_center - tf.cast(cells_x, tf.float32) * cell_width) / cell_width
    Y_cells_offset = (Y_center - tf.cast(cells_y, tf.float32) * cell_height) / cell_height

    class_targets = tf.one_hot(tf.cast(annotations[:, visDrone.category_idx], tf.int32), depth=C, dtype=tf.float32)
    class_targets = (1 - 1e-6) * class_targets + 1e-6 / tf.cast(C, tf.float32)

    gt_w = annotations[:, visDrone.width_idx] / cell_width
    gt_h = annotations[:, visDrone.height_idx] / cell_height
    gt_boxes = tf.stack([gt_w, gt_h], axis=1)

    iou_scores = _iou(gt_boxes, anchors)

    best_anchor_indices = tf.cast(tf.argmax(iou_scores, axis=1), dtype=tf.int32)

    Y_target = tf.zeros((S, S, B * 5 + C), dtype=tf.float32)

    gt_w_transformed = tf.math.log(gt_w / tf.gather(anchors[:, 0], best_anchor_indices) + 1e-10)
    gt_h_transformed = tf.math.log(gt_h / tf.gather(anchors[:, 1], best_anchor_indices) + 1e-10)

    bbox_targets = tf.stack([
        X_cells_offset,
        Y_cells_offset,
        gt_w_transformed,
        gt_h_transformed,
        tf.ones_like(gt_w)
    ], axis=1)

    for i in range(m):
        cell_y, cell_x = cells_y[i], cells_x[i]
        anchor_idx = best_anchor_indices[i]

        bbox_start_idx = anchor_idx * 5

        Y_target = tf.tensor_scatter_nd_update(
            Y_target,
            [[cell_y, cell_x, bbox_start_idx + j] for j in range(5)],
            bbox_targets[i]
        )

        class_start_idx = B * 5
        Y_target = tf.tensor_scatter_nd_update(
            Y_target,
            [[cell_y, cell_x, class_start_idx + j] for j in range(C)],
            class_targets[i]
        )

    return Y_target


def _uncook_annotations(Y_target, model_input_img_shape, S, B, C, anchors):
    """
    Convert network target tensor back to annotation format.

    Args:
        Y_target (tf.Tensor): Cooked tensor of shape (S, S, B*5 + C)
        model_input_img_shape (tuple): Original input image shape (height, width)
        S (int): Number of grid cells
        B (int): Number of anchors per cell
        C (int): Number of classes
        anchors (tf.Tensor): Anchor boxes (B, 2)
        confidence_threshold (float): Minimum confidence score to consider

    Returns:
        tf.Tensor: Uncooked annotations in format (N, 6) [x1, y1, w, h, class, confidence]
    """
    cell_height = model_input_img_shape[0] / S
    cell_width = model_input_img_shape[1] / S

    boxes = []
    confidences = []
    class_ids = []

    # Convert grid coordinates to image coordinates
    for cy in range(S):
        for cx in range(S):
            for b in range(B):
                # Extract bounding box parameters
                bbox_start = b * 5
                x_offset = Y_target[cy, cx, bbox_start + 0].numpy()
                y_offset = Y_target[cy, cx, bbox_start + 1].numpy()
                w = Y_target[cy, cx, bbox_start + 2].numpy()
                h = Y_target[cy, cx, bbox_start + 3].numpy()
                confidence = Y_target[cy, cx, bbox_start + 4].numpy()
                
                if confidence < 0.5:
                    continue

                # Reverse transformations
                x_center = (cx + x_offset) * cell_width
                y_center = (cy + y_offset) * cell_height
                width = np.exp(w) * anchors[b, 0] * cell_width
                height = np.exp(h) * anchors[b, 1] * cell_height

                # Convert to top-left coordinates
                x1 = x_center - width/2
                y1 = y_center - height/2

                # Get class probabilities
                class_start = B * 5
                class_probs = Y_target[cy, cx, class_start:class_start+C].numpy()
                class_id = np.argmax(class_probs)

                boxes.append([x1, y1, width, height])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    if len(boxes) > 0:
        boxes = np.array(boxes)
        confidences = np.array(confidences)
        class_ids = np.array(class_ids)

        annotations = np.zeros((len(boxes), 6))
        annotations[:, 0] = boxes[:, 0]  # x1
        annotations[:, 1] = boxes[:, 1]  # y1
        annotations[:, 2] = boxes[:, 2]  # width
        annotations[:, 3] = boxes[:, 3]  # height
        annotations[:, 4] = class_ids    # class
        annotations[:, 5] = confidences  # confidence

        return annotations

    return np.zeros((0, 6))


def _kmeans(boxes, k, dist=tf.reduce_mean, max_iter=300):
    """
    Perform k-means clustering on the given boxes using TensorFlow.

    Args:
    boxes (tf.Tensor): A tensor of shape (num_boxes, 2) containing box dimensions (width, height).
    k (int): The number of clusters.
    dist (callable): A function to compute the centroid of a cluster (e.g., tf.reduce_mean or tf.reduce_median).
    max_iter (int): The maximum number of iterations.

    Returns:
    tf.Tensor: A tensor of shape (k, 2) containing the cluster centroids.
    """
    num_boxes = tf.shape(boxes)[0]

    # Randomly initialize clusters
    clusters = tf.gather(boxes, tf.random.shuffle(tf.range(num_boxes))[:k])

    last_clusters = tf.zeros((num_boxes,), dtype=tf.int64)

    for _ in range(max_iter):
        iou_scores = _iou(boxes, clusters)  # Shape: (num_boxes, k)

        distances = 1 - iou_scores

        # Assign each box to the nearest cluster
        current_clusters = tf.argmin(distances, axis=1)

        # Check for convergence
        if tf.reduce_all(tf.equal(last_clusters, current_clusters)):
            break

        # Update cluster centroids
        def compute_new_centroid(cluster_idx):
            mask = tf.equal(current_clusters, cluster_idx)
            cluster_boxes = tf.boolean_mask(boxes, mask)
            return dist(cluster_boxes, axis=0) if tf.shape(cluster_boxes)[0] > 0 else clusters[cluster_idx]

        clusters = tf.stack([compute_new_centroid(i) for i in range(k)])
        last_clusters = current_clusters

    return clusters
