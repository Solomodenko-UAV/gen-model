import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from metadata import visDrone
from nn.pkg.models.tinysimmoYOLO import TinysimmoYOLOModel


class YOLOActorPhoto():
    def __init__(self,
                 data_folder: str,
                 annotation_folder: str,
                 model_input_img_res: tuple,
                 model: TinysimmoYOLOModel,
                 mini_batch_size: int
                 ):

        self.data_folder = data_folder
        self.annotation_folder = annotation_folder
        self.model_input_img_res = model_input_img_res
        self.model = model
        self.mini_batch_size = mini_batch_size
        self.data_idx = 0

    def func_for_tests(self, show_model_boxes=False, evaluate=False):
        entries = os.listdir(self.data_folder)
        img_files = {}
        annotation_files = {}
        for entry in entries:
            file_name = entry.split('.')[0]
            img_files[file_name] = os.path.join(self.data_folder, entry)
            annotation_files[file_name] = os.path.join(self.annotation_folder, file_name + '.txt')

        # just show annotations and boxes on the image
        # for file_name, img_path in img_files.items():
        #     img, img_resized = self._prepare_single_image(img_path)
        #     annotations = _extract_visDrone_annotations(annotation_files[file_name])
        #     annotations_resized = _downscale_annotation(annotations, img.shape[0] // img_resized.shape[0], img.shape[1] // img_resized.shape[1])

        #     fig, ax = plt.subplots(1)
        #     ax.imshow(img_resized)
        #     for i in range(len(annotations)):
        #         x, y, w, h, _, c, _, _ = annotations_resized[i]
        #         rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')
        #         ax.add_patch(rect)
        #         label = visDrone.categories.get(c)
        #         ax.text(x, y - 20, label, color='r', fontsize=10, bbox=dict(facecolor='white', alpha=0.5))

        #     plt.axis('off')
        #     plt.show()

        # show model output
            for file_name, img_path in img_files.items():
                img, img_resized = self._prepare_single_image(img_path)
                images = img_resized.reshape(1, *img_resized.shape)

                if show_model_boxes:
                    boxes_data = self.predict(images)
                    fig, ax = plt.subplots(1)
                    ax.imshow(img_resized)
                    for i in range(len(boxes_data)):
                        boxes_per_image = boxes_data[i]
                        for box in boxes_per_image:
                            x, y, w, h, obj_class, score = box
                            rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')
                            ax.add_patch(rect)

                    # plt.axis('off')
                    plt.show()

                if evaluate:
                    annotations = _extract_visDrone_annotations(annotation_files[file_name])
                    annotations_resized = _downscale_annotation(annotations, img.shape[0] // img_resized.shape[0], img.shape[1] // img_resized.shape[1])

                    annotations = [annotations]
                    annotations_resized = [annotations_resized]
                    mAp, aps = self.evaluate_model(images, annotations_resized)
                    print("resized annotations")
                    print(f"mAP: {mAp}, APs: {aps}")
                    print("--------------")

                    mAp, aps = self.evaluate_model(images, annotations)
                    print("NOT resized annotations")
                    print(f"mAP: {mAp}, APs: {aps}")
                    print("--------------")

    def load_data(self):
        """
        Load the data from the data folder

        Returns:
            tuple: tuple containing the images and the annotations.
                    Images are of shape (m, height, width, num_channels) and
                    annotations are of shape (m, n, 8)
        """
        entries = os.listdir(self.data_folder)[self.data_idx:self.data_idx + self.mini_batch_size]
        img_files = {}
        annotation_files = {}
        for entry in entries:
            file_name = entry.split('.')[0]
            img_files[file_name] = os.path.join(self.data_folder, entry)
            annotation_files[file_name] = os.path.join(self.annotation_folder, file_name + '.txt')

        I = []
        IR = np.zeros((len(img_files), *self.model_input_img_res, 3))
        annotations_list = [[] for _ in range(len(img_files))]
        i = 0

        for file_name, img_path in img_files.items():
            img, img_resized = self._prepare_single_image(img_path)
            annotations = _extract_visDrone_annotations(annotation_files[file_name])

            I.append(np.array(img))
            IR[i] = img_resized
            annotations_list[i] = np.array(annotations).astype(np.float64)
            i += 1

        print(f"Loaded images from {self.data_idx} to {self.data_idx + self.mini_batch_size} idx")
        self.data_idx += self.mini_batch_size

        return np.array(I).astype(np.float64), IR, annotations_list

    def run_training_loop(self, epochs=10, learning_rate=0.01):
        losses = []
        for i in range(epochs):
            original_images, resized_images, annotations = self.load_data()
            if resized_images.shape[0] == 0:
                print("No more data to train on")
                return

            Y = self.model.forward(resized_images)
            loss = self.train_on_multiple_images(
                Y=Y,
                Y_hat=annotations,
                original_img_shape=(original_images[0].shape[0], original_images[0].shape[1]),
                learning_rate=learning_rate
            )

            losses.append(loss)

            print(f"Epoch {i}, loss: {loss}")

        plt.plot(range(epochs), losses, label="Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Training Loss Over Epochs")
        plt.legend()
        plt.grid(True)
        # plt.show()

    def train_on_multiple_images(self,
                                 Y: np.ndarray,
                                 Y_hat: list,
                                 original_img_shape: tuple,
                                 learning_rate: float):
        """
        Calculate the loss
        Args:
            Y (np.ndarray): predicted values - matrix of shape (m, S, S, B*5+C), represents a batch of m images
            Y_hat (np.ndarray): true values - matrix of shape (m, n, 8), represents a batch of m annotations
            original_img_shape (tuple): shape of the original image (height, width)
            model_input_img_shape (tuple): shape of the image that the model accepts (height, width)

        Returns:
            loss(float): loss value
        """

        Y_target = np.zeros((len(Y_hat), self.model.S, self.model.S, self.model.B * 5 + self.model.C))
        for i in range(len(Y_hat)):
            Y_target[i] = self._cook_annotations(Y_hat[i], original_img_shape, self.model_input_img_res)

        loss, grad_A = self._calc_loss_and_gradient(Y, Y_target)
        self.model.backward(grad_A, learning_rate)

        return loss

    def _cook_annotations(self, annotations: np.ndarray, original_img_shape: tuple, model_input_img_shape: tuple):
        """
        convert list of annotations to the model output shape

        Args:
            annotations (np.ndarray): true values - matrix of shape (m, 8), represents a batch of m annotations
            original_img_shape (tuple): shape of the original image (width, height)
            model_input_img_shape (tuple): shape of the image that the model accepts (width, height)

        Returns:
            Y_target (np.ndarray): true values - matrix of shape (S, S, B*5+C), represents a batch of m images
        """
        m = annotations.shape[0]
        W = original_img_shape[1] / model_input_img_shape[1]
        H = original_img_shape[0] / model_input_img_shape[0]

        Y_norm = annotations.copy()
        Y_norm[:, visDrone.top_left_x_idx] /= W
        Y_norm[:, visDrone.top_left_y_idx] /= H
        Y_norm[:, visDrone.width_idx] /= W
        Y_norm[:, visDrone.height_idx] /= H

        # assign annotations to a grid cells
        X_center = (Y_norm[:, visDrone.top_left_x_idx] + Y_norm[:, visDrone.width_idx] / 2)
        Y_center = (Y_norm[:, visDrone.top_left_y_idx] + Y_norm[:, visDrone.height_idx] / 2)
        cell_height = model_input_img_shape[0] / self.model.S
        cell_width = model_input_img_shape[1] / self.model.S

        # assign bounding box to a cell
        cells_x = X_center // cell_width
        cells_y = Y_center // cell_height

        # calc offset of the x and y inside the cell where the center of the box is. Relative values [0, 1]
        X_cells_offset = (X_center - cells_x * cell_width) / cell_width
        Y_cells_offset = (Y_center - cells_y * cell_height) / cell_height

        # create one-hot encoding vector for the class
        class_targets = np.zeros((m, self.model.C))
        class_targets[Y_norm[:, visDrone.category_idx].astype(np.int8)] = 1

        # construct output matrix - aka reshape annotations to the model output shape
        Y_target = np.zeros((self.model.S, self.model.S, self.model.B * 5 + self.model.C))

        for i in range(m):
            cell_x = int(cells_x[i])
            cell_y = int(cells_y[i])

            # bounding box target
            bbox_target = np.array([
                X_cells_offset[i],
                Y_cells_offset[i],
                Y_norm[i, visDrone.width_idx],
                Y_norm[i, visDrone.height_idx],
                1,
            ])

            # TODO: for now it's just 1st box, but I should choose the best somehow (IoU based?)
            bbox_slot = 0
            start_index = bbox_slot * 5
            Y_target[cell_y, cell_x, start_index:start_index+5] = bbox_target
            Y_target[cell_y, cell_x, self.model.B * 5:] = class_targets[i]

        return Y_target

    def _calc_loss_and_gradient(self, A: np.ndarray, Y_target: np.ndarray):
        """
        Calculate the loss and gradient

        Args:
            A (np.ndarray): predicted values - matrix of shape (m, S, S, B*5+C). For this function m == 1 is required
            Y_target (np.ndarray): true values - matrix of shape (m, S, S, B*5+C)

        Returns:
            mean_loss(float): mean loss value over the image
            grad_A(np.ndarray): gradient of the loss with respect to A - matrix of shape (m, S, S, B*5+C)
        """
        loss = 0.0
        grad_A = np.zeros_like(A)

        # Constants for loss weighting
        lambda_coord = 5.0
        lambda_noobj = 0.5

        m, S, _, _ = A.shape
        B = self.model.B
        C = self.model.C

        for i in range(m):
            for row in range(S):
                for col in range(S):
                    prediction = A[i, row, col]  # [B * 5 + C] array
                    target = Y_target[i, row, col]

                    # for each bounding box predictor in this cell
                    for b in range(B):
                        idx = b * 5
                        pred_bbox = prediction[idx:idx+5]  # [6] array
                        target_bbox = target[idx:idx+5]

                        # if model thinks there is no object in this box
                        if target_bbox[visDrone.object_existence_idx] == 0:
                            conf_diff = pred_bbox[visDrone.object_existence_idx]
                            loss += lambda_noobj * (conf_diff ** 2)
                            grad_A[i, row, col, idx + visDrone.object_existence_idx] = 2 * lambda_noobj * conf_diff
                            continue

                        # if model thinks there is an object in this box
                        # localization loss for x, y
                        # calc squared error
                        for j in range(visDrone.top_left_y_idx + 1):
                            diff = pred_bbox[j] - target_bbox[j]
                            loss += lambda_coord * (diff ** 2)
                            grad_A[i, row, col, idx + j] = 2 * lambda_coord * diff

                        # for width and height, apply square root transformation to stabilize small boxes
                        for j in range(visDrone.width_idx, visDrone.height_idx + 1):
                            # avoid division by zero
                            pred_sqrt = np.sqrt(np.maximum(pred_bbox[j], 1e-6))
                            target_sqrt = np.sqrt(target_bbox[j])

                            diff = pred_sqrt - target_sqrt
                            loss += lambda_coord * (diff ** 2)
                            # derivative of sqrt (that we've just applied couple lines above) is 1/(2*sqrt(x))
                            grad_A[i, row, col, idx+j] = 2 * lambda_coord * diff * (1/(2*np.sqrt(np.maximum(pred_bbox[j], 1e-6))))

                        # confidence loss
                        conf_diff = pred_bbox[visDrone.object_existence_idx] - target_bbox[visDrone.object_existence_idx]
                        loss += (conf_diff ** 2)
                        grad_A[i, row, col, idx + visDrone.object_existence_idx] = 2 * conf_diff

                        # classification loss
                        pred_class = prediction[B * 5:]  # [C] array
                        target_class = target[B * 5:]
                        class_diff = pred_class - target_class
                        loss += np.sum(class_diff ** 2)
                        grad_A[i, row, col, B * 5:] = 2 * class_diff

        mean_loss = loss / (m * S * S * C)
        return mean_loss, grad_A

    def predict(self, images: np.ndarray):
        """
        Predict the bounding boxes for the images

        Args:
            images (np.ndarray): images to predict the bounding boxes for - matrix of shape (m, height, width, num_channels), represents a batch of m images

        Returns:
            boxes(np.ndarray): predicted bounding boxes - matrix of shape (m, S * S * B, 6), represents a batch of m images. Each box is [x, y, w, h, class, score]
        """
        model_output = self.model.forward(images)

        m = model_output.shape[0]
        B = self.model.B
        S = self.model.S
        boxes_pred = model_output[..., :B * 5]  # shape (m, S, S, B*5)
        boxes_pred = boxes_pred.reshape(m, S, S, B, 5)

        classes_probs = model_output[..., B * 5:]  # shape (m, S, S, C)
        predicted_class_indices = np.argmax(classes_probs, axis=-1)  # shape (m, S, S)
        predicted_class_probabilities = np.max(classes_probs, axis=-1)  # shape (m, S, S)

        boxes = np.zeros((m, S, S, B, 6))

        cell_width = self.model_input_img_res[0] // S
        cell_height = self.model_input_img_res[1] // S

        for i in range(boxes_pred.shape[0]):
            for row in range(boxes_pred.shape[1]):
                for col in range(boxes_pred.shape[2]):
                    for b in range(B):
                        x, y, w, h, confidence = boxes_pred[i, row, col, b, :]

                        x_center_abs = (row + x) * cell_width
                        y_center_abs = (i + y) * cell_height

                        w_abs = w * self.model_input_img_res[0]
                        h_abs = h * self.model_input_img_res[1]

                        x1 = x_center_abs - w_abs / 2
                        y1 = y_center_abs - h_abs / 2

                        obj_class_probability = predicted_class_probabilities[i, row, col]
                        obj_class = predicted_class_indices[i, row, col]
                        score = obj_class_probability * confidence

                        boxes[i, row, col, b] = [x1, y1, w_abs, h_abs, obj_class, score]

        return boxes.reshape(m, -1, boxes.shape[-1])

    def evaluate_model(self, images: np.ndarray, annotations: list, iou_threshold: float = 0.5, score_threshold: float = 0.5):
        """
        Evaluate the model

        Args:
            images (np.ndarray): images to predict the bounding boxes for - matrix of shape (m, height, width, num_channels), represents a batch of m images
            annotations (np.ndarray): true values - matrix of shape (m, n, 8), represents a batch of m annotations

        Returns:
            float: mean average precision
            dict: average precision per class
        """
        boxes_pred = self.predict(images)
        boxes_pred = _non_max_suppression(boxes_pred, iou_threshold=iou_threshold, score_threshold=score_threshold)

        per_class_results = _calc_precision_recall(annotations, boxes_pred, iou_threshold)
        mAP, ap_per_class = _calculate_mean_average_precision(per_class_results)
        return mAP, ap_per_class

    def _prepare_single_image(self, img_path: str):
        """
        Prepare the image for the model

        Args:
            img_path (str): path to the image

        Returns:
            tuple: tuple containing the original image and the resized image
        """

        img = Image.open(img_path)

        img_array = np.array(img)
        img_resized = _downscale_img(img_array, self.model_input_img_res)
        return (img_array, img_resized)


def _downscale_img(image: np.ndarray, new_shape: tuple):
    """
    Downscale the image to the new shape

    Args:
        image (np.ndarray): matrix of shape (height, width, num_channels), represents an image
        new_shape (np.ndarray): tuple (new_height, new_width), represents the new shape of the image

    Returns:
        np.ndarray: downsampled image of shape(new_height, new_width, num_channels)
    """

    (new_height, new_width) = new_shape

    factor_H = image.shape[0] // new_height
    factor_W = image.shape[1] // new_width

    image_cropped = image[:new_height * factor_H, :new_width * factor_W, :]
    reshaped = image_cropped.reshape(new_height, factor_H, new_width, factor_W, image.shape[2])

    return reshaped.mean(axis=(1, 3)).astype(np.uint8)


def _downscale_annotation(annotations: list, factor_H: int, factor_W: int):
    """
    Downscale the annotation to the new shape

    Args:
        annotation (list): list of annotations
        new_shape (np.ndarray): tuple (new_height, new_width), represents the new shape of the image

    Returns:
        list: downsampled annotation
    """

    annotation_downscaled = [list(annotation) for annotation in annotations]
    for i in range(len(annotations)):
        annotation_downscaled[i][visDrone.top_left_x_idx] //= factor_W
        annotation_downscaled[i][visDrone.top_left_y_idx] //= factor_H
        annotation_downscaled[i][visDrone.width_idx] //= factor_W
        annotation_downscaled[i][visDrone.height_idx] //= factor_H

    return annotation_downscaled


def _extract_visDrone_annotations(file_path: str):
    """
    Extract annotations from the file

    Returns:
        list: list of dictionaries, each dictionary contains the annotation for a
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()

    annotations = []

    for line in lines:
        values = list(map(int, line.split(',')))
        annotations.append(values)

    return annotations


def _non_max_suppression(boxes: np.ndarray, iou_threshold: float = 0.5, score_threshold=0.5):
    """
    Perform non-max suppression on the boxes

    Args:
        boxes (np.ndarray): matrix of shape (m, S * S * B, 6), represents a batch of m images. Each box is [x, y, w, h, class, score]
        iou_threshold (float): threshold for the intersection over union

    Returns:
        np.ndarray: matrix of shape (m, S * S * B, 6), represents a batch of m images. Each box is [x, y, w, h, class, score]
    """
    m = boxes.shape[0]
    final_boxes = [[] for _ in range(m)]

    filtered_boxes = []
    for i in range(boxes.shape[0]):  # iterate over the batch
        # Apply the mask for each image individually
        mask = boxes[i, :, 5] > score_threshold
        filtered_boxes.append(boxes[i][mask])

    boxes = np.array(filtered_boxes, dtype=np.float64)

    classes = np.unique(boxes[:, :, 4])

    for i in range(m):
        single_image_boxes = boxes[i]  # (S * S * B, 6)

        for cls in classes:
            this_class_boxes = single_image_boxes[single_image_boxes[:, 4] == cls]

            # sort by score
            indices = np.argsort(this_class_boxes[:, 5])[::-1]
            this_class_boxes = this_class_boxes[indices]

            boxes_to_keep = []
            while len(this_class_boxes) > 0:
                current_box = this_class_boxes[0]  # box with the highest score. Shape (6,)
                boxes_to_keep.append(current_box)

                remaining_boxes_this_class = []
                for box in this_class_boxes[1:]:
                    if _compute_iou(current_box[:4], box[:4]) < iou_threshold:  # if not the same object
                        remaining_boxes_this_class.append(box)  # remain this box in the pool

                this_class_boxes = np.array(remaining_boxes_this_class)

            final_boxes[i].extend(boxes_to_keep)  # write this class boxes to the final list

    return np.array(final_boxes)


def _compute_iou(boxA: np.ndarray, boxB: np.ndarray):
    """
    Compute the intersection over union of the two boxes

    Args:
        boxA (np.ndarray): matrix of shape (4,), represents the coordinates of the top-left and bottom-right corners of the box
        boxB (np.ndarray): matrix of shape (4,), represents the coordinates of the top-left and bottom-right corners of the box

    Returns:
        float: intersection over union of the two boxes
    """
    x1_inner = max(boxA[0], boxB[0])
    y1_inner = max(boxA[1], boxB[1])
    x2_inner = min(boxA[2], boxB[2])
    y2_inner = min(boxA[3], boxB[3])
    inner_area = max(0, x2_inner - x1_inner) * max(0, y2_inner - y1_inner)

    if inner_area == 0:
        return 0

    boxA_area = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxB_area = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    return inner_area / (boxA_area + boxB_area - inner_area)


def _calc_precision_recall(true_boxes: list, pred_boxes: np.ndarray, iou_threshold: float = 0.5):
    """
    Calculate the precision and recall

    Args:
        true_boxes (list): list of true boxes - matrix of shape (m, n, 8), represents a batch of m annotations
        pred_boxes (np.ndarray): predicted boxes - matrix of shape (m, S * S * B, 6), represents a batch of m images. Each box is [x, y, w, h, class, score]
        iou_threshold (float, optional): iou threshold. Defaults to 0.5.
    """
    # determine all unique classes
    classes = set()
    for image_boxes in true_boxes:
        for box in image_boxes:
            classes.add(box[visDrone.category_idx])

    for image_boxes in pred_boxes:
        for box in image_boxes:
            classes.add(box[4])

    classes = list(classes)
    per_class_results = {}

    m = len(true_boxes)

    for cls in classes:
        this_class_tp_list = []
        this_class_fp_list = []
        this_class_total_true_boxes = 0

        for i in range(m):
            this_image_and_class_true_boxes = [box for box in true_boxes[i] if box[4] == cls]
            this_image_and_class_pred_boxes = [box for box in pred_boxes[i] if box[visDrone.category_idx] == cls]

            this_class_total_true_boxes += len(this_image_and_class_true_boxes)

            # sort by score in descending order
            this_image_and_class_pred_boxes = sorted(this_image_and_class_pred_boxes, key=lambda x: x[5], reverse=True)

            this_image_and_class_true_boxes_detected = [False] * len(this_image_and_class_true_boxes)

            for this_image_and_class_pred_box in this_image_and_class_pred_boxes:
                pred_box_coords = this_image_and_class_pred_box[:4]

                max_iou = 0.
                max_iou_idx = -1

                for idx, true_box in enumerate(this_image_and_class_true_boxes):
                    true_box_x1 = true_box[visDrone.top_left_x_idx]
                    true_box_y1 = true_box[visDrone.top_left_y_idx]
                    true_box_x2 = true_box[visDrone.top_left_x_idx] + true_box[visDrone.width_idx]
                    true_box_y2 = true_box[visDrone.top_left_y_idx] + true_box[visDrone.height_idx]
                    true_box = np.ndarray([true_box_x1, true_box_y1, true_box_x2, true_box_y2])

                    iou = _compute_iou(pred_box_coords, true_box)
                    if iou > max_iou:
                        max_iou = iou
                        max_iou_idx = idx

                if max_iou <= iou_threshold or this_image_and_class_true_boxes_detected[max_iou_idx]:  # duplicate detection
                    this_class_tp_list.append(0)
                    this_class_fp_list.append(1)
                    continue

                this_class_tp_list.append(1)
                this_class_fp_list.append(0)
                this_image_and_class_true_boxes_detected[max_iou_idx] = True

        cum_tp = np.cumsum(np.array(this_class_tp_list))
        cum_fp = np.cumsum(np.array(this_class_fp_list))

        recall = cum_tp / (this_class_total_true_boxes + 1e-6)
        precision = cum_tp / (cum_tp + cum_fp)

        per_class_results[cls] = (recall, precision)

    return per_class_results


def _calculate_average_precision(recall: np.ndarray, precision: np.ndarray):
    """
    Calculate the average precision

    Args:
        recall (np.ndarray): array of recall values
        precision (np.ndarray): array of precision values

    Returns:
        float: average precision
    """

    mrecall = np.concatenate(([0.], recall, [1.]))
    mprecision = np.concatenate(([0.], precision, [0.]))

    for i in range(len(mprecision) - 2, -1, -1):
        mprecision[i] = max(mprecision[i], mprecision[i + 1])

    indices = np.where(mrecall[1:] != mrecall[:-1])[0]

    ap = np.sum((mrecall[indices + 1] - mrecall[indices]) * mprecision[indices + 1])
    return ap


def _calculate_mean_average_precision(per_class_results: dict):
    """
    Calculate the mean average precision

    Args:
        per_class_results (dict): dictionary containing the precision and recall values for each class
    Returns:
        float: The mean Average Precision (mAP) over all classes.
        dict: The Average Precision (AP) per class.
    """
    ap_per_class = {}

    for cls, (recall, precision) in per_class_results.items():
        ap = _calculate_average_precision(recall, precision)
        ap_per_class[cls] = ap

    # average of the AP values over all classes
    mAP = np.mean(list(ap_per_class.values()))
    return mAP, ap_per_class
