import random
from nn.pkg.activations.activations import sigmoid
from nn.pkg.models.tinysimmoYOLO import TinysimmoYOLOModel
from metadata import visDrone
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os

on_cpu = os.environ.get("USE_GPU") != False and os.environ.get("USE_GPU") != 'True' and os.environ.get("USE_GPU") != 'True'

if on_cpu:
    import numpy as cp
else:
    import cupy as cp


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

    def func_for_tests(self, print_shrunk_image=False, print_orig_image=False, evaluate=False, iou_threshold: float = 0.5, score_threshold: float = 0.5, check=False):
        entries = os.listdir(self.data_folder)
        random.shuffle(entries)

        img_files = {}
        annotation_files = {}
        for entry in entries:
            file_name = entry.split('.')[0]
            img_files[file_name] = os.path.join(self.data_folder, entry)
            annotation_files[file_name] = os.path.join(self.annotation_folder, file_name + '.txt')

        # just show annotations and boxes on the image
        if check:
            self.default_bounding_box_offsets = cp.array([[0.5, 0.5], [0.5, 0.5]])

            for file_name, img_path in img_files.items():
                img, img_resized = self._prepare_single_image(img_path)
                _, cooked_resized_annotations, orig_annotations = self.load_data_without_orig_image(entries)

                cooked_orig_annotations = cp.array(self._cook_annotations(orig_annotations[0], img.shape, img.shape))
                uncooked_orig_annotations = self._uncook_annotations(cooked_orig_annotations, img.shape, img.shape)

                uncooked_resized_annotations = self._uncook_annotations(cooked_resized_annotations[0], self.model_input_img_res, self.model_input_img_res)

                annotations = _extract_visDrone_annotations(annotation_files[file_name])
                annotations_resized = _downscale_annotation(annotations, img.shape[0] // img_resized.shape[0], img.shape[1] // img_resized.shape[1])

                _, ((ax, ay), (az, ad)) = plt.subplots(2, 2, figsize=(12, 12))

                ax.imshow(img_resized)
                ay.imshow(img)
                az.imshow(img_resized)
                ad.imshow(img)

                ax.set_title("orig downscaled annotations")
                for i in range(len(annotations_resized)):
                    x, y, w, h, _, c, _, _ = annotations_resized[i]
                    rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')
                    ax.add_patch(rect)
                    label = visDrone.categories.get(c)
                    # ax.text(x, y - 20, label, color='r', fontsize=10, bbox=dict(facecolor='white', alpha=0.5))

                ay.set_title("orig annotations")
                for i in range(len(annotations)):
                    x, y, w, h, _, c, _, _ = annotations[i]
                    rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')
                    ay.add_patch(rect)
                    label = visDrone.categories.get(c)
                    # ax.text(x, y - 20, label, color='r', fontsize=10, bbox=dict(facecolor='white', alpha=0.5))

                az.set_title("uncooked downscaled annotations")
                for i in range(len(uncooked_resized_annotations)):
                    x, y, w, h, _, c, _, _ = uncooked_resized_annotations[i]
                    rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')
                    az.add_patch(rect)
                    label = visDrone.categories.get(c)
                    # ax.text(x, y + 5, label, color='r', fontsize=10, bbox=dict(facecolor='white', alpha=0.5))

                ad.set_title("uncooked orig annotations")
                for i in range(len(uncooked_orig_annotations)):
                    x, y, w, h, _, c, _, _ = uncooked_orig_annotations[i]
                    rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')
                    ad.add_patch(rect)
                    label = visDrone.categories.get(c)
                    # ax.text(x, y + 5, label, color='r', fontsize=10, bbox=dict(facecolor='white', alpha=0.5))

                plt.axis('off')
                plt.show()

        # show model output
        for file_name, img_path in img_files.items():
            img, img_resized = self._prepare_single_image(img_path)
            images = img_resized.reshape(1, *img_resized.shape)

            if print_shrunk_image or print_orig_image:
                if on_cpu:
                    boxes_data = self.predict(images, iou_threshold, score_threshold)
                else:
                    boxes_data = cp.asnumpy(self.predict(cp.array(images), iou_threshold, score_threshold))

                _, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

                if print_shrunk_image:
                    ax1.imshow(img_resized)
                    ax1.set_title("Resized Image")

                    for i in range(len(boxes_data)):
                        boxes_per_image = boxes_data[i]
                        for box in boxes_per_image:
                            x, y, w, h, obj_class, score = box
                            rect = patches.Rectangle((x, y + h), w, h, linewidth=1, edgecolor='r', facecolor='none')
                            ax1.add_patch(rect)
                            ax1.text(x, y + 5, f"{visDrone.categories.get(int(obj_class))} {score:.2f}", color='r', fontsize=5, bbox=dict(facecolor='white', alpha=0.5))

                if print_orig_image:
                    ax2.imshow(img)
                    ax2.set_title("Original Image")
                    boxes_data = _upscale_predicted_boxes(boxes_data, img.shape[0] // img_resized.shape[0], img.shape[1] // img_resized.shape[1])

                    for i in range(len(boxes_data)):
                        boxes_per_image = boxes_data[i]
                        for box in boxes_per_image:
                            x, y, w, h, obj_class, score = box
                            rect = patches.Rectangle((x, y + h), w, h, linewidth=1, edgecolor='b', facecolor='none')
                            ax2.add_patch(rect)
                            ax2.text(x, y + 5, f"{visDrone.categories.get(int(obj_class))} {score:.2f}", color='b', fontsize=5, bbox=dict(facecolor='white', alpha=0.5))

                plt.tight_layout()

                plt.axis('off')
                plt.show()
                return

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
            annotations_list[i] = np.array(annotations).astype(np.float32)
            i += 1

        self.data_idx += self.mini_batch_size

        return np.array(I).astype(np.float32), IR, annotations_list

    def load_data_without_orig_image(self, entries=None):
        """
        Load the data from the data folder

        Returns:
            tuple: tuple containing the images and the annotations.
                    Images are of shape np.shape(m, height, width, num_channels) and
                    annotations are of shape cp.shape(m, n, 8)
        """
        if entries is None:
            entries = os.listdir(self.data_folder)

        entries = entries[self.data_idx:self.data_idx + self.mini_batch_size]
        img_files = {}
        annotation_files = {}
        for entry in entries:
            file_name = entry.split('.')[0]
            img_files[file_name] = os.path.join(self.data_folder, entry)
            annotation_files[file_name] = os.path.join(self.annotation_folder, file_name + '.txt')

        resized_images = np.zeros((len(img_files), *self.model_input_img_res, 3))

        orig_annotations_list = [[] for _ in range(len(img_files))]
        resized_annotations_list = cp.zeros((len(img_files), self.model.S, self.model.S, self.model.B * 5 + self.model.C)).astype(np.float32)
        i = 0

        for file_name, img_path in img_files.items():
            img, img_resized = self._prepare_single_image(img_path)
            annotations_list = _extract_visDrone_annotations(annotation_files[file_name])

            resized_images[i] = img_resized

            annotations_list = np.array(annotations_list).astype(np.float32)

            orig_annotations_list[i] = np.array(annotations_list).astype(np.float32)
            resized_annotations_list[i] = cp.array(self._cook_annotations(orig_annotations_list[i], (img.shape[0], img.shape[1]), self.model_input_img_res))

            i += 1

        self.data_idx += self.mini_batch_size

        return resized_images, resized_annotations_list, orig_annotations_list

    def run_training_loop(self, start_image_idx=0, epochs=10, learning_rate=0.01, print_loss=False, num_of_loops=1):
        losses = []

        if num_of_loops <= 0:
            num_of_loops = 1

        for i in range(num_of_loops):
            print(f"Loop {i} from {num_of_loops}")

            if epochs == -1:
                epochs = int(np.ceil(len(os.listdir(self.data_folder)) / self.mini_batch_size))

            self.data_idx = 0

            if not hasattr(self, "default_bounding_box_offsets"):
                self._find_out_default_anchors()

            print("Starting training loop")
            print("number of epochs: ", epochs)

            start_epoch = start_image_idx // self.mini_batch_size
            for i in range(epochs):
                resized_images, resized_annotations, _ = self.load_data_without_orig_image()
                if i < start_epoch:
                    continue

                if resized_images.shape[0] == 0:
                    print("No more data to train on")
                    epochs = i
                    break

                # normalize image
                mean = cp.array([0.485, 0.456, 0.406])  # ImageNet mean
                std = cp.array([0.229, 0.224, 0.225])   # ImageNet std
                resized_images = (cp.array(resized_images) / 255.0 - mean) / std

                Y = self.model.forward(resized_images, self.default_bounding_box_offsets)

                loss = self.train_on_multiple_images(
                    Y=Y,
                    Y_hat=resized_annotations,
                    learning_rate=learning_rate
                )

                losses.append(loss)

                # if i % (epochs * 0.1) == 0:  # each 10%
                #     print(f"Epoch {i}, loss: {loss}")
                print(f"Epoch {i}, loss: {loss}")

        if print_loss:
            plt.plot(range(epochs * num_of_loops), losses, label="Loss")
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.title("Training Loss Over Epochs")
            plt.legend()
            plt.grid(True)
            plt.show()

        return losses[-1] if losses else None

    def debug(self):
        # img_path = self.data_folder + "/0000001_02999_d_0000005.jpg"

        # self._find_out_default_anchors()
        # img, img_downscaled = self._prepare_single_image(img_path)
        # grid_cell_size = (img_downscaled.shape[0] / self.model.S, img_downscaled.shape[1] / self.model.S)
        # _, ax1 = plt.subplots(1, 1, figsize=(12, 12))  # Ensure figsize is a tuple
        # ax1.imshow(img)
        # for i, anchor in enumerate(self.default_bounding_box_offsets):
        #     w = anchor[0] * grid_cell_size[1] * img.shape[1] // img_downscaled.shape[1]
        #     h = anchor[1] * grid_cell_size[0] * img.shape[0] // img_downscaled.shape[0]

        #     rect = patches.Rectangle((i * 10, i * 20), w, h, linewidth=1, edgecolor='r', facecolor='none')
        #     ax1.add_patch(rect)

        # plt.show()

        losses = []
        for i in range(10**1):
            print(i)
            loss = self.run_training_loop(epochs=-1, learning_rate=0.0025, start_image_idx=0, print_loss=False)
            losses.append(loss)

        norm_weights, activations = self.model.gather_debug_data()

        # Assuming `losses` is a list of arrays or a single array
        # plt.figure(figsize=(10, 6))
        # fig, axs = plt.subplots(1, len(norm_weights), figsize=(15, 5))

        # norm_weights
        # for idx, norm_weight in enumerate(norm_weights):
        #     norm_values, label = norm_weight
        #     plt.figure(idx + 1)
        #     plt.plot(norm_values, label=f"{label} weights norms")
        #     plt.title(f"Whiteboard {idx + 1}")
        #     plt.legend()
        #     plt.grid(True)

        # activations

        # print(activations)

        self.func_for_tests(
            print_shrunk_image=True,
            print_orig_image=True,
            # evaluate=True,
            iou_threshold=0.,
            score_threshold=0.,
        )

        # plt.xlabel("Epochs")
        # plt.ylabel("Loss")
        # plt.title("Loss Curves for All Runs")
        # plt.legend()
        # plt.grid(True)
        plt.show()

    def train_on_multiple_images(self,
                                 Y: np.ndarray,
                                 Y_hat: list,
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
        loss, grad_A = self._calc_loss_and_gradient(Y, cp.asarray(Y_hat))
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
        X_cells_offset = (X_center - cells_x * cell_width) / cell_width  # shape (m,)
        Y_cells_offset = (Y_center - cells_y * cell_height) / cell_height  # shape (m,)

        # create one-hot encoding vector for the class
        class_targets = np.zeros((m, self.model.C))
        categories = Y_norm[:, visDrone.category_idx].astype(np.int8)
        class_targets[np.arange(m), categories] = 1
        # smooth one-hot encoding vector
        class_targets = (1 - 1e-6) * class_targets + 1e-6 / self.model.C

        # construct output matrix - aka reshape annotations to the model output shape
        Y_target = np.zeros((self.model.S, self.model.S, self.model.B * 5 + self.model.C)).astype(np.float32)

        for i in range(m):
            cell_x = int(cells_x[i])
            cell_y = int(cells_y[i])

            # bounding box target
            bbox_target = np.array([
                X_cells_offset[i],
                Y_cells_offset[i],
                Y_norm[i, visDrone.width_idx] / cell_width,
                Y_norm[i, visDrone.height_idx] / cell_height,
                1,
            ])

            true_center_offset = np.array([X_cells_offset[i], Y_cells_offset[i]])
            distances = np.linalg.norm(self.default_bounding_box_offsets - true_center_offset, axis=1)

            best_bbox_slot = np.argmin(distances)
            start_index = best_bbox_slot * 5
            Y_target[cell_y, cell_x, start_index:start_index+5] = bbox_target
            Y_target[cell_y, cell_x, self.model.B * 5:] = class_targets[i]

        return Y_target

    def _uncook_annotations(self, Y_target: np.ndarray,
                            original_img_shape: tuple,
                            model_input_img_shape: tuple,
                            object_thresh: float = -1) -> np.ndarray:
        """
        Convert a YOLO output (of shape (S, S, B*5+C)) back to a list of annotations.

        Args:
            Y_target (np.ndarray): model output of shape (S, S, B*5+C).
            original_img_shape (tuple): (height, width) of the original image.
            model_input_img_shape (tuple): (height, width) of the model input image.
            S (int): grid size.
            B (int): number of bounding box predictors per cell.
            C (int): number of classes.
            object_thresh (float): threshold for deciding if a cell contains an object.

        Returns:
            np.ndarray: array of annotations of shape (N, 8), where each annotation is:
                        [top_left_x, top_left_y, width, height, object_flag, category, 0, 0]
                        (The last two entries are placeholders, as your original annotations had 8 numbers.)
        """
        # Calculate cell dimensions (model input scale)
        cell_height = model_input_img_shape[0] / self.model.S
        cell_width = model_input_img_shape[1] / self.model.S

        # Factors to convert from model input scale back to original image scale.
        # In _cook_annotations, original coordinates were scaled by dividing:
        #    X_norm = original_x / (original_img_width/model_input_img_width)
        # so here we reverse that.
        scale_x = original_img_shape[1] / model_input_img_shape[1]
        scale_y = original_img_shape[0] / model_input_img_shape[0]

        annotations_list = []

        # Loop over every grid cell
        for i in range(self.model.S):        # i is grid row (y-coordinate)
            for j in range(self.model.S):    # j is grid column (x-coordinate)
                cell_data = Y_target[i, j]  # shape (B*5 + C,)
                # Get class vector (same for all boxes in this cell)
                class_vector = cell_data[self.model.B*5:]
                category = int(np.argmax(class_vector))

                # For each bounding box predictor in this cell:
                for b in range(self.model.B):
                    start_idx = b * 5
                    # object_existence_idx is 4 as in your visDrone definitions
                    object_conf = cell_data[start_idx + 4]
                    if object_conf > object_thresh:
                        # Extract bounding box parameters (all in model input scale)
                        x_offset = cell_data[start_idx]   # relative offset within cell [0,1]
                        y_offset = cell_data[start_idx+1]   # relative offset within cell [0,1]
                        box_width = cell_data[start_idx+2]  # already normalized to model input scale
                        box_height = cell_data[start_idx+3]

                        # Compute center of the box in model input coordinates:
                        center_x = j * cell_width + x_offset * cell_width
                        center_y = i * cell_height + y_offset * cell_height

                        # Convert center and box dims to top-left corner (still in model input scale)
                        top_left_x_model = center_x - box_width / 2.0
                        top_left_y_model = center_y - box_height / 2.0

                        # Convert coordinates from model input scale back to original image scale:
                        top_left_x = top_left_x_model * scale_x
                        top_left_y = top_left_y_model * scale_y
                        orig_width = box_width * scale_x
                        orig_height = box_height * scale_y

                        # Build the annotation: here we assume annotation format is:
                        # [top_left_x, top_left_y, width, height, object_flag, category, 0, 0]
                        annotation = [top_left_x, top_left_y, orig_width, orig_height, 1, category, 0, 0]
                        annotations_list.append(annotation)

        return np.array(annotations_list)

    def _calc_loss_and_gradient(self, A: cp.ndarray, Y_target: cp.ndarray):
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
        grad_A = cp.zeros_like(A)

        eps = 1e-6

        # Constant for loss weighting
        lambda_coord = 10.0
        lambda_noobj = 5.0

        # focal loss hyperparameters:
        gamma_focal = 5.0
        alpha_focal = 0.25

        m, S, _, _ = A.shape
        B = self.model.B

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

                        # if there should not be any object
                        if target_bbox[visDrone.object_existence_idx] == 0:
                            pred_conf = pred_bbox[visDrone.object_existence_idx]

                            # focal loss
                            loss_conf = - (1 - alpha_focal) * (pred_conf ** gamma_focal) * cp.log(1 - pred_conf + 1e-6)
                            loss += lambda_noobj * loss_conf

                            # Compute gradient for no-object case (derivative from loss_conf function).
                            grad_conf = (1 - alpha_focal) * (
                                (pred_conf ** gamma_focal) / (1 - pred_conf + 1e-6)
                                - gamma_focal * (pred_conf ** (gamma_focal - 1)) * cp.log(1 - pred_conf + 1e-6)
                            )

                            # -------- quality focal loss --------

                            # loss_conf = - ((pred_conf ** gamma_focal) * cp.log(1 - pred_conf + eps))
                            # # Derivative with respect to p (pred_conf):
                            # grad_conf = - gamma_focal * (pred_conf ** (gamma_focal - 1)) * cp.log(1 - pred_conf + eps) \
                            #             - (pred_conf ** gamma_focal) / (1 - pred_conf + eps)
                            # loss += lambda_noobj * loss_conf

                            # -------- quality focal loss --------

                            grad_A[i, row, col, idx + visDrone.object_existence_idx] += lambda_noobj * grad_conf

                            continue

                        pred_box_coords = pred_bbox[0:4]  # [x, y, w, h]
                        target_box_coords = target_bbox[0:4]  # [x, y, w, h]

                        ciou_val, iou_val = _ciou(pred_box_coords, target_box_coords)

                        loss += lambda_coord * (1 - ciou_val)

                        grad_ciou = _ciou_gradient(pred_box_coords, target_box_coords)
                        grad_A[i, row, col, idx:idx+4] += lambda_coord * grad_ciou

                       
                       
                        # confidence loss. Quality focal loss for positives: y=1
                        
                        # common == focal loss. Advanced == quality focal loss
                        quality = ciou_val
                        pred_conf = pred_bbox[visDrone.object_existence_idx]

                        loss_common = - cp.log(pred_conf + eps) * ((1 - pred_conf) ** gamma_focal)
                        loss_advanced = - quality * (cp.abs(quality - pred_conf) ** gamma_focal) * cp.log(pred_conf + eps)

                        loss_conf = (1 - alpha_focal) * loss_common + alpha_focal * loss_advanced
                        
                        grad_common = alpha_focal * (
                            gamma_focal * ((1 - pred_conf) ** (gamma_focal - 1)) * cp.log(pred_conf + eps)
                            - ((1 - pred_conf) ** gamma_focal) / (pred_conf + eps)
                        )

                        grad_advanced = alpha_focal * (
                            alpha_focal * cp.sign(pred_conf - quality) * (cp.abs(quality - pred_conf) ** (gamma_focal - 1)) * cp.log(pred_conf + eps)
                            + (cp.abs(quality - pred_conf) ** gamma_focal) / (pred_conf + eps)
                        )
                        
                        # TODO: as far as I can see, the more weights of common grad, the more confident moodel is about prediction. I suppose it's not totally right
                        common_weight = (1 - alpha_focal) * 0.5 
                        advanced_weight = 1 - common_weight
                        
                        grad_conf = common_weight * grad_common + advanced_weight * grad_advanced

                        loss += loss_conf
                        grad_A[i, row, col, idx + visDrone.object_existence_idx] += grad_conf

                        # classification loss
                        pred_class = prediction[B * 5:]  # [C] array
                        target_class = target[B * 5:]  # [C] array
                        loss += -cp.sum(target_class * cp.log(pred_class + eps))
                        grad_class = pred_class - target_class  # Derivative of cross-entropy with softmax
                        grad_A[i, row, col, B * 5:] += grad_class

        # max_grad_norm = 1.0
        # grad_A = cp.clip(grad_A, -max_grad_norm, max_grad_norm)

        mean_loss = loss / (m * S * S)
        return mean_loss.item(), grad_A

    def predict(self, images: cp.ndarray, iou_threshold: float = 0.5, score_threshold: float = 0.5):
        """
        Predict the bounding boxes for the images

        Args:
            images (np.ndarray): images to predict the bounding boxes for - matrix of shape (m, height, width, num_channels), represents a batch of m images

        Returns:
            boxes(np.ndarray): predicted bounding boxes - matrix of shape (m, S * S * B, 6), represents a batch of m images. Each box is [x, y, w, h, class, score]
        """
        model_output = self.model.forward(images, self.default_bounding_box_offsets)

        m = model_output.shape[0]
        B = self.model.B
        S = self.model.S

        cell_width = self.model_input_img_res[0] / S
        cell_height = self.model_input_img_res[1] / S

        boxes = cp.zeros((m, S, S, B, 6))

        for i in range(m):
            for row in range(S):
                for col in range(S):
                    for b in range(B):
                        tx = model_output[i, row, col, b * 5 + 0]
                        ty = model_output[i, row, col, b * 5 + 1]
                        tw = model_output[i, row, col, b * 5 + 2]
                        th = model_output[i, row, col, b * 5 + 3]
                        confidence = model_output[i, row, col, b * 5 + 4]

                        # constrain offsets to [0, 1] within the grid cell
                        x = sigmoid(tx)
                        y = sigmoid(ty)

                        x_center_abs = (col + x) * cell_width
                        y_center_abs = (row + y) * cell_height

                        # already scaled by anchor in feed_forward
                        w_abs = tw * cell_width
                        h_abs = th * cell_height

                        x1 = x_center_abs - w_abs / 2
                        y1 = y_center_abs - h_abs / 2

                        class_probs = model_output[i, row, col, B*5:]
                        class_id = cp.argmax(class_probs)
                        score = cp.max(class_probs) * confidence

                        boxes[i, row, col, b] = cp.array([x1, y1, w_abs, h_abs, class_id, score])

        predicted_boxes = boxes.reshape(m, -1, boxes.shape[-1])

        if on_cpu:
            predicted_boxes = _non_max_suppression(predicted_boxes, iou_threshold=iou_threshold, score_threshold=score_threshold)
        else:
            predicted_boxes = _non_max_suppression(cp.asnumpy(predicted_boxes), iou_threshold=iou_threshold, score_threshold=score_threshold)

        return predicted_boxes

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
        boxes_pred = self.predict(cp.array(images), iou_threshold, score_threshold)

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

    def _find_out_default_anchors(self):
        print("Finding out default anchors")
        entries = os.listdir(self.data_folder)
        boxes = []

        for i, image_file_name in enumerate(entries):
            img, img_downscaled = self._prepare_single_image(os.path.join(self.data_folder, image_file_name))
            annotations = _extract_visDrone_annotations(os.path.join(self.annotation_folder, image_file_name.split('.')[0]) + ".txt")
            annotations_downscaled = _downscale_annotation(annotations, img.shape[0] / img_downscaled.shape[0], img.shape[1] / img_downscaled.shape[1])

            grid_cell_size = (img_downscaled.shape[0] / self.model.S, img_downscaled.shape[1] / self.model.S)

            for annotation in annotations_downscaled:
                boxes.append([annotation[visDrone.width_idx] / grid_cell_size[1],
                              annotation[visDrone.height_idx] / grid_cell_size[0]])

        res = _kmeans(boxes=np.array(boxes), k=self.model.B)
        self.default_bounding_box_offsets = res

    def _test_gradient(self):
        """
        Test the gradient
        """
        self._find_out_default_anchors()
        _, resized_annotations, _ = self.load_data_without_orig_image()

        input_array = cp.random.randn(1, 88, 88, 3)

        Y = self.model.forward(input_array, self.default_bounding_box_offsets)
        _, grad = self._calc_loss_and_gradient(Y, cp.asarray(resized_annotations))
        dW = self.model.backward(grad, 0.)

        epsilon = 1e-5
        for i in range(3):
            for j in range(3):
                weight = self.model.conv_blocks[0].conv1.filters[i, j, 0, 0]
                orig_weight = weight.copy()

                self.model.conv_blocks[0].conv1.filters[i, j, 0, 0] = orig_weight + epsilon
                Y_plus = self.model.forward(input_array, self.default_bounding_box_offsets)
                loss_plus, _ = self._calc_loss_and_gradient(Y_plus, cp.asarray(resized_annotations))

                self.model.conv_blocks[0].conv1.filters[i, j, 0, 0] = orig_weight - epsilon
                Y_minus = self.model.forward(input_array, self.default_bounding_box_offsets)
                loss_minus, _ = self._calc_loss_and_gradient(Y_minus, cp.asarray(resized_annotations))

                numerical_grad = (loss_plus - loss_minus) / (2 * epsilon)

                analytical_grad = dW[i, j, 0, 0]
                difference = abs(analytical_grad - numerical_grad) / max(1, abs(analytical_grad), abs(numerical_grad))
                print(f"i: {i}, j:{j}, Relative difference:", difference)

                self.model.conv_blocks[0].conv1.filters[i, j, 0, 0] = orig_weight


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


def _upscale_predicted_boxes(boxes: np.ndarray, factor_H: int, factor_W: int):
    """
    Downscale the annotation to the new shape

    Args:
        boxes(np.ndarray): predicted bounding boxes - matrix of shape (m, S * S * B, 6), represents a batch of m images. Each box is [x, y, w, h, class, score]

    Returns:
        upscaled_boxes(np.ndarray): predicted bounding boxes - matrix of shape (m, S * S * B, 6), represents a batch of m images. Each box is [x, y, w, h, class, score]
    """

    upscaled_boxes = cp.copy(boxes)
    upscaled_boxes[:, :, 0] *= factor_W
    upscaled_boxes[:, :, 1] *= factor_H
    upscaled_boxes[:, :, 2] *= factor_W
    upscaled_boxes[:, :, 3] *= factor_H

    return upscaled_boxes


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
        annotation_downscaled[i][visDrone.top_left_x_idx] = annotation_downscaled[i][visDrone.top_left_x_idx] / factor_W
        annotation_downscaled[i][visDrone.top_left_y_idx] = annotation_downscaled[i][visDrone.top_left_y_idx] / factor_H
        annotation_downscaled[i][visDrone.width_idx] = annotation_downscaled[i][visDrone.width_idx] / factor_W
        annotation_downscaled[i][visDrone.height_idx] = annotation_downscaled[i][visDrone.height_idx] / factor_H

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
        values = list(map(int, ''.join(c for c in line.strip().rstrip(',') if c.isdigit() or c == ',').split(',')))
        if values[visDrone.object_existence_idx] == 0 or values[visDrone.category_idx] > 10:
            continue

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


def _ciou(box1: cp.ndarray, box2: cp.ndarray, eps: float = 1e-7):
    """
    Compute the Complete IoU (CIoU) between a predicted box and a ground truth box.

    Args:
        pred_box (cp.ndarray): Array of shape (4,) representing [x, y, w, h] for the predicted box.
        gt_box (cp.ndarray): Array of shape (4,) representing [x, y, w, h] for the ground truth box.
        eps (float): A small value to avoid division by zero.

    Returns:
        cp.ndarray: The CIoU value.
    """
    x_idx = 0
    y_idx = 1
    w_idx = 2
    h_idx = 3

    # Convert center coordinates to (x1, y1, x2, y2)
    box1_x1 = box1[x_idx] - box1[w_idx] / 2
    box1_y1 = box1[y_idx] - box1[h_idx] / 2
    box1_x2 = box1[x_idx] + box1[w_idx] / 2
    box1_y2 = box1[y_idx] + box1[h_idx] / 2

    box2_x1 = box2[x_idx] - box2[w_idx] / 2
    box2_y1 = box2[y_idx] - box2[h_idx] / 2
    box2_x2 = box2[x_idx] + box2[w_idx] / 2
    box2_y2 = box2[y_idx] + box2[h_idx] / 2

    # Intersection rectangle
    inter_x1 = cp.maximum(box1_x1, box2_x1)
    inter_y1 = cp.maximum(box1_y1, box2_y1)
    inter_x2 = cp.minimum(box1_x2, box2_x2)
    inter_y2 = cp.minimum(box1_y2, box2_y2)

    inter_w = cp.maximum(inter_x2 - inter_x1, 0)
    inter_h = cp.maximum(inter_y2 - inter_y1, 0)
    inter_area = inter_w * inter_h

    # Areas of boxes
    area_pred = box1[w_idx] * box1[h_idx]
    area_gt = box2[w_idx] * box2[h_idx]
    union_area = area_pred + area_gt - inter_area + eps

    iou = inter_area / union_area

    # Compute the squared distance between the centers
    center_distance = (box1[x_idx] - box2[x_idx]) ** 2 + (box1[y_idx] - box2[y_idx]) ** 2

    # Compute the diagonal length squared of the smallest enclosing box
    enclose_x1 = cp.minimum(box1_x1, box2_x1)
    enclose_y1 = cp.minimum(box1_y1, box2_y1)
    enclose_x2 = cp.maximum(box1_x2, box2_x2)
    enclose_y2 = cp.maximum(box1_y2, box2_y2)
    enclose_diagonal = (enclose_x2 - enclose_x1) ** 2 + (enclose_y2 - enclose_y1) ** 2 + eps

    # Aspect ratio consistency term
    # v measures the similarity of the aspect ratios
    v = (4 / (cp.pi ** 2)) * (cp.arctan(box2[w_idx] / (box2[h_idx] + eps)) - cp.arctan(box1[w_idx] / (box1[h_idx] + eps))) ** 2
    alpha = v / (1 - iou + v + eps)

    ciou = iou - (center_distance / enclose_diagonal + alpha * v)

    return ciou, iou


def _ciou_gradient(pred_box: cp.ndarray, gt_box: cp.ndarray, eps: float = 1e-6) -> cp.ndarray:
    """
    Approximates the gradient of the CIoU

    Args:
        pred_box (cp.ndarray): Predicted box [x, y, w, h]
        gt_box (cp.ndarray): Ground truth box [x, y, w, h]
        eps (float): epsilon

    Returns:
        cp.ndarray: gradient of the CIoU with respect to the predicted box
    """
    grad = cp.zeros_like(pred_box)

    # For loss L = 1 - CIoU, the gradient is -d(CIoU)/d(pred_box).
    for i in range(len(pred_box)):
        delta = cp.zeros_like(pred_box)
        delta[i] = eps

        ciou_plus, _ = _ciou(pred_box + delta, gt_box)
        ciou_minus, _ = _ciou(pred_box - delta, gt_box)

        # Central difference approximation
        dciou_dp = (ciou_plus - ciou_minus) / (2 * eps)
        grad[i] = -dciou_dp  # negative sign because L = 1 - CIoU

    return grad


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


def _kmeans_iou(box, clusters):
    """
    Calculate the Intersection over Union (IoU) between a box and k clusters.
    box: tuple or array, (width, height)
    clusters: numpy array of shape (k, 2)
    """
    x = np.minimum(clusters[:, 0], box[0])
    y = np.minimum(clusters[:, 1], box[1])

    intersection = x * y

    box_area = box[0] * box[1]
    clusters_area = clusters[:, 0] * clusters[:, 1]

    res = intersection / (box_area + clusters_area - intersection)
    return res


def _kmeans(boxes, k, dist=np.median, max_iter=300):
    """
    Runs k-means clustering with an IoU-based distance metric.

    boxes: numpy array of shape (n, 2) where n is the number of boxes and each box is (width, height)
    k: number of clusters (anchors)
    dist: function to calculate the cluster center, here median is used
    max_iter: maximum iterations to run
    """
    num_boxes = boxes.shape[0]

    # initialize clusters by randomly choosing k boxes from the dataset.
    clusters = boxes[np.random.choice(num_boxes, k, replace=False)]

    last_clusters = np.zeros((num_boxes,))

    for iteration in range(max_iter):
        distances = np.zeros((num_boxes, k))

        for i in range(num_boxes):
            distances[i] = 1 - _kmeans_iou(boxes[i], clusters)

        current_clusters = np.argmin(distances, axis=1)
        if (last_clusters == current_clusters).all():
            break

        for cluster in range(k):
            if np.sum(current_clusters == cluster) == 0:
                continue

            clusters[cluster] = dist(boxes[current_clusters == cluster], axis=0)

        last_clusters = current_clusters

    return clusters
