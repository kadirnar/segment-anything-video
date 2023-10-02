"""Copyright (c) Metaseg Contributors.

All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from cv2 import Mat
from PIL import Image

from metaseg.generator import SAM_MODEL_REGISTRY, SamPredictor
from metaseg.utils import (
    download_model,
    load_image,
    multi_boxes,
    plt_load_box,
    plt_load_mask,
)


def sahi_sliced_predict(
    image_path,
    detection_model_type,
    detection_model_path,
    conf_th,
    image_size,
    slice_height,
    slice_width,
    overlap_height_ratio,
    overlap_width_ratio,
):
    """Predicts object detection results for a large image by slicing.

    Args:
        image_path (str): The path to the input image file.
        detection_model_type (str): The type of object detection model to use.
        detection_model_path (str): The path to the object detection model file.
        conf_th (float): The confidence threshold for object detection.
        image_size (int): The size of the input image.
        slice_height (int): The height of each slice.
        slice_width (int): The width of each slice.
        overlap_height_ratio (float): The overlap ratio for the height dimension.
        overlap_width_ratio (float): The overlap ratio for the width dimension.

    Returns:
        List[Dict[str, Union[str, np.ndarray]]]: A list of dictionaries representing
            the object detection results for each slice.
    """
    try:
        from sahi import AutoDetectionModel
        from sahi.predict import get_prediction, get_sliced_prediction
    except ImportError:
        raise ImportError("Please install SAHI library using 'pip install sahi'.")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    detection_model = AutoDetectionModel.from_pretrained(
        image_size=image_size,
        model_type=detection_model_type,
        model_path=detection_model_path,
        confidence_threshold=conf_th,
        device=device,
    )
    result = get_sliced_prediction(
        image_path,
        detection_model,
        slice_height=slice_height,
        slice_width=slice_width,
        overlap_height_ratio=overlap_height_ratio,
        overlap_width_ratio=overlap_width_ratio,
    )

    result = get_prediction(image_path, detection_model)
    output = result.object_prediction_list
    boxes = []
    for i in output:
        boxes.append(i.bbox.to_xyxy())

    return boxes


class SahiAutoSegmentation:
    """A class for performing automatic segmentation using SAHI library."""

    def __init__(self):
        """Initializes the segmentation model."""
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load_model(self, model_type):
        """Loads the segmentation model."""
        if self.model is None:
            self.model_path = download_model(model_type)
            self.model = SAM_MODEL_REGISTRY[model_type](checkpoint=self.model_path)
            self.model.to(device=self.device)

        return self.model

    def image_predict(
        self,
        source: str | Mat,
        model_type,
        input_box=None,
        input_point=None,
        input_label=None,
        multimask_output=False,
        random_color=False,
        show=False,
        save=False,
    ):
        """Performs automatic segmentation on an image."""
        read_image = load_image(source)
        model = self.load_model(model_type)
        predictor = SamPredictor(model)
        predictor.set_image(read_image)

        if type(input_box[0]) == list:
            input_boxes, new_boxes = multi_boxes(input_box, predictor, read_image)

            masks, _, _ = predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=new_boxes,
                multimask_output=False,
            )

        elif type(input_box[0]) == int:
            input_boxes = np.array(input_box)[None, :]

            masks, _, _ = predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                box=input_boxes,
                multimask_output=multimask_output,
            )

        plt.figure(figsize=(10, 10))
        plt.imshow(read_image)
        for mask in masks:
            plt_load_mask(mask.cpu().numpy(), plt.gca(), random_color=random_color)
        for box in input_boxes:
            plt_load_box(box.cpu().numpy(), plt.gca())
        plt.axis("off")
        if save:
            plt.savefig("output.png", bbox_inches="tight")
            output_image = cv2.imread("output.png")
            output_image = Image.fromarray(output_image)
            return output_image
        if show:
            plt.show()

        return masks
