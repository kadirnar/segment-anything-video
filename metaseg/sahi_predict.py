import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from metaseg import SamPredictor, sam_model_registry
from metaseg.utils import download_model, load_image, multi_boxes, plt_load_box, plt_load_mask


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
    def __init__(self):
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load_model(self, model_type):
        if self.model is None:
            self.model_path = download_model(model_type)
            self.model = sam_model_registry[model_type](checkpoint=self.model_path)
            self.model.to(device=self.device)

        return self.model

    def predict(
        self,
        source,
        model_type,
        input_box=None,
        input_point=None,
        input_label=None,
        multimask_output=False,
        random_color=False,
        show=False,
        save=False,
    ):

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
