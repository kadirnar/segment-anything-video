import cv2
import numpy as np
import torch

from metaseg import SamPredictor, sam_model_registry
from metaseg.utils import download_model, load_image


class SegManualMaskGenerator:
    def __init__(self):
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load_model(self, model_type):
        if self.model is None:
            self.model_path = download_model(model_type)
            self.model = sam_model_registry[model_type](checkpoint=self.model_path)
            self.model.to(device=self.device)

        return self.model

    def load_mask(self, mask, random_color):
        if random_color:
            color = np.random.rand(3) * 255
        else:
            color = np.array([255, 200, 0])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        mask_image = mask_image.astype(np.uint8)
        return mask_image

    def load_box(self, box, image):
        x0, y0 = box[0], box[1]
        x1, y1 = box[2], box[3]
        cv2.rectangle(image, (x0, y0), (x1, y1), (0, 255, 0), 2)
        return image

    def predict(self, frame, model_type, x0, y0, x1, y1):
        model = self.load_model(model_type)
        predictor = SamPredictor(model)
        predictor.set_image(frame)
        input_box = np.array([x0, y0, x1, y1])
        masks, _, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_box[None, :],
            multimask_output=False,
        )

        return frame, masks, input_box

    def save_image(self, source, model_type, x0, y0, x1, y1):
        read_image = load_image(source)
        image, anns, input_box = self.predict(read_image, model_type, x0, y0, x1, y1)
        if len(anns) == 0:
            return

        mask_image = self.load_mask(anns, True)
        image = self.load_box(input_box, image)
        combined_mask = cv2.add(image, mask_image)
        cv2.imwrite("output.jpg", combined_mask)

        return "output.jpg"


if __name__ == "__main__":
    model_type = "vit_l"
    seg_manual_mask_generator = SegManualMaskGenerator()
    seg_manual_mask_generator.save_image(
        source="data.jpg",
        model_type="vit_l",
        x0=100,
        y0=100,
        x1=200,
        y1=200,
    )
