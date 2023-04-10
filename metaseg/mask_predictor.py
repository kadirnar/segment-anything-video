from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import torch
from tqdm import tqdm

from metaseg import SamAutomaticMaskGenerator, SamPredictor, sam_model_registry
from metaseg.utils import download_model, load_image, load_video


class SegAutoMaskPredictor:
    def __init__(self):
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load_model(self, model_type):
        if self.model is None:
            self.model_path = download_model(model_type)
            self.model = sam_model_registry[model_type](checkpoint=self.model_path)
            self.model.to(device=self.device)

        return self.model

    def predict(self, frame, model_type, points_per_side, points_per_batch, min_area):
        model = self.load_model(model_type)
        mask_generator = SamAutomaticMaskGenerator(
            model, points_per_side=points_per_side, points_per_batch=points_per_batch, min_mask_region_area=min_area
        )

        masks = mask_generator.generate(frame)

        return frame, masks

    def save_image(self, source, model_type, points_per_side, points_per_batch, min_area):
        read_image = load_image(source)
        image, anns = self.predict(read_image, model_type, points_per_side, points_per_batch, min_area)
        if len(anns) == 0:
            return

        sorted_anns = sorted(anns, key=(lambda x: x["area"]), reverse=True)
        mask_image = np.zeros((anns[0]["segmentation"].shape[0], anns[0]["segmentation"].shape[1], 3), dtype=np.uint8)
        colors = np.random.randint(0, 255, size=(256, 3), dtype=np.uint8)
        for i, ann in enumerate(sorted_anns):
            m = ann["segmentation"]
            img = np.ones((m.shape[0], m.shape[1], 3), dtype=np.uint8)
            color = colors[i % 256]
            for i in range(3):
                img[:, :, 0] = color[0]
                img[:, :, 1] = color[1]
                img[:, :, 2] = color[2]
            img = cv2.bitwise_and(img, img, mask=m.astype(np.uint8))
            img = cv2.addWeighted(img, 0.35, np.zeros_like(img), 0.65, 0)
            mask_image = cv2.add(mask_image, img)

        combined_mask = cv2.add(image, mask_image)
        cv2.imwrite("output.jpg", combined_mask)

        return "output.jpg"

    def save_video(self, source, model_type, points_per_side, points_per_batch, min_area):
        cap, out = load_video(source)
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        colors = np.random.randint(0, 255, size=(256, 3), dtype=np.uint8)

        for _ in tqdm(range(length)):
            ret, frame = cap.read()
            if not ret:
                break

            image, anns = self.predict(frame, model_type, points_per_side, points_per_batch, min_area)
            if len(anns) == 0:
                continue

            sorted_anns = sorted(anns, key=(lambda x: x["area"]), reverse=True)
            mask_image = np.zeros(
                (anns[0]["segmentation"].shape[0], anns[0]["segmentation"].shape[1], 3), dtype=np.uint8
            )

            for i, ann in enumerate(sorted_anns):
                m = ann["segmentation"]
                color = colors[i % 256]  # Her nesne için farklı bir renk kullan
                img = np.zeros((m.shape[0], m.shape[1], 3), dtype=np.uint8)
                img[:, :, 0] = color[0]
                img[:, :, 1] = color[1]
                img[:, :, 2] = color[2]
                img = cv2.bitwise_and(img, img, mask=m.astype(np.uint8))
                img = cv2.addWeighted(img, 0.35, np.zeros_like(img), 0.65, 0)
                mask_image = cv2.add(mask_image, img)

            combined_mask = cv2.add(frame, mask_image)
            out.write(combined_mask)

        out.release()
        cap.release()
        cv2.destroyAllWindows()

        return "output.mp4"


class SegManualMaskPredictor:
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
            color = np.array([100, 50, 0])

        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        mask_image = mask_image.astype(np.uint8)
        return mask_image

    def load_box(self, box, image):
        x, y, w, h = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        cv2.rectangle(image, (x, y), (w, h), (0, 255, 0), 2)
        return image

    def multi_boxes(self, boxes, predictor, image):
        input_boxes = torch.tensor(boxes, device=predictor.device)
        transformed_boxes = predictor.transform.apply_boxes_torch(input_boxes, image.shape[:2])
        return input_boxes, transformed_boxes

    def predict(
        self,
        frame,
        model_type,
        input_box=None,
        input_point=None,
        input_label=None,
        multimask_output=False,
    ):
        model = self.load_model(model_type)
        predictor = SamPredictor(model)
        predictor.set_image(frame)

        if type(input_box[0]) == list:
            input_boxes, new_boxes = self.multi_boxes(input_box, predictor, frame)

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

        return frame, masks, input_boxes

    def save_image(
        self,
        source,
        model_type,
        input_box=None,
        input_point=None,
        input_label=None,
        multimask_output=False,
        output_path="v0.jpg",
    ):
        read_image = load_image(source)
        image, anns, boxes = self.predict(read_image, model_type, input_box, input_point, input_label, multimask_output)
        if len(anns) == 0:
            return

        if type(input_box[0]) == list:
            for mask in anns:
                mask_image = self.load_mask(mask.cpu().numpy(), False)

            for box in boxes:
                image = self.load_box(box.cpu().numpy(), image)

        elif type(input_box[0]) == int:
            mask_image = self.load_mask(anns, True)
            image = self.load_box(input_box, image)

        combined_mask = cv2.add(image, mask_image)
        cv2.imwrite(output_path, combined_mask)

        return output_path
