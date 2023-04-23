from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import torch
from tqdm import tqdm

from metaseg import SamAutomaticMaskGenerator, SamPredictor, sam_model_registry
from metaseg.utils import (
    download_model,
    load_box,
    load_image,
    load_mask,
    load_video,
    multi_boxes,
    save_image,
    show_image,
)


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

    def image_predict(
        self,
        source,
        model_type,
        points_per_side,
        points_per_batch,
        min_area,
        output_path="output.png",
        show=False,
        save=False,
    ):
        read_image = load_image(source)
        model = self.load_model(model_type)
        mask_generator = SamAutomaticMaskGenerator(
            model, points_per_side=points_per_side, points_per_batch=points_per_batch, min_mask_region_area=min_area
        )

        masks = mask_generator.generate(read_image)

        sorted_anns = sorted(masks, key=(lambda x: x["area"]), reverse=True)
        mask_image = np.zeros((masks[0]["segmentation"].shape[0], masks[0]["segmentation"].shape[1], 3), dtype=np.uint8)
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

        combined_mask = cv2.add(read_image, mask_image)
        self.combined_mask = combined_mask
        if show:
            show_image(combined_mask)

        if save:
            save_image(output_path=output_path, output_image=combined_mask)

        return masks

    def video_predict(self, source, model_type, points_per_side, points_per_batch, min_area, output_path="output.mp4"):
        cap, out = load_video(source, output_path)
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        colors = np.random.randint(0, 255, size=(256, 3), dtype=np.uint8)

        for _ in tqdm(range(length)):
            ret, frame = cap.read()
            if not ret:
                break

            model = self.load_model(model_type)
            mask_generator = SamAutomaticMaskGenerator(
                model, points_per_side=points_per_side, points_per_batch=points_per_batch, min_mask_region_area=min_area
            )
            masks = mask_generator.generate(frame)

            if len(masks) == 0:
                continue

            sorted_anns = sorted(masks, key=(lambda x: x["area"]), reverse=True)
            mask_image = np.zeros(
                (masks[0]["segmentation"].shape[0], masks[0]["segmentation"].shape[1], 3), dtype=np.uint8
            )

            for i, ann in enumerate(sorted_anns):
                m = ann["segmentation"]
                color = colors[i % 256]
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

        return output_path


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

    def image_predict(
        self,
        source,
        model_type,
        input_box=None,
        input_point=None,
        input_label=None,
        multimask_output=False,
        output_path="output.png",
        random_color=False,
        show=False,
        save=False,
    ):
        image = load_image(source)
        model = self.load_model(model_type)
        predictor = SamPredictor(model)
        predictor.set_image(image)

        if type(input_box[0]) == list:
            input_boxes, new_boxes = multi_boxes(input_box, predictor, image)

            masks, _, _ = predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=new_boxes,
                multimask_output=False,
            )
            for mask in masks:
                mask_image = load_mask(mask.cpu().numpy(), random_color)

            for box in input_boxes:
                image = load_box(box.cpu().numpy(), image)

        elif type(input_box[0]) == int:
            input_boxes = np.array(input_box)[None, :]

            masks, _, _ = predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                box=input_boxes,
                multimask_output=multimask_output,
            )
            mask_image = load_mask(masks, random_color)
            image = load_box(input_box, image)

        combined_mask = cv2.add(image, mask_image)
        if save:
            save_image(output_path=output_path, output_image=combined_mask)

        if show:
            show_image(combined_mask)

        return masks

    def video_predict(
        self,
        source,
        model_type,
        input_box=None,
        input_point=None,
        input_label=None,
        multimask_output=False,
        output_path="output.mp4",
        random_color=False,
    ):
        cap, out = load_video(source, output_path)
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        for _ in tqdm(range(length)):
            ret, frame = cap.read()
            if not ret:
                break

            model = self.load_model(model_type)
            predictor = SamPredictor(model)
            predictor.set_image(frame)

            if type(input_box[0]) == list:
                input_boxes, new_boxes = multi_boxes(input_box, predictor, frame)

                masks, _, _ = predictor.predict_torch(
                    point_coords=None,
                    point_labels=None,
                    boxes=new_boxes,
                    multimask_output=False,
                )
                for mask in masks:
                    mask_image = load_mask(mask.cpu().numpy(), random_color)

                for box in input_boxes:
                    frame = load_box(box.cpu().numpy(), frame)

            elif type(input_box[0]) == int:
                input_boxes = np.array(input_box)[None, :]

                masks, _, _ = predictor.predict(
                    point_coords=input_point,
                    point_labels=input_label,
                    box=input_boxes,
                    multimask_output=multimask_output,
                )
                mask_image = load_mask(masks, random_color)
                frame = load_box(input_box, frame)

            combined_mask = cv2.add(frame, mask_image)
            out.write(combined_mask)

        out.release()
        cap.release()
        cv2.destroyAllWindows()
        return output_path
