from typing import Optional

import cv2
import numpy as np
import torch

from metaseg import SamAutomaticMaskGenerator, sam_model_registry
from metaseg.utils.file import download_model


class SegAutoMaskGenerator:
    def __init__(self):
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load_model(self, model_type):
        if self.model is None:
            model_path = download_model(model_type)
            model = sam_model_registry[self.model_type](checkpoint=model_path)
            model.to(device=self.device)
            self.model = model

        return self.model

    def load_image(self, image_path):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def load_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        out = cv2.VideoWriter("output.mp4", fourcc, fps, (frame_width, frame_height))

        return cap, out

    def predict(self, frame, model_type, points_per_side, points_per_batch):
        model = self.load_model(model_type)
        mask_generator = SamAutomaticMaskGenerator(
            model, points_per_side=points_per_side, points_per_batch=points_per_batch
        )
        masks = mask_generator.generate(frame)

        return frame, masks

    def save_image(self, source, model_type, points_per_side, points_per_batch):
        read_image = self.load_image(source)
        image, anns = self.predict(read_image, model_type, points_per_side, points_per_batch)
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

    def save_video(self, source, model_type, points_per_side, points_per_batch):
        cap, out = self.load_video()
        colors = np.random.randint(0, 255, size=(256, 3), dtype=np.uint8)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            image, anns = self.predict(frame)
            if len(anns) == 0:
                continue

            sorted_anns = sorted(anns, key=(lambda x: x["area"]), reverse=True)
            mask_image = np.zeros(
                (anns[0]["segmentation"].shape[0], anns[0]["segmentation"].shape[1], 3), dtype=np.uint8
            )

            for i, ann in enumerate(sorted_anns):
                if ann["area"] > 5000:
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
