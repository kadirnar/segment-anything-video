import cv2
import matplotlib.pyplot as plt
import numpy as np

from metaseg import SamAutomaticMaskGenerator, sam_model_registry


class SegAutoMaskGenerator:
    def __init__(self, model_type, checkpoint_path, image_path, device, show_mask=True):
        self.model_type = model_type
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.image_path = image_path

        if show_mask:
            self.show_mask()

        if self.model is None:
            self.model = self.load_model()

    def load_image(self):
        image = cv2.imread(self.image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def load_model(self):
        model = sam_model_registry[self.model_type](checkpoint=self.checkpoint_path)
        model.to(device=self.device)
        return model

    def show_anns(self, anns):
        if len(anns) == 0:
            return
        sorted_anns = sorted(anns, key=(lambda x: x["area"]), reverse=True)
        ax = plt.gca()
        ax.set_autoscale_on(False)
        polygons = []
        color = []
        for ann in sorted_anns:
            m = ann["segmentation"]
            img = np.ones((m.shape[0], m.shape[1], 3))
            color_mask = np.random.random((1, 3)).tolist()[0]
            for i in range(3):
                img[:, :, i] = color_mask[i]
            ax.imshow(np.dstack((img, m * 0.35)))

    def predict(self):
        model = self.load_model()
        image = self.load_image()
        mask_generator = SamAutomaticMaskGenerator(model)
        masks = mask_generator.generate(image)

        return image, masks

    def show_mask(self):
        image, masks = self.predict()
        plt.figure(figsize=(20, 20))
        plt.imshow(image)
        self.show_anns(masks)
        plt.axis("off")
        plt.show()
