from io import BytesIO
from os import system
from os.path import isfile as isfile
from typing import Union
from uuid import uuid4

import cv2
import matplotlib.pyplot as plt
import numpy as np
from cv2 import Mat
from PIL import Image
from torch import tensor


def load_image(image: Union[str, Mat]) -> Mat:
    """
    Load image from path
    :param image_path: path to image file or image as Mat or np.ndarray
    :return: image as Mat
    """
    if isfile(image):
        image = cv2.imread(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
    elif isinstance(image, Mat) or isinstance(image, np.ndarray):
        return image
    else:
        raise ValueError("image must be a path or cv2.Mat")


def load_server_image(image_path):
    imagedir = str(uuid4())
    system(f"mkdir -p {imagedir}")
    image = Image.open(BytesIO(image_path))
    if image.mode != "RGB":
        image = image.convert("RGB")

    image_path = f"{imagedir}/base_image_v0.png"
    output_path = f"{imagedir}/output_v0.png"
    image.save(image_path, format="PNG")
    return image_path, output_path


def load_video(video_path, output_path="output.mp4"):
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    return cap, out


def load_mask(mask, random_color):
    if random_color:
        color = np.random.rand(3) * 255
    else:
        color = np.array([100, 50, 0])

    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    mask_image = mask_image.astype(np.uint8)
    return mask_image


def load_box(box, image):
    x, y, w, h = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    cv2.rectangle(image, (x, y), (w, h), (0, 255, 0), 2)
    return image


def plt_load_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def plt_load_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(
        plt.Rectangle((x0, y0), w, h, edgecolor="green", facecolor=(0, 0, 0, 0), lw=2)
    )


def multi_boxes(boxes, predictor, image):
    input_boxes = tensor(boxes, device=predictor.device)
    transformed_boxes = predictor.transform.apply_boxes_torch(
        input_boxes, image.shape[:2]
    )
    return input_boxes, transformed_boxes


def show_image(output_image):
    cv2.imshow("output", output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
