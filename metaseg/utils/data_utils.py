"""Copyright (c) Metaseg Contributors.

All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import logging as log
from io import BytesIO
from os import makedirs
from os.path import exists, isfile
from uuid import uuid4

import cv2
import matplotlib.pyplot as plt
import numpy as np
from cv2 import Mat
from PIL import Image
from torch import tensor

# set the logging level
log.basicConfig(level=log.INFO)


def load_image(image: str | Mat) -> Mat:
    """Load image from path.

    :param image_path: path to image file or image as Mat or np.ndarray
    :return: image as Mat.
    """
    if isfile(str(image)):
        img = cv2.imread(str(image))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    elif isinstance(image, Mat | np.ndarray):
        return image
    else:
        raise ValueError("image must be a path or cv2.Mat")


def load_server_image(image_path):
    """Load image from path and create output image."""
    imagedir = str(uuid4())
    # use try-except block to handle errors
    try:
        # create the directory if it doesn't exist
        if not exists(imagedir):
            makedirs(imagedir)
            log.info("Directory '%s' created successfully.", imagedir)
        else:
            log.info("Directory '%s' already exists.", imagedir)
    except OSError as error:
        log.info("Error creating directory: %s", error)

    image = Image.open(BytesIO(image_path))
    if image.mode != "RGB":
        image = image.convert("RGB")

    image_path = f"{imagedir}/base_image_v0.png"
    output_path = f"{imagedir}/output_v0.png"
    image.save(image_path, format="PNG")
    return image_path, output_path


def load_video(video_path, output_path="output.mp4"):
    """Load video and create output video."""
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    return cap, out


def load_mask(mask, random_color):
    """Load mask and create mask image."""
    if random_color:
        color = np.random.rand(3) * 255
    else:
        color = np.array([100, 50, 0])

    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    mask_image = mask_image.astype(np.uint8)
    return mask_image


def load_box(box, image):
    """Load box and create box image."""
    x, y, w, h = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    cv2.rectangle(image, (x, y), (w, h), (0, 255, 0), 2)
    return image


def plt_load_mask(mask, ax, random_color=False):
    """Load mask and create mask image for matplotlib."""
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def plt_load_box(box, ax):
    """Load box and create box image for matplotlib."""
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(
        plt.Rectangle((x0, y0), w, h, edgecolor="green", facecolor=(0, 0, 0, 0), lw=2)
    )


def multi_boxes(boxes, predictor, image):
    """Load boxes and create box image."""
    input_boxes = tensor(boxes, device=predictor.device)
    transformed_boxes = predictor.transform.apply_boxes_torch(
        input_boxes, image.shape[:2]
    )
    return input_boxes, transformed_boxes


def show_image(output_image):
    """Show image in cv2."""
    cv2.imshow("output", output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
