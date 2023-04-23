def load_image(image_path):
    import cv2

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def load_server_image(image_path):
    import os
    from io import BytesIO
    from uuid import uuid4

    from PIL import Image

    imagedir = str(uuid4())
    os.system(f"mkdir -p {imagedir}")
    image = Image.open(BytesIO(image_path))
    if image.mode != "RGB":
        image = image.convert("RGB")

    image_path = f"{imagedir}/base_image_v0.png"
    output_path = f"{imagedir}/output_v0.png"
    image.save(image_path, format="PNG")
    return image_path, output_path


def load_video(video_path, output_path="output.mp4"):
    import cv2

    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    return cap, out


def read_image(image_path):
    import cv2

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def load_mask(mask, random_color):
    import numpy as np

    if random_color:
        color = np.random.rand(3) * 255
    else:
        color = np.array([100, 50, 0])

    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    mask_image = mask_image.astype(np.uint8)
    return mask_image


def load_box(box, image):
    import cv2

    x, y, w, h = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    cv2.rectangle(image, (x, y), (w, h), (0, 255, 0), 2)
    return image


def plt_load_mask(mask, ax, random_color=False):
    import numpy as np

    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def plt_load_box(box, ax):
    import matplotlib.pyplot as plt

    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor="green", facecolor=(0, 0, 0, 0), lw=2))


def multi_boxes(boxes, predictor, image):
    import torch

    input_boxes = torch.tensor(boxes, device=predictor.device)
    transformed_boxes = predictor.transform.apply_boxes_torch(input_boxes, image.shape[:2])
    return input_boxes, transformed_boxes


def show_image(output_image):
    import cv2

    cv2.imshow("output", output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def save_image(output_image, output_path):
    import cv2

    cv2.imwrite(output_path, output_image)
