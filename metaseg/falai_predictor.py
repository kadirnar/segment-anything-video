"""Copyright (c) Metaseg Contributors.

All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

from io import BytesIO

from PIL import Image

from .sam_predictor import SegAutoMaskPredictor, SegManualMaskPredictor
from .utils.data_utils import load_server_image

try:
    from fal_serverless import isolated
except ImportError:
    raise ImportError(
        "Please install FalAI library using 'pip install fal_serverless'."
    )


@isolated(requirements=["metaseg"], keep_alive=1800, machine_type="GPU-T4")
def automask_image(
    data, model_type="vit_b", points_per_side=16, points_per_batch=32, min_area=0
):
    """Generates masks for input images using a vision transformer model.

    Args:
        data (List[Dict[str, Union[str, np.ndarray]]]):
            A list of dictionaries representing input images.
        model_type (str, optional):
            The type of vision transformer model to use. Defaults to "vit_b".
        points_per_side (int, optional):
            The number of points to use per side of the image. Defaults to 16.
        points_per_batch (int, optional):
            The number of points to process per batch. Defaults to 32.
        min_area (int, optional):
            The minimum area of a mask to keep. Defaults to 0.

    Returns:
        List[Dict[str, Union[str, np.ndarray]]]: A list of dictionaries
            representing input images with generated masks.
    """
    image_path, output_path = load_server_image(data)
    SegAutoMaskPredictor().image_predict(
        source=image_path,
        model_type=model_type,
        points_per_side=points_per_side,
        points_per_batch=points_per_batch,
        min_area=min_area,
        output_path=output_path,
        show=False,
        save=True,
    )
    with open(output_path, "rb") as f:
        result = f.read()

    return result


@isolated(requirements=["metaseg"], keep_alive=1800, machine_type="GPU-T4")
def manuelmask_image(
    data,
    model_type="vit_b",
    input_point=[[100, 100], [200, 200]],
    input_label=[0, 1],
    input_box=[100, 100, 200, 200],
    multimask_output=False,
    random_color=False,
    min_area=0,
):
    """Generates masks for input images using manual annotations.

    Args:
        data (List[Dict[str, Union[str, np.ndarray]]]): A list of dictionaries
            representing input images.
        model_type (str, optional): The type of vision transformer model to use.
            Defaults to "vit_b".
        input_point (List[List[int]], optional): A list of points to
            annotate the input images. Defaults to [[100, 100], [200, 200]].
        input_label (List[int], optional): A list of labels
            corresponding to the input points. Defaults to [0, 1].
        input_box (List[int], optional): A list of bounding
            box coordinates for the input images.
            Defaults to [100, 100, 200, 200].
        multimask_output (bool, optional): Whether to output multiple masks per image.
            Defaults to False.
        random_color (bool, optional): Whether to use
            random colors for the generated masks. Defaults to False.
        min_area (int, optional): The minimum area of a
            mask to keep. Defaults to 0.

    Returns:
        List[Dict[str, Union[str, np.ndarray]]]: A list of dictionaries representing
                input images with generated masks.
    """
    image_path, output_path = load_server_image(data)
    SegManualMaskPredictor().image_predict(
        source=image_path,
        model_type=model_type,
        input_point=input_point,
        input_label=input_label,
        input_box=input_box,  #
        multimask_output=multimask_output,
        random_color=random_color,
        min_area=min_area,  #
        output_path=output_path,
        show=False,
        save=True,
    )
    with open(output_path, "rb") as f:
        result = f.read()

    return result


def falai_automask_image(
    image_path,
    model_type="vit_b",
    points_per_side=16,
    points_per_batch=32,
    min_area=0,
):
    """Automatically generates masks.

    for the given input image using a vision transformer model.

    Args:
        image_path (str): The path to the input image file.
        model_type (str, optional): The type of vision
            transformer model to use. Defaults to "vit_b".
        points_per_side (int, optional): The number of points
            to use per side of the image. Defaults to 16.
        points_per_batch (int, optional): The number of points
            to process per batch. Defaults to 32.
        min_area (int, optional): The minimum area
            of a mask to keep. Defaults to 0.

    Returns:
        PIL.Image.Image: An image object with generated masks.
    """
    with open(image_path, "rb") as f:
        data = f.read()

    image = automask_image(
        data=data,
        model_type=model_type,
        points_per_side=points_per_side,
        points_per_batch=points_per_batch,
        min_area=min_area,
    )
    image = Image.open(BytesIO(image))
    return image


def falai_manuelmask_image(
    image_path,
    model_type="vit_b",
    input_point=[[100, 100], [200, 200]],
    input_label=[0, 1],
    input_box=[100, 100, 200, 200],
    multimask_output=False,
    random_color=False,
    min_area=0,
):
    """Generates masks for the given input image using manual annotations.

    Args:
        image_path (str): The path to the input image file.
        model_type (str, optional): The type of vision
            transformer model to use. Defaults to "vit_b".
        input_point (List[List[int]], optional): A list of points
            to annotate the input image. Defaults to [[100, 100], [200, 200]].
        input_label (List[int], optional): A list of labels
            corresponding to the input points. Defaults to [0, 1].
        input_box (List[int], optional): A list of bounding box coordinates
            for the input image. Defaults to [100, 100, 200, 200].
        multimask_output (bool, optional): Whether to output multiple
             masks per image. Defaults to False.
        random_color (bool, optional): Whether to use
            random colors for the generated masks. Defaults to False.
        min_area (int, optional): The minimum area of a mask to keep. Defaults to 0.

    Returns:
        PIL.Image.Image: An image object with generated masks.
    """
    with open(image_path, "rb") as f:
        data = f.read()

    image = manuelmask_image(
        data=data,
        model_type=model_type,
        input_point=input_point,
        input_label=input_label,
        input_box=input_box,
        multimask_output=multimask_output,
        random_color=random_color,
        min_area=min_area,
    )
    image = Image.open(BytesIO(image))
    return image
