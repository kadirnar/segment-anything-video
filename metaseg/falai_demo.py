from io import BytesIO

from fal_serverless import isolated
from PIL import Image

from metaseg import SegAutoMaskPredictor
from metaseg.utils.data_utils import load_server_image

try:
    from fal_serverless import isolated
except ImportError:
    raise ImportError("Please install FalAI library using 'pip install fal_serverless'.")


@isolated(requirements=["metaseg"], keep_alive=1800, machine_type="GPU-T4")
def automask_server(data, model_type="vit_b", points_per_side=16, points_per_batch=32, min_area=0):
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


def falai_automask(image_path, model_type="vit_b", points_per_side=16, points_per_batch=32, min_area=0):
    with open(image_path, "rb") as f:
        data = f.read()

    image = falai_automask(
        data=data,
        model_type=model_type,
        points_per_side=points_per_side,
        points_per_batch=points_per_batch,
        min_area=min_area,
    )
    image = Image.open(BytesIO(image))
    return image
