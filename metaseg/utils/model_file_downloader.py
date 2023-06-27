import os
from functools import partial
from hashlib import md5
from pathlib import Path
from shutil import copyfileobj

from requests import Response, get
from tqdm.auto import tqdm

# A dictionary containing model types as keys and their respective URLs as values
MODEL_URLS: dict[str : tuple[str]] = {
    "vit_h": (
        "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
        "green",
        "01ec64d29a2fca3f0661936605ae66f8",
    ),
    "vit_l": (
        "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
        "red",
        "0b3195507c641ddb6910d2bb5adee89c",
    ),
    "vit_b": (
        "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
        "cyan",
        "4b8939a88964f0f4ff5f5b2642c598a6",
    ),
}


# md5 check function
def _check_md5(filename: str, orig_md5: str) -> bool:
    """
    filename: str, A string representing the path to the file.
    orig_md5: str, A string representing the original md5 hash.
    """
    if not os.path.exists(filename):
        return False
    with open(filename, "rb") as file_to_check:
        # read contents of the file
        data = file_to_check.read()
        # pipe contents of the file through
        md5_returned = md5(data).hexdigest()
        # Return True if the computed hash matches the original one
        if md5_returned == orig_md5:
            return True
        return False


def download_model(model_type):
    """
    model_type: str, A string representing the model type.
    It can be 'vit_h', 'vit_l', or 'vit_b'.
    """

    # Check if the model file already exists and model_type is in MODEL_URLS
    filename = f"{model_type}.pth"
    if not os.path.exists(filename) and model_type in MODEL_URLS:
        print(f"Downloading {filename} model \n")
        res: Response = get(
            MODEL_URLS[model_type][0], stream=True, allow_redirects=True
        )
        if res.status_code != 200:
            res.raise_for_status()
            raise RuntimeError(
                f"Request to {MODEL_URLS[model_type][0]} "
                f"returned status code {res.status_code}"
            )

        file_size: int = int(res.headers.get("Content-Length", 0))
        folder_path: Path = Path(filename).expanduser().resolve()
        folder_path.parent.mkdir(parents=True, exist_ok=True)

        desc = "(Unknown total file size)" if file_size == 0 else ""
        res.raw.read = partial(
            res.raw.read, decode_content=True
        )  # Decompress if needed
        with tqdm.wrapattr(
            res.raw,
            "read",
            total=file_size,
            desc=desc,
            colour=MODEL_URLS[model_type][1],
        ) as r_raw:
            with folder_path.open("wb") as f:
                copyfileobj(r_raw, f)

    elif os.path.exists(filename):
        if not _check_md5(filename, MODEL_URLS[model_type][2]):
            print("File corrupted. Re-downloading... \n")
            os.remove(filename)
            download_model(model_type)

        print(f"{filename} model download complete. \n")
    else:
        raise ValueError(
            "Invalid model type. It should be 'vit_h', 'vit_l', or 'vit_b'."
        )

    return filename
