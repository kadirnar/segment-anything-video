"""Copyright (c) Metaseg Contributors.

All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import os
from functools import partial
from hashlib import md5
from pathlib import Path
from shutil import copyfileobj

# A dictionary containing model types as keys and their respective URLs as values
from requests import Response, get, status_codes
from tqdm.auto import tqdm

MODEL_URLS: dict[str, tuple[str, ...]] = {
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


def _check_md5(filename: str, orig_md5: str) -> bool:
    """Check MD5 hash of the given file matches the original MD5 hash.

    Args:
        filename (str): A string representing the path to the file.
        orig_md5 (str): A string representing the original MD5 hash.

    Returns:
        bool: True if the MD5 hash of the
            file matches the original MD5 hash, False otherwise.
    """
    if not os.path.exists(filename):
        return False
    with open(filename, "rb") as file_to_check:
        # read contents of the file
        data = file_to_check.read()
        # pipe contents of the file through
        md5_returned = md5(data, usedforsecurity=False).hexdigest()
        # Return True if the computed hash matches the original one
        if md5_returned == orig_md5:
            return True
        return False


def download_model(model_type):
    """Download the model file for the given model type.

    Args:
        model_type (str): A string representing the model type.
            It can be 'vit_h', 'vit_l', or 'vit_b'.

    Returns:
        None
    """
    filename = f"{model_type}.pth"
    if not os.path.exists(filename) and model_type in MODEL_URLS:
        res: Response = get(
            MODEL_URLS[model_type][0], stream=True, allow_redirects=True, timeout=20
        )
        if res.status_code != status_codes.codes.ok:
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
            os.remove(filename)
            download_model(model_type)

    else:
        raise ValueError(
            "Invalid model type. It should be 'vit_h', 'vit_l', or 'vit_b'."
        )

    return filename
