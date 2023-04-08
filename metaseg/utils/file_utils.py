import os
import urllib.request


def download_model(model_type):
    """
    model_type: str, A string representing the model type. It can be 'vit_h', 'vit_l', or 'vit_b'.
    """

    # A dictionary containing model types as keys and their respective URLs as values
    model_urls = {
        "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
        "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
        "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
    }

    # Check if the model file already exists and model_type is in model_urls
    filename = f"{model_type}.pth"
    if not os.path.exists(filename) and model_type in model_urls:
        url = model_urls[model_type]
        print(f"Downloading {model_type} model from {url}...")
        urllib.request.urlretrieve(url, filename)
        print(f"{model_type} model has been successfully downloaded and saved as '{filename}'.")
    elif os.path.exists(filename):
        print(f"{model_type} model already exists as '{filename}'. Skipping download.")
    else:
        raise ValueError("Invalid model type. It should be 'vit_h', 'vit_l', or 'vit_b'.")

    return filename
