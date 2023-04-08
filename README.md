<div align="center">
<h2>
     MetaSeg: Packaged version of the Segment Anything repository
</h2>
<div>
    <img width="1000" alt="teaser" src="https://github.com/kadirnar/segment-anything-pip/releases/download/v0.2.2/metaseg_demo.gif">
</div>
    <a href="https://pepy.tech/project/metaseg"><img src="https://pepy.tech/badge/metaseg" alt="downloads"></a>
    <a href="https://badge.fury.io/py/metaseg"><img src="https://badge.fury.io/py/metaseg.svg" alt="pypi version"></a>
    <a href="https://huggingface.co/spaces/ArtGAN/metaseg-webui"><img src="https://img.shields.io/badge/%20MetaSeg%20-Demo-blue.svg" alt="HuggingFace Spaces"></a>

</div>

This repo is a packaged version of the [segment-anything](https://github.com/facebookresearch/segment-anything) model.


### Installation
```bash
pip install metaseg
```

### Usage
```python
from metaseg import SegAutoMaskGenerator

# If gpu memory is not enough, reduce the points_per_side and points_per_batch.

# For image

autoseg_image = SegAutoMaskGenerator().save_image(
    source="image.jpg",
    model_type="vit_l",
    points_per_side=16, 
    points_per_batch=64
)

# For video

autoseg_video = SegAutoMaskGenerator().save_video(
    source="video.mp4",
    model_type="vit_l",
    points_per_side=16, 
    points_per_batch=64
)
```

# Extra Features

- [x] Support for video files
- [x] Support for pip installation
- [x] Support for web application
- [x] Support for automatic download model weights

