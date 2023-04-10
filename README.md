<div align="center">
<h2>
     MetaSeg: Packaged version of the Segment Anything repository
</h2>
<div>
    <img width="1000" alt="teaser" src="https://github.com/kadirnar/segment-anything-pip/releases/download/v0.2.2/metaseg_demo.gif">
</div>
    <a href="https://pepy.tech/project/metaseg"><img src="https://pepy.tech/badge/metaseg" alt="downloads"></a>
    <a href="https://badge.fury.io/py/metaseg"><img src="https://badge.fury.io/py/metaseg.svg" alt="pypi version"></a>
    <a href="https://huggingface.co/spaces/ArtGAN/metaseg-webui"><img src="https://huggingface.co/datasets/huggingface/badges/raw/main/open-in-hf-spaces-sm.svg" alt="HuggingFace Spaces"></a>

</div>

This repo is a packaged version of the [segment-anything](https://github.com/facebookresearch/segment-anything) model.

### Installation
```bash
pip install metaseg
```

### Usage
```python
from metaseg import SegAutoMaskPredictor, SegManualMaskPredictor

# If gpu memory is not enough, reduce the points_per_side and points_per_batch.

# For image

autoseg_image = SegAutoMaskPredictor().save_image(
    source="image.jpg",
    model_type="vit_l", # vit_l, vit_h, vit_b
    points_per_side=16, 
    points_per_batch=64,
    min_area=0,
)

# For video

autoseg_video = SegAutoMaskPredictor().save_video(
    source="video.mp4",
    model_type="vit_l", # vit_l, vit_h, vit_b
    points_per_side=16, 
    points_per_batch=64,
    min_area=1000,
)

# For manuel box and point selection

seg_manual_mask_generator = SegManualMaskPredictor().save_image(
    source="image.jpg",
    model_type="vit_l", # vit_l, vit_h, vit_b
    input_point=[[100, 100], [200, 200]],
    input_label=[0, 1],
    input_box=[100, 100, 200, 200], # x,y,w,h
    multimask_output=False,

)

# For multi box selection

seg_manual_mask_generator = SegManualMaskPredictor().save_image(
    source="data/brain.png",
    model_type="vit_l",
    input_point=None,
    input_label=None,
    input_box= [[100, 100, 400, 400]],
    multimask_output=False,

)
```
# Extra Features

- [x] Support for Yolov5/8, Detectron2, Mmdetection, Torchvision models
- [x] Support for video and web application(Huggingface Spaces)
- [x] Support for manual single multi box and point selection
- [x] Support for pip installation
- [x] Support for SAHI library
