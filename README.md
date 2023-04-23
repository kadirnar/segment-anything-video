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
results = SegAutoMaskPredictor().image_predict(
    source="image.jpg",
    model_type="vit_l", # vit_l, vit_h, vit_b
    points_per_side=16, 
    points_per_batch=64,
    min_area=0,
    output_path="output.jpg",
    show=True,
    save=False,
)

# For video
results = SegAutoMaskPredictor().video_predict(
    source="video.mp4",
    model_type="vit_l", # vit_l, vit_h, vit_b
    points_per_side=16, 
    points_per_batch=64,
    min_area=1000,
    output_path="output.mp4",
)

# For manuel box and point selection

# For image
results = SegManualMaskPredictor().image_predict(
    source="image.jpg",
    model_type="vit_l", # vit_l, vit_h, vit_b
    input_point=[[100, 100], [200, 200]],
    input_label=[0, 1],
    input_box=[100, 100, 200, 200], # or [[100, 100, 200, 200], [100, 100, 200, 200]]
    multimask_output=False,
    random_color=False,
    show=True,
    save=False,
)

# For video

results = SegManualMaskPredictor().video_predict(
    source="test.mp4",
    model_type="vit_l", # vit_l, vit_h, vit_b
    input_point=[0, 0, 100, 100]
    input_label=[0, 1],
    input_box=None,
    multimask_output=False,
    random_color=False,
    output_path="output.mp4",
)
```

### [SAHI](https://github.com/obss/sahi) + Segment Anything

```python
from metaseg import sahi_sliced_predict, SahiAutoSegmentation

image_path = "test.jpg"
boxes = sahi_sliced_predict(
    image_path=image_path,
    detection_model_type="yolov5", #yolov8, detectron2, mmdetection, torchvision
    detection_model_path="yolov5l6.pt",
    conf_th=0.25,
    image_size=1280,
    slice_height=256,
    slice_width=256,
    overlap_height_ratio=0.2,
    overlap_width_ratio=0.2,
)

SahiAutoSegmentation().predict(
    source=image_path,
    model_type="vit_b",
    input_box=boxes,
    multimask_output=False,
    random_color=False,
    show=True,
    save=False,
)
```
<img width="700" alt="teaser" src="https://github.com/kadirnar/segment-anything-pip/releases/download/v0.5.0/sahi_autoseg.png">

### [FalAI(Cloud GPU)](https://docs.fal.ai/fal-serverless/quickstart) + Segment Anything
```bash
pip install fal_serverless
```

```python
# For Auto Mask
from metaseg import falai_automask_image

image = falai_automask_image(
    image_path="data.jpg",
    model_type="vit_b",
    points_per_side=16,
    points_per_batch=32,
    min_area=0,
)   
image.show() # Show image
image.save("output.jpg") # Save image

# For Manual Mask
from metaseg import falai_manualmask_image

image = falai_manualmask_image(
    image_path="data.jpg",
    model_type="vit_b",
    input_point=[[100, 100], [200, 200]],
    input_label=[0, 1],
    input_box=[100, 100, 200, 200], # or [[100, 100, 200, 200], [100, 100, 200, 200]]
    multimask_output=False,
    random_color=False,
)
image.show() # Show image
image.save("output.jpg") # Save image
```
# Extra Features

- [x] Support for Yolov5/8, Detectron2, Mmdetection, Torchvision models
- [x] Support for video and web application(Huggingface Spaces)
- [x] Support for manual single multi box and point selection
- [x] Support for pip installation
- [x] Support for SAHI library
- [x] Support for FalAI
