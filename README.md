<div align="center">
<h2>
     MetaSeg: Packaged version of the Segment Anything repository
</h2>
<div>
    <img width="1000" alt="teaser" src="https://github.com/kadirnar/segment-anything-pip/releases/download/v0.1.2/metaseg_demo.png">
</div>
    <a href="https://badge.fury.io/py/metaseg"><img src="https://badge.fury.io/py/metaseg.svg" alt="pypi version"></a>

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

SegAutoMaskGenerator(
        model_type="vit_h", # "vit_l", "vit_b"
        source= "test.png", # test.mp4
        device="cuda", # "cpu" or "cuda"
        show=True, 
        points_per_side=16, # Optional
        points_per_batch=64, # Optional
)
```

# Extra Features

- [x] Support for video files
- [x] Support for pip installation
- [x] Support for automatic download model weights
