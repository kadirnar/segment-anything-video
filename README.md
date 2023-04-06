<div align="center">
<h1>
     MetaSeg:Packaged version of the Segment Anything repository
</h1>
<div>
    <a href="https://pepy.tech/project/metaseg"><img src="https://pepy.tech/badge/metaseg" alt="downloads"></a>
    <a href="https://badge.fury.io/py/metaseg"><img src="https://badge.fury.io/py/metaseg.svg" alt="pypi version"></a>
</div>
</div>

This repo is a packaged version of the [segment-anything](https://github.com/facebookresearch/segment-anything) model.


### Installation
```bash
pip install metaseg
```

### Usage
```python
from metaseg import SegAutoMaskGenerator

SegAutoMaskGenerator(
        model_type="default", 
        checkpoint_path="sam_vit_h_4b8939.pth",
        image_path= "test.png",
        device="cuda",
        show_mask=True, 
```
