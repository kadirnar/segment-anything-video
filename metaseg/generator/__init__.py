"""Copyright (c) Meta Platforms, Inc. and affiliates.

All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

from .automatic_mask_generator import (
    SamAutomaticMaskGenerator,
)
from .build_sam import (
    SAM_MODEL_REGISTRY,
    build_sam_vit_b,
    build_sam_vit_h,
    build_sam_vit_l,
)
from .predictor import SamPredictor
