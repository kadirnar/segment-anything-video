# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from metaseg.automatic_mask_generator import SamAutomaticMaskGenerator
from metaseg.build_sam import build_sam, build_sam_vit_b, build_sam_vit_h, build_sam_vit_l, sam_model_registry
from metaseg.predictor import SamPredictor

__version__ = "0.1.0"
