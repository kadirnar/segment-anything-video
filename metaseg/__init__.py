# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from .falai_demo import automask_image as automask_image
from .falai_demo import falai_automask_image as falai_automask_image
from .falai_demo import falai_manuelmask_image as falai_manuelmask_image
from .falai_demo import manuelmask_image as manuelmask_image
from .mask_predictor import SegAutoMaskPredictor as SegAutoMaskPredictor
from .mask_predictor import SegManualMaskPredictor as SegManualMaskPredictor
from .sahi_predict import SahiAutoSegmentation as SahiAutoSegmentation
from .sahi_predict import sahi_sliced_predict as sahi_sliced_predict

__version__ = "0.7.6"
