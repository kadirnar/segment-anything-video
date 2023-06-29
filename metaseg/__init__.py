"""Copyright (c) Metaseg Contributors.

All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

from .falai_predictor import (
    automask_image,
    falai_automask_image,
    falai_manuelmask_image,
    manuelmask_image,
)
from .sahi_predictor import SahiAutoSegmentation, sahi_sliced_predict
from .sam_predictor import SegAutoMaskPredictor, SegManualMaskPredictor

__version__ = "0.7.8"
