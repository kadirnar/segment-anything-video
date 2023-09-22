# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from .falai_predictor import automask_image as automask_image
from .falai_predictor import falai_automask_image as falai_automask_image
from .falai_predictor import falai_manuelmask_image as falai_manuelmask_image
from .falai_predictor import manuelmask_image as manuelmask_image
from .sahi_predictor import SahiAutoSegmentation as SahiAutoSegmentation
from .sahi_predictor import sahi_sliced_predict as sahi_sliced_predict
from .sam_predictor import SegAutoMaskPredictor as SegAutoMaskPredictor
from .sam_predictor import SegManualMaskPredictor as SegManualMaskPredictor

import importlib.metadata as importlib_metadata

try:
    # This will read version from pyproject.toml
    __version__ = importlib_metadata.version(__package__ or __name__)
except importlib_metadata.PackageNotFoundError:
    __version__ = "development"
