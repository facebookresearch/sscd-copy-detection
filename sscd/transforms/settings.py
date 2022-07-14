# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import enum
import json
from classy_vision.dataset.transforms import build_transforms


_simclr_config = """
[
    {"name": "RandomResizedCrop", "size": %(train_image_size)d},
    {"name": "RandomHorizontalFlip"},
    {
        "name": "MaybeApply",
        "p": 0.8,
        "transform": {
            "name": "ColorJitter",
            "brightness": 0.8,
            "contrast": 0.8,
            "saturation": 0.8,
            "hue": 0.2
        }
    },
    {"name": "RandomGrayscale", "p": 0.2},
    {
        "name": "MaybeApply",
        "p": 0.5,
        "transform": {
            "name": "Blur",
            "radius": {
                "sampler_type": "uniform",
                "low": 0.1,
                "high": 2
            }
        }
    },
    {"name": "ToTensor"},
    {"name": "Normalize", "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
]
"""

_strong_blur_config = """
[
    {"name": "RandomResizedCrop", "size": %(train_image_size)d},
    {"name": "RandomHorizontalFlip"},
    {
        "name": "MaybeApply",
        "p": 0.8,
        "transform": {
            "name": "ColorJitter",
            "brightness": 0.8,
            "contrast": 0.8,
            "saturation": 0.8,
            "hue": 0.2
        }
    },
    {"name": "RandomGrayscale", "p": 0.2},
    {
        "name": "MaybeApply",
        "p": 0.5,
        "transform": {
            "name": "Blur",
            "radius": {
                "sampler_type": "uniform",
                "low": 1,
                "high": 5
            }
        }
    },
    {"name": "ToTensor"},
    {"name": "Normalize", "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
]
"""

_advanced_augs_config = """
[
    {"name": "RandomHorizontalFlip"},
    {
        "name": "MaybeApply",
        "p": 0.05,
        "transform": {
            "name": "Rotate",
            "degrees_ccw": {
                "sampler_type": "choice",
                "values": [90, 180, 270]
            }
        }
    },
    {
        "name": "MaybeApply",
        "p": 0.1,
        "transform": {
            "name": "OverlayText",
            "font_size": {"sampler_type": "uniform", "low": 0.1, "high": 0.3},
            "opacity": {"sampler_type": "uniform", "low": 0.1, "high": 1},
            "color": {
                "sampler_type": "tuple",
                "samplers": [
                    {"sampler_type": "uniformint", "low": 0, "high": 255},
                    {"sampler_type": "uniformint", "low": 0, "high": 255},
                    {"sampler_type": "uniformint", "low": 0, "high": 255}
                ]
            },
            "fx": {"sampler_type": "uniform", "low": 0, "high": 1},
            "fy": {"sampler_type": "uniform", "low": 0, "high": 1}
        }
    },
    {
        "name": "MaybeApply",
        "p": 0.2,
        "transform": {
            "name": "OverlayEmoji",
            "emoji_size": {"sampler_type": "uniform", "low": 0.1, "high": 0.5},
            "opacity": {"sampler_type": "uniform", "low": 0.7, "high": 1},
            "fx": {"sampler_type": "uniform", "low": 0, "high": 1},
            "fy": {"sampler_type": "uniform", "low": 0, "high": 1}
        }
    },
    {
      "name": "MaybeApply",
      "p": 0.05,
      "transform": {
        "name": "Rotate",
        "degrees_ccw": {
          "sampler_type": "uniformint",
          "low": 0,
          "high": 359
        }
      }
    },
    {"name": "RandomResizedCrop", "size": %(train_image_size)d},
    {
        "name": "MaybeApply",
        "p": 0.8,
        "transform": {
            "name": "ColorJitter",
            "brightness": 0.8,
            "contrast": 0.8,
            "saturation": 0.8,
            "hue": 0.2
        }
    },
    {"name": "RandomGrayscale", "p": 0.2},
    {
        "name": "MaybeApply",
        "p": 0.5,
        "transform": {
            "name": "Blur",
            "radius": {
                "sampler_type": "uniform",
                "low": 1,
                "high": 5
            }
        }
    },
    {
        "name": "MaybeApply",
        "p": 0.2,
        "transform": {
            "name": "JpegCompress",
            "quality": {
                "sampler_type": "uniformint",
                "low": 0,
                "high": 100
            }
        }
    },
    {"name": "ToTensor"},
    {"name": "Normalize", "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
]
"""


class AugmentationSetting(enum.Enum):
    """Augmentation configs explored in the SSCD paper."""

    SIMCLR = enum.auto()
    STRONG_BLUR = enum.auto()
    ADVANCED = enum.auto()

    def get_transformations(self, image_size):
        config = self._get_config(self) % {"train_image_size": image_size}
        config = json.loads(config)
        return build_transforms(config)

    def _get_config(self, value):
        return {
            self.SIMCLR: _simclr_config,
            self.STRONG_BLUR: _strong_blur_config,
            self.ADVANCED: _advanced_augs_config,
        }[value]
