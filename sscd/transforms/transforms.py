# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import random
from typing import Any, Dict
from classy_vision.dataset.transforms import (
    ClassyTransform,
    build_transform,
    register_transform,
)
from augly.image import encoding_quality
from PIL import Image, ImageFilter
from torchvision import transforms
from .samplers import Sampler


@register_transform("Blur")
class BlurTransform(ClassyTransform):
    """
    Applies Gaussian blur to image
    """

    def __init__(self, radius_sampler: Sampler):
        self._r_sampler = radius_sampler

    def __call__(self, image: Image.Image):
        im_filter = ImageFilter.GaussianBlur(radius=self._r_sampler())
        image_filtered = image.filter(im_filter)
        return image_filtered

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "BlurTransform":
        """
        Load blur transform from a config
        Examples:
        1. Deterministic blur with radius = 3:
            {
                "name": "Blur",
                "radius": 3.0
            }
        2. Blur with uniform random radius:
            {
                "name": "Blur",
                "radius": {
                    "sampler_type": "uniform",
                    "low": 0.0,
                    "high": 5.0
                }
            }
        """
        radius_sampler = Sampler.from_config(config["radius"])
        return cls(radius_sampler=radius_sampler)


@register_transform("Rotate")
class RotateTransform(ClassyTransform):
    """
    Applies rotation to image
    """

    def __init__(self, angle_deg_sampler: Sampler):
        self._angle_deg_sampler = angle_deg_sampler

    def __call__(self, image: Image.Image):
        image_rotated = image.rotate(
            angle=self._angle_deg_sampler(), resample=Image.BILINEAR
        )
        return image_rotated

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "RotateTransform":
        """
        Load rotate transform from a config
        Examples:
        1. Deterministic rotation on 45 degrees:
            {
                "name": "Rotate",
                "degrees_ccw": 45.0
            }
        2. Random uniform rotation:
            {
                "name": "Rotate",
                "degrees_ccw": {
                    "sampler_type": "uniform",
                    "low": 0.0,
                    "high": 90.0
                }
            }
        """
        angle_sampler = Sampler.from_config(config["degrees_ccw"])
        return cls(angle_deg_sampler=angle_sampler)


@register_transform("JpegCompress")
class JpegCompressTransform(ClassyTransform):
    """
    Compresses an image with lower bitrate JPEG to make compression
    artifacts appear on the resulting image
    """

    def __init__(self, quality_sampler: Sampler):
        """
        Args:
          quality_sampler: sampler of JPEG quality values (integers in [0, 100])
        """
        self._q_sampler = quality_sampler

    def __call__(self, image: Image.Image):
        image_transformed = encoding_quality(image, quality=self._q_sampler())
        return image_transformed

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "JpegCompressTransform":
        """
        Load JPEG compression transform from a config
        Examples:
        1. Deterministic compression with quality = 15:
            {
                "name": "JpegCompress",
                "quality": 15
            }
        2. Compression with uniformly sampled quality:
            {
                "name": "JpegCompress",
                "quality": {
                    "sampler_type": "uniformint",
                    "low": 0,
                    "high": 100
                }
            }
        """
        quality_sampler = Sampler.from_config(config["quality"])
        return cls(quality_sampler=quality_sampler)


@register_transform("MaybeApply")
class MaybeApplyTransform(ClassyTransform):
    """A Classy version of RandomApply.

    This is just shorthand for the `n = 1` case of BinomialWrapper, which is a
    common case.
    """

    def __init__(self, p, transform):
        self.p = p
        self.transform = transform

    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() < self.p:
            image = self.transform(image)
        return image

    @classmethod
    def from_config(cls, config: Dict[str, Any]):
        p = config["p"]
        transform = build_transform(config["transform"])
        return cls(p, transform)


@register_transform("ResizeLongEdge")
class ResizeLongEdge(ClassyTransform):
    """Resize the long edge of an image to a target size."""

    def __init__(self, size):
        self.size = size

    def __call__(self, image: Image.Image) -> Image.Image:
        scale = self.size / max(image.size)
        h, w = image.size
        if h > w:
            h = self.size
            w = int(scale * w + 0.5)
        else:
            w = self.size
            h = int(scale * h + 0.5)
        return transforms.Resize((w, h))(image)

    @classmethod
    def from_config(cls, config: Dict[str, Any]):
        return cls(config["size"])
