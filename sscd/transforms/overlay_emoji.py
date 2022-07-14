# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import functools
import random
import glob
from typing import Any, Dict

import numpy as np
from classy_vision.dataset.transforms import ClassyTransform, register_transform
from PIL import Image
from augly.utils import EMOJI_DIR

from .samplers import Sampler


class EmojiRepository:
    def __init__(self, path):
        self._emoji_fpaths = glob.glob(os.path.join(path, "*/*.png"))
        self._emoji_images = {}

    def map_path(self, path, local_path):
        path = path.strip()
        if local_path:
            local_mapped = os.path.join(local_path, os.path.basename(path))
            if os.path.isfile(local_mapped):
                return local_mapped
        return path

    def random_emoji(self) -> Image.Image:
        emoji_fpath = random.choice(self._emoji_fpaths)
        return self.get_emoji(emoji_fpath)

    @functools.lru_cache(maxsize=None)
    def get_emoji(self, emoji_fpath: str) -> Image.Image:
        return Image.open(open(emoji_fpath, "rb"))

    @classmethod
    @functools.lru_cache(maxsize=None)
    def get(cls, path) -> "EmojiRepository":
        return cls(path)

    def size(self):
        return len(self._emoji_fpaths)


@register_transform("OverlayEmoji")
class OverlayEmojiTransform(ClassyTransform):
    """
    Overlays (random) emoji on image
    """

    def __init__(
        self,
        emoji_vault: str,
        emoji_size_sampler: Sampler,
        opacity_sampler: Sampler,
        fx_sampler: Sampler,
        fy_sampler: Sampler,
    ):
        self._emojis = EmojiRepository.get(emoji_vault)
        assert self._emojis.size() > 0
        self._emoji_size_sampler = emoji_size_sampler
        self._opacity_sampler = opacity_sampler
        self._fx_sampler = fx_sampler
        self._fy_sampler = fy_sampler

    def __call__(self, image: Image.Image):
        emoji: Image.Image = self._emojis.random_emoji()
        emoji_w, emoji_h = emoji.size
        image_w, image_h = image.size
        max_scale = min(image_w / emoji_w, image_h / emoji_h)
        emoji_size_frac = self._emoji_size_sampler()
        emoji_scale = max_scale * emoji_size_frac
        emoji = emoji.resize(
            (int(emoji_w * emoji_scale), int(emoji_h * emoji_scale)),
            resample=Image.BILINEAR,
        )
        fx = self._fx_sampler()
        fy = self._fy_sampler()
        topleft_x = int(fx * (image_w - emoji.width))
        topleft_y = int(fy * (image_h - emoji.height))
        opacity = self._opacity_sampler()
        # perform overlay
        image_rgba = image.copy().convert("RGBA")
        # Get the mask of the emoji if it has one, otherwise create it
        try:
            mask = emoji.getchannel("A")
            mask = Image.fromarray((np.array(mask) * opacity).astype(np.uint8))
        except ValueError:
            mask = Image.new(mode="L", size=emoji.size, color=int(opacity * 255))
        image_rgba.paste(emoji, box=(topleft_x, topleft_y), mask=mask)
        return image_rgba.convert("RGB")

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "OverlayEmojiTransform":
        emoji_vault = config.get("emoji_vault", EMOJI_DIR)
        emoji_size_sampler = Sampler.from_config(config["emoji_size"])
        opacity_sampler = Sampler.from_config(config["opacity"])
        fx_sampler = Sampler.from_config(config["fx"])
        fy_sampler = Sampler.from_config(config["fy"])
        transform = cls(
            emoji_vault, emoji_size_sampler, opacity_sampler, fx_sampler, fy_sampler
        )
        return transform
