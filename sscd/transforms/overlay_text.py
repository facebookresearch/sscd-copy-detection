# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
import functools
import io
import logging
import os
import pickle
import random
import numpy as np
from typing import Any, Dict, List
from PIL import Image, ImageFont, ImageDraw

from classy_vision.dataset.transforms import ClassyTransform, register_transform
from .samplers import Sampler
from augly.utils import FONTS_DIR


@dataclasses.dataclass
class Font:
    name: str
    path: str
    ttf_bytes: bytes
    charset: Any  # numpy array

    def ttf(self):
        return io.BytesIO(self.ttf_bytes)

    def image_font(self, size) -> ImageFont:
        return ImageFont.truetype(self.ttf(), size)

    @classmethod
    def load(cls, path) -> "Font":
        prefix, ext = os.path.splitext(path)
        assert ext in [".ttf", ".pkl"]
        ttf_path = f"{prefix}.ttf"
        name = os.path.basename(ttf_path)
        with open(ttf_path, "rb") as f:
            ttf_bytes = f.read()
        with open(f"{prefix}.pkl", "rb") as f:
            charset = np.array(pickle.load(f), dtype=np.int64)
        return cls(name=name, path=ttf_path, ttf_bytes=ttf_bytes, charset=charset)

    def sample_chars(self, length) -> List[int]:
        return random.choices(self.charset, k=length)

    def sample_string(self, length) -> str:
        characters = self.sample_chars(length)
        return "".join(chr(x) for x in characters)


class FontRepository:

    fonts = List[Font]

    def __init__(self, path):
        filenames = [
            os.path.join(path, filename)
            for filename in os.listdir(path)
            if filename.endswith(".ttf")
        ]
        logging.info("Loading %d fonts from %s.", len(filenames), path)
        self.fonts = [Font.load(filename) for filename in filenames]
        logging.info("Finished loading %d fonts.", len(filenames))

    def random_font(self) -> Font:
        return random.choice(self.fonts)

    @classmethod
    @functools.lru_cache(maxsize=None)
    def get(cls, path) -> "FontRepository":
        return cls(path)

    def size(self):
        return len(self.fonts)


@register_transform("OverlayText")
class OverlayTextTransform(ClassyTransform):
    """
    Overlays text on image
    """

    def __init__(
        self,
        font_vault: str,
        font_size_sampler: Sampler,
        opacity_sampler: Sampler,
        color_sampler: Sampler,
        fx_sampler: Sampler,
        fy_sampler: Sampler,
    ):
        self._fonts = FontRepository.get(font_vault)
        assert self._fonts.size() > 0
        self._font_size_sampler = font_size_sampler
        self._opacity_sampler = opacity_sampler
        self._color_sampler = color_sampler
        self._fx_sampler = fx_sampler
        self._fy_sampler = fy_sampler

    def __call__(self, image: Image.Image):
        # instantiate font
        font: Font = self._fonts.random_font()
        font_size_frac = self._font_size_sampler()
        font_size = int(min(image.width, image.height) * font_size_frac)
        image_font = font.image_font(font_size)
        # sample a string of fixed length from charset
        _SAMPLE_STR_LEN = 100
        text_str = font.sample_string(_SAMPLE_STR_LEN)
        # compute maximum length that fits into image
        # TODO: binary search over a lazy list of fixed length
        # (tw and th are monotonically increasing)
        maxlen = 0
        for i in range(1, len(text_str)):
            substr = text_str[:i]
            try:
                tw, th = image_font.getsize(substr)
            except OSError as e:
                # Safeguard against invalid chars in charset
                # that produce "invalid composite glyph" error
                logging.warning(f"Error, font={font.path}, char_i={ord(substr[-1])}")
                logging.warning(e)
                # don't overlay text in case of invalid glyphs
                return image
            if (tw > image.width) or (th > image.height):
                maxlen = i - 1
                break
        if maxlen == 0:
            return image
        # sample text length and get definitive text size
        text_len = random.randint(1, maxlen)
        text_str = text_str[:text_len]
        text_width, text_height = image_font.getsize(text_str)
        assert (text_width <= image.width) and (text_height <= image.height), (
            f"Text has size (H={text_height}, W={text_width}) which does "
            f"not fit into image of size (H={image.height}, W={image.width})"
        )
        # sample text location
        fx = self._fx_sampler()
        fy = self._fy_sampler()
        topleft_x = fx * (image.width - text_width)
        topleft_y = fy * (image.height - text_height)
        opacity = self._opacity_sampler()
        alpha = int(opacity * 255 + 0.5)
        color = tuple(self._color_sampler())
        color_w_opacity = color + (alpha,)
        # create output image
        image_base = image.convert("RGBA")
        image_txt = Image.new("RGBA", image_base.size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(image_txt)
        draw.text(
            xy=(topleft_x, topleft_y),
            text=text_str,
            fill=color_w_opacity,
            font=image_font,
        )
        image_out = Image.alpha_composite(image_base, image_txt).convert("RGB")
        return image_out

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "OverlayTextTransform":
        font_vault = config.get("font_vault", FONTS_DIR)
        font_size_sampler = Sampler.from_config(config["font_size"])
        opacity_sampler = Sampler.from_config(config["opacity"])
        color_sampler = Sampler.from_config(config["color"])
        fx_sampler = Sampler.from_config(config["fx"])
        fy_sampler = Sampler.from_config(config["fy"])
        transform = cls(
            font_vault,
            font_size_sampler,
            opacity_sampler,
            color_sampler,
            fx_sampler,
            fy_sampler,
        )
        return transform
