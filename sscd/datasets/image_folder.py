# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import functools
import logging
import os.path

from torchvision.datasets.folder import is_image_file
from torchvision.datasets.folder import default_loader


@functools.lru_cache()
def get_image_paths(path):
    logging.info(f"Resolving files in: {path}")
    filenames = [f"{path}/{file}" for file in os.listdir(path)]
    return sorted([fn for fn in filenames if is_image_file(fn)])


class ImageFolder:
    """An image folder dataset intended for self-supervised learning."""

    def __init__(self, path, transform=None, img_transform=None, loader=default_loader):
        self.files = get_image_paths(path)
        self.loader = loader
        self.transform = transform
        self.img_transform = img_transform

    def __getitem__(self, idx: int):
        assert 0 <= idx < len(self)
        img = self.loader(self.files[idx])
        record = {"input": img, "instance_id": idx}
        if self.img_transform:
            record["input"] = self.img_transform(record["input"])
        if self.transform:
            record = self.transform(record)
        return record

    def __len__(self):
        return len(self.files)
