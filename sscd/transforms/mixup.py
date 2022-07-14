# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Any, Dict

import torch
from classy_vision.dataset.transforms import (
    ClassyTransform,
    register_transform,
)
from classy_vision.dataset.transforms.mixup import MixupTransform


@register_transform("contrastive_mixup")
class ContrastiveMixup(ClassyTransform):
    """Mixup / cutmix augmentations for contrastive learning."""

    MIXUP_DEFAULTS = {
        "mixup_alpha": 1.0,
        "cutmix_alpha": 0.0,
        "cutmix_minmax": None,
        "switch_prob": 0.5,
        "mode": "elem",
        "correct_lam": True,
        "label_smoothing": 0.0,
    }

    def __init__(
        self,
        image_column: str,
        target_column: str,
        mixup_args: Dict[str, Any],
        repeated_augmentations: int,
    ):
        self.image_column = image_column
        self.target_column = target_column
        self.repeated_augmentations = repeated_augmentations
        self.mixup_args = mixup_args

    def transcode_targets(self, target):
        R = self.repeated_augmentations
        M = target.size(0)
        assert (
            M % R == 0
        ), "Config error: Batch size not divisible by repeated augmentations"
        N = M // R
        transcoded = torch.arange(M, dtype=target.dtype, device=target.device) % N
        # Sanity checking
        old_matches = target.unsqueeze(1) == target.unsqueeze(0)
        new_matches = transcoded.unsqueeze(1) == transcoded.unsqueeze(0)
        mismatches = new_matches ^ old_matches
        if mismatches.any():
            num_mismatches = mismatches.sum().item()
            logging.warning(
                f"Target transcoding introduced {num_mismatches} mismatches. "
                f"Batch size {N} * R={R} = {M}."
            )
        return transcoded, N

    def __call__(self, batch):
        targets, num_classes = self.transcode_targets(batch[self.target_column])
        # The mixup transform appears to mutate input tensors in place. This
        # produces surprising results for repeated augmentations. Clone the
        # input tensor before calling mixup.
        mixup_batch = {
            "input": batch[self.image_column].clone(),
            "target": targets,
        }
        mixup = MixupTransform(**self.mixup_args, num_classes=num_classes)
        mixed = mixup(mixup_batch)
        batch = batch.copy()
        batch[self.image_column] = mixed["input"]
        batch[self.target_column] = mixed["target"]
        return batch

    @classmethod
    def from_config(cls, config: Dict[str, Any]):
        # Needed to recode targets to per-GPU-batch relative identifiers
        image_column = config.get("image_column", "input")
        target_column = config.get("target_column", "target")
        repeated_augmentations = config["repeated_augmentations"]
        mixup_args = {
            "mix_prob": config["mix_prob"],  # required
        }
        for key, default in cls.MIXUP_DEFAULTS.items():
            mixup_args[key] = config.get(key, default)

        return cls(image_column, target_column, mixup_args, repeated_augmentations)
