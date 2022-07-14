# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


class RepeatedAugmentationTransform:
    """Applies a transform multiple times.

    Input: {"input": <image>, ...}
    Output: {"input0": <augmented tensor>, "input1": <augmented tensor>, ...}
    """

    def __init__(self, transform, copies=2, key="input"):
        self.transform = transform
        self.copies = copies
        self.key = key

    def __call__(self, record):
        record = record.copy()
        img = record.pop(self.key)
        for i in range(self.copies):
            record[f"{self.key}{i}"] = self.transform(img)
        return record
