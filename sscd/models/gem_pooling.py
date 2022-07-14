# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from torch import nn


class GlobalGeMPool2d(nn.Module):
    """Generalized mean pooling.

    Inputs should be non-negative.
    """

    def __init__(
        self,
        pooling_param: float,
    ):
        """
        Args:
            pooling_param: the GeM pooling parameter
        """
        super().__init__()
        self.pooling_param = pooling_param

    def forward(self, x):
        N, C, H, W = x.size()
        x = x.reshape(N, C, H * W)  # Combine spatial dimensions
        mean = x.clamp(min=1e-6).pow(self.pooling_param).mean(dim=2)
        r = 1.0 / self.pooling_param
        return mean.pow(r)
