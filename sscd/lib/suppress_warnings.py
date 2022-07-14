# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import warnings

# This relates to Classy Vision transforms that we don't use.
warnings.filterwarnings("ignore", module=".*_(functional|transforms)_video")
# Upstream Classy Vision issue; fix hasn't reached released package.
# https://github.com/facebookresearch/ClassyVision/pull/770
warnings.filterwarnings("ignore", message=".*To copy construct from a tensor.*")
# Lightning non-issue (warning false positive).
warnings.filterwarnings("ignore", message=".*overridden after .* initialization.*")
