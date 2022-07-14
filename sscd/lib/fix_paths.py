# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path
import sys

# Make sure SSCD is in PYTHONPATH.
base_path = str(Path(__file__).parent.parent.parent)

if base_path not in sys.path:
    sys.path.append(base_path)
