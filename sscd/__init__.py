# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path

from classy_vision.generic.registry_utils import import_all_modules

FILE_ROOT = Path(__file__).parent


def import_subdir(name):
    path = Path(FILE_ROOT, name)
    import_all_modules(path, f"sscd.{name}")


# Automatically import any Python files in selected directories.
import_subdir("transforms")
