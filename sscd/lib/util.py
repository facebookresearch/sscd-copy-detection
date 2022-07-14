# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import inspect
from typing import Callable
from distutils.util import strtobool


def call_using_args(function: Callable, args: argparse.Namespace):
    """Calls the callable using arguments from an argparse container."""
    signature = inspect.signature(function)
    arguments = {key: getattr(args, key) for key in signature.parameters}
    return function(**arguments)


def parse_bool(bool_str):
    return bool(strtobool(bool_str))
