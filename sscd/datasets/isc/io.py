# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Original source: https://github.com/facebookresearch/isc2021

from typing import List

import numpy as np

from .metrics import GroundTruthMatch


def read_ground_truth(filename: str) -> List[GroundTruthMatch]:
    """
    Read groundtruth csv file.
    Must contain query_image_id,db_image_id on each line.
    handles the no header version and DD's version with header
    """
    gt_pairs = []
    with open(filename, "r") as cfile:
        for line in cfile:
            line = line.strip()
            if line == "query_id,reference_id":
                continue
            q, db = line.split(",")
            if db == "":
                continue
            gt_pairs.append(GroundTruthMatch(q, db))
    return gt_pairs
