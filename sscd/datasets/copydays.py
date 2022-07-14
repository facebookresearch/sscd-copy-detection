# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import numpy as np
from torchvision.datasets.folder import default_loader


def score_ap_from_ranks_1(ranks, nres):
    """Compute the average precision of one search.
    ranks = ordered list of ranks of true positives
    nres  = total number of positives in dataset
    """

    # accumulate trapezoids in PR-plot
    ap = 0.0

    # All have an x-size of:
    recall_step = 1.0 / nres

    for ntp, rank in enumerate(ranks):

        # y-size on left side of trapezoid:
        # ntp = nb of true positives so far
        # rank = nb of retrieved items so far
        if rank == 0:
            precision_0 = 1.0
        else:
            precision_0 = ntp / float(rank)

        # y-size on right side of trapezoid:
        # ntp and rank are increased by one
        precision_1 = (ntp + 1) / float(rank + 1)

        ap += (precision_1 + precision_0) * recall_step / 2.0

    return ap


# Compute the macro-MAP, precision and recall (threshold based) from
# the groundtruth defined as nquery x results matrices (
# 0 for match and 1 for non-matches)
def score_macro_AP(gnd, dis, totpos=None):
    # Default is one possible positive result per query
    if totpos is None:
        totpos = gnd.shape[0]
    # Interleave the results from all queries and sort them by distances
    gnd = np.reshape(gnd, (-1))
    a = np.reshape(dis, (-1)).argsort()
    gnd = gnd[a]
    ntp = gnd.cumsum().astype("float")
    recall = ntp / float(totpos)
    precision = ntp / (np.arange(ntp.shape[0]) + 1)

    # Compute the macroMAP now
    ranks_pos = [i for i in range(gnd.shape[0]) if gnd[i] == 1]
    MAP = score_ap_from_ranks_1(ranks_pos, totpos)
    return MAP, recall, precision


def blocks_from_directories(imnames):
    """splits a list of filenames according to their direcotry"""
    imnames.sort()
    prev_dirname = None
    block_names = []
    per_block_images = []
    for name in imnames:
        if name.startswith("./"):
            name = name[2:]
        if "/" in name:
            dirname = name[: name.rfind("/")]
        else:
            dirname = ""
        if dirname != prev_dirname:
            prev_dirname = dirname
            block_names.append(dirname)
            block_images = []
            per_block_images.append(block_images)
        block_images.append(name)

    return block_names, per_block_images


def cluster_pr(imnos, ids):
    """
    The images in the list imnos are from a cluster.
    Return the recall @ cluster size for the the results ids.
    1 = perfect result list, each false positive costs 1/len(imnos).
    """
    prs = []
    npos = len(imnos)
    for qno in imnos:
        ranks = [rank for rank, rno in enumerate(ids[qno]) if rno in imnos]
        # print '  ', ranks,
        recall = len(ranks) / float(npos)
        precision = len(ranks) / float(len(ids[qno]))
        prs.append((precision, recall))
    # print
    return np.array(prs)


class CopydaysBlock:

    STRONG = "strong"
    STANDARD_SIZE = 157
    STRONG_SIZE = 229

    def __init__(self, name, path, start_id, transforms=None):
        self.name = name
        self.path = path
        self.size = self.STRONG_SIZE if name == self.STRONG else self.STANDARD_SIZE
        self.start_id = start_id
        self.files = sorted([f for f in os.listdir(path) if f.endswith(".jpg")])
        self.transforms = transforms

    def __getitem__(self, i):
        file = self.files[i]
        img = default_loader(os.path.join(self.path, file))
        if self.transforms:
            img = self.transforms(img)
        return dict(
            input=img, block=self.name, filename=file, instance_id=i + self.start_id
        )

    def __len__(self):
        return self.size


class Copydays:
    def __init__(self, path, transforms=None):
        self.basedir = path
        self.block_names = (
            ["original", "strong"]
            + ["jpegqual/%d" % i for i in [3, 5, 8, 10, 15, 20, 30, 50, 75]]
            + ["crops/%d" % i for i in [10, 15, 20, 30, 40, 50, 60, 70, 80]]
        )
        self.blocks = []
        instance_id = 0
        for name in self.block_names:
            block = CopydaysBlock(
                name, os.path.join(path, name), instance_id, transforms
            )
            self.blocks.append(block)
            instance_id += len(block)
        self.size = instance_id
        self.nblocks = len(self.block_names)
        self.query_blocks = list(range(self.nblocks))
        self.q_block_sizes = np.ones(self.nblocks, dtype=int) * 157
        self.q_block_sizes[1] = 229
        # search only among originals
        self.database_blocks = [0]

    def __len__(self):
        return self.size

    def __getitem__(self, i):
        assert i < self.size
        for block in self.blocks:
            if i < len(block):
                return block[i]
            i -= len(block)
        raise AssertionError("unreachable")

    def get_block(self, i):
        dirname = self.basedir + "/" + self.block_names[i]
        fnames = [
            dirname + "/" + fname
            for fname in sorted(os.listdir(dirname))
            if fname.endswith(".jpg")
        ]
        res = ("image_list", fnames)
        return res

    def get_block_filenames(self, subdir_name, absolute=False):
        dirname = self.basedir + "/" + subdir_name
        relative = [
            fname for fname in sorted(os.listdir(dirname)) if fname.endswith(".jpg")
        ]
        if absolute:
            return [os.path.join(dirname, fname) for fname in relative]
        return relative

    def get_block_embeddings(self, embeddings, block_name):
        assert embeddings.shape[0] == len(self)
        assert block_name in self.block_names
        i = self.block_names.index(block_name)
        block = self.blocks[i]
        start = block.start_id
        end = start + len(block)
        return embeddings[start:end, :]

    def eval_result(self, ids, distances):
        metrics = {}
        j0 = 0
        for i in range(self.nblocks):
            j1 = j0 + self.q_block_sizes[i]
            block_name = self.block_names[i]
            I = ids[j0:j1]  # block size
            assert I.shape[0] == self.q_block_sizes[i]  # check for partial slice
            sum_AP = 0
            if block_name != "strong":
                # 1:1 mapping of files to names
                positives_per_query = [[i] for i in range(j1 - j0)]
            else:
                originals = self.get_block_filenames("original")
                strongs = self.get_block_filenames("strong")

                # check if prefixes match
                positives_per_query = [
                    [j for j, bname in enumerate(originals) if bname[:4] == qname[:4]]
                    for qname in strongs
                ]

            for qno, Iline in enumerate(I):
                positives = positives_per_query[qno]
                ranks = []
                for rank, bno in enumerate(Iline):
                    if bno in positives:
                        ranks.append(rank)
                sum_AP += score_ap_from_ranks_1(ranks, len(positives))

            mAP = sum_AP / (j1 - j0)
            print("eval on %s mAP=%.3f" % (block_name, mAP))
            metrics["%s mAP" % block_name] = mAP
            j0 = j1

        self.eval_result_alt(ids, distances, metrics)
        return metrics

    def eval_result_alt(self, ids, distances, metrics):
        gnd = np.zeros(ids.shape, dtype="int")
        j0 = 0
        for i in range(self.nblocks):
            j1 = j0 + self.q_block_sizes[i]
            block_name = self.block_names[i]
            I = ids[j0:j1]  # block size
            if block_name != "strong":
                # 1:1 mapping of files to names
                positives_per_query = [[i] for i in range(j1 - j0)]
            else:
                originals = self.get_block_filenames("original")
                strongs = self.get_block_filenames("strong")

                # check if prefixes match
                positives_per_query = [
                    [j for j, bname in enumerate(originals) if bname[:4] == qname[:4]]
                    for qname in strongs
                ]

            for qno, Iline in enumerate(I):
                positives = positives_per_query[qno]
                for rank, bno in enumerate(Iline):
                    if bno in positives:
                        gnd[j0 + qno][rank] = 1
            j0 = j1
        MAP, recall, precision = score_macro_AP(gnd, distances)
        print("Macro-AP = %.4f" % MAP)
        metrics["macro AP"] = MAP
