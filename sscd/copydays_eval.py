#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging
import os

import faiss
import numpy as np
import pandas as pd

from lib import initialize  # noqa
from lib.inference import Inference
from sscd.datasets.copydays import Copydays
from sscd.datasets.image_folder import ImageFolder
from sscd.lib.util import parse_bool

# After initialize import to silence an error.
from classy_vision.dataset.transforms import build_transforms

parser = argparse.ArgumentParser()
inference_parser = parser.add_argument_group("Inference")
Inference.add_parser_args(inference_parser)
inference_parser.add_argument(
    "--resize_long_edge",
    default=False,
    type=parse_bool,
    help=(
        "Preprocess images by resizing the long edge to --size. "
        "Has no effect if --preserve_aspect_ratio is not set."
    ),
)

cd_parser = parser.add_argument_group("Copydays")
cd_parser.add_argument("--copydays_path", required=True)
cd_parser.add_argument("--distractor_path", required=True)
cd_parser.add_argument("--codec_train_path")
cd_parser.add_argument(
    "--codecs",
    default="Flat",
    help="FAISS codecs for postprocessing embeddings as ';' separated strings"
    "in index_factory format",
)
cd_parser.add_argument("--metadata", help="Metadata column to put in the result CSV")

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.WARNING,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("copydays_eval.py")
logger.setLevel(logging.INFO)


def get_transforms(size, preserve_aspect_ratio, resize_long_edge):
    resize_long_edge = preserve_aspect_ratio and resize_long_edge
    resize_name = "ResizeLongEdge" if resize_long_edge else "Resize"
    resize_size = size if preserve_aspect_ratio else [size, size]
    return build_transforms(
        [
            {"name": resize_name, "size": resize_size},
            {"name": "ToTensor"},
            {
                "name": "Normalize",
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
            },
        ]
    )


def evaluate(
    embeddings, distractors, train, copydays: Copydays, index_type: str, k=100
):
    D = embeddings.shape[1]
    faiss_index = faiss.index_factory(D, index_type)
    if not faiss_index.is_trained:
        assert (
            train is not None
        ), f"A training dataset must be provided for {index_type} search"
        faiss_index.train(train)
    faiss_index.add(copydays.get_block_embeddings(embeddings, "original"))
    faiss_index.add(distractors)
    distances, ids = faiss_index.search(embeddings, k)
    assert np.isfinite(
        distances
    ).all(), "Non-finite distances found; this often means whitening failed"
    metrics = copydays.eval_result(ids, distances)
    metrics = dict(strong_mAP=metrics["strong mAP"], overall_uAP=metrics["macro AP"])
    return metrics


def evaluate_all(
    copydays_outputs,
    distractor_outputs,
    train_outputs,
    copydays: Copydays,
    codecs: str,
    metadata=None,
    k=100,
):
    codecs = codecs.split(";")
    instance_ids = copydays_outputs["instance_id"]
    embeddings = copydays_outputs["embeddings"]
    order = np.argsort(instance_ids)
    instance_ids = instance_ids[order]
    embeddings = embeddings[order, :]
    N, D = embeddings.shape
    assert N == len(copydays)
    assert np.all(instance_ids == np.arange(N, dtype=np.int64))
    distractors = distractor_outputs["embeddings"]
    train = train_outputs["embeddings"] if train_outputs else None
    records = []
    for codec in codecs:
        record = {"codec": codec}
        metrics = evaluate(embeddings, distractors, train, copydays, codec, k=k)
        record.update(metrics)
        logger.info(f"Metrics: {record}")
        records.append(record)
    df = pd.DataFrame(records)
    if metadata:
        df["metadata"] = metadata
    return df


def main(args):
    logger.info("Setting up dataset")
    transforms = get_transforms(
        args.size, args.preserve_aspect_ratio, args.resize_long_edge
    )
    copydays = Copydays(args.copydays_path, transforms)
    copydays_embeddings = Inference.inference(args, copydays, "copydays")
    distractors = ImageFolder(args.distractor_path, img_transform=transforms)
    distractor_embeddings = Inference.inference(args, distractors, "distractors")
    if args.codec_train_path:
        codec_train = ImageFolder(args.codec_train_path, img_transform=transforms)
        train_embeddings = Inference.inference(args, codec_train, "codec_train")
    else:
        train_embeddings = None
    df = evaluate_all(
        copydays_embeddings,
        distractor_embeddings,
        train_embeddings,
        copydays,
        args.codecs,
        metadata=args.metadata,
    )
    csv_filename = os.path.join(args.output_path, "copydays_metrics.csv")
    df.to_csv(csv_filename, index=False)
    with open(csv_filename, "r") as f:
        logger.info("Metric CSV:\n%s", f.read())


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
