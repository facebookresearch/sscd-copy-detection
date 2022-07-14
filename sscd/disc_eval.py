#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import dataclasses
import json
import logging
import os
from typing import Optional

import faiss
import torch
import numpy as np
from numpy import linalg
import pandas as pd

from lib import initialize  # noqa
from lib.inference import Inference
from sscd.train import DISCData
from sscd.datasets.disc import DISCEvalDataset

parser = argparse.ArgumentParser()
inference_parser = parser.add_argument_group("Inference")
Inference.add_parser_args(inference_parser)

disc_parser = parser.add_argument_group("DISC")
disc_parser.add_argument("--disc_path", required=True)
disc_parser.add_argument(
    "--codecs",
    default=None,
    help="FAISS codecs for postprocessing embeddings as ';' separated strings "
    "in index_factory format",
)
disc_parser.add_argument(
    "--score_norm",
    default="1.0[0,2]",
    help="Score normalization settings, ';' separated, in format: "
    "<weight>[<first index>,<last index>]",
)
disc_parser.add_argument("--metadata", help="Metadata column to put in the result CSV")

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.WARNING,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("disc_eval.py")
logger.setLevel(logging.INFO)


class ProjectionError(Exception):
    """Projection returned non-finite values."""


def get_codecs(dims, is_l2_normalized, codecs_arg):
    if codecs_arg:
        return codecs_arg.split(";")
    if is_l2_normalized:
        return ["Flat", f"PCAW{dims},L2norm,Flat"]
    return ["Flat", "L2norm,Flat", f"PCAW{dims},L2norm,Flat", f"L2norm,PCAW{dims},Flat"]


def is_l2_normalized(embeddings):
    norms = linalg.norm(embeddings, axis=1)
    return np.abs(norms - 1).mean() < 0.01


@dataclasses.dataclass
class ScoreNormalization:
    weight: float
    start_index: int
    end_index: int

    @classmethod
    def parse(cls, spec):
        weight, spec = spec.split("[", 1)
        assert spec.endswith("]")
        spec = spec[:-1]
        if "," in spec:
            start, end = spec.split(",", 1)
        else:
            start = spec
            end = spec
        return cls(weight=float(weight), start_index=int(start), end_index=int(end))

    def __str__(self):
        return f"{self.weight:.2f}[{self.start_index},{self.end_index}]"

    __repr__ = __str__


@dataclasses.dataclass
class Embeddings:
    ids: np.ndarray
    embeddings: np.ndarray

    @property
    def size(self):
        return self.embeddings.shape[0]

    @property
    def dims(self):
        return self.embeddings.shape[1]

    def project(self, codec_index, codec_str) -> "Embeddings":
        projected = codec_index.sa_encode(self.embeddings)
        projected = np.frombuffer(projected, dtype=np.float32).reshape(self.size, -1)
        if not np.isfinite(projected).all():
            raise ProjectionError(
                f"Projection to {codec_str} resulted in non-finite values"
            )
        return dataclasses.replace(self, embeddings=projected)


def dataset_split(outputs, split_id) -> Embeddings:
    split = outputs["split"]
    this_split = split == split_id
    embeddings = outputs["embeddings"][this_split, :]
    image_num = outputs["image_num"][this_split]
    order = np.argsort(image_num)
    embeddings = embeddings[order, :]
    image_num = image_num[order]
    return Embeddings(ids=image_num, embeddings=embeddings)


def evaluate_all(dataset, outputs, codecs_arg, score_norm_arg):
    embeddings = outputs["embeddings"]
    codecs = get_codecs(embeddings.shape[1], is_l2_normalized(embeddings), codecs_arg)
    logger.info("Using codecs: %s", codecs)
    score_norms = [None]
    if score_norm_arg:
        score_norms.extend(
            [ScoreNormalization.parse(spec) for spec in score_norm_arg.split(";")]
        )
    logger.info("Using score_norm: %s", score_norms)
    queries = dataset_split(outputs, DISCEvalDataset.SPLIT_QUERY)
    refs = dataset_split(outputs, DISCEvalDataset.SPLIT_REF)
    training = dataset_split(outputs, DISCEvalDataset.SPLIT_TRAIN)
    logger.info(
        "Dataset size: %d query, %d ref, %d train",
        queries.size,
        refs.size,
        training.size,
    )
    all_metrics = []
    for score_norm in score_norms:
        for codec in codecs:
            record = dict(codec=codec, score_norm=str(score_norm))
            metrics = evaluate(dataset, queries, refs, training, score_norm, codec)
            if metrics:
                record.update(metrics)
                all_metrics.append(record)
    return all_metrics


def project(
    codec_str: str, queries: Embeddings, refs: Embeddings, training: Embeddings
):
    if codec_str != "Flat":
        assert codec_str.endswith(",Flat")
        codec = faiss.index_factory(training.dims, codec_str)
        codec.train(training.embeddings)
        queries = queries.project(codec, codec_str)
        refs = refs.project(codec, codec_str)
        training = training.project(codec, codec_str)
    return queries, refs, training


def evaluate(
    dataset: DISCEvalDataset,
    queries: Embeddings,
    refs: Embeddings,
    training: Embeddings,
    score_norm: Optional[ScoreNormalization],
    codec,
):
    try:
        queries, refs, training = project(codec, queries, refs, training)
    except ProjectionError as e:
        logger.error(f"DISC eval {codec}: {e}")
        return None
    eval_kwargs = {}
    use_gpu = torch.cuda.is_available()
    if score_norm:
        queries, refs = apply_score_norm(
            queries, refs, training, score_norm, use_gpu=use_gpu
        )
        eval_kwargs["metric"] = faiss.METRIC_INNER_PRODUCT
    metrics = dataset.retrieval_eval_splits(
        queries.ids,
        queries.embeddings,
        refs.ids,
        refs.embeddings,
        use_gpu=use_gpu,
        **eval_kwargs,
    )
    logger.info(
        f"DISC eval ({score_norm or 'no norm'}, {codec}): {json.dumps(metrics)}"
    )
    return metrics


def apply_score_norm(
    queries, refs, training, score_norm: ScoreNormalization, use_gpu=False
):
    index = faiss.IndexFlatIP(training.dims)
    index.add(training.embeddings)
    if use_gpu:
        index = faiss.index_cpu_to_all_gpus(index)
    D, I = index.search(queries.embeddings, score_norm.end_index + 1)
    adjustment = -score_norm.weight * np.mean(
        D[:, score_norm.start_index : score_norm.end_index + 1],
        axis=1,
        keepdims=True,
    )
    ones = np.ones_like(refs.embeddings[:, :1])
    adjusted_queries = np.concatenate([queries.embeddings, adjustment], axis=1)
    adjusted_refs = np.concatenate([refs.embeddings, ones], axis=1)
    queries = dataclasses.replace(queries, embeddings=adjusted_queries)
    refs = dataclasses.replace(refs, embeddings=adjusted_refs)
    return queries, refs


def main(args):
    logger.info("Setting up dataset")
    dataset = DISCData.make_validation_dataset(
        args.disc_path,
        size=args.size,
        include_train=True,
        preserve_aspect_ratio=args.preserve_aspect_ratio,
    )
    outputs = Inference.inference(args, dataset)
    logger.info("Retrieval eval")
    records = evaluate_all(dataset, outputs, args.codecs, args.score_norm)
    df = pd.DataFrame(records)
    if args.metadata:
        df["metadata"] = args.metadata
    csv_filename = os.path.join(args.output_path, "disc_metrics.csv")
    df.to_csv(csv_filename, index=False)
    with open(csv_filename, "r") as f:
        logger.info("DISC metrics:\n%s", f.read())


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
