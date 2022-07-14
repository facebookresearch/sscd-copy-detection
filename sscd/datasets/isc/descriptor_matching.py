# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Original source: https://github.com/facebookresearch/isc2021

import numpy as np
import faiss
from faiss.contrib import exhaustive_search
import logging

from .metrics import PredictedMatch


def query_iterator(xq):
    """produces batches of progressively increasing sizes"""
    nq = len(xq)
    bs = 32
    i = 0
    while i < nq:
        xqi = xq[i : i + bs]
        yield xqi
        if bs < 20000:
            bs *= 2
        i += len(xqi)


#########################
# These two functions are there because current Faiss contrib
# does not proporly support IP search
#########################


def threshold_radius_nres_IP(nres, dis, ids, thresh):
    """select a set of results"""
    mask = dis > thresh
    new_nres = np.zeros_like(nres)
    o = 0
    for i, nr in enumerate(nres):
        nr = int(nr)  # avoid issues with int64 + uint64
        new_nres[i] = mask[o : o + nr].sum()
        o += nr
    return new_nres, dis[mask], ids[mask]


def apply_maxres_IP(res_batches, target_nres):
    """find radius that reduces number of results to target_nres, and
    applies it in-place to the result batches used in range_search_max_results"""
    alldis = np.hstack([dis for _, dis, _ in res_batches])
    alldis.partition(len(alldis) - target_nres)
    radius = alldis[-target_nres]

    LOG = logging.getLogger(exhaustive_search.__name__)

    if alldis.dtype == "float32":
        radius = float(radius)
    else:
        radius = int(radius)
    LOG.debug("   setting radius to %s" % radius)
    totres = 0
    for i, (nres, dis, ids) in enumerate(res_batches):
        nres, dis, ids = threshold_radius_nres_IP(nres, dis, ids, radius)
        totres += len(dis)
        res_batches[i] = nres, dis, ids
    LOG.debug("   updated previous results, new nb results %d" % totres)
    return radius, totres


def search_with_capped_res(xq, xb, num_results, metric=faiss.METRIC_L2):
    """
    Searches xq into xb, with a maximum total number of results
    """
    index = faiss.IndexFlat(xb.shape[1], metric)
    index.add(xb)
    # logging.basicConfig()
    # logging.getLogger(exhaustive_search.__name__).setLevel(logging.DEBUG)

    if metric == faiss.METRIC_INNER_PRODUCT:
        # this is a very ugly hack because contrib.exhaustive_search does
        # not support IP search correctly. Do not use in a multithreaded env.
        apply_maxres_saved = exhaustive_search.apply_maxres
        exhaustive_search.apply_maxres = apply_maxres_IP

    radius, lims, dis, ids = exhaustive_search.range_search_max_results(
        index,
        query_iterator(xq),
        1e10
        if metric == faiss.METRIC_L2
        else -1e10,  # initial radius does not filter anything
        max_results=2 * num_results,
        min_results=num_results,
        ngpu=-1,  # use GPU if available
    )

    if metric == faiss.METRIC_INNER_PRODUCT:
        exhaustive_search.apply_maxres = apply_maxres_saved

    n = len(dis)
    nq = len(xq)
    if n > num_results:
        # crop to num_results exactly
        if metric == faiss.METRIC_L2:
            o = dis.argpartition(num_results)[:num_results]
        else:
            o = dis.argpartition(len(dis) - num_results)[-num_results:]
        mask = np.zeros(n, bool)
        mask[o] = True
        new_dis = dis[mask]
        new_ids = ids[mask]
        nres = [0] + [mask[lims[i] : lims[i + 1]].sum() for i in range(nq)]
        new_lims = np.cumsum(nres)
        lims, dis, ids = new_lims, new_dis, new_ids

    return lims, dis, ids


def match_and_make_predictions(
    xq, query_image_ids, xb, db_image_ids, num_results, ngpu=-1, metric=faiss.METRIC_L2
):
    lims, dis, ids = search_with_capped_res(xq, xb, num_results, metric=metric)
    nq = len(xq)

    if metric == faiss.METRIC_L2:
        # use negated distances as scores
        dis = -dis

    predictions = [
        PredictedMatch(query_image_ids[i], db_image_ids[ids[j]], dis[j])
        for i in range(nq)
        for j in range(lims[i], lims[i + 1])
    ]
    return predictions


def knn_match_and_make_predictions(
    xq, query_image_ids, xb, db_image_ids, k, ngpu=-1, metric=faiss.METRIC_L2
):
    if ngpu == 0 or faiss.get_num_gpus() == 0:
        D, I = faiss.knn(xq, xb, k, metric)
    else:
        d = xq.shape[1]
        index = faiss.IndexFlat(d, metric)
        index.add(xb)
        index = faiss.index_cpu_to_all_gpus(index)
        D, I = index.search(xq, k=k)
    nq = len(xq)

    if metric == faiss.METRIC_L2:
        # use negated distances as scores
        D = -D

    predictions = [
        PredictedMatch(query_image_ids[i], db_image_ids[I[i, j]], D[i, j])
        for i in range(nq)
        for j in range(k)
    ]
    return predictions


def range_result_read(fname):
    """read the range search result file format"""
    f = open(fname, "rb")
    nq, total_res = np.fromfile(f, count=2, dtype="int32")
    nres = np.fromfile(f, count=nq, dtype="int32")
    assert nres.sum() == total_res
    I = np.fromfile(f, count=total_res, dtype="int32")
    return nres, I
