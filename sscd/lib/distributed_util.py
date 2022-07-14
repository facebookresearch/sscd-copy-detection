# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import enum

import torch
from classy_vision.generic.distributed_util import (
    all_reduce_mean,
    all_reduce_sum,
    get_world_size,
    get_rank,
)
from torch import autograd, distributed


def multi_gather_batch(*tensors):
    """Gather tensors across nodes / GPUs.

    Tensors must have the same shape on all devices. Gathering for each
    tensor happens in parallel.
    """

    world_size = distributed.get_world_size()
    out = []
    handles = []

    for tensor in tensors:
        gathered_shape = (world_size * tensor.shape[0],) + tensor.shape[1:]

        gathered = torch.empty(gathered_shape, dtype=tensor.dtype, device=tensor.device)
        # Non-contiguous tensors seem to get scrambled. Source and dest memory layouts
        # may have to match.
        tensor = tensor.contiguous()
        handle = distributed.all_gather(
            list(torch.chunk(gathered, world_size)), tensor, async_op=True
        )

        out.append(gathered)
        handles.append(handle)

    for handle in handles:
        handle.wait()

    return out


class ReduceMethod(enum.Enum):
    SUM = enum.auto()
    MEAN = enum.auto()
    # No gradient aggregation (eg. where all GPUs compute the same loss)
    NONE = enum.auto()


class _CrossGPUBatch(autograd.Function):
    """Aggregates embeddings and labels across GPUs.

    This requires that batches have the same size on each GPU.
    """

    @staticmethod
    def forward(ctx, embeddings, target, reduce_method: ReduceMethod):
        ctx.n = embeddings.size(0)
        ctx.reduce_method = reduce_method
        ctx.world_size = get_world_size()
        if target is None:
            if ctx.world_size == 1:
                return embeddings
            else:
                return multi_gather_batch(embeddings)[0]

        assert ctx.n == target.size(0)
        if ctx.world_size == 1:
            ctx.mark_non_differentiable(target)
            return embeddings, target
        all_embeddings, all_target = multi_gather_batch(embeddings, target)
        ctx.mark_non_differentiable(all_target)
        return all_embeddings, all_target

    @staticmethod
    def backward(ctx, all_embeddings_gradient, ignored_target_grad=None):
        if ctx.world_size == 1:
            embeddings_gradient = all_embeddings_gradient
        else:
            # Aggregate gradients across nodes.
            if ctx.reduce_method == ReduceMethod.MEAN:
                all_reduce_mean(all_embeddings_gradient)
            elif ctx.reduce_method == ReduceMethod.SUM:
                all_reduce_sum(all_embeddings_gradient)
            else:
                # Do not accumulate.
                assert ctx.reduce_method == ReduceMethod.NONE
            rank = get_rank()
            start = ctx.n * rank
            end = start + ctx.n
            # Slice gradient for embeddings that belong to this node.
            embeddings_gradient = all_embeddings_gradient[start:end]
        return (embeddings_gradient, None, None)


cross_gpu_batch = _CrossGPUBatch.apply


def cross_gpu_batch(embeddings, targets, reduce_method=ReduceMethod.SUM):
    return _CrossGPUBatch.apply(embeddings, targets, reduce_method)
