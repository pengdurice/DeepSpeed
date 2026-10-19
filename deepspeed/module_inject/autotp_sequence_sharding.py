# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Sequence sharding for AutoTP training (``tensor_parallel.sequence_sharding``), the scheme Megatron-LM calls
sequence parallelism. It is unrelated to DeepSpeed's Ulysses sequence parallelism (``sequence_parallel_size``),
which splits attention over ranks of its own.

Without it, every tensor-parallel rank runs the norms, residual adds and other token-wise ops of a decoder layer
on the full sequence and keeps their activations at full size. With it, the hidden states between the
tensor-parallel blocks (the attention and MLP modules) are split over the tensor-parallel ranks, 1/tp of the tokens
each (one contiguous chunk of the flattened batch and sequence):

    block input   [b, s/tp, h]  -- all-gather -->      [b, s, h]  -> column-parallel linears -> ...
    row output    [b, s, h] partial sums  -- reduce-scatter -->  [b, s/tp, h]

In the forward pass the all-gather and the reduce-scatter replace the row-parallel all-reduce. In the backward
pass, the reduce-scatter of the all-gather sums the input gradients of all the column-parallel linears of a block
at once, instead of one all-reduce per linear (Qwen3: q, k and v share one; gate and up share one).

Requirements, checked when it is applied:
  - every tensor-parallel linear inside the decoder layers belongs to a block: the parent module of a row-parallel
    linear, whose first tensor argument (or ``hidden_states``) is the input of its column-parallel linears;
  - every decoder layer of the layer list has at least one block;
  - the sequence length is divisible by the tensor-parallel size.
Every replicated parameter inside the decoder layers (norm weights, row-parallel biases) then sees only this rank's
part of the sequence or of the heads, so its gradient is summed over the tensor-parallel group.

With ``tensor_parallel.sequence_sharding_copy_engine`` the all-gathers and reduce-scatters of the blocks move data
with the GPU copy engines, overlapped with the GEMMs (deepspeed.comm.copy_engine), instead of NCCL kernels.
"""

import torch
from torch import nn

from deepspeed import comm as dist
from deepspeed.accelerator import get_accelerator
from deepspeed.comm import pg_utils
from deepspeed.runtime.zero.linear import autocast_custom_bwd, autocast_custom_fwd
from deepspeed.runtime.zero.utils import is_zero_param
from deepspeed.utils.logging import print_dist

SEQUENCE_SHARDING_ATTR = "_ds_autotp_sequence_sharding"


def _as_dtype(t, dtype):
    # Tensor.to is a dispatched op even when the dtype already matches; the copy-engine paths call it per block.
    return t if t.dtype == dtype else t.to(dtype)


def _shape_with(x, dim, size):
    return (*x.shape[:dim], size, *x.shape[dim + 1:])


def _check_divisible(x, dim, world_size):
    if x.shape[dim] % world_size != 0:
        raise ValueError(f"AutoTP sequence sharding: sequence length {x.shape[dim]} (dim {dim}) is not divisible by "
                         f"the tensor-parallel size {world_size}")


# The ops between blocks act on each token alone, so a rank may hold any 1/tp of the tokens. Each rank holds one
# contiguous chunk of the flattened [batch * sequence] tokens: the collectives then work on contiguous buffers and
# need no reordering copy for any batch size (splitting each sequence instead needs two copies per collective at
# batch > 1). The local tensor keeps the shape [..., sequence / tp, ...].


def _split(x, dim, world_size, rank):
    _check_divisible(x, dim, world_size)
    flat = x.contiguous().view(-1, x.shape[-1])
    chunk = flat.shape[0] // world_size
    return flat[rank * chunk:(rank + 1) * chunk].view(_shape_with(x, dim, x.shape[dim] // world_size))


def _all_gather(x, dim, group, world_size):
    x = x.contiguous()
    hidden = x.shape[-1]
    gathered = x.new_empty((world_size * (x.numel() // hidden), hidden))
    dist.all_gather_into_tensor(gathered, x.view(-1, hidden), group=group)
    return gathered.view(_shape_with(x, dim, x.shape[dim] * world_size))


def _reduce_scatter(x, dim, group, world_size, async_op=False):
    _check_divisible(x, dim, world_size)
    x = x.contiguous()
    hidden = x.shape[-1]
    out = x.new_empty((x.numel() // hidden // world_size, hidden))
    handle = dist.reduce_scatter_tensor(out, x.view(-1, hidden), group=group, async_op=async_op)
    out = out.view(_shape_with(x, dim, x.shape[dim] // world_size))
    return (out, handle) if async_op else out


class ScatterToSequenceShards(torch.autograd.Function):
    """Enter the sequence-sharded region: keep this rank's part of a replicated tensor; backward all-gathers."""

    @staticmethod
    def forward(ctx, x, group, dim):
        ctx.group, ctx.dim = group, dim
        ctx.world_size = dist.get_world_size(group=group)
        return _split(x, dim, ctx.world_size, dist.get_rank(group=group))

    @staticmethod
    def backward(ctx, grad):
        return _all_gather(grad, ctx.dim, ctx.group, ctx.world_size), None, None


class GatherFromSequenceShards(torch.autograd.Function):
    """Leave the sequence-sharded region: all-gather into a replicated tensor; backward keeps this rank's part.

    The ops after it (final norm, output head, loss) are replicated, so every rank receives the same full gradient.
    """

    @staticmethod
    def forward(ctx, x, group, dim):
        ctx.group, ctx.dim = group, dim
        ctx.world_size = dist.get_world_size(group=group)
        return _all_gather(x, dim, group, ctx.world_size)

    @staticmethod
    def backward(ctx, grad):
        return _split(grad, ctx.dim, ctx.world_size, dist.get_rank(group=ctx.group)), None, None


class GatherToTensorParallelBlock(torch.autograd.Function):
    """All-gather a block input for its column-parallel linears; backward reduce-scatters their partial gradients."""

    @staticmethod
    def forward(ctx, x, group, dim):
        ctx.group, ctx.dim = group, dim
        ctx.world_size = dist.get_world_size(group=group)
        return _all_gather(x, dim, group, ctx.world_size)

    @staticmethod
    def backward(ctx, grad):
        return _reduce_scatter(grad, ctx.dim, ctx.group, ctx.world_size), None, None


class ReduceScatterFromTensorParallelBlock(torch.autograd.Function):
    """Sum the row-parallel partial outputs and keep this rank's part of the sequence; backward all-gathers."""

    @staticmethod
    def forward(ctx, x, group, dim):
        ctx.group, ctx.dim = group, dim
        ctx.world_size = dist.get_world_size(group=group)
        return _reduce_scatter(x, dim, group, ctx.world_size)

    @staticmethod
    def backward(ctx, grad):
        return _all_gather(grad, ctx.dim, ctx.group, ctx.world_size), None, None


def reduce_scatter_from_tensor_parallel_block(x, group, dim):
    return ReduceScatterFromTensorParallelBlock.apply(x, group, dim)


class ColumnGroupFromSequenceShards(torch.autograd.Function):
    """All-gather a block input and run every column-parallel linear that reads it, in one autograd node.

    Each linear keeps its own GEMM, so its output stays contiguous for the reshapes that follow (one GEMM over the
    concatenated weights gives strided outputs that later kernels copy). Backward sums the input gradients of the
    linears and reduce-scatters them while the weight-gradient GEMMs run, instead of after them. Returns the
    gathered input (for the block's own use of its shape) and one output per weight.

    Only this rank's part of the input is saved for backward, which gathers it again while the input-gradient GEMMs
    run: saving the gathered input would keep tp times as much activation memory per block.
    """

    @staticmethod
    @autocast_custom_fwd
    def forward(ctx, x_shard, group, dim, copy_engine, backward_copy_engine, *weights):
        ctx.set_materialize_grads(False)
        ctx.group, ctx.dim, ctx.copy_engine = group, dim, backward_copy_engine
        ctx.world_size = dist.get_world_size(group=group)
        if copy_engine is not None:
            hidden = x_shard.shape[-1]
            full_shape = _shape_with(x_shard, dim, x_shard.shape[dim] * ctx.world_size)
            x_local = x_shard.contiguous().view(-1, hidden)
            x_full_2d, outs_2d = copy_engine.all_gather_matmul(x_local,
                                                               [_as_dtype(w.t(), x_local.dtype) for w in weights])
            x_full = x_full_2d.view(full_shape)
            outs = [out.view(*full_shape[:-1], out.shape[-1]) for out in outs_2d]
        else:
            x_full = _all_gather(x_shard, dim, group, ctx.world_size)
            outs = [torch.matmul(x_full, w.t()) for w in weights]
        ctx.save_for_backward(x_shard, *weights)
        return (x_full, *outs)

    @staticmethod
    @autocast_custom_bwd
    def backward(ctx, grad_x_full, *grad_outs):
        x_shard, *weights = ctx.saved_tensors
        hidden = x_shard.shape[-1]
        x_local = x_shard.contiguous().view(-1, hidden)
        full_shape = _shape_with(x_shard, ctx.dim, x_shard.shape[ctx.dim] * ctx.world_size)
        if ctx.copy_engine is not None:
            return ColumnGroupFromSequenceShards._backward_copy_engine(ctx, x_local, x_shard.shape, weights,
                                                                       grad_x_full, grad_outs)
        x_2d = x_local.new_empty((ctx.world_size * x_local.shape[0], hidden))
        gather = dist.all_gather_into_tensor(x_2d, x_local, group=ctx.group, async_op=True)
        # Sum of the input gradients of all the linears; addmm_ accumulates inside each GEMM.
        grad_in_2d = None
        for grad_out, weight in zip(grad_outs, weights):
            if grad_out is None:
                continue
            grad_out_2d = grad_out.reshape(-1, grad_out.shape[-1])
            if grad_in_2d is None:
                grad_in_2d = torch.matmul(grad_out_2d, weight)
            else:
                # In-place ops are not autocast: match dtypes by hand (fp32 weights under torch_autocast).
                grad_in_2d.addmm_(grad_out_2d.to(grad_in_2d.dtype), weight.to(grad_in_2d.dtype))
        if grad_in_2d is None:
            grad_in_2d = x_2d.new_zeros(x_2d.shape)
        grad_in = grad_in_2d.view(full_shape)
        if grad_x_full is not None:
            grad_in = grad_in + grad_x_full
        grad_shard, handle = _reduce_scatter(grad_in, ctx.dim, ctx.group, ctx.world_size, async_op=True)
        gather.wait()
        grad_weights = []
        for grad_out in grad_outs:
            if grad_out is None:
                grad_weights.append(None)
            else:
                grad_weights.append(torch.matmul(grad_out.reshape(-1, grad_out.shape[-1]).t(), x_2d))
        handle.wait()
        return (grad_shard, None, None, None, None, *grad_weights)

    @staticmethod
    def _backward_copy_engine(ctx, x_local, shard_shape, weights, grad_x_full, grad_outs):
        """The input gradient of each peer's rows is computed and copied to that peer while the next is computed;
        the input is gathered again first, on the same copy stream."""
        hidden = x_local.shape[-1]
        x_full_2d = ctx.copy_engine.all_gather_start(x_local)
        grads_2d = [None if g is None else g.reshape(-1, g.shape[-1]) for g in grad_outs]
        grad_x_2d = None if grad_x_full is None else grad_x_full.reshape(-1, hidden)
        dtype = next((g.dtype for g in grads_2d if g is not None), x_local.dtype)
        weights = [_as_dtype(w, dtype) for w in weights]

        def partial_into(rows, out):
            first = True
            for grad, weight in zip(grads_2d, weights):
                if grad is None:
                    continue
                if first:
                    torch.matmul(grad[rows], weight, out=out)
                    first = False
                else:
                    out.addmm_(grad[rows], weight)
            if first:
                out.zero_()
            if grad_x_2d is not None:
                out.add_(grad_x_2d[rows])

        rows_total = x_full_2d.shape[0]
        pending = ctx.copy_engine.matmul_reduce_scatter_start(rows_total, hidden, dtype, x_local.device, partial_into)
        ctx.copy_engine.all_gather_wait()
        x_2d = _as_dtype(x_full_2d, dtype)
        # Before the reduce-scatter's barrier, so that these GEMMs run while its copies finish.
        grad_weights = [None if g is None else torch.matmul(g.t(), x_2d) for g in grads_2d]
        grad_shard = ctx.copy_engine.matmul_reduce_scatter_finish(pending)
        return (grad_shard.view(shard_shape), None, None, None, None, *grad_weights)


class RowParallelToSequenceShards(torch.autograd.Function):
    """Row-parallel linear whose summed output goes to the rank that holds those tokens, on the copy engines in the
    forward pass, the backward pass or both (NCCL in the other).

    Forward: the partial result for each peer's rows is copied to that peer while the next one is computed.
    Backward: the output gradient is gathered while the input-gradient GEMM runs on the blocks already here.
    """

    @staticmethod
    @autocast_custom_fwd
    def forward(ctx, x, weight, group, dim, copy_engine, backward_copy_engine):
        ctx.group, ctx.dim, ctx.copy_engine = group, dim, backward_copy_engine
        ctx.world_size = world_size = dist.get_world_size(group=group)
        _check_divisible(x, dim, world_size)
        ctx.save_for_backward(x, weight)
        if copy_engine is None:
            return _reduce_scatter(torch.matmul(x, weight.t()), dim, group, world_size)
        x_2d = x.contiguous().view(-1, x.shape[-1])
        weight_t = _as_dtype(weight.t(), x_2d.dtype)

        def partial_into(rows, out):
            torch.matmul(x_2d[rows], weight_t, out=out)

        out = copy_engine.matmul_reduce_scatter(x_2d.shape[0], weight.shape[0], x_2d.dtype, x.device, partial_into)
        return out.view(*_shape_with(x, dim, x.shape[dim] // world_size)[:-1], weight.shape[0])

    @staticmethod
    @autocast_custom_bwd
    def backward(ctx, grad_shard):
        x, weight = ctx.saved_tensors
        if ctx.copy_engine is None:
            grad_full = _all_gather(grad_shard, ctx.dim, ctx.group, ctx.world_size)
            grad_in = torch.matmul(grad_full, _as_dtype(weight, grad_full.dtype))
            grad_full_2d = grad_full.reshape(-1, grad_full.shape[-1])
            grad_weight = torch.matmul(grad_full_2d.t(), _as_dtype(x.reshape(-1, x.shape[-1]), grad_full.dtype))
            return grad_in, grad_weight, None, None, None, None
        grad_local = grad_shard.contiguous().view(-1, grad_shard.shape[-1])
        grad_full, (grad_in, ) = ctx.copy_engine.all_gather_matmul(grad_local, [_as_dtype(weight, grad_local.dtype)])
        grad_weight = torch.matmul(grad_full.t(), _as_dtype(x.reshape(-1, x.shape[-1]), grad_full.dtype))
        return grad_in.view(x.shape), grad_weight, None, None, None, None


def row_parallel_to_sequence_shards(x, weight, group, dim, ce_group):
    """``x @ weight.t()`` summed over ``group`` into this rank's tokens, with the copy engines of ``ce_group`` in the
    passes where they pay off and NCCL in the others."""
    # The row output holds 1/tp of the input's tokens, at the full hidden size.
    tokens_per_rank = x.numel() // x.shape[-1] // ce_group.world_size
    hidden = weight.shape[0]
    copy_engine = ce_group if copy_engine_pays_off(tokens_per_rank, hidden, ce_group.min_tokens) else None
    backward_copy_engine = None
    if copy_engine_pays_off(tokens_per_rank, hidden, ce_group.min_tokens, backward=True):
        backward_copy_engine = ce_group
    if copy_engine is None and backward_copy_engine is None:
        return reduce_scatter_from_tensor_parallel_block(torch.matmul(x, weight.t()), group, dim)
    return RowParallelToSequenceShards.apply(x, weight, group, dim, copy_engine, backward_copy_engine)


class _BlockState:
    """Which column-parallel linears of a block read the block input, learned on the block's first call."""

    def __init__(self, columns):
        self.columns = columns
        self.readers = []
        self.calls = 0

    def can_fuse(self):
        weights = [layer.weight for layer in self.readers]
        if not weights or any(is_zero_param(w) or w.dim() != 2 for w in weights):
            return False
        return len({(w.dtype, w.device, w.shape[1]) for w in weights}) == 1


def take_precomputed_column_output(layer, input):
    """The output the block pre-hook already computed for ``layer`` from ``input``, or None.

    On the block's first call nothing is precomputed; a layer that reads the gathered block input records itself
    instead, so that later calls compute it inside ColumnGroupFromSequenceShards.
    """
    pending = getattr(layer, "_ds_sequence_sharding_pending", None)
    if pending is None:
        return None
    layer._ds_sequence_sharding_pending = None
    state, x_full, out = pending
    if x_full is not input:
        return None
    if out is None:
        state.readers.append(layer)
    return out


def _replace_hidden_states(args, kwargs, fn, where):
    if args and torch.is_tensor(args[0]):
        return (fn(args[0]), *args[1:]), kwargs
    if torch.is_tensor(kwargs.get("hidden_states")):
        kwargs = dict(kwargs)
        kwargs["hidden_states"] = fn(kwargs["hidden_states"])
        return args, kwargs
    raise RuntimeError(f"AutoTP sequence sharding: {where} was called without a tensor as its first argument or "
                       "as hidden_states")


def _layer_list_of(block_name):
    """'model.layers.3.self_attn' -> ('model.layers', 3)."""
    parts = block_name.split(".")
    for i in range(len(parts) - 1, -1, -1):
        if parts[i].isdigit():
            return ".".join(parts[:i]), int(parts[i])
    return None, None


# With the automatic minimum, the copy engines are used when tokens per rank x hidden x hidden (the multiply-adds of
# one [tokens, hidden] x [hidden, hidden] GEMM) is at least this. Below it, the GEMM on each peer's rows is too short
# to hide the copies, events and barrier of the operation. On 8 x H200 the forward pass was slower than NCCL at 34.4e9
# (Qwen3-8B, 2,048 tokens per rank) and 26.8e9 (Qwen3-32B, 1,024) and faster from 53.7e9; tokens x hidden alone does
# not separate the cases. The backward pass has more GEMM work per collective and was faster from 34.4e9 (slower at
# 17.2e9).
COPY_ENGINE_MIN_MULTIPLY_ADDS = 45e9
COPY_ENGINE_MIN_MULTIPLY_ADDS_BACKWARD = 25e9


def copy_engine_pays_off(tokens, hidden, min_tokens, backward=False):
    if min_tokens >= 0:
        return tokens >= min_tokens
    minimum = COPY_ENGINE_MIN_MULTIPLY_ADDS_BACKWARD if backward else COPY_ENGINE_MIN_MULTIPLY_ADDS
    return tokens * hidden * hidden >= minimum


def _copy_engine_for(x, ce_group, backward=False):
    """``ce_group`` when this rank's part of ``x`` is large enough for the copy engines in the forward (or the
    backward) pass, else None (the NCCL path)."""
    if ce_group is None:
        return None
    tokens = x.numel() // x.shape[-1]
    return ce_group if copy_engine_pays_off(tokens, x.shape[-1], ce_group.min_tokens, backward) else None


def _make_copy_engine_group(group, min_tokens):
    """A CopyEngineGroup for ``group``, or None when the copy engines cannot be used (the NCCL path)."""
    from deepspeed.comm.copy_engine import CopyEngineGroup, copy_engine_available, group_within_one_node

    if not copy_engine_available() or not group_within_one_node(group):
        print_dist(
            "AutoTP sequence sharding: copy-engine collectives need symmetric memory and a tensor-parallel "
            "group within one node; using NCCL",
            ranks=[0])
        return None
    ce_group = None
    try:
        ce_group = CopyEngineGroup(group)
        ce_group.min_tokens = min_tokens
        # A first allocation checks that the ranks can map each other's memory.
        ce_group.gather_buffers.next(1, torch.bfloat16, torch.device(get_accelerator().current_device_name()))
    except Exception as exc:  # noqa: BLE001 -- keep the NCCL collectives
        print_dist(f"AutoTP sequence sharding: copy-engine collectives unavailable ({exc}); using NCCL", ranks=[0])
        ce_group = None
    # Every rank of the group must take the same path: agree on the copy engines only if all ranks have them.
    usable = torch.tensor([1 if ce_group is not None else 0], device=get_accelerator().current_device_name())
    dist.all_reduce(usable, op=dist.ReduceOp.MIN, group=group)
    return ce_group if usable.item() == 1 else None


def apply_autotp_sequence_sharding(model,
                                   group,
                                   seq_dim=1,
                                   overlap=True,
                                   copy_engine=False,
                                   copy_engine_min_tokens=-1):
    """Split the hidden states between AutoTP blocks along ``seq_dim`` over ``group``. Call after AutoTP has
    replaced the linears. ``overlap``: run the column-parallel linears that read a block input in one autograd node
    and overlap their backward reduce-scatter (ColumnGroupFromSequenceShards). ``copy_engine`` (with ``overlap``):
    move the all-gathers and reduce-scatters of the blocks with the GPU copy engines, overlapped with the GEMMs
    (deepspeed.comm.copy_engine), in the passes where this rank holds at least ``copy_engine_min_tokens`` tokens of
    the micro-batch (negative: tokens x hidden x hidden at least COPY_ENGINE_MIN_MULTIPLY_ADDS forward and
    COPY_ENGINE_MIN_MULTIPLY_ADDS_BACKWARD backward). Returns the number of blocks."""
    from deepspeed.module_inject.layers import (DS_TENSOR_MODEL_PARALLEL, LinearAllreduce, LinearLayer,
                                                TensorParallel_Layer)

    if not pg_utils.is_parallel(group) or getattr(model, SEQUENCE_SHARDING_ATTR, False):
        return 0

    ce_group = _make_copy_engine_group(group, copy_engine_min_tokens) if copy_engine and overlap else None

    row_layers = {name: m for name, m in model.named_modules() if isinstance(m, LinearAllreduce)}
    if not row_layers:
        raise ValueError("AutoTP sequence sharding found no row-parallel linear in the model")
    blocks = {}
    for name in row_layers:
        if "." not in name:
            raise ValueError(f"AutoTP sequence sharding: row-parallel linear {name!r} has no parent module")
        parent = name.rsplit(".", 1)[0]
        blocks[parent] = model.get_submodule(parent)

    list_names = {_layer_list_of(name)[0] for name in blocks}
    if len(list_names) != 1 or None in list_names:
        raise ValueError("AutoTP sequence sharding needs all tensor-parallel blocks inside one numbered layer list; "
                         f"found blocks {sorted(blocks)}")
    list_name = list_names.pop()
    layers = model.get_submodule(list_name)
    if not isinstance(layers, nn.ModuleList) or len(layers) == 0:
        raise ValueError(f"AutoTP sequence sharding: {list_name!r} is not a non-empty nn.ModuleList")
    layers_with_blocks = {_layer_list_of(name)[1] for name in blocks}
    missing = sorted(set(range(len(layers))) - layers_with_blocks)
    if missing:
        raise ValueError(f"AutoTP sequence sharding: decoder layers {missing} of {list_name!r} have no "
                         "tensor-parallel block; their ops would see a split sequence")

    # Every tensor-parallel layer inside the decoder layers must belong to a block, and column outputs must stay
    # sharded (a gathered column output would mix a split sequence and a full one).
    for name, m in layers.named_modules(prefix=list_name):
        if not isinstance(m, TensorParallel_Layer):
            continue
        if not any(name.startswith(block + ".") for block in blocks):
            raise ValueError(f"AutoTP sequence sharding: tensor-parallel layer {name!r} is outside every block")
        if getattr(m, "defer_collectives_to_compiler", False):
            raise NotImplementedError("AutoTP sequence sharding does not support the DeepCompile 'autotp' pass")
        if isinstance(m, LinearLayer) and getattr(m, "gather_output", False):
            raise NotImplementedError(f"AutoTP sequence sharding: column-parallel layer {name!r} gathers its output")
        # The sequence-sharding paths live in LinearLayer.forward and LinearAllreduce.forward; a subclass with its
        # own forward would keep its all-reduce and reduce the gradients twice.
        if type(m).forward not in (LinearLayer.forward, LinearAllreduce.forward):
            raise NotImplementedError(f"AutoTP sequence sharding does not support {type(m).__name__} ({name!r})")
        m.sequence_sharding = True
        m.sequence_sharding_dim = seq_dim
        m.sequence_sharding_copy_engine = ce_group

    def make_gather_hook(state):

        def gather(x):
            state.calls += 1
            if overlap and state.calls > 1 and state.can_fuse():
                readers = state.readers
                weights = [layer.weight for layer in readers]
                copy_engine = _copy_engine_for(x, ce_group)
                backward_copy_engine = _copy_engine_for(x, ce_group, backward=True)
                x_full, *outs = ColumnGroupFromSequenceShards.apply(x, group, seq_dim, copy_engine,
                                                                    backward_copy_engine, *weights)
                for layer, layer_out in zip(readers, outs):
                    layer._ds_sequence_sharding_pending = (state, x_full, layer_out)
                return x_full
            x_full = GatherToTensorParallelBlock.apply(x, group, seq_dim)
            if overlap and state.calls == 1:
                for layer in state.columns:
                    layer._ds_sequence_sharding_pending = (state, x_full, None)
            return x_full

        def hook(module, args, kwargs):
            return _replace_hidden_states(args, kwargs, gather, type(module).__name__)

        return hook

    for block_name, block in blocks.items():
        columns = [
            m for name, m in block.named_modules(prefix=block_name)
            if isinstance(m, LinearLayer) and getattr(m, "sequence_sharding", False)
        ]
        block.register_forward_pre_hook(make_gather_hook(_BlockState(columns)), with_kwargs=True)

    def enter(module, args, kwargs):
        return _replace_hidden_states(args, kwargs, lambda x: ScatterToSequenceShards.apply(x, group, seq_dim),
                                      type(module).__name__)

    def leave(module, args, output):
        if torch.is_tensor(output):
            return GatherFromSequenceShards.apply(output, group, seq_dim)
        if isinstance(output, tuple) and output and torch.is_tensor(output[0]):
            return (GatherFromSequenceShards.apply(output[0], group, seq_dim), *output[1:])
        raise RuntimeError(f"AutoTP sequence sharding: {type(module).__name__} returned {type(output).__name__}")

    layers[0].register_forward_pre_hook(enter, with_kwargs=True)
    layers[-1].register_forward_hook(leave)

    # Replicated parameters in the decoder layers see this rank's part of the sequence (norms, row biases) or of
    # the heads (q_norm/k_norm, which AutoTP may already reduce), so their gradients are partial sums.
    def make_hook():

        def hook(grad):
            grad = grad.contiguous()
            dist.all_reduce(grad, group=group)
            return grad

        return hook

    reduced = []
    for name, param in layers.named_parameters(prefix=list_name):
        if not param.requires_grad or getattr(param, DS_TENSOR_MODEL_PARALLEL, False):
            continue
        if getattr(param, "_ds_grad_allreduce_registered", False):
            continue
        param.register_hook(make_hook())
        param._ds_grad_allreduce_registered = True
        reduced.append(name)

    setattr(model, SEQUENCE_SHARDING_ATTR, True)
    print_dist(
        f"AutoTP sequence sharding: {len(blocks)} blocks in {len(layers)} layers of {list_name!r}, sequence dim "
        f"{seq_dim}, overlap={overlap}, copy engine={ce_group is not None} (minimum tokens per rank: "
        f"{copy_engine_min_tokens if copy_engine_min_tokens >= 0 else 'automatic'}); "
        f"tensor-parallel gradient sum for {len(reduced)} replicated parameters"
        f"{', e.g. ' + repr(reduced[0]) if reduced else ''}",
        ranks=[0])
    return len(blocks)
