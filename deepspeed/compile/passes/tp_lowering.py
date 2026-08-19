# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Post-grad lowering of the AutoTP marker collectives into functional collectives.

pass_insert_tp_collectives inserts `autotp::copy_to_tp_region` / `autotp::reduce_from_tp_region`
before AOT autograd, because that is where autograd can still derive the matching backward. Those
are opaque `torch.library.custom_op`s: they run a blocking `dist.all_reduce`, so the compute stream
re-serializes on the collective the moment it is launched, and Inductor cannot see them at all: its
`is_collective` check requires an `ir._CollectiveKernel`, and a `torch.library.custom_op` lowers to
a plain `FallbackKernel` -- which is _CollectiveKernel's base class, not a subclass of it, so it
never qualifies. (The check is an exact `type(...) ==` comparison up to torch 2.8 and an isinstance
test from 2.9; either way an opaque custom op fails it.)

This pass runs after AOT autograd, on both the forward and the backward graph, and rewrites them
into `_c10d_functional.all_reduce` + a separate `wait_tensor`. That single change:

  * makes the nodes visible to Inductor's `sink_waits` / `raise_comms` /
    `reorder_compute_for_overlap` scheduling passes, which are what actually open the overlap
    window. An earlier version of this pass also sank the waits itself at the FX level; measurement
    showed it moved nothing (0% overlap without Inductor's scheduler), so it was removed;
  * makes them visible to `_micro_pipeline_tp`, which is why this is registered as
    `post_grad_custom_pre_pass` (Inductor runs custom_pre_pass, then micro_pipeline_tp, then
    custom_post_pass).

It also deletes `copy_to_tp_region` from the forward graph. That op is an identity whose only
purpose is to carry a backward formula; AOT autograd has already extracted the backward by the time
this runs, so all that is left is a full-activation-sized clone the custom-op aliasing rules forced.
"""

import torch
from torch.fx import Graph, GraphModule

COPY_TO_TP = torch.ops.autotp.copy_to_tp_region.default
REDUCE_FROM_TP = torch.ops.autotp.reduce_from_tp_region.default

ALL_REDUCE = torch.ops._c10d_functional.all_reduce.default
WAIT_TENSOR = torch.ops._c10d_functional.wait_tensor.default

# Every graph this pass rewrites appends its counters here. Kept so tests and benchmarks can assert
# the lowering actually fired rather than inferring it from a timing difference.
LOWERING_STATS = []


def _group_name() -> str:
    from torch.distributed._functional_collectives import _resolve_group_name
    from deepspeed.compile.custom_ops.tp_collectives import get_tp_group
    return _resolve_group_name(get_tp_group())


def _drop_identity_copies(graph: Graph) -> int:
    """Remove copy_to_tp_region nodes; forward-identity, and their backward is already derived."""
    removed = 0
    for node in list(graph.nodes):
        if node.op == "call_function" and node.target is COPY_TO_TP:
            node.replace_all_uses_with(node.args[0])
            graph.erase_node(node)
            removed += 1
    return removed


def _lower_reduces(graph: Graph, group_name: str) -> int:
    """reduce_from_tp_region -> all_reduce + wait_tensor as two independent nodes."""
    lowered = 0
    for node in list(graph.nodes):
        if node.op != "call_function" or node.target is not REDUCE_FROM_TP:
            continue
        src = node.args[0]
        with graph.inserting_before(node):
            started = graph.call_function(ALL_REDUCE, args=(src, "sum", group_name))
            waited = graph.call_function(WAIT_TENSOR, args=(started, ))
        for meta_key in ("val", "tensor_meta"):
            if meta_key in node.meta:
                started.meta[meta_key] = node.meta[meta_key]
                waited.meta[meta_key] = node.meta[meta_key]
        node.replace_all_uses_with(waited)
        graph.erase_node(node)
        lowered += 1
    return lowered


def lower_tp_collectives(gm: GraphModule) -> dict:
    """Inductor post_grad_custom_pre_pass entry point. Mutates the graph in place."""
    graph = gm.graph if isinstance(gm, GraphModule) else gm
    stats = {"dropped_copies": _drop_identity_copies(graph)}
    reduces = sum(1 for n in graph.nodes if n.op == "call_function" and n.target is REDUCE_FROM_TP)
    if reduces:
        stats["lowered"] = _lower_reduces(graph, _group_name())
    else:
        stats["lowered"] = 0
    if stats["dropped_copies"] or stats["lowered"]:
        graph.lint()
        if isinstance(gm, GraphModule):
            gm.recompile()
        LOWERING_STATS.append(stats)
    return stats
