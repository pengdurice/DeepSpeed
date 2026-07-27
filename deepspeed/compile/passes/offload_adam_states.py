# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import copy
import os
import time
from typing import List, Tuple

import torch
from torch.fx import Graph, GraphModule

from deepspeed.accelerator import get_accelerator
from deepspeed.runtime.zero.offload_states import _make_offload_state_key

try:
    from torch._subclasses.fake_tensor import unset_fake_temporarily
except ImportError:
    # Unsupported torch version
    pass

try:
    from torch._higher_order_ops.effects import _EffectType, _register_effectful_op
except ImportError:
    # Torch without the effects registry: the ops then survive inductor only through the
    # scheduler-DCE patch in inductor.py (see register_offload_ops for why that matters).
    _register_effectful_op = None

from ..profilers import ProfilingResult
from ..graph_param import DSGraphParamManager
from ..fx import move_primals_to_head
from .contract import PassContract

import deepspeed.comm as dist

NAME = "offload_adam_states"
# Offloads optimizer state and does not depend on the graph-rewriting passes.
CONTRACT = PassContract()


def print_r0(msg):
    if dist.get_rank() == 0:
        print(msg)


MARGIN = 0.2

# Optional floor on how many pieces the planner offloads. The memory-necessity count
# computed from the budget walk can raise the final number but never drops below this.
# Extra pieces beyond necessity are the lever of the combined schedule: each one frees
# its bytes for prefetch/selective gather to spend. None keeps budget-only planning.
offload_piece_target = None

copy_stream = None
offload_event = None
reload_event = None

max_memory = 0


def lazy_init():
    global copy_stream
    global offload_event
    global reload_event

    if copy_stream is None:

        copy_stream = get_accelerator().Stream()
        offload_event = get_accelerator().Event()
        reload_event = get_accelerator().Event()


optimizer = None
device = None
nz3 = None


def set_offload_piece_target(target):
    global offload_piece_target
    offload_piece_target = target


def _resolve_piece_target():
    env_target = os.environ.get("DS_DC_OFFLOAD_PIECE_TARGET")
    if env_target is not None:
        return int(env_target)
    return offload_piece_target


def move_key(state, key, key_event=None):
    # Nothing to copy when the key is already offloaded (e.g. a second offload call in the
    # same phase); check before touching state[key] so the pinned-buffer setup cannot raise.
    if key not in state:
        return
    offload_buf_key = _make_offload_state_key(key)
    if offload_buf_key not in state:
        state[offload_buf_key] = get_accelerator().pin_memory(torch.empty_like(state[key], device="cpu"))

    with get_accelerator().stream(copy_stream):
        state[offload_buf_key].copy_(state[key], non_blocking=True)
        # Callers may drop their reference to state[key] without waiting for this copy; keep the
        # allocator from recycling the block until copy_stream has finished reading it.
        if state[key].device.type != "cpu":
            state[key].record_stream(copy_stream)

    if key_event is None:
        offload_event.record(stream=copy_stream)
    else:
        key_event.record(stream=copy_stream)


def _wait_for_compute_before_write(compute_stream):
    # The caching allocator hands out blocks freed on the compute stream while that
    # stream's last kernel may still be reading them: same-stream reuse is safe because a
    # later kernel cannot overtake the earlier one, but a write from copy_stream can.
    # Reload buffers come from exactly such blocks, so a reload placed mid-backward would
    # otherwise overwrite a live activation and produce NaN losses (measured: partial
    # offload with reloads placed at mid-graph nodes went NaN one step after the pass
    # engaged, while runs whose reloads landed at the end of backward were unaffected).
    copy_stream.wait_stream(compute_stream)


def _alloc_reload_buffer(like_tensor):
    # Allocate the reload destination in the pool that has room at the moment the reload
    # runs -- measured both ways to fail otherwise. Default placement reloads at the end
    # of backward, right after the activations were freed from the compute stream's pool:
    # allocate there so the buffers recycle those blocks (forcing them into the copy
    # stream's pool grew it by ~100 driver requests/step at seq 2400). The experimental
    # early-reload path (DS_DC_RELOAD_EARLY) fires at backward START, when activations
    # still fill the compute pool: allocate on copy_stream so the buffers recycle the
    # morning's freed copy blocks instead (the compute-pool form thrashed there).
    if os.environ.get("DS_DC_RELOAD_EARLY") == "1":
        with get_accelerator().stream(copy_stream):
            return torch.empty_like(like_tensor, device=device)
    return torch.empty_like(like_tensor, device=device)


def move_back_key(state, key, key_event=None):
    # The buffer is written on copy_stream; record_stream keeps the allocator from
    # recycling it for compute-pool work until the copy has finished. Its later
    # compute-stream reads need no extra guard: they are ordered before the next
    # offload's D2H copy by the launch op's wait_stream, and the D2H read is protected
    # at free time by move_key's record_stream.
    compute_stream = get_accelerator().current_stream()
    buf = _alloc_reload_buffer(state[_make_offload_state_key(key)])
    with get_accelerator().stream(copy_stream):
        _wait_for_compute_before_write(compute_stream)
        buf.copy_(state[_make_offload_state_key(key)], non_blocking=True)
    buf.record_stream(copy_stream)
    state[key] = buf

    if key_event is None:
        reload_event.record(stream=copy_stream)
    else:
        key_event.record(stream=copy_stream)


def move_hp_param(src_tensor, dest_buf, key_event=None):
    with get_accelerator().stream(copy_stream):
        dest_buf.copy_(src_tensor, non_blocking=True)
        # The .data rebind drops the only reference to the GPU storage while the async copy may
        # still be reading it; keep the allocator from recycling it until copy_stream catches up.
        # Skip tensors already offloaded to host memory (e.g. re-offload after a profiling run,
        # where reloads are skipped): there is no GPU storage at risk and CPU tensors do not
        # support record_stream.
        if src_tensor.device.type != "cpu":
            src_tensor.record_stream(copy_stream)
        src_tensor.data = dest_buf

    if key_event is None:
        reload_event.record(stream=copy_stream)
    else:
        key_event.record(stream=copy_stream)


def move_back_hp_param(src_tensor, dest_buf, key_event=None):
    # Same allocation/ownership discipline and safety argument as move_back_key.
    compute_stream = get_accelerator().current_stream()
    buf = _alloc_reload_buffer(src_tensor)
    with get_accelerator().stream(copy_stream):
        _wait_for_compute_before_write(compute_stream)
        buf.copy_(src_tensor, non_blocking=True)
    buf.record_stream(copy_stream)
    dest_buf.data = buf

    if key_event is None:
        reload_event.record(stream=copy_stream)
    else:
        key_event.record(stream=copy_stream)


def offload_adam_states_sync():

    with unset_fake_temporarily():

        if not hasattr(optimizer, "hp_params_pin_buffers"):
            optimizer.hp_params_pin_buffers = [
                get_accelerator().pin_memory(torch.empty_like(t, device="cpu"))
                for t in optimizer.fp32_partitioned_groups_flat
            ]

        for i, (k, state) in enumerate(optimizer.state.items()):
            if "exp_avg" in state:
                move_key(state, "exp_avg")
            if "exp_avg_sq" in state:
                move_key(state, "exp_avg_sq")

        for _, state in optimizer.state.items():
            if "exp_avg" in state:
                del state["exp_avg"]
            if "exp_avg_sq" in state:
                del state["exp_avg_sq"]

        for src_tensor, dest_buf in zip(optimizer.fp32_partitioned_groups_flat, optimizer.hp_params_pin_buffers):
            move_hp_param(src_tensor, dest_buf)

        get_accelerator().synchronize()


def reload_adam_states_sync():

    with unset_fake_temporarily():

        for _, state in optimizer.state.items():
            if _make_offload_state_key("exp_avg") in state:
                move_back_key(state, "exp_avg")
            if _make_offload_state_key("exp_avg_sq") in state:
                move_back_key(state, "exp_avg_sq")

        for src, dest in zip(optimizer.hp_params_pin_buffers, optimizer.fp32_partitioned_groups_flat):
            move_back_hp_param(src, dest)

        get_accelerator().synchronize()


def sync_offload_states(event=None):
    if nz3.is_profiling():
        offload_adam_states_sync()
    else:
        if event is None:
            offload_event.wait(copy_stream)
        else:
            event.wait(copy_stream)


def sync_reload_states(event=None):
    if nz3.is_profiling():
        reload_adam_states_sync()
    else:
        if event is None:
            reload_event.wait(copy_stream)
        else:
            event.wait(copy_stream)


# The offload/reload work used to be inserted into FX graphs as Python closures.
# That works in eager mode (FX just calls the objects) but not under inductor:
# FxGraphCache serializes call_function targets by qualified name (a closure's
# `<locals>` qualname is unimportable), and lowering requires a registered op
# with a schema and a Meta kernel. The dc.* ops below carry only an int index
# into this registry; the task tuples (which hold live tensors) never cross the
# op boundary. The anchor tensor argument is ignored at runtime -- it exists so
# the dispatcher and fake-tensor tracing have a tensor to route on (see the
# "Undefined" fallback for end_backward in csrc/compile/init.cpp for what
# happens without one).
_op_task_registry = []
_offload_ops_lib = None

# Rank-local counts of op executions. Launches also run during pass-time memory profiling;
# reloads are skipped while profiling, so "reloads" only counts real training steps. Tests use
# these to assert the offload machinery actually ran: if any dead-code elimination silently
# dropped the ops from the compiled graph (see register_offload_ops), training would proceed
# with resident states and identical loss -- these counters are the only cheap detector.
_offload_op_stats = {"launches": 0, "reloads": 0}


def get_offload_op_stats():
    return dict(_offload_op_stats)


def reset_offload_op_stats():
    for key in _offload_op_stats:
        _offload_op_stats[key] = 0


def _register_op_task(task) -> int:
    _op_task_registry.append(task)
    return len(_op_task_registry) - 1


def _offload_opt_launch_impl(anchor, idx):
    _offload_op_stats["launches"] += 1
    task = _op_task_registry[idx]
    # The states were last written by the optimizer step on the compute stream; make the
    # D2H reads wait for those writes. Stream-level dependency only, no host wait.
    copy_stream.wait_stream(get_accelerator().current_stream())
    if task[2] == "hp_param":
        move_hp_param(task[1][0], task[1][1])
    else:
        assert task[1] in optimizer.state, f"State {task[1]} not found in optimizer"
        state = optimizer.state[task[1]]
        move_key(state, task[2])
        # move_key record_stream'd the source on copy_stream, so the allocator will not hand the
        # block out again until the copy completes. Drop the reference right here instead of at a
        # separately placed host-blocking sync node: waiting on the copy event from the launch
        # thread stalled the whole kernel-submission pipeline for the duration of the drain.
        if task[2] in state:
            del state[task[2]]


def _reload_opt_impl(anchor, idx):
    if nz3.is_profiling():
        return

    _offload_op_stats["reloads"] += 1
    task = _op_task_registry[idx]
    if task[2] == "hp_param":
        move_back_hp_param(task[1][1], task[1][0])
    else:
        state = optimizer.state[task[1]]
        move_back_key(state, task[2])


# Re-armed each time the pass runs (i.e. once per compile phase) and cleared on first execution.
_empty_cache_pending = False


def _opt_empty_cache_impl(anchor):
    # Emptying the cache on every training step forces the backward working set back through
    # cudaMalloc each step (measured at +28% step time on an 8xH200 14B run). One call after
    # each recompile is enough to return the segments freed by offloading to the driver; the
    # pass re-arms the flag whenever it runs.
    global _empty_cache_pending
    if not _empty_cache_pending:
        return
    _empty_cache_pending = False
    get_accelerator().empty_cache()


# Previous cumulative allocator counters, for per-step deltas in the timing print.
_prev_alloc_stats = {}


def _offload_copy_stream_sync_impl(anchor):
    # DS_DC_TIMING=1 prints the host wait here (the exposed reload tail) each step plus the
    # caching allocator's slow-path counters, separating exposed transfer time from
    # allocator-induced stalls in overlap experiments.
    if os.environ.get("DS_DC_TIMING") == "1":
        start = time.perf_counter()
        copy_stream.synchronize()
        if dist.get_rank() == 0:
            wait = time.perf_counter() - start
            stats = get_accelerator().memory_stats()
            keys = ("num_alloc_retries", "num_sync_all_streams", "num_device_alloc", "num_device_free")
            current = {key: stats.get(key, 0) for key in keys}
            delta = {key: current[key] - _prev_alloc_stats.get(key, 0) for key in keys}
            _prev_alloc_stats.update(current)
            print(
                f"[DeepCompile][timing] reload tail wait: {wait:.3f}s "
                f"alloc_retries=+{delta['num_alloc_retries']} sync_all_streams=+{delta['num_sync_all_streams']} "
                f"device_alloc=+{delta['num_device_alloc']} device_free=+{delta['num_device_free']}",
                flush=True)
        return
    copy_stream.synchronize()


_OFFLOAD_OP_SPECS = [
    ("offload_opt_launch", "offload_opt_launch(Tensor anchor, int idx) -> ()", _offload_opt_launch_impl),
    ("reload_opt", "reload_opt(Tensor anchor, int idx) -> ()", _reload_opt_impl),
    ("opt_empty_cache", "opt_empty_cache(Tensor anchor) -> ()", _opt_empty_cache_impl),
    ("offload_copy_stream_sync", "offload_copy_stream_sync(Tensor anchor) -> ()", _offload_copy_stream_sync_impl),
]


def register_offload_ops():
    global _offload_ops_lib
    if _offload_ops_lib is not None:
        return

    # FRAGMENT extends the "dc" namespace defined by the C++ extension, so this
    # must only run after get_deepcompile_handle() has loaded it.
    lib = torch.library.Library("dc", "FRAGMENT")
    for name, schema, impl in _OFFLOAD_OP_SPECS:
        lib.define(schema)
        lib.impl(name, impl, "CompositeExplicitAutograd")
        lib.impl(name, lambda *args: None, "Meta")
        overload = getattr(torch.ops.dc, name).default

        # These ops return nothing, so no node consumes their output and two independent
        # dead-code eliminations would drop them:
        #  1. FX GraphModule.eliminate_dead_code() -- guarded by _side_effectful_functions.
        #  2. Inductor's scheduler DCE -- keys off the op schema (mutation/effects), not the
        #     FX impurity set. Registering an ORDERED effect makes stock inductor keep the
        #     ops AND preserve their program order relative to each other (EffectfulKernel
        #     StarDep chaining). The reload-before-sync order is correctness-critical: the
        #     optimizer reads the states right after the graph returns. Without the effect
        #     registration the ops survive only because inductor.py disables
        #     Scheduler.dead_node_elimination process-wide, which this pass must not rely on.
        torch.fx.node._side_effectful_functions.add(overload)
        if _register_effectful_op is not None:
            _register_effectful_op(overload, _EffectType.ORDERED)

    # The ops deregister if the library object is garbage collected.
    _offload_ops_lib = lib


def _find_graph_anchor(graph: Graph):
    for node in graph.nodes:
        if node.op == 'placeholder' and isinstance(node.meta.get("val"), torch.Tensor):
            return node
    # A non-tensor anchor (e.g. a SymInt placeholder) would violate the op schemas at
    # runtime; fail here instead of falling back silently.
    raise AssertionError("no tensor placeholder found to anchor the offload ops on")


def update_max_memory(name):

    global max_memory
    mem = get_accelerator().max_memory_allocated()
    max_memory = max(max_memory, mem)


offload_tasks = []
offload_tasks_scheduled = []
# How many entries of offload_tasks_scheduled already have launch nodes: with graph breaks
# the pass runs once per forward graph and must not re-insert the whole list each time.
offload_tasks_inserted = 0
reload_tasks_remaining = []
total_reload_mem = 0


def offload_opt_states_inc(graph: Graph, graph_id: int, graph_order: List[Tuple[int, bool]],
                           profiling_results: ProfilingResult, mem_budget: float, param_manager: DSGraphParamManager,
                           bwd: bool) -> Graph:
    global _empty_cache_pending, offload_tasks_inserted, reload_tasks_remaining, total_reload_mem

    to_remove = []
    for node in graph.nodes:
        if node.op == 'call_function' and \
            node.target in [offload_adam_states_sync, sync_offload_states, reload_adam_states_sync, sync_reload_states, update_max_memory]:
            to_remove.append(node)

    for node in to_remove:
        graph.erase_node(node)

    register_offload_ops()
    anchor = _find_graph_anchor(graph)

    accelerator = get_accelerator()
    budget_override = os.environ.get("DS_DC_OFFLOAD_OPT_BUDGET_GB")
    if budget_override is not None:
        # Test/debug hook: pretend the device has this much usable memory so task scheduling
        # can be forced (or suppressed) independently of the hardware the run lands on.
        total_mem = float(budget_override) * 1e9
    else:
        total_mem = accelerator.total_memory() * (1 - MARGIN)
    print_r0(f"offload_opt_states_inc start graph {graph_id} bwd={bwd} max_memory={max_memory} total_mem={total_mem}")

    mem = profiling_results[graph_id].bwd_mem if bwd else profiling_results[graph_id].fwd_mem
    mem_dict = {name: peak for name, alloc_mem, delta, peak in mem}

    current_peak_mem = 0
    peak_mem = {}

    ordered_node = reversed(graph.nodes) if bwd else graph.nodes
    for node in ordered_node:
        # Nodes without a profiled entry (inserted by a pass whose profiling was skipped)
        # inherit the running peak instead of raising.
        if node.name in mem_dict and mem_dict[node.name] > current_peak_mem:
            current_peak_mem = mem_dict[node.name]
        peak_mem[node.name] = current_peak_mem

    if not bwd:
        is_first_graph = graph_id == graph_order[0][0]

        # At the beginning of the first graph, we schedule offload tasks to launch all offloading
        if is_first_graph:
            # This one-shot module state survives across compile phases; reset it so re-running
            # the pass (a later phase or a second engine.compile) does not double-append tasks.
            offload_tasks.clear()
            offload_tasks_scheduled.clear()
            offload_tasks_inserted = 0
            _op_task_registry.clear()
            total_reload_mem = 0

            with unset_fake_temporarily():
                offload_adam_states_sync()
                reload_adam_states_sync()
                sync_reload_states()

            for i, ((k, state), hp_param, hp_param_cpu) in enumerate(
                    zip(optimizer.state.items(), optimizer.fp32_partitioned_groups_flat,
                        optimizer.hp_params_pin_buffers)):

                if _make_offload_state_key("exp_avg") in state:
                    key = _make_offload_state_key("exp_avg")
                    offload_tasks.append(
                        (i, k, "exp_avg", state[key].numel() * state[key].element_size(), state[key].dtype))

                if _make_offload_state_key("exp_avg_sq") in state:
                    key = _make_offload_state_key("exp_avg_sq")
                    offload_tasks.append(
                        (i, k, "exp_avg_sq", state[key].numel() * state[key].element_size(), state[key].dtype))

                offload_tasks.append((i, (hp_param, hp_param_cpu), "hp_param",
                                      hp_param.numel() * hp_param.element_size(), hp_param.dtype))

        for node in graph.nodes:
            if node.name not in peak_mem \
                    or node.op == 'placeholder' \
                    or "offload_opt_" in node.name:
                continue

            to_offload = []
            optim_size = sum([task[3] for task in offload_tasks])

            # The peaks were profiled after offload_adam_states_for_init emptied the GPU of
            # optimizer state, so keeping optim_size bytes resident adds on top of them.
            while total_mem - peak_mem[node.name] - optim_size < 0:
                if len(offload_tasks) == 0:
                    break

                task = offload_tasks.pop(0)
                to_offload.append(task)
                optim_size = sum([task[3] for task in offload_tasks])

            # No sync/free node is inserted: the launch op drops the reference itself and
            # record_stream makes the free completion-driven, so this loop only decides which
            # tasks are offloaded at all.
            for task in to_offload:
                print_r0(f"Scheduling offload of optimizer state {task[0]}_{task[2]}")
                offload_tasks_scheduled.append(task)

        # Raise the count to the requested piece target. Memory necessity above can only
        # be exceeded, never undercut: a target below it is ignored.
        piece_target = _resolve_piece_target()
        if piece_target is not None:
            while len(offload_tasks_scheduled) < piece_target and len(offload_tasks) > 0:
                task = offload_tasks.pop(0)
                print_r0(f"Scheduling offload of optimizer state {task[0]}_{task[2]} (piece target {piece_target})")
                offload_tasks_scheduled.append(task)

        # Only tasks scheduled since the last insertion get launch nodes: with graph breaks
        # this runs once per forward graph, and earlier graphs already carry the launches
        # for their share of the list.
        new_tasks = offload_tasks_scheduled[offload_tasks_inserted:]
        for node in graph.nodes:
            if node.op != 'placeholder':
                print_r0(f"Inserting {len(new_tasks)} offload tasks before {node.name}")
                for task in new_tasks:
                    name = f"offload_opt_{task[0]}_{task[2]}"
                    with graph.inserting_before(node):
                        graph.create_node('call_function',
                                          torch.ops.dc.offload_opt_launch.default, (anchor, _register_op_task(task)),
                                          {},
                                          name=name)
                break
        offload_tasks_inserted = len(offload_tasks_scheduled)

        # Physically offload the scheduled pieces now instead of waiting for the first
        # runtime step: passes running after this one in the same phase (prefetch,
        # selective gather) measure memory during compilation and must see the residency
        # the runtime will actually have -- otherwise the freed bytes stay invisible to
        # their budgets and the combination gains nothing. The step-1 launch ops then
        # find the keys already gone and no-op gracefully. copy_stream is None only in
        # unit tests that bypass init_offload_opt_states; skip there.
        if len(offload_tasks_scheduled) > 0 and copy_stream is not None:
            with unset_fake_temporarily():
                for task in offload_tasks_scheduled:
                    if task[2] == "hp_param":
                        move_hp_param(task[1][0], task[1][1])
                    else:
                        state = optimizer.state[task[1]]
                        move_key(state, task[2])
                        if task[2] in state:
                            del state[task[2]]
                get_accelerator().synchronize()

        print_r0(f"offload_opt_states_inc finish graph {graph_id}")
    else:

        graph_order_with_backward = [g[0] for g in graph_order if g[1]]
        is_first_graph = graph_id == graph_order_with_backward[-1]
        is_last_graph = graph_id == graph_order_with_backward[0]

        if is_first_graph:
            _empty_cache_pending = True
            inserted_sync = False
            for node in graph.nodes:
                if node.op != 'placeholder' and not inserted_sync:
                    with graph.inserting_before(node):
                        graph.create_node('call_function',
                                          torch.ops.dc.opt_empty_cache.default, (anchor, ), {},
                                          name="empty_cache")

                    inserted_sync = True
        if is_first_graph:
            # Reset once per step's backward, not once per backward graph: with graph breaks
            # each graph reloads its share and later graphs continue from the remainder.
            reload_tasks_remaining = copy.copy(offload_tasks_scheduled)

        # DS_DC_RELOAD_EARLY=1 queues every reload at the start of backward instead of using the
        # budget-driven placement, giving the transfers the whole backward to hide under. Used to
        # separate structural exposure (placement) from irreducible transfer cost in experiments;
        # costs peak memory, so it is not the default.
        if os.environ.get("DS_DC_RELOAD_EARLY") == "1" and is_first_graph:
            for node in graph.nodes:
                if node.op != 'placeholder':
                    for task in reload_tasks_remaining:
                        with graph.inserting_before(node):
                            graph.create_node('call_function',
                                              torch.ops.dc.reload_opt.default, (anchor, _register_op_task(task)), {},
                                              name=f"reload_opt_{task[0]}_{task[2]}")
                    reload_tasks_remaining = []
                    break

        for node in graph.nodes:
            if node.name not in peak_mem \
                or node.op == 'placeholder' \
                or node.op == 'output':
                continue

            if len(reload_tasks_remaining) > 0:
                task = reload_tasks_remaining[0]
                next_reload_mem = task[3]

                insert_pos = node
                while total_mem > peak_mem[node.name] + total_reload_mem + next_reload_mem:
                    expected_mem = peak_mem[node.name] + total_reload_mem
                    print_r0(
                        f" Inserting reload_opt reload_opt_{task[0]}_{task[2]} after {insert_pos.name} next_inc={next_reload_mem} peak_mem[{node.name}]={peak_mem[node.name]} inc_total={total_reload_mem} expected_mem={expected_mem}"
                    )

                    with graph.inserting_after(insert_pos):
                        insert_pos = graph.create_node('call_function',
                                                       torch.ops.dc.reload_opt.default,
                                                       (anchor, _register_op_task(task)), {},
                                                       name=f"reload_opt_{task[0]}_{task[2]}")

                    total_reload_mem += next_reload_mem
                    reload_tasks_remaining.pop(0)
                    if len(reload_tasks_remaining) == 0:
                        break

                    task = reload_tasks_remaining[0]
                    next_reload_mem = task[3]

        if is_last_graph:
            for node in graph.nodes:
                if node.op == 'output':
                    for task in reload_tasks_remaining:
                        with graph.inserting_before(node):
                            graph.create_node('call_function',
                                              torch.ops.dc.reload_opt.default, (anchor, _register_op_task(task)), {},
                                              name=f"reload_opt_{task[0]}_{task[2]}")

                    with graph.inserting_before(node):
                        graph.create_node('call_function',
                                          torch.ops.dc.offload_copy_stream_sync.default, (anchor, ), {},
                                          name="sync_offload_copy_stream")

        print_r0(
            f"offload_opt_states_inc graph {graph_id} graph_order {graph_order} bwd is_first_graph {is_first_graph} is_last_graph {is_last_graph}"
        )

    return graph


def add_record_max_mem_nodes(graph: Graph):

    nodes = list(graph.nodes)
    for node in nodes:
        if node.op == "output" or node.op == "placeholder":
            continue

        with graph.inserting_after(node):
            name = f"update_max_memory_{node.name}"
            graph.create_node('call_function', update_max_memory, (name, ), {}, name=name)


def insert_offload_opt_states(graph: Graph, graph_id: int, graph_order: List[Tuple[int, bool]],
                              profiling_results: ProfilingResult, mem_budget: float,
                              param_manager: DSGraphParamManager, bwd: bool) -> Graph:

    if bwd:
        graph_order_with_backward = [g[0] for g in graph_order if g[1]]
        is_last_graph = graph_id == graph_order_with_backward[0]

        inserted_reload = False
        for node in graph.nodes:
            if node.op == 'output' and not inserted_reload and is_last_graph:
                with graph.inserting_before(node):
                    graph.create_node('call_function', reload_adam_states_sync, (), {}, name="reload_opt")
                inserted_reload = True

    else:
        is_first_graph = graph_id == graph_order[0][0]

        graph = move_primals_to_head(graph)

        inserted_offload = False
        for node in graph.nodes:
            if node.op != 'placeholder' and not inserted_offload and is_first_graph:
                print_r0(f"Inserting offload_opt before {node.name}")
                with graph.inserting_before(node):
                    graph.create_node('call_function', offload_adam_states_sync, (), {}, name="offload_opt")
                inserted_offload = True

    add_record_max_mem_nodes(graph)

    return graph


def move_opt_states(gm: GraphModule, graph_id: int, graph_order: List[Tuple[int, bool]], profiling_results,
                    create_inputs_fn, mem_budget: float, param_manager: DSGraphParamManager, bwd: bool) -> GraphModule:
    gm.graph = offload_opt_states_inc(gm.graph, graph_id, graph_order, profiling_results, mem_budget, param_manager,
                                      bwd)
    return gm


def move_opt_states_sync(gm: GraphModule, graph_id: int, graph_order: List[Tuple[int, bool]], profiling_results,
                         create_inputs_fn, mem_budget: float, param_manager: DSGraphParamManager,
                         bwd: bool) -> GraphModule:
    gm.graph = insert_offload_opt_states(gm.graph, graph_id, graph_order, profiling_results, mem_budget, param_manager,
                                         bwd)
    return gm


def offload_adam_states_for_init(gm: GraphModule, graph_id: int, graph_order: List[Tuple[int, bool]],
                                 profiling_results, create_inputs_fn, mem_budget: float,
                                 param_manager: DSGraphParamManager, bwd: bool) -> GraphModule:
    if not bwd and graph_id == graph_order[0][0]:
        with unset_fake_temporarily():
            offload_adam_states_sync()
    # returns None, and profiling will be skipped


def init_offload_opt_states(adam_optimizer, _nz3):
    lazy_init()
    register_offload_ops()

    global optimizer
    optimizer = adam_optimizer
    global device
    device = torch.device(get_accelerator().current_device())
    global nz3
    nz3 = _nz3
