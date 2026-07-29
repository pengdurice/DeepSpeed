# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import os
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

import deepspeed.compile.passes.offload_adam_states as offload_pass
from deepspeed.accelerator import get_accelerator
from deepspeed.runtime.zero.offload_states import _make_offload_state_key
from deepspeed.utils.torch import required_torch_version

from unit.common import DistributedTest
from unit.util import bf16_required_version_check, skip_on_arch
from unit.v1.compile.util import compare_loss

pytestmark = pytest.mark.skipif(not required_torch_version(min_version=2.1),
                                reason="Compile tests requires Pytorch version 2.1 or above")


@pytest.fixture(autouse=True)
def _reset_offload_pass_globals():
    # The pass keeps its planning state in module globals; reset after every test so tests
    # pass in any order and in isolation.
    yield
    offload_pass.offload_tasks.clear()
    offload_pass.offload_tasks_scheduled.clear()
    offload_pass.offload_tasks_inserted = 0
    offload_pass.reload_tasks_remaining = []
    offload_pass.total_reload_mem = 0
    offload_pass._op_task_registry.clear()
    offload_pass._empty_cache_pending = False
    offload_pass.set_offload_piece_target(None)
    offload_pass.set_offload_piece_cap(None)
    offload_pass.reset_offload_op_stats()


def _ensure_dc_ops():
    # Load the C++ dc library first where the environment supports it so the Python FRAGMENT
    # extends it in the same order as production; on CPU-only environments the fragment creates
    # the namespace standalone.
    from deepspeed.compile.util import is_deepcompile_supported
    if is_deepcompile_supported():
        from deepspeed.compile.util import get_deepcompile_handle
        get_deepcompile_handle()
    offload_pass.register_offload_ops()


def _make_fake_optimizer():
    param_key = torch.zeros(4)
    state = {
        _make_offload_state_key("exp_avg"): torch.zeros(8),
        _make_offload_state_key("exp_avg_sq"): torch.zeros(8),
    }
    return SimpleNamespace(state={param_key: state},
                           fp32_partitioned_groups_flat=[torch.zeros(8)],
                           hp_params_pin_buffers=[torch.zeros(8)])


def _make_fwd_graph():
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    x.meta["val"] = torch.empty(4)
    a = graph.call_function(torch.relu, (x, ))
    b = graph.call_function(torch.relu, (a, ))
    graph.output(b)
    return graph


class _null_context:

    def __enter__(self):
        return None

    def __exit__(self, *args):
        return False


def _mem_rows(graph):
    return [(node.name, 100, 0, 100) for node in graph.nodes]


def _run_fwd_pass(monkeypatch, graph, budget_gb="0", moved=None):
    monkeypatch.setattr(offload_pass.dist, "get_rank", lambda: 0)
    # The pass physically offloads the pieces it schedules; these tests exercise the
    # planner, not the copies, so record the calls instead of running them.
    monkeypatch.setattr(offload_pass, "_offload_task", lambda task: (moved if moved is not None else []).append(task))
    monkeypatch.setattr(offload_pass, "unset_fake_temporarily", _null_context)
    monkeypatch.setattr(offload_pass, "offload_adam_states_sync", lambda: None)
    monkeypatch.setattr(offload_pass, "reload_adam_states_sync", lambda: None)
    monkeypatch.setattr(offload_pass, "sync_reload_states", lambda: None)
    monkeypatch.setattr(offload_pass, "optimizer", _make_fake_optimizer())
    # The default zero budget forces every task to be scheduled at the first compute node.
    monkeypatch.setenv("DS_DC_OFFLOAD_OPT_BUDGET_GB", budget_gb)
    prof = SimpleNamespace(fwd_mem=_mem_rows(graph), bwd_mem=[])
    return offload_pass.offload_opt_states_inc(graph, 0, [(0, True)], {0: prof}, 0.0, None, bwd=False)


def test_register_offload_ops_idempotent():
    _ensure_dc_ops()
    lib_first = offload_pass._offload_ops_lib
    offload_pass.register_offload_ops()
    assert offload_pass._offload_ops_lib is lib_first

    for name, _, _ in offload_pass._OFFLOAD_OP_SPECS:
        overload = getattr(torch.ops.dc, name).default
        assert overload is not None
        assert overload in torch.fx.node._side_effectful_functions


def test_offload_ops_registered_with_ordered_effects():
    _ensure_dc_ops()
    from torch._higher_order_ops.effects import SIDE_EFFECTS

    for name, _, _ in offload_pass._OFFLOAD_OP_SPECS:
        overload = getattr(torch.ops.dc, name).default
        # The ORDERED effect is what makes stock inductor keep these ops and preserve their
        # program order; _side_effectful_functions alone only protects FX-level DCE, and the
        # scheduler's own DCE would silently drop the ops (states never move, loss unchanged).
        # test_side_effect_ops_survive_stock_inductor proves the mechanism itself.
        assert overload in SIDE_EFFECTS


def test_side_effect_ops_survive_stock_inductor():
    # Mechanism guard: an anchor-only `-> ()` op is dropped by stock inductor's scheduler DCE
    # unless it carries an ORDERED effect, and with the effect its program order is preserved.
    # Uses a throwaway namespace so it needs no dc C++ extension and runs on CPU.
    from torch.fx.experimental.proxy_tensor import make_fx
    from torch._higher_order_ops.effects import _EffectType, _register_effectful_op
    from torch._inductor.scheduler import Scheduler

    if getattr(Scheduler, "is_dc_patched", False):
        pytest.skip("inductor DCE already patched out in this process; mechanism not observable")

    if not hasattr(test_side_effect_ops_survive_stock_inductor, "_probe"):
        lib = torch.library.Library("dctest_probe", "DEF")
        lib.define("side_op(Tensor anchor, int idx) -> ()")
        calls = []
        lib.impl("side_op", lambda anchor, idx: calls.append(idx), "CompositeExplicitAutograd")
        lib.impl("side_op", lambda anchor, idx: None, "Meta")
        op = torch.ops.dctest_probe.side_op.default
        torch.fx.node._side_effectful_functions.add(op)
        _register_effectful_op(op, _EffectType.ORDERED)
        # Keep the library object alive or the ops deregister on garbage collection.
        test_side_effect_ops_survive_stock_inductor._probe = (lib, op, calls)
    _, op, calls = test_side_effect_ops_survive_stock_inductor._probe

    def f(x):
        return ((x * 2).relu().sin().sum(), )

    gm = make_fx(f)(torch.randn(8, 8))
    nodes = list(gm.graph.nodes)
    placeholder = nodes[0]
    compute = [n for n in nodes if n.op == "call_function"]
    out = next(n for n in nodes if n.op == "output")
    with gm.graph.inserting_before(compute[0]):
        gm.graph.create_node("call_function", op, (placeholder, 0), {})
    with gm.graph.inserting_before(compute[-1]):
        gm.graph.create_node("call_function", op, (placeholder, 1), {})
    with gm.graph.inserting_before(out):
        gm.graph.create_node("call_function", op, (placeholder, 2), {})
    gm.recompile()

    compiled = torch._inductor.compile(gm, [torch.randn(8, 8)])
    calls.clear()
    compiled(torch.randn(8, 8))
    assert calls == [0, 1, 2], f"side-effect ops dropped or reordered by stock inductor: {calls}"


def test_fwd_insertion_schedules_all_tasks_under_forced_budget(monkeypatch):
    _ensure_dc_ops()
    graph = _make_fwd_graph()
    _run_fwd_pass(monkeypatch, graph)

    names = [n.name for n in graph.nodes]
    launch_names = [n for n in names if n.startswith("offload_opt_")]

    assert len(launch_names) == 3, f"expected exp_avg/exp_avg_sq/hp_param launches, got {launch_names}"
    assert any("hp_param" in n for n in launch_names)
    # Frees are completion-driven via record_stream inside the launch op; no sync nodes exist.
    assert not any("sync" in n for n in launch_names)

    # All copies launch at the top of the graph, before all compute.
    first_compute = names.index("relu")
    assert max(names.index(n) for n in launch_names) < first_compute


def test_piece_target_raises_scheduled_count(monkeypatch):
    # A generous budget schedules nothing by necessity; the piece target raises the count
    # so the freed bytes become budget for the passes that run after this one.
    _ensure_dc_ops()
    graph = _make_fwd_graph()
    monkeypatch.setenv("DS_DC_OFFLOAD_PIECE_TARGET", "2")
    _run_fwd_pass(monkeypatch, graph, budget_gb="1")

    launch_names = [n.name for n in graph.nodes if n.name.startswith("offload_opt_")]
    assert len(launch_names) == 2
    assert len(offload_pass.offload_tasks_scheduled) == 2


def test_piece_target_cannot_undercut_memory_necessity(monkeypatch):
    # A zero budget forces all three pieces out; a smaller target must not reduce that.
    _ensure_dc_ops()
    graph = _make_fwd_graph()
    monkeypatch.setenv("DS_DC_OFFLOAD_PIECE_TARGET", "1")
    _run_fwd_pass(monkeypatch, graph, budget_gb="0")

    launch_names = [n.name for n in graph.nodes if n.name.startswith("offload_opt_")]
    assert len(launch_names) == 3


def test_piece_target_setter_used_when_env_absent(monkeypatch):
    _ensure_dc_ops()
    graph = _make_fwd_graph()
    offload_pass.set_offload_piece_target(2)
    _run_fwd_pass(monkeypatch, graph, budget_gb="1")

    launch_names = [n.name for n in graph.nodes if n.name.startswith("offload_opt_")]
    assert len(launch_names) == 2


def test_scheduled_pieces_are_offloaded_at_planning_time(monkeypatch):
    # Passes that run after this one in the same phase (prefetch, selective gather) profile
    # memory during compilation, so the pieces must be off the device by the time the pass
    # returns -- not only when the graph first executes.
    _ensure_dc_ops()
    graph = _make_fwd_graph()
    moved = []
    _run_fwd_pass(monkeypatch, graph, moved=moved)

    assert len(moved) == 3
    assert [task[2] for task in moved] == ["exp_avg", "exp_avg_sq", "hp_param"]


def test_piece_cap_holds_the_count_below_what_the_budget_asks(monkeypatch):
    # A zero budget would offload all three pieces. The cap is set only when the tuner reverts to
    # a count this run already executed, so it must win over the estimate.
    _ensure_dc_ops()
    graph = _make_fwd_graph()
    monkeypatch.setenv("DS_DC_OFFLOAD_PIECE_CAP", "1")
    _run_fwd_pass(monkeypatch, graph, budget_gb="0")

    assert len([n for n in graph.nodes if n.name.startswith("offload_opt_")]) == 1


def test_piece_cap_also_limits_the_target(monkeypatch):
    _ensure_dc_ops()
    graph = _make_fwd_graph()
    monkeypatch.setenv("DS_DC_OFFLOAD_PIECE_TARGET", "3")
    monkeypatch.setenv("DS_DC_OFFLOAD_PIECE_CAP", "2")
    _run_fwd_pass(monkeypatch, graph, budget_gb="1")

    assert len([n for n in graph.nodes if n.name.startswith("offload_opt_")]) == 2


def test_partial_offload_when_budget_allows_residency(monkeypatch):
    # The for_init-first schedule profiles with every state already offloaded, so keeping tasks
    # resident adds on top of the profiled peak: peak=100B, tasks=3x32B, budget=150B admits only
    # 32B of residency and two tasks must stay scheduled for offload.
    _ensure_dc_ops()
    graph = _make_fwd_graph()
    _run_fwd_pass(monkeypatch, graph, budget_gb="1.5e-7")

    launch_names = [n.name for n in graph.nodes if n.name.startswith("offload_opt_")]
    assert len(launch_names) == 2
    assert len(offload_pass.offload_tasks_scheduled) == 2


def test_for_init_offloads_before_profiling(monkeypatch):
    calls = []
    monkeypatch.setattr(offload_pass, "offload_adam_states_sync", lambda: calls.append(1))

    ret = offload_pass.offload_adam_states_for_init(None, 0, [(0, True)], None, None, 0.0, None, bwd=False)
    assert ret is None
    assert calls == [1]

    ret = offload_pass.offload_adam_states_for_init(None, 0, [(0, True)], None, None, 0.0, None, bwd=True)
    assert ret is None
    assert calls == [1]


def test_pass_reruns_do_not_double_append(monkeypatch):
    _ensure_dc_ops()
    graph_first = _make_fwd_graph()
    _run_fwd_pass(monkeypatch, graph_first)
    assert len(offload_pass.offload_tasks_scheduled) == 3

    graph_second = _make_fwd_graph()
    _run_fwd_pass(monkeypatch, graph_second)
    assert len(offload_pass.offload_tasks_scheduled) == 3

    launch_names = [n.name for n in graph_second.nodes if n.name.startswith("offload_opt_")]
    assert len(launch_names) == 3


def test_bwd_insertion_reloads_at_graph_end(monkeypatch):
    _ensure_dc_ops()
    fwd_graph = _make_fwd_graph()
    _run_fwd_pass(monkeypatch, fwd_graph)

    bwd_graph = torch.fx.Graph()
    tangent = bwd_graph.placeholder("tangent")
    tangent.meta["val"] = torch.empty(4)
    a = bwd_graph.call_function(torch.relu, (tangent, ))
    bwd_graph.output(a)
    prof = SimpleNamespace(fwd_mem=[], bwd_mem=_mem_rows(bwd_graph))

    offload_pass.offload_opt_states_inc(bwd_graph, 0, [(0, True)], {0: prof}, 0.0, None, bwd=True)

    names = [n.name for n in bwd_graph.nodes]
    reload_names = [n for n in names if n.startswith("reload_opt_")]

    assert "empty_cache" in names
    assert len(reload_names) == 3
    assert "sync_offload_copy_stream" in names
    # With a zero budget there is no mid-graph headroom, so every reload lands at the end of the
    # last backward graph, followed by the copy-stream sync.
    assert names.index("sync_offload_copy_stream") > max(names.index(n) for n in reload_names)
    # Running the backward pass re-arms the once-per-phase empty_cache.
    assert offload_pass._empty_cache_pending is True


def test_empty_cache_runs_once_per_phase(monkeypatch):
    calls = []
    monkeypatch.setattr(offload_pass, "get_accelerator", lambda: SimpleNamespace(empty_cache=lambda: calls.append(1)))

    offload_pass._empty_cache_pending = True
    offload_pass._opt_empty_cache_impl(None)
    offload_pass._opt_empty_cache_impl(None)

    assert len(calls) == 1


class TestOffloadOptStates(DistributedTest):
    world_size = 2
    non_daemonic_procs = True

    # forced=True offloads every piece (reloads land at the end of backward); piece_target
    # offloads a subset with memory to spare, which places reloads MID-backward -- the case
    # that exposed a cross-stream write into a live activation (NaN losses).
    @pytest.mark.parametrize('dtype', [torch.bfloat16])
    @pytest.mark.parametrize('mode', ['forced_full', 'piece_target', 'combined'])
    def test_offload_opt_states_correctness(self, dtype, mode):
        from deepspeed.compile.util import is_deepcompile_supported

        skip_on_arch(min_arch=8)
        if not bf16_required_version_check():
            pytest.skip(
                "DeepSpeed BFloat16 tests need NCCL >= 2.10.3, CUDA >=11.0, and HW support for BFloat16 to run correctly"
            )
        if get_accelerator().device_name() == "cpu":
            pytest.skip("CPU does not support this test yet")
        if not is_deepcompile_supported():
            pytest.skip("DeepCompile is not supported in this environment")

        # 'combined' adds prefetch and selective gather on top of the offload pass in one
        # phase: prefetch rewrites the graph after the offload/reload/sync nodes are in it
        # and selective gather then pins parameters resident. That rewrite interaction is
        # the highest-risk path in the combined schedule, so it gets its own loss check.
        config = {
            "train_micro_batch_size_per_gpu": 1,
            "steps_per_print": 1,
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 0.00015
                }
            },
            "zero_optimization": {
                "stage": 3
            },
            "compile": {
                "deepcompile": True,
                "offload_opt_states": True,
                "offload_opt_states_combine": mode == 'combined'
            },
            "bf16": {
                "enabled": True
            },
        }

        # Reference run: identical stage/compile configuration with offloading off. Comparing
        # against it isolates the offload pass alone, so a much tighter tolerance applies than
        # compare_loss's cross-stage default. Silently-stale states (a missed stream ordering)
        # would show up here as a small persistent loss drift.
        config_no_offload = deepcopy(config)
        config_no_offload["compile"]["offload_opt_states"] = False
        losses_no_offload = compare_loss(self, config_no_offload, dtype, iteration=8)

        # Force every optimizer-state tensor to be scheduled for offload (including hp_param,
        # which covers the event-key regression) regardless of the device's actual memory.
        if mode in ('forced_full', 'combined'):
            os.environ["DS_DC_OFFLOAD_OPT_BUDGET_GB"] = "0.000001"
        else:
            # Plenty of budget, so nothing is offloaded out of necessity: the piece target
            # alone schedules pieces and the planner places their reloads early.
            os.environ["DS_DC_OFFLOAD_OPT_BUDGET_GB"] = "1000"
            os.environ["DS_DC_OFFLOAD_PIECE_TARGET"] = "2"
        try:
            offload_pass.reset_offload_op_stats()
            # The offload schedule engages at step 1; the combined phase is added at WARMUP
            # (5), so run past it to exercise the recompiled graph.
            losses_offload = compare_loss(self, config, dtype, iteration=8)
        finally:
            del os.environ["DS_DC_OFFLOAD_OPT_BUDGET_GB"]
            os.environ.pop("DS_DC_OFFLOAD_PIECE_TARGET", None)

        stats = offload_pass.get_offload_op_stats()
        assert stats["launches"] > 0, "offload launch ops never executed"
        # This is the assertion that carries the weight: launch ops also execute while the
        # pass-time profilers interpret the graph eagerly, but _reload_opt_impl early-returns
        # during profiling, so a nonzero reload count proves the ops ran in the compiled graph
        # (i.e. no dead-code elimination silently removed them).
        assert stats["reloads"] > 0, "reload ops never executed outside profiling"
        assert stats["reloads"] <= stats["launches"]

        # compare_loss seeds identically at entry, so both runs saw the same data and the same
        # kernel sequence; moving states off-GPU and back must not change the arithmetic.
        for step, (ref, got) in enumerate(zip(losses_no_offload, losses_offload)):
            assert got == pytest.approx(ref, rel=1e-4, abs=1e-5), \
                f"offloading changed the loss at step {step}: {ref} vs {got}"
