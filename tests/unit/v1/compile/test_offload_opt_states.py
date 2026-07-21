# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import os
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


def _mem_rows(graph):
    return [(node.name, 100, 0, 100) for node in graph.nodes]


def _run_fwd_pass(monkeypatch, graph, budget_gb="0"):
    monkeypatch.setattr(offload_pass.dist, "get_rank", lambda: 0)
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

    @pytest.mark.parametrize('dtype', [torch.bfloat16])
    def test_offload_opt_states_correctness(self, dtype):
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
                "offload_opt_states": True
            },
            "bf16": {
                "enabled": True
            },
        }

        # Force every optimizer-state tensor to be scheduled for offload (including hp_param,
        # which covers the event-key regression) regardless of the device's actual memory.
        os.environ["DS_DC_OFFLOAD_OPT_BUDGET_GB"] = "0.000001"
        try:
            offload_pass.reset_offload_op_stats()
            # WARMUP is 5, so run enough iterations for the offload phase to engage.
            compare_loss(self, config, dtype, iteration=8)
        finally:
            del os.environ["DS_DC_OFFLOAD_OPT_BUDGET_GB"]

        stats = offload_pass.get_offload_op_stats()
        assert stats["launches"] > 0, "offload launch ops never executed"
        assert stats["reloads"] > 0, "reload ops never executed outside profiling"
        assert stats["reloads"] <= stats["launches"]
