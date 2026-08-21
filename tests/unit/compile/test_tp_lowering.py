# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import pytest
import torch

import deepspeed
from deepspeed.accelerator import get_accelerator
from deepspeed.compile.init_tp import AUTOTP_MIN_TORCH_VERSION
from deepspeed.utils.torch import required_torch_version

from unit.common import DistributedTest

pytestmark = pytest.mark.skipif(not required_torch_version(min_version=AUTOTP_MIN_TORCH_VERSION),
                                reason=f"The AutoTP compile pass requires PyTorch >= {AUTOTP_MIN_TORCH_VERSION}")

HIDDEN_DIM = 64
INTERMEDIATE_DIM = 128


class MLPBlock(torch.nn.Module):
    """gate/up are column-parallel, so their collective is in the backward; down is row-parallel,
    so its collective is in the forward. One block exercises both markers."""

    def __init__(self):
        super().__init__()
        self.gate_proj = torch.nn.Linear(HIDDEN_DIM, INTERMEDIATE_DIM, bias=False)
        self.up_proj = torch.nn.Linear(HIDDEN_DIM, INTERMEDIATE_DIM, bias=False)
        self.down_proj = torch.nn.Linear(INTERMEDIATE_DIM, HIDDEN_DIM, bias=False)

    def forward(self, x):
        return x + self.down_proj(torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))


class MLPModel(torch.nn.Module):

    def __init__(self, nlayers=2):
        super().__init__()
        self.layers = torch.nn.ModuleList([MLPBlock() for _ in range(nlayers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def build_config(tp_size, overlap):
    return {
        "train_micro_batch_size_per_gpu": 1,
        "optimizer": {
            "type": "Adam",
            "params": {
                "lr": 1e-6
            }
        },
        "tensor_parallel": {
            "autotp_size": tp_size,
            "partition_config": {
                "use_default_specs":
                False,
                "layer_specs": [{
                    "patterns": [".*\\.gate_proj\\.weight$", ".*\\.up_proj\\.weight$"],
                    "partition_type": "column",
                }, {
                    "patterns": [".*\\.down_proj\\.weight$"],
                    "partition_type": "row",
                }],
            },
        },
        "zero_optimization": {
            "stage": 0
        },
        "compile": {
            "deepcompile": True,
            "passes": ["autotp"],
            "autotp_overlap": overlap,
        },
    }


def build_engine(tp_size, overlap):
    torch.manual_seed(42)
    model = MLPModel()
    engine, _, _, _ = deepspeed.initialize(model=model,
                                           model_parameters=model.parameters(),
                                           config=build_config(tp_size, overlap))
    engine.compile()
    return engine


def step(engine, device):
    torch.manual_seed(1234)  # the TP group must see identical inputs on every rank
    x = torch.randn(1, 8, HIDDEN_DIM, device=device, dtype=torch.float32, requires_grad=True)
    out = engine(x)
    engine.backward(out.sum())
    return out.detach().clone()


class TestAutoTPLoweringRewritesTheGraph(DistributedTest):
    """The markers must be gone from the compiled graph, replaced by functional collectives.

    Without this rewrite the collective is a single opaque custom op: it blocks the compute stream
    the moment it is issued, and Inductor's overlap passes cannot see it at all, because they
    require an ir._CollectiveKernel and a custom op lowers to a plain FallbackKernel.
    """

    world_size = 2
    non_daemonic_procs = True

    @pytest.mark.sequential
    def test_markers_are_lowered_and_clones_dropped(self):
        if get_accelerator().device_name() == "cpu":
            pytest.skip("CPU does not support this test yet")
        from deepspeed.compile.passes import tp_lowering

        device = torch.device(get_accelerator().current_device_name())
        tp_lowering.LOWERING_STATS.clear()
        step(build_engine(self.world_size, overlap=True), device)

        lowered = sum(s["lowered"] for s in tp_lowering.LOWERING_STATS)
        dropped = sum(s["dropped_copies"] for s in tp_lowering.LOWERING_STATS)
        assert lowered > 0, "no reduce_from_tp_region was rewritten into all_reduce + wait_tensor"
        assert dropped > 0, "the copy_to_tp_region identity clones were not removed"

    @pytest.mark.sequential
    def test_disabled_leaves_the_markers_alone(self):
        """autotp_overlap=False must be the previous behaviour exactly, so it is a usable escape hatch."""
        if get_accelerator().device_name() == "cpu":
            pytest.skip("CPU does not support this test yet")
        from deepspeed.compile.passes import tp_lowering

        device = torch.device(get_accelerator().current_device_name())
        tp_lowering.LOWERING_STATS.clear()
        step(build_engine(self.world_size, overlap=False), device)

        assert not tp_lowering.LOWERING_STATS, "the lowering pass ran even though autotp_overlap is False"


class TestAutoTPLoweringEquivalence(DistributedTest):
    """Rewriting the collectives must not change a single number.

    The lowering replaces a blocking all-reduce with an issue/wait pair and deletes an identity
    clone, so the arithmetic is untouched and the two paths should agree bitwise.
    """

    world_size = 2
    non_daemonic_procs = True

    @pytest.mark.sequential
    def test_overlap_matches_no_overlap(self):
        if get_accelerator().device_name() == "cpu":
            pytest.skip("CPU does not support this test yet")

        device = torch.device(get_accelerator().current_device_name())
        reference_engine = build_engine(self.world_size, overlap=False)
        lowered_engine = build_engine(self.world_size, overlap=True)

        reference_out = step(reference_engine, device)
        lowered_out = step(lowered_engine, device)

        assert torch.equal(reference_out, lowered_out), \
            "lowering the collectives changed the forward result"
        for (name, reference_param), (_, lowered_param) in zip(reference_engine.module.named_parameters(),
                                                               lowered_engine.module.named_parameters()):
            assert torch.equal(reference_param.grad, lowered_param.grad), \
                f"lowering the collectives changed the gradient of {name}"


@pytest.mark.parametrize("tp_size, expected_reorder", [(1, False), (2, False), (4, True), (8, True)])
def test_reorder_defaults_to_on_only_from_tp4(tp_size, expected_reorder):
    """Inductor's reordering is measured net negative below tp=4.

    There it disrupts more fusion than it recovers, because the collective is too small a share of
    the step to hide anything behind. The default has to depend on the tensor-parallel size.
    """
    from deepspeed.compile.config import CompileConfig
    from deepspeed.compile.init_tp import init_autotp

    inductor_config = torch._inductor.config
    saved = (inductor_config.post_grad_custom_pre_pass, inductor_config.reorder_for_compute_comm_overlap)
    inductor_config.post_grad_custom_pre_pass = None
    inductor_config.reorder_for_compute_comm_overlap = False
    try:
        init_autotp(torch.nn.Module(), CompileConfig(), tp_size=tp_size)
        assert inductor_config.reorder_for_compute_comm_overlap is expected_reorder
        assert inductor_config.post_grad_custom_pre_pass is not None, "the lowering pass was not installed"
    finally:
        inductor_config.post_grad_custom_pre_pass, inductor_config.reorder_for_compute_comm_overlap = saved


@pytest.mark.parametrize("forced", [True, False])
def test_reorder_can_be_forced_either_way(forced):
    from deepspeed.compile.config import CompileConfig
    from deepspeed.compile.init_tp import init_autotp

    inductor_config = torch._inductor.config
    saved = (inductor_config.post_grad_custom_pre_pass, inductor_config.reorder_for_compute_comm_overlap)
    inductor_config.post_grad_custom_pre_pass = None
    inductor_config.reorder_for_compute_comm_overlap = False
    try:
        # tp_size=2 would default the reorder off; an explicit setting has to win.
        init_autotp(torch.nn.Module(), CompileConfig(autotp_reorder=forced), tp_size=2)
        assert inductor_config.reorder_for_compute_comm_overlap is forced
    finally:
        inductor_config.post_grad_custom_pre_pass, inductor_config.reorder_for_compute_comm_overlap = saved


def test_lowering_pass_chains_an_existing_pass():
    """Installing the hook must not silently displace a pass someone else registered."""
    from deepspeed.compile.init_tp import _AutoTPLoweringPass, _install_inductor_hooks

    inductor_config = torch._inductor.config
    saved = (inductor_config.post_grad_custom_pre_pass, inductor_config.reorder_for_compute_comm_overlap)
    calls = []
    try:
        inductor_config.post_grad_custom_pre_pass = lambda graph: calls.append("existing")
        _install_inductor_hooks(reorder=False)
        installed = inductor_config.post_grad_custom_pre_pass
        assert isinstance(installed, _AutoTPLoweringPass)

        installed(torch.fx.Graph())  # no autotp nodes: the lowering is a no-op, the chain is not
        assert calls == ["existing"], "the previously installed pass was dropped"

        # Installing twice must not nest the wrapper and run the chain again.
        _install_inductor_hooks(reorder=False)
        assert inductor_config.post_grad_custom_pre_pass is installed
    finally:
        inductor_config.post_grad_custom_pre_pass, inductor_config.reorder_for_compute_comm_overlap = saved
