# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
from packaging.version import Version
from torch.fx import GraphModule

from deepspeed.utils.torch import required_torch_version
from deepspeed.utils import logger

from .passes.tp_compile import apply_autotp, defer_collectives_to_compiler
from .passes.tp_lowering import lower_tp_collectives

AUTOTP_MIN_TORCH_VERSION = 2.6
BROKEN_TRANSFORMERS_MOE_VERSIONS = ("5.8.0", "5.10.1")


def _check_autotp_compatibility(model):
    if not required_torch_version(min_version=AUTOTP_MIN_TORCH_VERSION):
        raise RuntimeError(f"The AutoTP compile pass requires PyTorch >= {AUTOTP_MIN_TORCH_VERSION}, found "
                           f"{torch.__version__}.")
    _check_broken_transformers_moe(model)


def _check_broken_transformers_moe(model):
    """Reject models whose experts forward the installed transformers cannot capture in a graph."""

    try:
        import transformers
    except ImportError:
        return
    first_broken, first_fixed = BROKEN_TRANSFORMERS_MOE_VERSIONS
    if not (Version(first_broken) <= Version(transformers.__version__) < Version(first_fixed)):
        return
    experts_modules = [
        name for name, module in model.named_modules()
        if hasattr(module, "_apply_gate") and hasattr(module, "is_concatenated")
        and getattr(getattr(module, "config", None), "_experts_implementation", None) == "batched_mm"
    ]
    if experts_modules:
        raise RuntimeError(
            f"transformers {transformers.__version__} mutates the MoE routing tensor in place inside "
            "batched_mm_experts_forward (huggingface/transformers#45621, fixed by #45634), and this model "
            f"routes through it (e.g. '{experts_modules[0]}'), so the AutoTP compile pass cannot capture a "
            f"full graph. Upgrade to transformers >= {first_fixed}.")


class _AutoTPLoweringPass:
    """Inductor post_grad_custom_pre_pass that lowers the AutoTP markers to functional collectives.

    Registered as a *pre* pass on purpose: Inductor runs post_grad_custom_pre_pass, then
    micro_pipeline_tp_pass, then post_grad_custom_post_pass. Lowering first is what would let any
    later collective-aware pass see real collectives instead of an opaque custom op.

    Chains whatever pass was already installed so this does not silently displace it.
    """

    def __init__(self, previous=None):
        self.previous = previous

    def __call__(self, graph):
        if self.previous is not None:
            self.previous(graph)
        lower_tp_collectives(graph)

    def uuid(self):
        return "deepspeed.autotp.lower_tp_collectives.v1"


def _install_inductor_hooks(reorder: bool):
    """Wire the lowering pass and the collective-overlap scheduling into Inductor."""
    inductor_config = torch._inductor.config

    existing = inductor_config.post_grad_custom_pre_pass
    if not isinstance(existing, _AutoTPLoweringPass):
        inductor_config.post_grad_custom_pre_pass = _AutoTPLoweringPass(existing)

    # Only meaningful once the markers are lowered: these scheduler passes key off
    # torch._inductor.utils.is_collective, which requires an ir._CollectiveKernel. An opaque
    # custom op lowers to a plain FallbackKernel and never qualifies.
    if reorder:
        inductor_config.reorder_for_compute_comm_overlap = True


def init_autotp(model, compile_config=None, tp_size=1):
    """Hand the tensor-parallel collectives of an AutoTP-partitioned model over to the compiler.

    The model is expected to have been partitioned already by the regular AutoTP path, so this only
    suppresses the module-level collectives and returns a backend that emits them as graph nodes.
    """
    _check_autotp_compatibility(model)
    defer_collectives_to_compiler(model)
    if getattr(compile_config, "autotp_overlap", True):
        reorder = getattr(compile_config, "autotp_reorder", None)
        if reorder is None:
            # Below tp=4 the collective is too small a share of the step for the reordering to pay
            # for the fusion it disrupts; measured net negative at tp=2.
            reorder = tp_size >= 4
        _install_inductor_hooks(reorder)
        logger.info(f"AutoTP compile pass: functional collectives on, Inductor reordering "
                    f"{'on' if reorder else 'off'} (tp={tp_size}).")

    def backend_fn(gm: GraphModule, real_inputs):
        apply_autotp(gm, real_inputs)
        return torch._inductor.compile(gm, real_inputs)

    return backend_fn
