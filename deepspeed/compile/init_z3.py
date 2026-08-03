# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import os

import torch

from deepspeed import comm as dist
from deepspeed.accelerator import get_accelerator
from deepspeed.runtime.zero.partition_parameters import InsertPostInitMethodToModuleSubClasses
from deepspeed.runtime.zero.parameter_offload import DeepSpeedZeRoOffload

from .passes import zero3_compile, prefetch, selective_gather, offload_parameters
from .backend import make_backend, launch_compile_passes, init_schedule
from .patch_fake_tensor import patch_fake_tensor
from .util import get_deepcompile_handle, add_pre_backward_hook, add_post_backward_hook
from .z3_eager_fallback import DeepCompileZ3EagerFallback

WARMUP = 5

_MISSING = object()


def _resolve_expected_grad_dtype(param):
    # Match PyTorch's leaf grad accumulation contract. grad_dtype can be a
    # dtype, or None to allow any incoming gradient dtype:
    # https://docs.pytorch.org/docs/main/generated/torch.sparse.semi_structured.SparseSemiStructuredTensorCUSPARSELT.html#torch.sparse.semi_structured.SparseSemiStructuredTensorCUSPARSELT.grad_dtype
    grad_dtype = getattr(param, "grad_dtype", _MISSING)
    if grad_dtype is None:
        return None
    if grad_dtype is not _MISSING:
        return grad_dtype
    return param.dtype


def init_z3(engine, backend, compile_config, compile_kwargs, schedule=None):

    optimizer = engine.optimizer
    use_opt = not isinstance(optimizer, DeepSpeedZeRoOffload)

    if use_opt and hasattr(optimizer, "ipg_buckets"):
        optimizer.ipg_buckets.clear()
        get_accelerator().empty_cache()

    dc = get_deepcompile_handle()
    dc.init(engine.data_parallel_group, compile_config, engine.zero_reduce_bucket_size())

    engine._deepcompile_z3_eager_fallback = DeepCompileZ3EagerFallback(engine)
    add_post_backward_hook(engine._deepcompile_z3_eager_fallback.release_gathered_params)

    if use_opt:
        optimizer.parameter_offload._remove_module_hooks()

        for hook in optimizer._grad_acc_hooks:
            hook.remove()
        optimizer._grad_acc_hooks.clear()

    # Unpatch linear
    if hasattr(InsertPostInitMethodToModuleSubClasses, "linear_bk"):
        torch.nn.functional.linear = InsertPostInitMethodToModuleSubClasses.linear_bk

    if compile_config.symmetric_memory:
        group_name = engine.data_parallel_group.group_name
        dist.enable_symm_mem_for_group(group_name)

    for p in engine.module.parameters():
        grad_buffer = torch.Tensor()
        if use_opt:
            grad_buffer = optimizer._DeepSpeedZeroOptimizer_Stage3__param_id_to_grad_partition[p.ds_id]

        # Disable persistent param
        p.ds_persist = False
        dc.register_z3_param(p.ds_id, p.ds_shape, p.ds_tensor, grad_buffer, p.ds_persist,
                             _resolve_expected_grad_dtype(p))

    if schedule is None:
        if compile_config.offload_parameters and compile_config.offload_opt_states:
            raise ValueError("offload_parameters and offload_opt_states cannot be enabled together; "
                             "choose one offloading target per run. Note that offload_parameters may have "
                             "been enabled implicitly: the engine turns it on when the ZeRO config "
                             "offloads both optimizer and parameters to CPU.")
        schedule = []
        if (compile_config.offload_parameters):
            schedule.append((0, [zero3_compile.add_z3_gather_release, offload_parameters.offload_parameter_fwd]))
        elif compile_config.offload_opt_states:
            from .passes.offload_adam_states import move_opt_states, offload_adam_states_for_init
            schedule.append((0, [zero3_compile.add_z3_gather_release]))
            # Optimizer states materialize at step 0's optimizer step, so offloading can engage at
            # step 1. for_init empties the GPU of optimizer state before the profilers run: the
            # plan is made against the floor, and a job that only fits with offloading never has
            # to survive a step with every state resident alongside the activations.
            schedule.append((1, [offload_adam_states_for_init, zero3_compile.add_z3_gather_release, move_opt_states]))
            if compile_config.offload_opt_states_combine:
                # Combined phase: offload plans (and physically moves) first so prefetch and
                # selective gather compile against the freed memory, then spend it.
                combined = [
                    offload_adam_states_for_init, zero3_compile.add_z3_gather_release, move_opt_states,
                    prefetch.schedule_prefetch, selective_gather.selective_gather
                ]
                # Experiment hook: run the offload planner a second time, after the gather
                # passes have spent their budget. Its first pass plans reload positions and
                # the piece count against peaks measured BEFORE those passes raise them; the
                # second pass sees the final memory picture. One iteration of the fixed-point
                # the three passes really form.
                if os.environ.get("DS_DC_COMBINE_REPLAN") == "1":
                    combined.append(move_opt_states)
                schedule.append((WARMUP, combined))
                if compile_config.offload_opt_states_tune:
                    from .backend import init_offload_tuner
                    from .offload_tuner import OffloadTuner
                    from .passes.offload_adam_states import resolve_margin
                    # Tuning rounds rerun every pass EXCEPT selective gather: set_persistent()
                    # has no release path (invalidate_gathered_param and clear_all_gathered_params
                    # both skip persistent params), so running the pass again only ever adds more
                    # resident parameters. That raises the memory floor each round, which raises
                    # the piece count the offload planner must schedule, and the tuner can never
                    # return to a smaller count. Persisting what the first combined phase chose
                    # keeps every round comparable and leaves the piece count as the only variable.
                    # DS_DC_TUNE_KEEP_SG=1 keeps it in, so the cost of freezing the persistent
                    # set can be measured against the cost of letting it grow every round.
                    if os.environ.get("DS_DC_TUNE_KEEP_SG") == "1":
                        tune_passes = combined
                    else:
                        tune_passes = [p for p in combined if p is not selective_gather.selective_gather]
                    budget = get_accelerator().total_memory() * (1 - resolve_margin())
                    init_offload_tuner(OffloadTuner(compile_config.offload_opt_states_tune_rounds, budget),
                                       tune_passes)
        else:
            schedule.append((0, [zero3_compile.add_z3_gather_release]))
            schedule.append(
                (WARMUP,
                 [zero3_compile.add_z3_gather_release, prefetch.schedule_prefetch, selective_gather.selective_gather]))

    init_schedule(schedule)

    if use_opt:

        def set_grad_buffer(_is_gradient_accumulation_boundary):
            for i, sub_group in enumerate(optimizer.fp16_groups):
                optimizer.averaged_gradients[i] = [
                    optimizer._DeepSpeedZeroOptimizer_Stage3__param_id_to_grad_partition[param.ds_id]
                    if param.requires_grad else torch.zeros_like(param.ds_tensor) for param in sub_group
                ]

        add_pre_backward_hook(set_grad_buffer)

        # offloading opt states need additional setup
        from .passes.offload_adam_states import (move_opt_states, move_opt_states_sync, offload_adam_states_for_init,
                                                 init_offload_opt_states)
        for _, passes in schedule:
            if move_opt_states in passes or move_opt_states_sync in passes or offload_adam_states_for_init in passes:
                init_offload_opt_states(optimizer, dc)

    engine.launch_compile_passes = launch_compile_passes

    patch_fake_tensor()
    torch._inductor.config.size_asserts = False

    return make_backend(backend, compile_config, compile_kwargs=compile_kwargs)
