# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

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


def _print_rank0(msg):
    if dist.get_rank() == 0:
        print(f"[DeepCompile] {msg}", flush=True)


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

    _print_rank0(f"init_z3: backend={backend} offload_parameters={compile_config.offload_parameters} "
                 f"offload_opt_states={compile_config.offload_opt_states} "
                 f"free_activation={compile_config.free_activation} double_buffer={compile_config.double_buffer} "
                 f"symmetric_memory={compile_config.symmetric_memory} debug_log={compile_config.debug_log}")
    _print_rank0(f"init_z3: optimizer={type(optimizer).__module__}.{type(optimizer).__name__} use_opt={use_opt} "
                 f"(use_opt=False only when ZeRO-3 was initialized without a real optimizer, "
                 f"i.e. engine.optimizer is DeepSpeedZeRoOffload)")

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

    n_params = 0
    shard_bytes = 0
    for p in engine.module.parameters():
        grad_buffer = torch.Tensor()
        if use_opt:
            grad_buffer = optimizer._DeepSpeedZeroOptimizer_Stage3__param_id_to_grad_partition[p.ds_id]

        # Disable persistent param
        p.ds_persist = False
        dc.register_z3_param(p.ds_id, p.ds_shape, p.ds_tensor, grad_buffer, p.ds_persist,
                             _resolve_expected_grad_dtype(p))
        n_params += 1
        shard_bytes += p.ds_tensor.numel() * p.ds_tensor.element_size()

    _print_rank0(f"init_z3: registered {n_params} ZeRO-3 params, "
                 f"shard bytes on this rank={shard_bytes / 1e9:.2f}GB (world_size={dist.get_world_size()}), "
                 f"persistent params disabled at init")

    schedule_source = "default" if schedule is None else "user-provided"
    if schedule is None:
        schedule = []
        if (compile_config.offload_parameters):
            schedule.append((0, [zero3_compile.add_z3_gather_release, offload_parameters.offload_parameter_fwd]))
        else:
            from .passes.offload_adam_states import move_opt_states, move_opt_states_sync, init_offload_opt_states
            schedule.append((0, [zero3_compile.add_z3_gather_release]))
            schedule.append(
                (WARMUP,
                 [zero3_compile.add_z3_gather_release, prefetch.schedule_prefetch, selective_gather.selective_gather, move_opt_states]))

    for step, passes in schedule:
        _print_rank0(f"init_z3: {schedule_source} schedule from step {step}: "
                     f"passes={[opt_pass.__name__ for opt_pass in passes]}")
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
        from .passes.offload_adam_states import move_opt_states, move_opt_states_sync, init_offload_opt_states
        for _, passes in schedule:
            if move_opt_states in passes or move_opt_states_sync in passes:
                init_offload_opt_states(optimizer, dc)

    engine.launch_compile_passes = launch_compile_passes


    for step, passes in schedule:
        _print_rank0(f"second print: init_z3: {schedule_source} schedule from step {step}: "
                     f"second print: passes={[opt_pass.__name__ for opt_pass in passes]}")
    patch_fake_tensor()
    torch._inductor.config.size_asserts = False

    return make_backend(backend, compile_config, compile_kwargs=compile_kwargs)
