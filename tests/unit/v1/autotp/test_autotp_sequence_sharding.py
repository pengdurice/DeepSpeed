# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import contextlib
import copy

import pytest
import torch
import deepspeed
import deepspeed.comm as dist
from deepspeed.accelerator import get_accelerator
from deepspeed.utils import groups
from unit.common import DistributedTest


def skip_on_device():
    if get_accelerator().device_name() == "xpu":
        pytest.skip("XPU requires a higher version for test")


def build_qwen3_model():
    from transformers import AutoModelForCausalLM, Qwen3Config

    # Construct locally so the test does not download model configuration or weights. q_norm/k_norm act on the
    # sharded heads and the decoder norms on the split sequence, so both gradient reductions are exercised.
    config = Qwen3Config(
        vocab_size=1000,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=4,
        head_dim=32,
        max_position_embeddings=128,
        tie_word_embeddings=False,
    )
    torch.manual_seed(0)
    return config, AutoModelForCausalLM.from_config(config)


# plain: separate collectives; overlap: one autograd node per block with the backward reduce-scatter overlapped;
# copy-engine: the block all-gathers and reduce-scatters on the copy engines in both passes, for any size (the
# automatic minimum would choose NCCL for this small model); copy-engine-backward: NCCL in the forward pass and the
# copy engines in the backward pass (the per-pass automatic choice, forced here by setting its two minimums).
MODES = {
    "plain": {
        "sequence_sharding_overlap": False
    },
    "overlap": {
        "sequence_sharding_overlap": True
    },
    "copy-engine": {
        "sequence_sharding_overlap": True,
        "sequence_sharding_copy_engine": True,
        "sequence_sharding_copy_engine_min_tokens": 0
    },
    "copy-engine-backward": {
        "sequence_sharding_overlap": True,
        "sequence_sharding_copy_engine": True
    },
}


@contextlib.contextmanager
def copy_engine_minimums(mode):
    """For copy-engine-backward: no size uses the copy engines forward, every size uses them backward."""
    from deepspeed.module_inject import autotp_sequence_sharding as sharding
    saved = sharding.COPY_ENGINE_MIN_MULTIPLY_ADDS, sharding.COPY_ENGINE_MIN_MULTIPLY_ADDS_BACKWARD
    if mode == "copy-engine-backward":
        sharding.COPY_ENGINE_MIN_MULTIPLY_ADDS, sharding.COPY_ENGINE_MIN_MULTIPLY_ADDS_BACKWARD = float("inf"), 0
    try:
        yield
    finally:
        sharding.COPY_ENGINE_MIN_MULTIPLY_ADDS, sharding.COPY_ENGINE_MIN_MULTIPLY_ADDS_BACKWARD = saved


@contextlib.contextmanager
def count_copy_engine_calls(counter):
    """Count the calls of the copy-engine collectives into counter["calls"]."""
    from deepspeed.comm.copy_engine import CopyEngineGroup
    names = ("all_gather_matmul", "all_gather_start", "matmul_reduce_scatter_start")
    saved = {name: getattr(CopyEngineGroup, name) for name in names}

    def counted(method):

        def wrapper(*args, **kwargs):
            counter["calls"] += 1
            return method(*args, **kwargs)

        return wrapper

    for name in names:
        setattr(CopyEngineGroup, name, counted(saved[name]))
    try:
        yield
    finally:
        for name in names:
            setattr(CopyEngineGroup, name, saved[name])


def build_config(zero_stage, mode):
    return {
        "train_micro_batch_size_per_gpu": 2,
        "tensor_parallel": {
            "autotp_size": 2,
            "sequence_sharding": True,
            **MODES[mode],
        },
        # Plain SGD keeps a weight difference proportional to the gradient difference; Adam's normalized update
        # turns rounding-level gradient differences into 1e-5 weight differences.
        "optimizer": {
            "type": "SGD",
            "params": {
                "lr": 0.1
            },
        },
        "zero_allow_untested_optimizer": True,
        # The reference does not clip; DeepSpeed clips the global gradient norm to 1.0 by default.
        "gradient_clipping": 0.0,
        "zero_optimization": {
            "stage": zero_stage
        },
    }


def tp_shared_input_ids(vocab_size, seq_len, step):
    generator = torch.Generator().manual_seed(step)
    input_ids = torch.randint(0, vocab_size, (2, seq_len), generator=generator)
    input_ids = input_ids.to(get_accelerator().current_device_name())
    dist.broadcast(input_ids,
                   src=groups.get_tensor_model_parallel_src_rank(),
                   group=groups.get_tensor_model_parallel_group())
    return input_ids


@pytest.mark.parametrize("mode", list(MODES))
@pytest.mark.parametrize("zero_stage", [1, 3])
class TestAutoTPSequenceSharding(DistributedTest):
    world_size = 2

    def test_matches_single_device_training(self, zero_stage, mode):
        """AutoTP with sequence sharding trains like the unsharded model under plain PyTorch SGD."""
        skip_on_device()
        pytest.importorskip("transformers")
        uses_copy_engine = MODES[mode].get("sequence_sharding_copy_engine", False)
        if uses_copy_engine:
            from deepspeed.comm.copy_engine import copy_engine_available
            if not copy_engine_available():
                pytest.skip("copy-engine collectives need PyTorch symmetric memory")

        config, model = build_qwen3_model()
        reference = copy.deepcopy(model).to(get_accelerator().current_device_name())
        reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.1)
        engine, _, _, _ = deepspeed.initialize(model=model,
                                               model_parameters=model.parameters(),
                                               config=build_config(zero_stage, mode))
        # The decoder norms must run on this rank's half of the sequence, not on the whole of it.
        norm_lengths = []
        engine.module.model.layers[0].input_layernorm.register_forward_hook(
            lambda module, args, output: norm_lengths.append(args[0].shape[1]))

        # The first step learns which linears read each block input; later steps run them in one autograd node.
        forward_calls, backward_calls = {"calls": 0}, {"calls": 0}
        for step in range(3):
            input_ids = tp_shared_input_ids(config.vocab_size, 16, step)
            with copy_engine_minimums(mode):
                with count_copy_engine_calls(forward_calls):
                    loss = engine(input_ids=input_ids, labels=input_ids).loss
                with count_copy_engine_calls(backward_calls):
                    engine.backward(loss)
            engine.step()

            reference_loss = reference(input_ids=input_ids, labels=input_ids).loss
            reference_loss.backward()
            reference_optimizer.step()
            reference_optimizer.zero_grad()

            torch.testing.assert_close(loss.detach(), reference_loss.detach(), rtol=1e-5, atol=1e-5)

        assert norm_lengths == [16 // self.world_size] * 3
        # The copy engines must actually run where the mode asks for them, not fall back to NCCL.
        if mode == "copy-engine":
            assert forward_calls["calls"] > 0 and backward_calls["calls"] > 0
        elif mode == "copy-engine-backward":
            assert forward_calls["calls"] == 0 and backward_calls["calls"] > 0
        else:
            assert forward_calls["calls"] == 0 and backward_calls["calls"] == 0
        if zero_stage < 3:
            # The decoder norms see half of the sequence on each rank; their summed gradient keeps them equal to
            # the reference after the updates.
            norm = engine.module.model.layers[0].input_layernorm.weight
            reference_norm = reference.model.layers[0].input_layernorm.weight
            torch.testing.assert_close(norm.detach(), reference_norm.detach(), rtol=1e-5, atol=1e-6)


class TestAutoTPSequenceShardingRejects(DistributedTest):
    world_size = 2

    def test_sequence_not_divisible_by_tp(self):
        skip_on_device()
        pytest.importorskip("transformers")

        config, model = build_qwen3_model()
        engine, _, _, _ = deepspeed.initialize(model=model,
                                               model_parameters=model.parameters(),
                                               config=build_config(zero_stage=1, mode="overlap"))
        input_ids = tp_shared_input_ids(config.vocab_size, 15, 0)
        # pytest.raises inside a DistributedTest worker hangs the run when nothing is raised.
        try:
            engine(input_ids=input_ids, labels=input_ids)
            raised = False
        except ValueError as error:
            raised = "not divisible by the tensor-parallel size" in str(error)
        assert raised
