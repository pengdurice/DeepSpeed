# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from typing import List, Optional, Literal
from deepspeed.runtime.config_utils import DeepSpeedConfigModel

PassName = Literal["z1", "z3", "autosp"]


class CompileConfig(DeepSpeedConfigModel):
    """ Configure compile settings """

    deepcompile: bool = False
    """ Turn on/off the DeepCompile mode """

    free_activation: bool = False
    """ Turn on/off the free activation mode """

    free_activation_threshold: int = 10 * 1024 * 1024
    """ In free activation mode, activations no less than this threshold (in byte) are eagerly freed """

    offload_activation: bool = False
    """ Turn on/off the activation offloading """

    offload_opt_states: bool = False
    """ Offload optimizer states (fp32 master parameters and Adam moments) to pinned host memory
    during forward/backward and reload them for the optimizer step. Offloading engages at step 1
    with the states emptied before planning, so jobs that only fit with offloading are supported;
    the planner then keeps resident whatever the memory budget allows. Schedules the
    move_opt_states pass in place of the default prefetch/selective-gather passes; mutually
    exclusive with offload_parameters. Designed for gradient_accumulation_steps=1: the compiled
    graph runs once per micro-batch, so with accumulation the full offload/reload cycle repeats
    every micro-batch instead of once per optimizer step. """

    offload_opt_states_tune: bool = False
    """ With offload_opt_states_combine: after each compile phase, measure real step times and
    try one more offloaded piece, keeping the fastest configuration seen. Also offloads more when
    a phase's peak exceeds the memory budget the plan assumed. Costs one recompile per round. """

    offload_opt_states_tune_rounds: int = 3
    """ Maximum tuning rounds (recompiles) for offload_opt_states_tune. """

    offload_opt_states_combine: bool = False
    """ With offload_opt_states: after the offload-only phase, recompile once more at the
    warmup step with all passes combined -- offload first (it frees the memory), then
    prefetch and selective gather spending the freed budget. """

    double_buffer: bool = True
    """ Turn on/off the double buffering """

    symmetric_memory: bool = False
    """ Turn on/off the symmetric memory """

    debug_log: bool = False
    """ Turn on/off the graph dumping """

    offload_parameters: bool = False
    """ Turn on/off the parameter offloading """

    sync_before_reduce: bool = False
    """ Turn on/off the sync before reduce """

    sync_after_reduce: bool = False
    """ Turn on/off the sync after reduce """

    sync_before_allgather: bool = False
    """ Turn on/off the sync before allgather """

    sync_after_allgather: bool = False
    """ Turn on/off the sync after allgather """

    keep_int_input_tensors: bool = True
    """ Keep real values for int tensors in InputStorage instead of using dummy values """

    keep_all_input_tensors: bool = False
    """ Keep real values for all input tensors in InputStorage instead of using dummy values """

    passes: Optional[List[PassName]] = None
    """ Composes different optimizations. """
