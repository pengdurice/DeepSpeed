# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Deliver the parameters updated by a ZeRO stage 1/2 step while the next forward pass runs
(``zero_optimization.overlap_param_gather``).

After a stage 1/2 step each data-parallel rank holds the new bit16 values of its own contiguous partition of every
flat parameter buffer, and the other ranks need them before their next forward pass. One all-gather (the default)
delivers every part of the model at the same time, so the forward pass cannot start until all of it has arrived.
Here each owner broadcasts its partition in pieces, in the order in which the forward pass reads the parameters, and
a forward pre-hook waits only for the pieces that hold the parameters of its module. A rank receives and sends as much data as in the all-gather,
but in each broadcast the owner receives nothing and one rank sends nothing, so the pieces together take up to
dp / (dp - 1) as long; the forward pass runs meanwhile.
"""

from torch import nn

from deepspeed import comm as dist
from deepspeed.utils import log_dist


def _runs_layer_list(module):
    """True when ``module`` or a module inside it is a list of two or more layers with parameters (the decoder layers
    of a transformer). Waiting for all its parameters before it runs would wait for most of the model."""
    for inner in module.modules():
        if isinstance(inner, nn.ModuleList):
            layers_with_params = [layer for layer in inner if next(layer.parameters(), None) is not None]
            if len(layers_with_params) >= 2:
                return True
    return False


class ForwardOrderParamGather:
    """Broadcasts of the ZeRO 1/2 bit16 partitions, ordered and waited for by the forward pass of ``module``.

    ``groups_flat[i]``: the flat bit16 buffer of parameter group i; ``partitioned_groups[i][r]``: the view of data-parallel
    rank r's partition of it; ``flat_params[i]`` and ``flat_offsets[i]``: the parameters of group i in buffer order and
    their element offsets; ``dp_groups[i]``: its data-parallel process group; ``piece_numel``: elements per broadcast.
    """

    def __init__(self, module, groups_flat, partitioned_groups, flat_params, flat_offsets, dp_groups, piece_numel):
        model_order = {id(param): index for index, param in enumerate(module.parameters())}
        unused = len(model_order)
        pieces = []
        params_of_piece = []
        for group_index, (flat, partitions) in enumerate(zip(groups_flat, partitioned_groups)):
            group = dp_groups[group_index]
            if dist.get_world_size(group=group) == 1:
                continue
            spans = [(offset, offset + param.numel(), param)
                     for param, offset in zip(flat_params[group_index], flat_offsets[group_index])]
            for rank, partition in enumerate(partitions):
                partition_start = (partition.data_ptr() - flat.data_ptr()) // flat.element_size()
                src = dist.get_global_rank(group, rank)
                for piece_start in range(0, partition.numel(), piece_numel):
                    piece_numel_here = min(piece_numel, partition.numel() - piece_start)
                    start = partition_start + piece_start
                    end = start + piece_numel_here
                    overlapping = []
                    for span_start, span_end, param in spans:
                        if span_start < end and span_end > start:
                            overlapping.append(param)
                    # The forward pass needs a piece as soon as it needs the first parameter in it.
                    first_use = min((model_order.get(id(param), unused) for param in overlapping), default=unused)
                    view = partition.narrow(0, piece_start, piece_numel_here)
                    pieces.append((first_use, group_index, start, view, src, group))
                    params_of_piece.append(overlapping)
        order = sorted(range(len(pieces)), key=lambda k: pieces[k][:3])
        self.pieces = [pieces[k][3:] for k in order]
        # The last piece each parameter needs, in broadcast order.
        self.last_piece = {}
        for position, k in enumerate(order):
            for param in params_of_piece[k]:
                self.last_piece[id(param)] = position
        self.works = []
        self.waited = 0
        self.launch_pending = False
        self.hooks = []
        for submodule in module.modules():
            params = submodule.parameters(recurse=not _runs_layer_list(submodule))
            needed = max((self.last_piece.get(id(param), -1) for param in params), default=-1)
            if needed < 0:
                continue
            # Before the module's other pre-hooks, which may already read its parameters.
            self.hooks.append(submodule.register_forward_pre_hook(self._make_hook(needed), prepend=True))
        log_dist(
            f"ZeRO overlap_param_gather: {len(self.pieces)} broadcasts of up to {piece_numel} elements per step, "
            f"waited for by {len(self.hooks)} module pre-hooks",
            ranks=[0])

    def _make_hook(self, needed):

        def hook(module, args):
            self.wait_until(needed)

        return hook

    def start(self):
        """Called after the owners have written their updated partitions. The broadcasts are issued by the first wait,
        normally the first forward pre-hook: a device synchronize between the step and the next forward pass (for
        example the timers of wall_clock_breakdown) would otherwise wait for all of them."""
        self.wait_all()
        self.launch_pending = True

    def _launch(self):
        self.works = [dist.broadcast(view, src, group=group, async_op=True) for view, src, group in self.pieces]
        self.waited = 0
        self.launch_pending = False

    def wait_until(self, position):
        """Make the current stream wait for the pieces up to ``position`` in broadcast order."""
        if self.launch_pending:
            self._launch()
        while self.waited <= position and self.waited < len(self.works):
            self.works[self.waited].wait()
            self.waited += 1

    def wait_all(self):
        self.wait_until(len(self.pieces) - 1)
