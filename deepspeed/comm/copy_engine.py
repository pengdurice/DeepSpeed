# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""All-gather + GEMM and GEMM + reduce-scatter that move data with the GPU copy engines.

An NCCL collective runs as a kernel that occupies SMs, so a GEMM running at the same time slows down and overlapping
the two hides little. Copies between GPUs of one node through PyTorch symmetric memory run on the copy engines
instead, so the GEMMs keep their SMs. Both operations work on the rows of a 2-D activation split into one contiguous
block of rows per rank.

    all_gather_matmul:      gather the row blocks of all ranks while the GEMMs run on the blocks already here
    matmul_reduce_scatter:  compute the partial result for each peer's rows and copy it into that peer's buffer
                            while the next partial is computed; then each rank sums the partials it received

Each operation uses one of two symmetric buffers in turn, with one GPU barrier per call: a rank reaches the barrier
of call i + 1 only after its copies of call i are done, so the buffer of call i is free again at call i + 2.
"""

import os

import torch

from deepspeed import comm as dist
from deepspeed.accelerator import get_accelerator

# all_gather_matmul runs its GEMMs on groups of at least this many rows: on H200, GEMMs with an 8,192-wide input
# split into 2,048-row blocks ran 14-18% slower than one GEMM, and 4,096-row blocks lost about 1%.
_MIN_GEMM_ROWS = 4096


def copy_engine_available():
    """Whether this PyTorch has symmetric memory and the accelerator has streams to run the copies on."""
    try:
        import torch.distributed._symmetric_memory  # noqa: F401
    except ImportError:
        return False
    return not get_accelerator().is_synchronized_device()


def group_within_one_node(group):
    """Whether every rank of ``group`` is on this node; symmetric memory maps peers' memory, so it works only there.
    Decided from the global ranks alone, so every rank of the group reaches the same answer."""
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", get_accelerator().device_count()))
    world_size = dist.get_world_size(group=group)
    nodes = {dist.get_global_rank(group, rank) // local_world_size for rank in range(world_size)}
    return len(nodes) == 1


class _SymmetricBuffers:
    """Two symmetric buffers, used in turn, grown (a collective on every rank of the group) when a call needs more."""

    def __init__(self, group_name):
        self.group_name = group_name
        self.buffers = []
        self.calls = 0

    def next(self, numel, dtype, device):
        """The next buffer as (tensor, rendezvous handle, view): ``view(peer, shape, offset)`` is a peer's part of it."""
        import torch.distributed._symmetric_memory as symm_mem

        if not self.buffers or self.buffers[0][0].numel() < numel or self.buffers[0][0].dtype != dtype:
            self.buffers = []
            for _ in range(2):
                tensor = symm_mem.empty(numel, dtype=dtype, device=device)
                handle = symm_mem.rendezvous(tensor, self.group_name)
                self.buffers.append((tensor, handle, _PeerViews(handle, dtype)))
        buffer = self.buffers[self.calls % 2]
        self.calls += 1
        return buffer


class _PeerViews:
    """Views of the peers' parts of one symmetric buffer, made once: ``get_buffer`` builds a new tensor on every call,
    and at small sizes the host time of the copy-engine collectives limits the step."""

    def __init__(self, handle, dtype):
        self.handle = handle
        self.dtype = dtype
        self.cache = {}

    def __call__(self, peer, shape, offset):
        key = (peer, shape, offset)
        view = self.cache.get(key)
        if view is None:
            view = self.handle.get_buffer(peer, shape, self.dtype, offset)
            self.cache[key] = view
        return view


class CopyEngineGroup:
    """Copy-engine collectives over a group whose ranks share one node."""

    def __init__(self, group):
        # These collectives only copy between peers and need no multicast. With multicast on, symmetric memory asks
        # for the multicast objects that NCCL's NVLink SHARP collectives also use, and the rendezvous failed with
        # "CUDA driver error: out of memory" with most of the GPU memory free (8 x H200, torch 2.8).
        os.environ.setdefault("TORCH_SYMM_MEM_DISABLE_MULTICAST", "1")
        self.group = group
        self.rank = dist.get_rank(group=group)
        self.world_size = dist.get_world_size(group=group)
        self.copy_stream = get_accelerator().Stream()
        self.gather_done = get_accelerator().Event()
        self.gather_buffers = _SymmetricBuffers(group.group_name)
        self.scatter_buffers = _SymmetricBuffers(group.group_name)
        self.events = [get_accelerator().Event() for _ in range(2 * self.world_size)]
        self._groups_cache = {}

    def _peers(self):
        return [(self.rank + step) % self.world_size for step in range(1, self.world_size)]

    def _row_groups(self, n):
        """(first, last) peer blocks of each GEMM of all_gather_matmul, in arrival order."""
        groups = self._groups_cache.get(n)
        if groups is None:
            blocks_per_gemm = max(1, -(-_MIN_GEMM_ROWS // n))
            groups = []
            for run in (range(self.rank + 1, self.world_size), range(0, self.rank)):
                for first in range(0, len(run), blocks_per_gemm):
                    group = run[first:first + blocks_per_gemm]
                    groups.append((group[0], group[-1]))
            self._groups_cache[n] = groups
        return groups

    def all_gather_matmul(self, local, weights_t):
        """``local``: this rank's [n, k] rows. Returns the gathered [world * n, k] rows and ``local_full @ w`` for
        each [k, m] matrix in ``weights_t``; the GEMMs on each peer's rows run once that block has arrived."""
        n, k = local.shape
        full = local.new_empty((self.world_size * n, k))
        outs = [local.new_empty((self.world_size * n, w.shape[1])) for w in weights_t]
        tensor, handle, view = self.gather_buffers.next(n * k, local.dtype, local.device)
        tensor[:n * k].view(n, k).copy_(local)
        handle.barrier(channel=0)
        compute = get_accelerator().current_stream()
        self.copy_stream.wait_stream(compute)
        # The peer blocks arrive in the order rank + 1, ..., world - 1, 0, ..., rank - 1: two runs of adjacent rows. One
        # GEMM per group of adjacent blocks with at least _MIN_GEMM_ROWS rows, once its last block has arrived.
        groups = self._row_groups(n)
        with get_accelerator().stream(self.copy_stream):
            for index, (first, last) in enumerate(groups):
                for peer in range(first, last + 1):
                    full[peer * n:(peer + 1) * n].copy_(view(peer, (n, k), 0))
                self.events[index].record(self.copy_stream)
        own = slice(self.rank * n, (self.rank + 1) * n)
        full[own].copy_(local)
        for w, out in zip(weights_t, outs):
            torch.matmul(local, w, out=out[own])
        for index, (first, last) in enumerate(groups):
            compute.wait_event(self.events[index])
            rows = slice(first * n, (last + 1) * n)
            for w, out in zip(weights_t, outs):
                torch.matmul(full[rows], w, out=out[rows])
        # No record_stream for ``full``: the compute stream has waited for every copy into it, so freeing it in compute
        # order is safe. record_stream made the caching allocator hold such blocks until the copy stream passed them,
        # and near the memory limit it then reserved new memory and retried several times per step.
        return full, outs

    def all_gather_start(self, local):
        """Start gathering the [n, k] rows of all ranks into a new [world * n, k] tensor, without GEMMs. Call
        all_gather_wait before the tensor is read and before the next all-gather; one may be in flight at a time."""
        n, k = local.shape
        full = local.new_empty((self.world_size * n, k))
        tensor, handle, view = self.gather_buffers.next(n * k, local.dtype, local.device)
        tensor[:n * k].view(n, k).copy_(local)
        handle.barrier(channel=0)
        compute = get_accelerator().current_stream()
        full[self.rank * n:(self.rank + 1) * n].copy_(local)
        # On the copy stream, ahead of the copies of a reduce-scatter started next: copies on two streams at once take
        # about twice as long as on one.
        self.copy_stream.wait_stream(compute)
        with get_accelerator().stream(self.copy_stream):
            for peer in self._peers():
                full[peer * n:(peer + 1) * n].copy_(view(peer, (n, k), 0))
            self.gather_done.record(self.copy_stream)
        # No record_stream (see all_gather_matmul): the caller waits in all_gather_wait before it reads or frees it.
        return full

    def all_gather_wait(self):
        # Also keeps the rule of the two buffers: this rank's copies are done before its next barrier.
        get_accelerator().current_stream().wait_event(self.gather_done)

    def matmul_reduce_scatter(self, rows_total, width, dtype, device, partial_into):
        """Sum over the ranks of a [rows_total, width] partial result and keep this rank's rows.
        ``partial_into(rows, out)`` writes this rank's partial result for the row slice ``rows`` into ``out``."""
        return self.matmul_reduce_scatter_finish(
            self.matmul_reduce_scatter_start(rows_total, width, dtype, device, partial_into))

    def matmul_reduce_scatter_start(self, rows_total, width, dtype, device, partial_into):
        """The partial results and their copies to the peers of matmul_reduce_scatter. Kernels the caller queues before
        matmul_reduce_scatter_finish run while the copies finish; no other reduce-scatter may run in between."""
        n = rows_total // self.world_size
        tensor, handle, view = self.scatter_buffers.next(self.world_size * n * width, dtype, device)
        compute = get_accelerator().current_stream()
        # One staging buffer per peer, so that a partial GEMM shorter than its copy never waits for the copy of an
        # earlier partial.
        staging = torch.empty((self.world_size - 1, n, width), dtype=dtype, device=device)
        for step, peer in enumerate(self._peers()):
            partial = staging[step]
            partial_into(slice(peer * n, (peer + 1) * n), partial)
            self.events[step].record(compute)
            self.copy_stream.wait_event(self.events[step])
            with get_accelerator().stream(self.copy_stream):
                view(peer, (n, width), self.rank * n * width).copy_(partial)
        own = view(self.rank, (n, width), self.rank * n * width)
        partial_into(slice(self.rank * n, (self.rank + 1) * n), own)
        # ``staging`` stays referenced until matmul_reduce_scatter_finish has made the compute stream wait for its
        # copies, instead of record_stream (see all_gather_matmul).
        return tensor, handle, n, width, device, staging

    def matmul_reduce_scatter_finish(self, pending):
        tensor, handle, n, width, device, _staging = pending
        get_accelerator().current_stream().wait_stream(self.copy_stream)
        handle.barrier(channel=0)
        received = tensor[:self.world_size * n * width].view(self.world_size, n, width)
        # Under torch.autocast, sum is one of the ops that run in fp32: the result would come back fp32 and turn
        # the residual stream and every later GEMM fp32. A bf16 sum accumulates in fp32 and returns bf16.
        with torch.autocast(device_type=device.type, enabled=False):
            return received.sum(dim=0)
