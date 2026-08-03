# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Pick how many optimizer-state pieces to offload by measuring real steps.

The offload pass, prefetch and selective gather form a coupled plan: offloading frees memory,
the gather passes spend it, and their spending changes the peaks the offload plan was made
against. Re-planning from the post-gather memory picture does not converge -- the gather passes
consume whatever is freed, so each round offloads more until everything is offloaded, measured
at 2x slower than stopping at what memory requires.

This controller steers on measured step time instead. It starts at the count memory requires,
tries one more piece per round, keeps the best round and stops as soon as a round is worse. The
exception is memory pressure: if a round's peak exceeds the budget the plan assumed, the next
round offloads more whatever the timing says, because the alternative is running out of memory.
"""

import statistics
from typing import List, Optional

import deepspeed.comm as dist


def _log(msg: str) -> None:
    # Not log_rank0: it calls dist.get_rank() unconditionally, and the decision logic is
    # exercised by single-process tests where distributed is never initialized.
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(msg)


# Steps to discard after a recompile before the timings mean anything.
SETTLE_STEPS = 2
# Steps to average over. Several configurations alternate fast and slow steps, so a couple of
# samples cannot tell two rounds apart.
MEASURE_STEPS = 6
# Above this multiple of the window median a sample is a stall or a compile, not a step. Trimming
# them lets the round be scored by its mean -- what total training time actually depends on --
# rather than by a median, which on a series alternating fast and slow just returns whichever
# value happened to appear more often in the window.
OUTLIER_FACTOR = 3.0


class OffloadTuner:
    """Chooses the piece count for the next compile phase from the current phase's step times."""

    def __init__(self, max_rounds: int = 3, budget_bytes: Optional[float] = None):
        self.max_rounds = max_rounds
        self.budget_bytes = budget_bytes
        self.round = 0
        self.done = False
        self.current_pieces: Optional[int] = None
        self.samples: List[float] = []
        self.best_pieces: Optional[int] = None
        self.best_time: Optional[float] = None

    def observe(self, pieces: int, step_time: float, peak_bytes: float, new_phase: bool = False) -> Optional[int]:
        """Feed one training step. Returns a piece count to recompile with, or None to continue."""
        if self.done:
            return None

        # Restart on any phase launch, not only when the piece count changes. The count is the
        # same before and after the combined phase engages, so keying off it alone let the first
        # round average steps from two different schedules and score the wrong configuration.
        if new_phase or pieces != self.current_pieces:
            self.current_pieces = pieces
            self.samples = []

        self.samples.append(step_time)
        if len(self.samples) < SETTLE_STEPS + MEASURE_STEPS:
            return None

        window = self.samples[SETTLE_STEPS:]
        floor = statistics.median(window)
        kept = [t for t in window if t <= OUTLIER_FACTOR * floor]
        measured = statistics.mean(kept)
        self.samples = []
        self.round += 1
        improved = self.best_time is None or measured < self.best_time
        if improved:
            self.best_time, self.best_pieces = measured, pieces

        over_budget = self.budget_bytes is not None and peak_bytes > self.budget_bytes
        if over_budget:
            # Safety beats speed: the run exceeded the memory the plan assumed, so free more even
            # if this round was slower. Accept the current count as the fallback either way.
            self.best_time, self.best_pieces = measured, pieces
            return self._advance(pieces + 1, f"peak {peak_bytes / 1e9:.1f}GB over budget")

        if not improved:
            return self._settle(pieces, f"{measured:.3f}s did not beat {self.best_time:.3f}s")

        return self._advance(pieces + 1, f"{measured:.3f}s improved, trying one more piece")

    def _advance(self, pieces: int, why: str) -> Optional[int]:
        if self.round >= self.max_rounds:
            return self._settle(self.current_pieces, f"reached max_rounds={self.max_rounds}")
        _log(f"[DeepCompile] offload tuner: round {self.round}, recompiling with {pieces} pieces ({why})")
        return pieces

    def _settle(self, pieces: int, why: str) -> Optional[int]:
        self.done = True
        _log(f"[DeepCompile] offload tuner: settling on {self.best_pieces} pieces ({why})")
        # Recompile only when the winner is not what is already running.
        return None if self.best_pieces == pieces else self.best_pieces
