# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import pytest

from deepspeed.compile.offload_tuner import OffloadTuner, SETTLE_STEPS, MEASURE_STEPS

GB = 1024**3
ROUND = SETTLE_STEPS + MEASURE_STEPS


def _run_round(tuner, pieces, step_time, peak=1 * GB):
    """Feed one full round of steps and return the tuner's verdict."""
    verdict = None
    for _ in range(ROUND):
        verdict = tuner.observe(pieces, step_time, peak)
    return verdict


def test_stops_when_first_round_is_already_best():
    # The common case on one node: what memory requires is also the fastest, so the second
    # round is slower and the tuner reverts to the first without a third recompile. The caller
    # pins that count exactly, so a risen estimate cannot override a configuration that ran.
    tuner = OffloadTuner(max_rounds=3, budget_bytes=100 * GB)

    assert _run_round(tuner, pieces=1, step_time=2.0) == 2
    assert _run_round(tuner, pieces=2, step_time=2.5) == 1
    assert tuner.done
    assert tuner.best_pieces == 1


def test_keeps_climbing_while_each_round_improves():
    tuner = OffloadTuner(max_rounds=3, budget_bytes=100 * GB)

    assert _run_round(tuner, pieces=1, step_time=3.0) == 2
    assert _run_round(tuner, pieces=2, step_time=2.5) == 3
    # Third round hits max_rounds and settles on the best seen so far.
    assert _run_round(tuner, pieces=3, step_time=2.0) is None
    assert tuner.done
    assert tuner.best_pieces == 3


def test_offloads_more_when_peak_exceeds_the_budget():
    # Safety beats speed: over budget, the next round frees more even though it was slower.
    tuner = OffloadTuner(max_rounds=3, budget_bytes=10 * GB)

    assert _run_round(tuner, pieces=2, step_time=2.0, peak=9 * GB) == 3
    assert _run_round(tuner, pieces=3, step_time=5.0, peak=11 * GB) == 4
    assert not tuner.done
    assert tuner.best_pieces == 3


def test_ignores_settle_steps_and_trims_stalls():
    # The first steps after a recompile are warmup, and one stalled step must not decide a round.
    tuner = OffloadTuner(max_rounds=2, budget_bytes=100 * GB)
    for step_time in [90.0] * SETTLE_STEPS + [2.0] * (MEASURE_STEPS - 1) + [90.0]:
        verdict = tuner.observe(1, step_time, 1 * GB)
    assert verdict == 2
    assert tuner.best_time == 2.0


def test_scores_a_round_by_its_mean_not_one_mode():
    # Several configurations alternate fast and slow steps with period 2. A median returns
    # whichever value the window happened to hold more of; the mean reflects what the run costs.
    tuner = OffloadTuner(max_rounds=2, budget_bytes=100 * GB)
    alternating = [2.0, 4.0] * MEASURE_STEPS
    for step_time in [90.0] * SETTLE_STEPS + alternating[:MEASURE_STEPS]:
        tuner.observe(1, step_time, 1 * GB)
    assert tuner.best_time == pytest.approx(3.0)


def test_restarts_the_window_when_a_phase_launches():
    # The piece count is the same before and after the combined phase engages, so a window keyed
    # only on the count spanned two schedules and scored the wrong one.
    tuner = OffloadTuner(max_rounds=3, budget_bytes=100 * GB)
    for _ in range(ROUND - 1):
        tuner.observe(1, 9.0, 1 * GB)  # the previous schedule, slower
    assert tuner.observe(1, 9.0, 1 * GB, new_phase=True) is None
    assert len(tuner.samples) == 1

    for _ in range(ROUND - 1):
        tuner.observe(1, 2.0, 1 * GB)
    assert tuner.best_time == pytest.approx(2.0)


def test_waits_for_a_full_round_before_deciding():
    tuner = OffloadTuner(max_rounds=3, budget_bytes=100 * GB)
    for _ in range(ROUND - 1):
        assert tuner.observe(1, 2.0, 1 * GB) is None


def test_restarts_the_window_when_the_piece_count_changes():
    tuner = OffloadTuner(max_rounds=3, budget_bytes=100 * GB)
    for _ in range(ROUND - 1):
        tuner.observe(1, 2.0, 1 * GB)
    # A recompile lands before the window filled; the partial samples must not carry over.
    assert tuner.observe(2, 2.0, 1 * GB) is None
    assert len(tuner.samples) == 1
