#!/usr/bin/env python

from argparse import Namespace

import pytest
import torch

from examples.inference.evaluate_pi05_rtc_replay import (
    begin_queue_transition,
    build_parser,
    finish_queue_transition,
    replay_initial_chunk,
    validate_args,
)


def _chunks() -> tuple[torch.Tensor, torch.Tensor]:
    previous = torch.arange(50, dtype=torch.float32).unsqueeze(1)
    current = (100 + torch.arange(50, dtype=torch.float32)).unsqueeze(1)
    return previous, current


@pytest.mark.parametrize(
    ("queue_threshold", "expected_interval", "expected_first", "expected_last"),
    [
        (20, 30, 105.0, 134.0),
        (30, 20, 105.0, 124.0),
        (35, 15, 105.0, 119.0),
        (40, 10, 105.0, 114.0),
        (45, 5, 105.0, 109.0),
    ],
)
def test_fixed_delay_replay_uses_real_action_queue_indices(
    queue_threshold: int,
    expected_interval: int,
    expected_first: float,
    expected_last: float,
) -> None:
    previous, current = _chunks()
    start = begin_queue_transition(
        previous,
        previous,
        queue_threshold=queue_threshold,
        execution_horizon=10,
    )

    trace = finish_queue_transition(
        start,
        current,
        current,
        consumed_during_inference=5,
    )

    assert trace.initial_merge.actual_consumed_steps == 0
    assert trace.initial_merge.merge_skip == 0
    assert start.replan_interval_steps == expected_interval
    assert start.inference_start.original_leftover[0, 0].item() == expected_interval
    assert trace.old_actions_during_inference[:, 0].tolist() == list(
        map(float, range(expected_interval, expected_interval + 5))
    )
    assert trace.transition_merge.actual_consumed_steps == 5
    assert trace.transition_merge.merge_skip == 5
    assert trace.executed_model_indices == list(range(5, 5 + expected_interval))
    assert trace.executed_current_actions[0, 0].item() == expected_first
    assert trace.executed_current_actions[-1, 0].item() == expected_last
    assert trace.underflow_count == 0


def test_q20_q40_and_q45_encode_expected_cadence_windows() -> None:
    previous, current = _chunks()
    q20 = begin_queue_transition(
        previous,
        previous,
        queue_threshold=20,
        execution_horizon=10,
    )
    q40 = begin_queue_transition(
        previous,
        previous,
        queue_threshold=40,
        execution_horizon=10,
    )
    q45 = begin_queue_transition(
        previous,
        previous,
        queue_threshold=45,
        execution_horizon=10,
    )

    trace20 = finish_queue_transition(q20, current, current, consumed_during_inference=4)
    trace40 = finish_queue_transition(q40, current, current, consumed_during_inference=4)
    trace45 = finish_queue_transition(q45, current, current, consumed_during_inference=4)

    assert 30 / q20.replan_interval_steps == 1.0
    assert 30 / q40.replan_interval_steps == 3.0
    assert 30 / q45.replan_interval_steps == 6.0
    assert trace20.executed_model_indices == list(range(4, 34))
    assert trace40.executed_model_indices == list(range(4, 14))
    assert trace45.executed_model_indices == list(range(4, 9))


@pytest.mark.parametrize(
    ("queue_threshold", "next_consumed", "expected_indices"),
    [
        (20, 5, list(range(35))),
        (45, 4, list(range(9))),
        (45, 5, list(range(10))),
    ],
)
def test_initial_chunk_runs_until_second_inference_result(
    queue_threshold: int,
    next_consumed: int,
    expected_indices: list[int],
) -> None:
    previous, _ = _chunks()

    trace = replay_initial_chunk(
        previous,
        previous,
        queue_threshold=queue_threshold,
        next_inference_consumed_steps=next_consumed,
        execution_horizon=10,
    )

    assert trace.initial_merge.merge_skip == 0
    assert trace.executed_model_indices == expected_indices
    assert trace.executed_current_actions[:, 0].tolist() == list(map(float, expected_indices))
    assert trace.underflow_count == 0


@pytest.mark.parametrize("queue_threshold", [-1, 50, 51])
def test_queue_threshold_outside_chunk_is_rejected(queue_threshold: int) -> None:
    previous, _ = _chunks()
    with pytest.raises(ValueError, match="queue_threshold"):
        begin_queue_transition(
            previous,
            previous,
            queue_threshold=queue_threshold,
            execution_horizon=10,
        )


def test_cli_rejects_delay_at_or_beyond_execution_horizon() -> None:
    args: Namespace = build_parser().parse_args(["--delays", "5", "10"])

    with pytest.raises(ValueError, match="actual-consumed-steps must be in"):
        validate_args(args)
