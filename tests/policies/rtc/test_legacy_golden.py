#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CPU golden tests for the legacy RTC timing and queue behavior."""

import logging
import math

import pytest
import torch

from lerobot.configs.types import RTCAttentionSchedule
from lerobot.policies.rtc.action_queue import ActionQueue
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.policies.rtc.latency_tracker import LatencyTracker
from lerobot.policies.rtc.modeling_rtc import RTCProcessor
from lerobot.rollout.inference.rtc import _normalize_prev_actions_length

_ACTION_QUEUE_LOGGER = "lerobot.policies.rtc.action_queue"
_BASE_ORIGINAL = torch.arange(0, 30, dtype=torch.float32).reshape(10, 3)
_BASE_PROCESSED = _BASE_ORIGINAL + 1000
_NEW_ORIGINAL = torch.arange(100, 130, dtype=torch.float32).reshape(10, 3)
_NEW_PROCESSED = _NEW_ORIGINAL + 1000


def _legacy_config() -> RTCConfig:
    return RTCConfig(
        enabled=True,
        prefix_attention_schedule=RTCAttentionSchedule.EXP,
        max_guidance_weight=10.0,
        execution_horizon=10,
    )


def _peek_next_action(queue: ActionQueue) -> torch.Tensor | None:
    """Read the next processed action without advancing the legacy queue."""
    with queue.lock:
        if queue.queue is None or queue.last_index >= len(queue.queue):
            return None
        return queue.queue[queue.last_index].clone()


def _queue_warning_messages(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [
        record.getMessage()
        for record in caplog.records
        if record.name == _ACTION_QUEUE_LOGGER and record.levelno >= logging.WARNING
    ]


def test_legacy_first_empty_queue_uses_wall_delay(caplog: pytest.LogCaptureFixture) -> None:
    queue = ActionQueue(_legacy_config())

    assert queue.queue is None
    assert queue.original_queue is None
    assert queue.get_action_index() == 0
    assert queue.qsize() == 0
    assert queue.empty() is True
    assert queue.get() is None
    assert queue.get_left_over() is None
    assert queue.get_processed_left_over() is None

    caplog.set_level(logging.WARNING, logger=_ACTION_QUEUE_LOGGER)
    index_before_inference = queue.get_action_index()
    queue.merge(
        _NEW_ORIGINAL,
        _NEW_PROCESSED,
        real_delay=4,
        action_index_before_inference=index_before_inference,
    )

    # Legacy behavior skips four actions even though the empty queue consumed none.
    assert queue.get_action_index() == 0
    assert queue.qsize() == 6
    assert queue.empty() is False
    assert torch.equal(_peek_next_action(queue), torch.tensor([1112.0, 1113.0, 1114.0]))
    assert torch.equal(queue.get_left_over(), _NEW_ORIGINAL[4:])
    assert torch.equal(queue.get_processed_left_over(), _NEW_PROCESSED[4:])
    assert _queue_warning_messages(caplog) == [
        "Indexes diff is not equal to real delay. indexes_diff=0, real_delay=4"
    ]


@pytest.mark.parametrize(
    (
        "consumed",
        "wall_delay",
        "expected_qsize",
        "expected_next_action",
        "expected_warning",
    ),
    [
        pytest.param(0, 0, 10, [1100.0, 1101.0, 1102.0], None, id="consume-0-match"),
        pytest.param(
            0,
            1,
            9,
            [1103.0, 1104.0, 1105.0],
            "Indexes diff is not equal to real delay. indexes_diff=0, real_delay=1",
            id="consume-0-mismatch",
        ),
        pytest.param(1, 1, 9, [1103.0, 1104.0, 1105.0], None, id="consume-1-match"),
        pytest.param(
            1,
            2,
            8,
            [1106.0, 1107.0, 1108.0],
            "Indexes diff is not equal to real delay. indexes_diff=1, real_delay=2",
            id="consume-1-mismatch",
        ),
        pytest.param(3, 3, 7, [1109.0, 1110.0, 1111.0], None, id="consume-3-match"),
        pytest.param(
            3,
            1,
            9,
            [1103.0, 1104.0, 1105.0],
            "Indexes diff is not equal to real delay. indexes_diff=3, real_delay=1",
            id="consume-3-mismatch",
        ),
        pytest.param(5, 5, 5, [1115.0, 1116.0, 1117.0], None, id="consume-5-match"),
        pytest.param(
            5,
            4,
            6,
            [1112.0, 1113.0, 1114.0],
            "Indexes diff is not equal to real delay. indexes_diff=5, real_delay=4",
            id="consume-5-mismatch",
        ),
    ],
)
def test_legacy_merge_uses_wall_delay_after_consumption(
    caplog: pytest.LogCaptureFixture,
    consumed: int,
    wall_delay: int,
    expected_qsize: int,
    expected_next_action: list[float],
    expected_warning: str | None,
) -> None:
    queue = ActionQueue(_legacy_config())
    queue.merge(_BASE_ORIGINAL, _BASE_PROCESSED, real_delay=0)
    index_before_inference = queue.get_action_index()

    popped = [queue.get() for _ in range(consumed)]
    assert all(action is not None for action in popped)
    assert all(torch.equal(action, _BASE_PROCESSED[index]) for index, action in enumerate(popped))
    index_after_inference = queue.get_action_index()
    assert index_after_inference - index_before_inference == consumed

    caplog.set_level(logging.WARNING, logger=_ACTION_QUEUE_LOGGER)
    caplog.clear()
    queue.merge(
        _NEW_ORIGINAL,
        _NEW_PROCESSED,
        real_delay=wall_delay,
        action_index_before_inference=index_before_inference,
    )

    # A mismatch only warns in legacy mode; real_delay remains the merge authority.
    assert queue.get_action_index() == 0
    assert queue.qsize() == expected_qsize
    assert torch.equal(_peek_next_action(queue), torch.tensor(expected_next_action))
    assert torch.equal(queue.get_left_over(), _NEW_ORIGINAL[wall_delay:])
    assert torch.equal(queue.get_processed_left_over(), _NEW_PROCESSED[wall_delay:])

    warnings = _queue_warning_messages(caplog)
    if expected_warning is None:
        assert warnings == []
    else:
        assert warnings == [expected_warning]


def test_legacy_clear_resets_all_queue_state() -> None:
    queue = ActionQueue(_legacy_config())
    queue.merge(_BASE_ORIGINAL, _BASE_PROCESSED, real_delay=0)
    for _ in range(3):
        assert queue.get() is not None

    assert queue.get_action_index() == 3
    assert queue.qsize() == 7
    assert torch.equal(_peek_next_action(queue), torch.tensor([1009.0, 1010.0, 1011.0]))
    assert torch.equal(queue.get_left_over(), _BASE_ORIGINAL[3:])
    assert torch.equal(queue.get_processed_left_over(), _BASE_PROCESSED[3:])

    queue.clear()

    assert queue.queue is None
    assert queue.original_queue is None
    assert queue.get_action_index() == 0
    assert queue.qsize() == 0
    assert queue.empty() is True
    assert queue.get() is None
    assert queue.get_left_over() is None
    assert queue.get_processed_left_over() is None


def test_legacy_latency_tracker_keeps_lifetime_max_after_window_eviction() -> None:
    tracker = LatencyTracker(maxlen=3)
    assert len(tracker) == 0
    assert tracker.max() == 0.0
    assert tracker.p95() == 0.0

    events = [
        (0.10, [0.10], 0.10, 0.10000000149011612),
        (0.20, [0.10, 0.20], 0.20, 0.19500000774860382),
        (0.30, [0.10, 0.20, 0.30], 0.30, 0.2900000214576721),
        (-1.0, [0.10, 0.20, 0.30], 0.30, 0.2900000214576721),
        (0.05, [0.20, 0.30, 0.05], 0.30, 0.2900000214576721),
        (0.06, [0.30, 0.05, 0.06], 0.30, 0.2759999930858612),
        (0.07, [0.05, 0.06, 0.07], 0.30, 0.0689999982714653),
    ]
    for latency, expected_window, expected_max, expected_p95 in events:
        tracker.add(latency)
        assert list(tracker._values) == pytest.approx(expected_window)
        assert tracker.max() == expected_max
        assert tracker.p95() == pytest.approx(expected_p95, rel=0.0, abs=1e-12)

    # The 0.30 sample has left the deque, but max() is a lifetime maximum until reset.
    assert list(tracker._values) == pytest.approx([0.05, 0.06, 0.07])
    assert tracker.max() == 0.30

    tracker.reset()
    assert len(tracker) == 0
    assert list(tracker._values) == []
    assert tracker.max() == 0.0
    assert tracker.p95() == 0.0


@pytest.mark.parametrize(
    ("latency_us", "expected_delay_steps"),
    [
        (0, 0),
        (124_000, 4),
        (125_000, 4),
        (133_333, 4),
        (133_334, 5),
        (140_000, 5),
    ],
)
def test_legacy_30hz_latency_buckets(latency_us: int, expected_delay_steps: int) -> None:
    latency = latency_us / 1_000_000
    time_per_chunk = 1.0 / 30.0
    delay_steps = math.ceil(latency / time_per_chunk) if latency else 0
    assert delay_steps == expected_delay_steps


@pytest.mark.parametrize(
    ("delay", "expected_prefix"),
    [
        (
            4,
            [
                1.0,
                1.0,
                1.0,
                1.0,
                0.676631987,
                0.433459193,
                0.256334066,
                0.133454680,
                0.054990519,
                0.012767323,
            ],
        ),
        (
            5,
            [
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                0.630948007,
                0.367706031,
                0.188770339,
                0.076745749,
                0.017591257,
            ],
        ),
    ],
)
def test_legacy_exp_prefix_weights(delay: int, expected_prefix: list[float]) -> None:
    processor = RTCProcessor(_legacy_config())
    actual = processor.get_prefix_weights(start=delay, end=10, total=50)
    expected = torch.tensor(expected_prefix + [0.0] * 40, dtype=torch.float32)

    assert actual.shape == (50,)
    assert actual.dtype == torch.float32
    assert actual.device.type == "cpu"
    assert torch.equal(actual, expected)


@pytest.mark.parametrize(
    ("target_steps", "expected"),
    [
        (2, [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]),
        (3, [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]]),
        (
            5,
            [
                [0.0, 1.0, 2.0],
                [3.0, 4.0, 5.0],
                [6.0, 7.0, 8.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
        ),
    ],
)
def test_legacy_normalize_previous_actions_length(target_steps: int, expected: list[list[float]]) -> None:
    previous_actions = torch.arange(9, dtype=torch.float32).reshape(3, 3)
    actual = _normalize_prev_actions_length(previous_actions, target_steps)

    assert actual.shape == (target_steps, 3)
    assert actual.dtype == previous_actions.dtype
    assert actual.device == previous_actions.device
    assert torch.equal(actual, torch.tensor(expected, dtype=torch.float32))
