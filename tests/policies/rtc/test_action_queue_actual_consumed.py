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

"""Tests for atomic actual-consumed ActionQueue merges."""

import logging

import pytest
import torch

from lerobot.policies.rtc.action_queue import ActionQueue, ActionQueueSnapshot
from lerobot.policies.rtc.configuration_rtc import RTCConfig


def _actions(steps: int = 10, offset: float = 0.0) -> torch.Tensor:
    return torch.arange(steps * 3, dtype=torch.float32).reshape(steps, 3) + offset


def _consume(queue: ActionQueue, count: int) -> None:
    for _ in range(count):
        assert queue.get() is not None


def test_first_empty_queue_merge_does_not_skip_wall_time_equivalent_actions() -> None:
    queue = ActionQueue(RTCConfig(enabled=True))
    inference_start = queue.snapshot()
    original = _actions()
    processed = _actions(offset=100.0)

    result = queue.merge_actual_consumed(original, processed, inference_start)

    assert result.merged is True
    assert result.stale is False
    assert result.actual_consumed_steps == 0
    assert result.merge_skip == 0
    assert result.skip_was_clamped is False
    assert result.expected_generation == 0
    assert result.observed_generation == 0
    assert result.generation_after == 1
    assert result.queue_size_after == 10
    assert torch.equal(queue.get(), processed[0])


@pytest.mark.parametrize("consumed", [0, 1, 3, 5])
def test_merge_skip_is_number_consumed_during_inference(consumed: int) -> None:
    queue = ActionQueue(RTCConfig(enabled=True))
    base = _actions()
    queue.merge(base, base + 100.0, real_delay=0)
    inference_start = queue.snapshot()
    _consume(queue, consumed)
    new_original = _actions(offset=1000.0)
    new_processed = _actions(offset=2000.0)

    result = queue.merge_actual_consumed(new_original, new_processed, inference_start)

    assert result.merged is True
    assert result.actual_consumed_steps == consumed
    assert result.merge_skip == consumed
    assert result.skip_was_clamped is False
    assert result.queue_size_after == len(new_processed) - consumed
    assert torch.equal(queue.get(), new_processed[consumed])


def test_consumption_after_diagnostic_snapshot_is_included_atomically() -> None:
    queue = ActionQueue(RTCConfig(enabled=True))
    base = _actions()
    queue.merge(base, base, real_delay=0)
    inference_start = queue.snapshot()
    _consume(queue, 2)

    # This emulates one more control-loop get after a non-authoritative
    # diagnostic read but before the producer acquires the merge lock.
    diagnostic_after_inference = queue.snapshot()
    assert diagnostic_after_inference.total_consumed - inference_start.total_consumed == 2
    _consume(queue, 1)

    new_actions = _actions(offset=1000.0)
    result = queue.merge_actual_consumed(new_actions, new_actions, inference_start)

    assert result.actual_consumed_steps == 3
    assert result.merge_skip == 3
    assert torch.equal(queue.get(), new_actions[3])


def test_clear_makes_inference_result_stale_and_does_not_merge(
    caplog: pytest.LogCaptureFixture,
) -> None:
    queue = ActionQueue(RTCConfig(enabled=True))
    base = _actions()
    queue.merge(base, base + 100.0, real_delay=0)
    inference_start = queue.snapshot()
    queue.clear()
    caplog.set_level(logging.WARNING, logger="lerobot.policies.rtc.action_queue")

    new_actions = _actions(offset=1000.0)
    result = queue.merge_actual_consumed(new_actions, new_actions, inference_start)

    assert result.merged is False
    assert result.stale is True
    assert result.expected_generation == 1
    assert result.observed_generation == 2
    assert result.generation_after == 2
    assert result.merge_skip == 0
    assert result.queue_size_after == 0
    assert queue.empty() is True
    assert queue.snapshot().generation == 2
    assert "Discarding stale inference result" in caplog.text


def test_intervening_merge_makes_inference_result_stale() -> None:
    queue = ActionQueue(RTCConfig(enabled=True))
    base = _actions()
    queue.merge(base, base, real_delay=0)
    inference_start = queue.snapshot()
    intervening = _actions(offset=500.0)
    queue.merge(intervening, intervening, real_delay=0)

    rejected = _actions(offset=1000.0)
    result = queue.merge_actual_consumed(rejected, rejected, inference_start)

    assert result.merged is False
    assert result.stale is True
    assert result.observed_generation == 2
    assert torch.equal(queue.get(), intervening[0])


def test_consumed_steps_beyond_chunk_are_clamped_and_reported(
    caplog: pytest.LogCaptureFixture,
) -> None:
    queue = ActionQueue(RTCConfig(enabled=True))
    base = _actions(steps=8)
    queue.merge(base, base, real_delay=0)
    inference_start = queue.snapshot()
    _consume(queue, 5)
    short_original = _actions(steps=3, offset=1000.0)
    short_processed = _actions(steps=4, offset=2000.0)
    caplog.set_level(logging.ERROR, logger="lerobot.policies.rtc.action_queue")

    result = queue.merge_actual_consumed(short_original, short_processed, inference_start)

    assert result.merged is True
    assert result.actual_consumed_steps == 5
    assert result.merge_skip == 3
    assert result.skip_was_clamped is True
    assert result.queue_size_after == 1
    assert torch.equal(queue.get(), short_processed[3])
    assert queue.get() is None
    assert "Actual consumed steps exceed the new action chunk" in caplog.text


def test_consumption_limit_refuses_merge_without_mutating_queue() -> None:
    queue = ActionQueue(RTCConfig(enabled=True, execution_horizon=4))
    base = _actions()
    queue.merge(base, base, real_delay=0)
    inference_start = queue.snapshot()
    _consume(queue, 4)
    before_rejected_merge = queue.snapshot()
    replacement = _actions(offset=1000.0)

    result = queue.merge_actual_consumed(
        replacement,
        replacement,
        inference_start,
        max_actual_consumed_steps=4,
    )

    assert result.merged is False
    assert result.stale is False
    assert result.consumption_limit_exceeded is True
    assert result.actual_consumed_steps == 4
    assert result.merge_skip == 0
    after_rejected_merge = queue.snapshot()
    assert after_rejected_merge.generation == before_rejected_merge.generation
    assert after_rejected_merge.total_consumed == before_rejected_merge.total_consumed
    assert torch.equal(after_rejected_merge.processed_leftover, before_rejected_merge.processed_leftover)
    assert torch.equal(queue.get(), base[4])


def test_snapshot_with_future_consumption_count_is_rejected() -> None:
    queue = ActionQueue(RTCConfig(enabled=True))
    start = queue.snapshot()
    invalid_start = ActionQueueSnapshot(
        generation=start.generation,
        next_action_index=start.next_action_index,
        total_consumed=start.total_consumed + 1,
        queue_size=start.queue_size,
        original_leftover=start.original_leftover,
        processed_leftover=start.processed_leftover,
    )
    actions = _actions()

    with pytest.raises(ValueError, match="total_consumed cannot exceed"):
        queue.merge_actual_consumed(actions, actions, invalid_start)

    assert queue.snapshot().generation == 0
    assert queue.empty() is True
