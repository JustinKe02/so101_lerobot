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

"""Tests for the Phase 1 atomic ActionQueue snapshot."""

import torch

from lerobot.policies.rtc.action_queue import ActionQueue, ActionQueueSnapshot
from lerobot.policies.rtc.configuration_rtc import RTCConfig


def _actions(offset: float = 0.0) -> torch.Tensor:
    return torch.arange(12, dtype=torch.float32).reshape(4, 3) + offset


def test_empty_snapshot_has_initial_counters() -> None:
    queue = ActionQueue(RTCConfig(enabled=True))

    snapshot = queue.snapshot()

    assert isinstance(snapshot, ActionQueueSnapshot)
    assert snapshot.generation == 0
    assert snapshot.next_action_index == 0
    assert snapshot.total_consumed == 0
    assert snapshot.queue_size == 0
    assert snapshot.original_leftover is None
    assert snapshot.processed_leftover is None


def test_snapshot_tracks_merge_consumption_and_clear() -> None:
    queue = ActionQueue(RTCConfig(enabled=True))
    original = _actions()
    processed = _actions(100.0)

    queue.merge(original, processed, real_delay=1)
    merged = queue.snapshot()
    assert merged.generation == 1
    assert merged.next_action_index == 0
    assert merged.total_consumed == 0
    assert merged.queue_size == 3
    assert merged.source_chunk_generation == 1
    assert merged.next_model_action_index == 1
    assert torch.equal(merged.original_leftover, original[1:])
    assert torch.equal(merged.processed_leftover, processed[1:])

    assert torch.equal(queue.get(), processed[1])
    assert torch.equal(queue.get(), processed[2])
    consumed = queue.snapshot()
    assert consumed.generation == 1
    assert consumed.next_action_index == 2
    assert consumed.total_consumed == 2
    assert consumed.queue_size == 1
    assert consumed.source_chunk_generation == 1
    assert consumed.next_model_action_index == 3
    assert torch.equal(consumed.original_leftover, original[3:])
    assert torch.equal(consumed.processed_leftover, processed[3:])

    queue.clear()
    cleared = queue.snapshot()
    assert cleared.generation == 2
    assert cleared.next_action_index == 0
    assert cleared.total_consumed == 2
    assert cleared.queue_size == 0
    assert cleared.original_leftover is None
    assert cleared.processed_leftover is None
    assert cleared.source_chunk_generation is None
    assert cleared.next_model_action_index is None


def test_atomic_pop_reports_source_chunk_and_model_action_index() -> None:
    queue = ActionQueue(RTCConfig(enabled=True))
    actions = _actions()
    queue.merge(actions, actions, real_delay=1)

    first = queue.get_with_diagnostics()
    second = queue.get_with_diagnostics()

    assert torch.equal(first.action, actions[1])
    assert first.source_chunk_generation == 1
    assert first.model_action_index == 1
    assert first.total_consumed == 1
    assert first.queue_size_after == 2
    assert torch.equal(second.action, actions[2])
    assert second.source_chunk_generation == 1
    assert second.model_action_index == 2
    assert second.total_consumed == 2
    assert second.queue_size_after == 1


def test_snapshot_tensors_are_clones() -> None:
    queue = ActionQueue(RTCConfig(enabled=True))
    original = _actions()
    processed = _actions(100.0)
    queue.merge(original, processed, real_delay=0)

    snapshot = queue.snapshot()
    snapshot.original_leftover.fill_(-1)
    snapshot.processed_leftover.fill_(-2)

    fresh = queue.snapshot()
    assert torch.equal(fresh.original_leftover, original)
    assert torch.equal(fresh.processed_leftover, processed)


def test_failed_or_empty_get_does_not_increment_total_consumed() -> None:
    queue = ActionQueue(RTCConfig(enabled=True))
    assert queue.get() is None
    assert queue.snapshot().total_consumed == 0

    one_action = torch.tensor([[1.0, 2.0, 3.0]])
    queue.merge(one_action, one_action, real_delay=0)
    assert queue.get() is not None
    assert queue.get() is None

    snapshot = queue.snapshot()
    assert snapshot.total_consumed == 1
    assert snapshot.queue_size == 0
    assert snapshot.original_leftover is not None
    assert snapshot.original_leftover.shape == (0, 3)
    assert snapshot.processed_leftover is not None
    assert snapshot.processed_leftover.shape == (0, 3)


def test_each_successful_merge_advances_generation() -> None:
    queue = ActionQueue(RTCConfig(enabled=False))
    actions = _actions()

    queue.merge(actions[:2], actions[:2], real_delay=0)
    first = queue.snapshot()
    queue.merge(actions[2:], actions[2:], real_delay=0)
    second = queue.snapshot()

    assert first.generation == 1
    assert second.generation == 2
    assert second.queue_size == 4
