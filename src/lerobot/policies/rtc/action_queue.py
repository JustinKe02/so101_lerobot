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

"""Action queue management for Real-Time Chunking (RTC).

This module provides ActionQueue, a thread-safe queue for managing action chunks
in real-time control scenarios. It supports both RTC-enabled and non-RTC modes,
handling action merging and leftover tracking.
"""

import logging
from dataclasses import dataclass
from threading import Lock

import torch
from torch import Tensor

from .configuration_rtc import RTCConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ActionQueueSnapshot:
    """Atomic, read-only view of queue state used by RTC timing diagnostics."""

    generation: int
    next_action_index: int
    total_consumed: int
    queue_size: int
    original_leftover: Tensor | None
    processed_leftover: Tensor | None
    source_chunk_generation: int | None = None
    next_model_action_index: int | None = None


@dataclass(frozen=True)
class ActionQueuePopResult:
    """One atomically dequeued action and its policy-chunk provenance."""

    action: Tensor | None
    source_chunk_generation: int | None
    model_action_index: int | None
    total_consumed: int
    queue_size_after: int


@dataclass(frozen=True)
class ActionQueueMergeResult:
    """Outcome of an atomic actual-consumed merge.

    A stale inference result is reported rather than merged. ``merge_skip`` is
    the number of new actions actually discarded and can be smaller than
    ``actual_consumed_steps`` only when the latter exceeds an input chunk.
    """

    merged: bool
    stale: bool
    expected_generation: int
    observed_generation: int
    generation_after: int
    start_total_consumed: int
    current_total_consumed: int
    actual_consumed_steps: int
    merge_skip: int
    skip_was_clamped: bool
    queue_size_after: int
    consumption_limit_exceeded: bool = False


class ActionQueue:
    """Thread-safe queue for managing action chunks in real-time control.

    This queue handles two types of action sequences:
    - Original actions: Used for RTC to compute leftovers from previous chunks
    - Processed actions: Post-processed actions ready for robot execution

    The queue operates in two modes:
    1. RTC-enabled: Replaces the entire queue with new actions, accounting for inference delay
    2. RTC-disabled: Appends new actions to the queue, maintaining continuity

    Args:
        cfg (RTCConfig): Configuration for Real-Time Chunking behavior.

    Attributes:
        queue (Tensor | None): Processed actions for robot rollout (time_steps, action_dim).
        original_queue (Tensor | None): Original actions for RTC computation (time_steps, action_dim).
        last_index (int): Current consumption index in the queue.
    """

    def __init__(self, cfg: RTCConfig):
        """Initialize the action queue.

        Args:
            cfg: RTC configuration controlling queue behavior.
        """
        self.queue = None  # Processed actions for robot rollout
        self.original_queue = None  # Original actions for RTC
        self.lock = Lock()
        self.last_index = 0
        self._generation = 0
        self._total_consumed = 0
        self._source_chunk_generation: int | None = None
        self._source_model_start_index: int | None = None
        self.cfg = cfg

    def get(self) -> Tensor | None:
        """Get the next action from the queue.

        Returns:
            Tensor | None: The next action (action_dim,) or None if queue is empty.
                          Returns a clone to prevent external modifications.
        """
        with self.lock:
            if self.queue is None or self.last_index >= len(self.queue):
                return None

            action = self.queue[self.last_index].clone()
            self.last_index += 1
            self._total_consumed += 1
            return action

    def get_with_diagnostics(self) -> ActionQueuePopResult:
        """Atomically dequeue an action together with its source model index."""
        with self.lock:
            if self.queue is None or self.last_index >= len(self.queue):
                return ActionQueuePopResult(
                    action=None,
                    source_chunk_generation=None,
                    model_action_index=None,
                    total_consumed=self._total_consumed,
                    queue_size_after=0,
                )

            model_action_index = (
                None
                if self._source_model_start_index is None
                else self._source_model_start_index + self.last_index
            )
            action = self.queue[self.last_index].clone()
            self.last_index += 1
            self._total_consumed += 1
            return ActionQueuePopResult(
                action=action,
                source_chunk_generation=self._source_chunk_generation,
                model_action_index=model_action_index,
                total_consumed=self._total_consumed,
                queue_size_after=self._queue_size_unlocked(),
            )

    def clear(self) -> None:
        """Clear queued actions and reset consumption index."""
        with self.lock:
            self.queue = None
            self.original_queue = None
            self.last_index = 0
            self._source_chunk_generation = None
            self._source_model_start_index = None
            self._generation += 1

    def snapshot(self) -> ActionQueueSnapshot:
        """Return one lock-consistent snapshot without consuming an action."""
        with self.lock:
            if self.queue is None:
                queue_size = 0
                processed_leftover = None
            else:
                queue_size = max(0, len(self.queue) - self.last_index)
                processed_leftover = self.queue[self.last_index :].clone()

            original_leftover = (
                None if self.original_queue is None else self.original_queue[self.last_index :].clone()
            )
            return ActionQueueSnapshot(
                generation=self._generation,
                next_action_index=self.last_index,
                total_consumed=self._total_consumed,
                queue_size=queue_size,
                original_leftover=original_leftover,
                processed_leftover=processed_leftover,
                source_chunk_generation=self._source_chunk_generation,
                next_model_action_index=(
                    None
                    if self._source_model_start_index is None or queue_size == 0
                    else self._source_model_start_index + self.last_index
                ),
            )

    def qsize(self) -> int:
        """Get the number of remaining actions in the queue.

        Returns:
            int: Number of unconsumed actions.
        """
        with self.lock:
            if self.queue is None:
                return 0
            return len(self.queue) - self.last_index

    def empty(self) -> bool:
        """Check if the queue is empty.

        Returns:
            bool: True if no actions remain, False otherwise.
        """
        with self.lock:
            if self.queue is None:
                return True
            return len(self.queue) - self.last_index <= 0

    def get_action_index(self) -> int:
        """Get the current action consumption index.

        Returns:
            int: Index of the next action to be consumed.
        """
        with self.lock:
            return self.last_index

    def get_left_over(self) -> Tensor | None:
        """Get leftover original actions for RTC prev_chunk_left_over.

        These are the unconsumed actions from the current chunk, which will be
        used by RTC to compute corrections for the next chunk.

        Returns:
            Tensor | None: Remaining original actions (remaining_steps, action_dim),
                          or None if no original queue exists.
        """
        with self.lock:
            if self.original_queue is None:
                return None
            return self.original_queue[self.last_index :].clone()

    def get_processed_left_over(self) -> Tensor | None:
        """Get leftover processed actions (the actions currently executed by the robot).

        Returns:
            Tensor | None: Remaining processed actions (remaining_steps, action_dim),
                or None if no processed queue exists.
        """
        with self.lock:
            if self.queue is None:
                return None
            return self.queue[self.last_index :].clone()

    def merge(
        self,
        original_actions: Tensor,
        processed_actions: Tensor,
        real_delay: int,
        action_index_before_inference: int | None = None,
    ):
        """Merge new actions into the queue.

        This method operates differently based on RTC mode:
        - RTC enabled: Replaces the queue, accounting for inference delay
        - RTC disabled: Appends to the queue, maintaining continuity

        Args:
            original_actions: Unprocessed actions from policy (time_steps, action_dim).
            processed_actions: Post-processed actions for robot (time_steps, action_dim).
            real_delay: Number of time steps of inference delay.
            action_index_before_inference: Index before inference started, for validation.
        """
        with self.lock:
            delay = self._check_and_resolve_delays(real_delay, action_index_before_inference)

            if self.cfg.enabled:
                self._replace_actions_queue(original_actions, processed_actions, delay)
            else:
                self._append_actions_queue(original_actions, processed_actions)
            self._generation += 1

    def merge_actual_consumed(
        self,
        original_actions: Tensor,
        processed_actions: Tensor,
        inference_start: ActionQueueSnapshot,
        max_actual_consumed_steps: int | None = None,
    ) -> ActionQueueMergeResult:
        """Atomically merge using actions consumed since ``inference_start``.

        Unlike :meth:`merge`, wall-clock latency is not used to select a slice
        of the new action chunk. Generation validation, consumption accounting,
        and queue replacement happen under the same lock, so a clear or another
        producer merge cannot race between validation and replacement.

        Args:
            original_actions: Unprocessed actions from policy.
            processed_actions: Post-processed actions for robot execution.
            inference_start: Snapshot captured immediately before inference.
            max_actual_consumed_steps: Optional exclusive safety limit. If the
                number consumed is greater than or equal to this value, report
                the violation without changing the queue.

        Returns:
            Diagnostics describing whether the merge was accepted and the skip
            applied to the new chunk.
        """
        if max_actual_consumed_steps is not None and max_actual_consumed_steps <= 0:
            raise ValueError("max_actual_consumed_steps must be positive")

        with self.lock:
            observed_generation = self._generation
            current_total_consumed = self._total_consumed
            actual_consumed_steps = max(0, current_total_consumed - inference_start.total_consumed)

            if observed_generation != inference_start.generation:
                logger.warning(
                    "Discarding stale inference result. expected_generation=%d, observed_generation=%d",
                    inference_start.generation,
                    observed_generation,
                )
                return ActionQueueMergeResult(
                    merged=False,
                    stale=True,
                    expected_generation=inference_start.generation,
                    observed_generation=observed_generation,
                    generation_after=observed_generation,
                    start_total_consumed=inference_start.total_consumed,
                    current_total_consumed=current_total_consumed,
                    actual_consumed_steps=actual_consumed_steps,
                    merge_skip=0,
                    skip_was_clamped=False,
                    queue_size_after=self._queue_size_unlocked(),
                )

            if current_total_consumed < inference_start.total_consumed:
                raise ValueError(
                    "inference_start.total_consumed cannot exceed the queue's current total: "
                    f"start={inference_start.total_consumed}, current={current_total_consumed}"
                )

            if max_actual_consumed_steps is not None and actual_consumed_steps >= max_actual_consumed_steps:
                logger.error(
                    "Actual consumed steps reached the configured safety limit; refusing merge. "
                    "actual_consumed_steps=%d, limit=%d",
                    actual_consumed_steps,
                    max_actual_consumed_steps,
                )
                return ActionQueueMergeResult(
                    merged=False,
                    stale=False,
                    expected_generation=inference_start.generation,
                    observed_generation=observed_generation,
                    generation_after=observed_generation,
                    start_total_consumed=inference_start.total_consumed,
                    current_total_consumed=current_total_consumed,
                    actual_consumed_steps=actual_consumed_steps,
                    merge_skip=0,
                    skip_was_clamped=False,
                    queue_size_after=self._queue_size_unlocked(),
                    consumption_limit_exceeded=True,
                )

            if self.cfg.enabled:
                merge_skip = min(actual_consumed_steps, len(original_actions), len(processed_actions))
                skip_was_clamped = merge_skip != actual_consumed_steps
                if skip_was_clamped:
                    logger.error(
                        "Actual consumed steps exceed the new action chunk and were clamped. "
                        "actual_consumed_steps=%d, original_steps=%d, processed_steps=%d, "
                        "merge_skip=%d",
                        actual_consumed_steps,
                        len(original_actions),
                        len(processed_actions),
                        merge_skip,
                    )
                self._replace_actions_queue(original_actions, processed_actions, merge_skip)
            else:
                merge_skip = 0
                skip_was_clamped = False
                self._append_actions_queue(original_actions, processed_actions)

            self._generation += 1
            return ActionQueueMergeResult(
                merged=True,
                stale=False,
                expected_generation=inference_start.generation,
                observed_generation=observed_generation,
                generation_after=self._generation,
                start_total_consumed=inference_start.total_consumed,
                current_total_consumed=current_total_consumed,
                actual_consumed_steps=actual_consumed_steps,
                merge_skip=merge_skip,
                skip_was_clamped=skip_was_clamped,
                queue_size_after=self._queue_size_unlocked(),
            )

    def _queue_size_unlocked(self) -> int:
        """Return remaining queue size while the caller holds ``self.lock``."""
        if self.queue is None:
            return 0
        return max(0, len(self.queue) - self.last_index)

    def _replace_actions_queue(self, original_actions: Tensor, processed_actions: Tensor, real_delay: int):
        """Replace the queue with new actions (RTC mode).

        Discards the first `real_delay` actions since they correspond to the time
        spent during inference, when the robot was executing previous actions.

        Args:
            original_actions: Unprocessed actions from policy.
            processed_actions: Post-processed actions for robot.
            real_delay: Number of time steps to skip due to inference delay.
        """
        clamped_delay = max(0, min(real_delay, len(original_actions), len(processed_actions)))
        self.original_queue = original_actions[clamped_delay:].clone()
        self.queue = processed_actions[clamped_delay:].clone()
        self._source_chunk_generation = self._generation + 1
        self._source_model_start_index = clamped_delay

        logger.debug(f"original_actions shape: {self.original_queue.shape}")
        logger.debug(f"processed_actions shape: {self.queue.shape}")
        logger.debug(f"real_delay: {real_delay}, clamped_delay: {clamped_delay}")

        self.last_index = 0

    def _append_actions_queue(self, original_actions: Tensor, processed_actions: Tensor):
        """Append new actions to the queue (non-RTC mode).

        Removes already-consumed actions and appends new ones, maintaining
        queue continuity without replacement.

        Args:
            original_actions: Unprocessed actions from policy.
            processed_actions: Post-processed actions for robot.
        """
        if self.queue is None:
            self.original_queue = original_actions.clone()
            self.queue = processed_actions.clone()
            self._source_chunk_generation = None
            self._source_model_start_index = None
            return

        self.original_queue = torch.cat([self.original_queue, original_actions.clone()])
        self.original_queue = self.original_queue[self.last_index :]

        self.queue = torch.cat([self.queue, processed_actions.clone()])
        self.queue = self.queue[self.last_index :]

        self.last_index = 0
        self._source_chunk_generation = None
        self._source_model_start_index = None

    def _check_and_resolve_delays(
        self, real_delay: int, action_index_before_inference: int | None = None
    ) -> int:
        """Validate that computed delays match expectations.

        Compares the delay computed from inference latency with the actual
        number of actions consumed during inference.

        Args:
            real_delay: Delay computed from inference latency.
            action_index_before_inference: Action index when inference started.

        Returns:
            int: Delay to use.
        """
        effective_delay = max(0, real_delay)

        if action_index_before_inference is not None:
            indexes_diff = max(0, self.last_index - action_index_before_inference)
            if indexes_diff != real_delay:
                logger.warning(
                    "Indexes diff is not equal to real delay. indexes_diff=%d, real_delay=%d",
                    indexes_diff,
                    real_delay,
                )
                return real_delay

        return effective_delay
