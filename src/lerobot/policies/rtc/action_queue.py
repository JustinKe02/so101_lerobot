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
    processed_leftover: Tensor | None = None


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


@dataclass(frozen=True)
class ActionQueueAnchorMergeResult:
    """Outcome of merging a postfix generated for a future queue anchor."""

    merged: bool
    stale: bool
    expected_generation: int
    observed_generation: int
    generation_after: int
    actual_consumed_steps: int
    anchor_steps_after_start: int
    preserved_steps: int
    postfix_skip: int
    queue_size_after: int
    insufficient_committed_actions: bool = False
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
                    processed_leftover=None,
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
                processed_leftover=self.queue[self.last_index :].clone(),
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

    def merge_postfix_at_anchor(
        self,
        original_postfix: Tensor,
        processed_postfix: Tensor,
        inference_start: ActionQueueSnapshot,
        anchor_steps_after_start: int,
        max_actual_consumed_steps: int | None = None,
    ) -> ActionQueueAnchorMergeResult:
        """Merge model postfix whose first action follows a predicted future anchor.

        ``anchor_steps_after_start`` is the zero-based index, in the queue
        snapshot taken at request time, of the final committed action supplied
        to the model. If inference returns before that action is consumed, the
        still-committed queue prefix is retained. If it returns later, only the
        now-stale part of the generated postfix is skipped.
        """

        if isinstance(anchor_steps_after_start, bool) or anchor_steps_after_start < -1:
            raise ValueError("anchor_steps_after_start must be an integer greater than or equal to -1")
        if max_actual_consumed_steps is not None and max_actual_consumed_steps <= 0:
            raise ValueError("max_actual_consumed_steps must be positive")
        if not self.cfg.enabled:
            raise ValueError("Anchor-based postfix merge requires RTC to be enabled")

        with self.lock:
            observed_generation = self._generation
            actual_consumed = max(0, self._total_consumed - inference_start.total_consumed)

            def result(
                *,
                merged: bool,
                stale: bool = False,
                preserved_steps: int = 0,
                postfix_skip: int = 0,
                insufficient: bool = False,
                limit_exceeded: bool = False,
            ) -> ActionQueueAnchorMergeResult:
                return ActionQueueAnchorMergeResult(
                    merged=merged,
                    stale=stale,
                    expected_generation=inference_start.generation,
                    observed_generation=observed_generation,
                    generation_after=self._generation,
                    actual_consumed_steps=actual_consumed,
                    anchor_steps_after_start=anchor_steps_after_start,
                    preserved_steps=preserved_steps,
                    postfix_skip=postfix_skip,
                    queue_size_after=self._queue_size_unlocked(),
                    insufficient_committed_actions=insufficient,
                    consumption_limit_exceeded=limit_exceeded,
                )

            if observed_generation != inference_start.generation:
                logger.warning(
                    "Discarding stale anchored inference result. expected_generation=%d, observed_generation=%d",
                    inference_start.generation,
                    observed_generation,
                )
                return result(merged=False, stale=True)
            if self._total_consumed < inference_start.total_consumed:
                raise ValueError(
                    "inference_start.total_consumed cannot exceed the queue's current total: "
                    f"start={inference_start.total_consumed}, current={self._total_consumed}"
                )
            if max_actual_consumed_steps is not None and actual_consumed >= max_actual_consumed_steps:
                return result(merged=False, limit_exceeded=True)

            remaining_through_anchor = anchor_steps_after_start + 1 - actual_consumed
            preserved_steps = max(0, remaining_through_anchor)
            postfix_skip = max(0, -remaining_through_anchor)

            current_original = None if self.original_queue is None else self.original_queue[self.last_index :]
            current_processed = None if self.queue is None else self.queue[self.last_index :]
            available = min(
                0 if current_original is None else len(current_original),
                0 if current_processed is None else len(current_processed),
            )
            if preserved_steps > available:
                logger.error(
                    "Committed anchor is no longer available in the action queue: required=%d available=%d",
                    preserved_steps,
                    available,
                )
                return result(
                    merged=False,
                    preserved_steps=preserved_steps,
                    postfix_skip=postfix_skip,
                    insufficient=True,
                )
            if postfix_skip >= len(original_postfix) or postfix_skip >= len(processed_postfix):
                logger.error(
                    "Anchored inference result is fully stale: postfix_skip=%d original=%d processed=%d",
                    postfix_skip,
                    len(original_postfix),
                    len(processed_postfix),
                )
                return result(
                    merged=False,
                    preserved_steps=preserved_steps,
                    postfix_skip=postfix_skip,
                    insufficient=True,
                )

            original_parts = []
            processed_parts = []
            if preserved_steps:
                original_parts.append(current_original[:preserved_steps].clone())
                processed_parts.append(current_processed[:preserved_steps].clone())
            original_parts.append(original_postfix[postfix_skip:].clone())
            processed_parts.append(processed_postfix[postfix_skip:].clone())
            self.original_queue = torch.cat(original_parts, dim=0)
            self.queue = torch.cat(processed_parts, dim=0)
            self.last_index = 0
            self._generation += 1
            if preserved_steps:
                self._source_chunk_generation = None
                self._source_model_start_index = None
            else:
                self._source_chunk_generation = self._generation
                self._source_model_start_index = postfix_skip
            return result(
                merged=True,
                preserved_steps=preserved_steps,
                postfix_skip=postfix_skip,
            )

    def merge_postfix_after_external_anchor(
        self,
        original_postfix: Tensor,
        processed_postfix: Tensor,
        inference_start: ActionQueueSnapshot,
        *,
        external_consumed_steps: int,
        committed_steps: int,
        max_actual_consumed_steps: int | None = None,
    ) -> ActionQueueAnchorMergeResult:
        """Merge a postfix whose committed prefix is owned by an external executor.

        The executor keeps dispatching its immutable preview trajectory while
        inference runs, so those committed commands must not be copied back into
        this raw action queue and smoothed a second time. The queue receives only
        the generated postfix; a caller-side heartbeat barrier prevents it from
        being consumed before the committed executor prefix completes.
        """

        if isinstance(external_consumed_steps, bool) or external_consumed_steps < 0:
            raise ValueError("external_consumed_steps must be a non-negative integer")
        if isinstance(committed_steps, bool) or committed_steps < 0:
            raise ValueError("committed_steps must be a non-negative integer")
        if max_actual_consumed_steps is not None and max_actual_consumed_steps <= 0:
            raise ValueError("max_actual_consumed_steps must be positive")
        if not self.cfg.enabled:
            raise ValueError("External-anchor postfix merge requires RTC to be enabled")

        with self.lock:
            observed_generation = self._generation
            postfix_skip = max(0, external_consumed_steps - committed_steps)

            def result(
                *,
                merged: bool,
                stale: bool = False,
                limit_exceeded: bool = False,
                insufficient: bool = False,
            ) -> ActionQueueAnchorMergeResult:
                return ActionQueueAnchorMergeResult(
                    merged=merged,
                    stale=stale,
                    expected_generation=inference_start.generation,
                    observed_generation=observed_generation,
                    generation_after=self._generation,
                    actual_consumed_steps=external_consumed_steps,
                    anchor_steps_after_start=committed_steps - 1,
                    preserved_steps=0,
                    postfix_skip=postfix_skip,
                    queue_size_after=self._queue_size_unlocked(),
                    insufficient_committed_actions=insufficient,
                    consumption_limit_exceeded=limit_exceeded,
                )

            if observed_generation != inference_start.generation:
                return result(merged=False, stale=True)
            if max_actual_consumed_steps is not None and external_consumed_steps >= max_actual_consumed_steps:
                return result(merged=False, limit_exceeded=True)
            if postfix_skip >= len(original_postfix) or postfix_skip >= len(processed_postfix):
                return result(merged=False, insufficient=True)

            self._replace_actions_queue(original_postfix, processed_postfix, postfix_skip)
            self._generation += 1
            return result(merged=True)

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
