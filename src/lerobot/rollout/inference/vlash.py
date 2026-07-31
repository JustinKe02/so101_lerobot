# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Future-state-aware asynchronous inference for action-chunking policies."""

from __future__ import annotations

import logging
import math
import time
from collections import deque
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from threading import Event, Lock, RLock, Thread
from typing import Any

import numpy as np
import torch

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import prepare_observation_for_inference
from lerobot.processor import PolicyProcessorPipeline
from lerobot.utils.constants import OBS_STATE
from lerobot.utils.feature_utils import build_dataset_frame

from ..robot_wrapper import ThreadSafeRobot
from .base import InferenceEngine

logger = logging.getLogger(__name__)

_IDLE_WAIT_S = 0.002
_JOIN_TIMEOUT_S = 10.0
_TRACKING_GAIN_WINDOW_SIZE = 8
_TRACKING_COMMAND_EPS = 1e-3
_ACTION_REWRITE_EPS = 1e-4


def _percentile(values: deque[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(q * len(ordered)) - 1))
    return ordered[index]


@dataclass(frozen=True)
class VLASHInferenceStats:
    inference_count: int
    deadline_misses: int
    active_actions: int
    pending_actions: int
    latency_p50_ms: float | None
    latency_p95_ms: float | None
    latency_max_ms: float | None
    action_feedback_count: int
    hardware_clamp_count: int
    action_filter_rewrite_count: int
    tracking_gain_mean: float | None


@dataclass(frozen=True)
class _VLASHInferenceRequest:
    observation: dict
    epoch: int
    sequence: int
    future_state: np.ndarray | None
    projected_steps: int


class VLASHInferenceEngine(InferenceEngine):
    """Execute one chunk while predicting the next from its execution-time state.

    The model output is already conditioned on the state expected at the next
    chunk boundary, so completed chunks are never latency-sliced as RTC chunks
    are. They remain pending until the current execution horizon is exhausted.
    """

    def __init__(
        self,
        *,
        policy: PreTrainedPolicy,
        preprocessor: PolicyProcessorPipeline,
        postprocessor: PolicyProcessorPipeline,
        robot_wrapper: ThreadSafeRobot,
        hw_features: dict,
        ordered_action_keys: list[str],
        task: str,
        fps: float,
        device: str | None,
        execution_horizon: int,
        inference_overlap_steps: int,
        max_future_state_delta: float | None = None,
        deadline_miss_limit: int = 1,
        latency_window_size: int = 64,
        timing_diagnostics: bool = False,
        use_torch_compile: bool = False,
        compile_warmup_inferences: int = 2,
        shutdown_event: Event | None = None,
    ) -> None:
        if (
            isinstance(execution_horizon, bool)
            or not isinstance(execution_horizon, int)
            or execution_horizon < 1
        ):
            raise ValueError("VLASH execution_horizon must be a positive integer")
        if (
            isinstance(inference_overlap_steps, bool)
            or not isinstance(inference_overlap_steps, int)
            or not 1 <= inference_overlap_steps <= execution_horizon
        ):
            raise ValueError("VLASH inference_overlap_steps must be in [1, execution_horizon]")
        if max_future_state_delta is not None and (
            isinstance(max_future_state_delta, bool)
            or not isinstance(max_future_state_delta, (int, float))
            or not math.isfinite(max_future_state_delta)
            or max_future_state_delta <= 0
        ):
            raise ValueError("VLASH max_future_state_delta must be finite and positive")
        if (
            isinstance(deadline_miss_limit, bool)
            or not isinstance(deadline_miss_limit, int)
            or deadline_miss_limit < 1
        ):
            raise ValueError("VLASH deadline_miss_limit must be a positive integer")
        if (
            isinstance(latency_window_size, bool)
            or not isinstance(latency_window_size, int)
            or latency_window_size < 1
        ):
            raise ValueError("VLASH latency_window_size must be a positive integer")
        if not ordered_action_keys or len(set(ordered_action_keys)) != len(ordered_action_keys):
            raise ValueError("VLASH ordered_action_keys must be non-empty and unique")
        policy_chunk_size = getattr(getattr(policy, "config", None), "chunk_size", None)
        if isinstance(policy_chunk_size, int) and execution_horizon > policy_chunk_size:
            raise ValueError(
                "VLASH execution_horizon cannot exceed policy chunk_size: "
                f"{execution_horizon} > {policy_chunk_size}"
            )

        self._policy = policy
        self._preprocessor = preprocessor
        self._postprocessor = postprocessor
        self._robot = robot_wrapper
        self._hw_features = hw_features
        self._ordered_action_keys = tuple(ordered_action_keys)
        self._task = task
        self._fps = float(fps)
        self._device = torch.device(device or "cpu")
        self._execution_horizon = execution_horizon
        self._inference_overlap_steps = inference_overlap_steps
        self._max_future_state_delta = max_future_state_delta
        self._deadline_miss_limit = deadline_miss_limit
        self._timing_diagnostics = timing_diagnostics
        self._requires_warmup = use_torch_compile
        self._warmups_remaining = max(1, compile_warmup_inferences) if use_torch_compile else 0

        self._queue_lock = Lock()
        self._active_actions: deque[torch.Tensor] = deque()
        self._pending_actions: torch.Tensor | None = None
        self._inference_request: _VLASHInferenceRequest | None = None
        self._initialized = False
        self._inference_inflight = False
        self._deadline_reported = False
        self._consecutive_deadline_misses = 0
        self._awaiting_action_feedback: torch.Tensor | None = None
        self._previous_feedback_state: torch.Tensor | None = None
        self._previous_applied_action: torch.Tensor | None = None
        self._tracking_gain_samples: list[deque[float]] = [
            deque(maxlen=_TRACKING_GAIN_WINDOW_SIZE) for _ in self._ordered_action_keys
        ]
        self._action_feedback_count = 0
        self._hardware_clamp_count = 0
        self._action_filter_rewrite_count = 0

        self._obs_lock = Lock()
        self._latest_observation: dict[str, Any] | None = None
        self._observation_epoch = 0
        self._observation_sequence = 0

        self._inference_lock = RLock()
        self._dispatch_lock = RLock()
        self._fatal_lock = Lock()
        self._fatal_error: BaseException | None = None
        self._failed = Event()
        self._ready = Event()
        self._active = Event()
        self._shutdown = Event()
        self._wake = Event()
        self._global_shutdown = shutdown_event
        self._thread: Thread | None = None

        self._stats_lock = Lock()
        self._inference_count = 0
        self._deadline_misses = 0
        self._latencies: deque[float] = deque(maxlen=latency_window_size)

    @property
    def ready(self) -> bool:
        return self._ready.is_set() if self._requires_warmup else True

    @property
    def failed(self) -> bool:
        return self._failed.is_set()

    @property
    def fatal_error(self) -> BaseException | None:
        return self._fatal_error

    def stats_snapshot(self) -> VLASHInferenceStats:
        with self._queue_lock:
            active_actions = len(self._active_actions)
            pending_actions = 0 if self._pending_actions is None else len(self._pending_actions)
            tracking_gain = self._tracking_gain_locked()
            action_feedback_count = self._action_feedback_count
            hardware_clamp_count = self._hardware_clamp_count
            action_filter_rewrite_count = self._action_filter_rewrite_count
        with self._stats_lock:
            p50 = _percentile(self._latencies, 0.50)
            p95 = _percentile(self._latencies, 0.95)
            return VLASHInferenceStats(
                inference_count=self._inference_count,
                deadline_misses=self._deadline_misses,
                active_actions=active_actions,
                pending_actions=pending_actions,
                latency_p50_ms=None if p50 is None else p50 * 1000,
                latency_p95_ms=None if p95 is None else p95 * 1000,
                latency_max_ms=None if not self._latencies else max(self._latencies) * 1000,
                action_feedback_count=action_feedback_count,
                hardware_clamp_count=hardware_clamp_count,
                action_filter_rewrite_count=action_filter_rewrite_count,
                tracking_gain_mean=(None if tracking_gain is None else float(tracking_gain.mean().item())),
            )

    def clear_latency_window(self) -> None:
        """Start a fresh latency window after deployment warmup."""
        with self._stats_lock:
            self._latencies.clear()

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._shutdown.clear()
        self._thread = Thread(target=self._inference_loop, daemon=True, name="VLASHInference")
        self._thread.start()
        logger.info(
            "VLASH inference thread started: execution_horizon=%d overlap_steps=%d",
            self._execution_horizon,
            self._inference_overlap_steps,
        )

    def stop(self) -> None:
        self._shutdown.set()
        self._active.clear()
        self._wake.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=_JOIN_TIMEOUT_S)
        if thread is not None and thread.is_alive():
            self._enter_fatal_state(TimeoutError("VLASH inference thread did not stop"))
        self._thread = None

    def pause(self) -> None:
        self._active.clear()

    def resume(self) -> None:
        self._active.set()
        self._wake.set()

    def reset(self) -> None:
        with self._inference_lock:
            with self._obs_lock:
                self._observation_epoch += 1
                self._latest_observation = None
            with self._queue_lock:
                self._active_actions.clear()
                self._pending_actions = None
                self._inference_request = None
                self._initialized = False
                self._inference_inflight = False
                self._deadline_reported = False
                self._consecutive_deadline_misses = 0
                self._awaiting_action_feedback = None
                self._previous_feedback_state = None
                self._previous_applied_action = None
                for samples in self._tracking_gain_samples:
                    samples.clear()
            self._policy.reset()
            self._preprocessor.reset()
            self._postprocessor.reset()
        self._wake.set()

    def notify_observation(self, obs: dict) -> None:
        with self._obs_lock:
            self._latest_observation = obs
            self._observation_sequence += 1
            epoch = self._observation_epoch
            sequence = self._observation_sequence
        with self._queue_lock:
            if not self._initialized:
                self._schedule_request_locked(obs, epoch, sequence, None, projected_steps=0)
        self._wake.set()

    def get_action(self, obs_frame: dict | None) -> torch.Tensor | None:
        del obs_frame
        if self._failed.is_set():
            return None
        with self._queue_lock:
            if not self._active_actions and self._pending_actions is not None:
                self._active_actions.extend(self._pending_actions)
                self._pending_actions = None
                self._deadline_reported = False

            if not self._active_actions:
                if self._initialized and not self._deadline_reported:
                    self._deadline_reported = True
                    with self._stats_lock:
                        self._deadline_misses += 1
                    self._consecutive_deadline_misses += 1
                    logger.error(
                        "VLASH missed chunk deadline (%d/%d)",
                        self._consecutive_deadline_misses,
                        self._deadline_miss_limit,
                    )
                    if self._consecutive_deadline_misses >= self._deadline_miss_limit:
                        self._enter_fatal_state_locked(
                            RuntimeError("VLASH inference missed the configured chunk deadline")
                        )
                return None

            action = self._active_actions.popleft().clone()
            self._awaiting_action_feedback = action.clone()
            return action

    def notify_action_result(self, requested: dict, sent: dict | None, observation: dict) -> None:
        """Project the next chunk boundary from actual dispatch and tracking feedback."""
        with self._obs_lock:
            latest_observation = self._latest_observation
            epoch = self._observation_epoch
            sequence = self._observation_sequence

        with self._queue_lock:
            policy_action = self._awaiting_action_feedback
            self._awaiting_action_feedback = None
            if policy_action is None:
                return

            observed_state = self._mapping_to_action_vector(observation)
            requested_action = self._mapping_to_action_vector(requested)
            applied_action = self._mapping_to_action_vector(sent if sent is not None else requested)
            if observed_state is None or requested_action is None or applied_action is None:
                logger.error(
                    "VLASH action feedback is missing or non-finite for one of the configured action keys"
                )
                return

            self._action_feedback_count += 1
            if torch.max(torch.abs(requested_action - applied_action)).item() > _ACTION_REWRITE_EPS:
                self._hardware_clamp_count += 1
            if torch.max(torch.abs(policy_action - requested_action)).item() > _ACTION_REWRITE_EPS:
                self._action_filter_rewrite_count += 1

            self._update_tracking_gain_locked(observed_state)
            tracking_gain = self._tracking_gain_locked()
            self._previous_feedback_state = observed_state.clone()
            self._previous_applied_action = applied_action.clone()

            if (
                latest_observation is None
                or not self._initialized
                or not 1 <= len(self._active_actions) <= self._inference_overlap_steps
            ):
                return

            remaining = torch.stack(tuple(self._active_actions))
            targets = torch.cat((applied_action.unsqueeze(0), remaining), dim=0)
            future_state = self._estimate_future_state(
                observed_state.numpy(),
                targets,
                tracking_gain=tracking_gain,
            )
            self._schedule_request_locked(
                latest_observation,
                epoch,
                sequence,
                future_state,
                projected_steps=len(targets),
            )

    def _mapping_to_action_vector(self, values: dict | None) -> torch.Tensor | None:
        if values is None or not self._ordered_action_keys:
            return None
        vector: list[float] = []
        for key in self._ordered_action_keys:
            try:
                value = float(values[key])
            except (KeyError, TypeError, ValueError):
                return None
            if not math.isfinite(value):
                return None
            vector.append(value)
        return torch.tensor(vector, dtype=torch.float32)

    def _update_tracking_gain_locked(self, observed_state: torch.Tensor) -> None:
        previous_state = self._previous_feedback_state
        previous_action = self._previous_applied_action
        if previous_state is None or previous_action is None:
            return
        commanded_delta = previous_action - previous_state
        observed_delta = observed_state - previous_state
        for index, (commanded, achieved) in enumerate(zip(commanded_delta, observed_delta, strict=True)):
            commanded_value = float(commanded.item())
            achieved_value = float(achieved.item())
            if abs(commanded_value) < _TRACKING_COMMAND_EPS:
                continue
            ratio = achieved_value / commanded_value
            if not math.isfinite(ratio) or ratio < 0:
                continue
            self._tracking_gain_samples[index].append(min(1.0, ratio))

    def _tracking_gain_locked(self) -> torch.Tensor | None:
        if not self._tracking_gain_samples or not any(self._tracking_gain_samples):
            return None
        populated = [value for samples in self._tracking_gain_samples for value in samples]
        fallback = float(np.median(populated))
        return torch.tensor(
            [float(np.median(samples)) if samples else fallback for samples in self._tracking_gain_samples],
            dtype=torch.float32,
        )

    @contextmanager
    def action_dispatch_guard(self) -> Iterator[bool]:
        with self._dispatch_lock:
            yield not self._failed.is_set()

    def _enter_fatal_state_locked(self, error: BaseException) -> None:
        # Caller holds _queue_lock. Keep lock acquisition order out of this
        # helper so deadline failures cannot deadlock with the worker.
        if self._fatal_error is None:
            self._fatal_error = error
        self._failed.set()
        self._active.clear()
        self._shutdown.set()
        self._active_actions.clear()
        self._pending_actions = None
        if self._global_shutdown is not None:
            self._global_shutdown.set()

    def _enter_fatal_state(self, error: BaseException) -> None:
        with self._dispatch_lock, self._fatal_lock, self._queue_lock:
            self._enter_fatal_state_locked(error)

    def _schedule_request_locked(
        self,
        observation: dict,
        epoch: int,
        sequence: int,
        future_state: np.ndarray | None,
        *,
        projected_steps: int,
    ) -> None:
        """Schedule one lock-consistent observation/future-state pair."""
        if self._inference_inflight or self._pending_actions is not None:
            return
        if self._initialized and future_state is None:
            return
        self._inference_request = _VLASHInferenceRequest(
            observation=observation,
            epoch=epoch,
            sequence=sequence,
            future_state=None if future_state is None else future_state.copy(),
            projected_steps=projected_steps,
        )
        self._inference_inflight = True
        self._deadline_reported = False
        self._wake.set()

    def _take_request(self) -> _VLASHInferenceRequest | None:
        with self._queue_lock:
            request = self._inference_request
            self._inference_request = None
            return request

    def _reschedule_initial_request(self) -> None:
        with self._obs_lock:
            observation = self._latest_observation
            epoch = self._observation_epoch
            sequence = self._observation_sequence
        if observation is None:
            return
        with self._queue_lock:
            self._schedule_request_locked(observation, epoch, sequence, None, projected_steps=0)

    def _estimate_future_state(
        self,
        current_state: np.ndarray,
        remaining: torch.Tensor,
        tracking_gain: torch.Tensor | None = None,
    ) -> np.ndarray:
        state = torch.as_tensor(current_state, dtype=torch.float32).clone()
        if remaining.ndim != 2 or state.ndim != 1:
            raise ValueError(
                "VLASH future-state estimation expects state [D] and actions [T,D], "
                f"got state={tuple(state.shape)}, actions={tuple(remaining.shape)}"
            )
        if remaining.shape[-1] != state.shape[-1]:
            raise ValueError(
                "VLASH requires matching state/action dimensions for absolute-state rollforward: "
                f"state={state.shape[-1]}, action={remaining.shape[-1]}"
            )
        if tracking_gain is not None:
            tracking_gain = tracking_gain.to(dtype=state.dtype, device=state.device)
            if tracking_gain.shape != state.shape or not torch.isfinite(tracking_gain).all():
                raise ValueError("VLASH tracking gain must be finite and match observation.state")
            tracking_gain = tracking_gain.clamp(0.0, 1.0)
        for target in remaining.to(dtype=state.dtype, device=state.device):
            if self._max_future_state_delta is None:
                delta = target - state
            else:
                delta = (target - state).clamp(-self._max_future_state_delta, self._max_future_state_delta)
            state = state + delta * (1.0 if tracking_gain is None else tracking_gain)
        return state.numpy()

    def _run_policy(
        self,
        observation: dict,
        future_state: np.ndarray | None,
    ) -> torch.Tensor:
        obs_batch = build_dataset_frame(self._hw_features, observation, prefix="observation")
        if future_state is not None:
            current_state = obs_batch.get(OBS_STATE)
            if current_state is None:
                raise ValueError("VLASH inference requires observation.state")
            if current_state.shape != future_state.shape:
                raise ValueError(
                    "VLASH future state must match observation.state: "
                    f"state={current_state.shape}, future={future_state.shape}"
                )
            obs_batch[OBS_STATE] = future_state.copy()

        obs_batch = prepare_observation_for_inference(
            obs_batch, self._device, self._task, self._robot.robot_type
        )
        obs_batch["task"] = [self._task]
        with self._inference_lock, torch.inference_mode():
            preprocessed = self._preprocessor(obs_batch)
            actions = self._policy.predict_action_chunk(preprocessed)
            processed = self._postprocessor(actions).squeeze(0).detach().cpu()
        if processed.ndim != 2 or len(processed) < self._execution_horizon:
            raise ValueError(
                "VLASH policy output is shorter than execution_horizon: "
                f"shape={tuple(processed.shape)}, horizon={self._execution_horizon}"
            )
        return processed[: self._execution_horizon].clone()

    def _inference_loop(self) -> None:
        try:
            while not self._shutdown.is_set():
                self._wake.wait(timeout=_IDLE_WAIT_S)
                self._wake.clear()
                if not self._active.is_set():
                    continue
                request = self._take_request()
                if request is None:
                    continue
                started = time.perf_counter()
                actions = self._run_policy(request.observation, request.future_state)
                latency = time.perf_counter() - started

                with self._stats_lock:
                    self._inference_count += 1
                    self._latencies.append(latency)

                with self._obs_lock:
                    stale_epoch = request.epoch != self._observation_epoch
                reschedule_initial = False
                with self._queue_lock:
                    self._inference_inflight = False
                    if self._failed.is_set() or self._shutdown.is_set():
                        continue
                    if stale_epoch:
                        logger.info("VLASH discarded result from reset epoch %d", request.epoch)
                        reschedule_initial = True
                    elif self._warmups_remaining > 0:
                        self._warmups_remaining -= 1
                        logger.info("VLASH compile warmup completed, remaining=%d", self._warmups_remaining)
                        reschedule_initial = True
                    elif not self._initialized:
                        self._active_actions.extend(actions)
                        self._initialized = True
                        self._ready.set()
                    else:
                        self._pending_actions = actions
                        if self._active_actions:
                            self._consecutive_deadline_misses = 0

                if reschedule_initial:
                    self._reschedule_initial_request()
                    continue

                if self._timing_diagnostics:
                    stats = self.stats_snapshot()
                    logger.info(
                        "VLASH timing: latency_ms=%.3f p50_ms=%s p95_ms=%s "
                        "future_steps=%d observation_sequence=%d active=%d pending=%d "
                        "feedback=%d hardware_clamps=%d filter_rewrites=%d tracking_gain_mean=%s",
                        latency * 1000,
                        "n/a" if stats.latency_p50_ms is None else f"{stats.latency_p50_ms:.3f}",
                        "n/a" if stats.latency_p95_ms is None else f"{stats.latency_p95_ms:.3f}",
                        request.projected_steps,
                        request.sequence,
                        stats.active_actions,
                        stats.pending_actions,
                        stats.action_feedback_count,
                        stats.hardware_clamp_count,
                        stats.action_filter_rewrite_count,
                        ("n/a" if stats.tracking_gain_mean is None else f"{stats.tracking_gain_mean:.3f}"),
                    )
        except BaseException as error:
            logger.exception("VLASH inference failed")
            self._enter_fatal_state(error)
