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

"""Real-Time Chunking inference engine.

A background thread produces action chunks asynchronously via
:meth:`policy.predict_action_chunk`.  The main control loop polls
``get_action`` for the next ready action; observations flow the other
way via ``notify_observation``.
"""

from __future__ import annotations

import logging
import math
import time
import traceback
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from threading import Event, Lock, RLock, Thread
from typing import Any

import torch

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.rtc import (
    ActionQueue,
    GuidanceDelayEstimator,
    LatencyTracker,
    reanchor_relative_rtc_prefix,
)
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.policies.utils import prepare_observation_for_inference
from lerobot.processor import (
    NormalizerProcessorStep,
    PolicyProcessorPipeline,
    RelativeActionsProcessorStep,
)
from lerobot.utils.feature_utils import build_dataset_frame

from ..robot_wrapper import ThreadSafeRobot
from .base import InferenceEngine

logger = logging.getLogger(__name__)

# How long the RTC loop sleeps when paused, idle, or backpressured by a full queue.
_RTC_IDLE_SLEEP_S: float = 0.01
# Backoff between transient inference errors (per consecutive failure).
_RTC_ERROR_RETRY_DELAY_S: float = 0.5
# Consecutive transient errors tolerated before giving up and propagating shutdown.
_RTC_MAX_CONSECUTIVE_ERRORS: int = 10
# Hard timeout for joining the RTC thread on stop().
_RTC_JOIN_TIMEOUT_S: float = 3.0


class _RTCFatalError(RuntimeError):
    """An RTC invariant violation that must stop the rollout immediately."""


# ---------------------------------------------------------------------------
# RTC helpers
# ---------------------------------------------------------------------------


def _normalize_prev_actions_length(prev_actions: torch.Tensor, target_steps: int) -> torch.Tensor:
    """Pad or truncate RTC prefix actions to a fixed length for stable compiled inference."""
    if prev_actions.ndim != 2:
        raise ValueError(f"Expected 2D [T, A] tensor, got shape={tuple(prev_actions.shape)}")
    steps, action_dim = prev_actions.shape
    if steps == target_steps:
        return prev_actions
    if steps > target_steps:
        return prev_actions[:target_steps]
    padded = torch.zeros((target_steps, action_dim), dtype=prev_actions.dtype, device=prev_actions.device)
    padded[:steps] = prev_actions
    return padded


def _max_action_residual(requested: dict, sent: dict | None) -> float | None:
    """Return the largest comparable robot-space action residual."""
    if not isinstance(requested, dict) or not isinstance(sent, dict):
        return None

    residuals: list[float] = []
    for key in requested.keys() & sent.keys():
        try:
            residual = abs(float(requested[key]) - float(sent[key]))
        except (TypeError, ValueError):
            continue
        residuals.append(residual if math.isfinite(residual) else math.inf)
    return max(residuals) if residuals else None


@dataclass(frozen=True)
class RTCPrefixHealthSnapshot:
    """Lock-consistent prefix-health counters for diagnostics and tests."""

    enabled: bool
    total_action_results: int
    comparable_action_results: int
    uncomparable_action_results: int
    severe_action_results: int
    consecutive_severe: int
    feedback_revision: int
    acknowledged_revision: int
    replan_requested: bool
    replan_requests: int
    replans_since_healthy: int
    unguided_inferences: int
    unguided_merges: int
    last_max_residual: float | None
    max_residual: float
    safety_stop_requested: bool


# ---------------------------------------------------------------------------
# RTCInferenceEngine
# ---------------------------------------------------------------------------


class RTCInferenceEngine(InferenceEngine):
    """Async RTC inference: a background thread produces action chunks.

    ``get_action`` pops the next action from the shared queue (or
    returns ``None`` if the queue is empty).  The main loop should call
    ``notify_observation`` every tick and ``pause``/``resume`` around
    human-intervention phases.
    """

    def __init__(
        self,
        policy: PreTrainedPolicy,
        preprocessor: PolicyProcessorPipeline,
        postprocessor: PolicyProcessorPipeline,
        robot_wrapper: ThreadSafeRobot,
        rtc_config: RTCConfig,
        hw_features: dict,
        task: str,
        fps: float,
        device: str | None,
        use_torch_compile: bool = False,
        compile_warmup_inferences: int = 2,
        rtc_queue_threshold: int = 30,
        rtc_timing_mode: str = "legacy",
        guidance_delay_mode: str = "legacy_max",
        fixed_guidance_delay_steps: int = 5,
        latency_warmup_inferences: int = 5,
        latency_window_size: int = 32,
        latency_percentile: float = 0.95,
        delay_hysteresis_steps: float = 0.25,
        delay_change_confirmations: int = 3,
        timing_diagnostics: bool = False,
        enforce_guided_execution_window: bool = False,
        prefix_health_enabled: bool = False,
        prefix_health_severe_residual_threshold: float = 5.0,
        prefix_health_consecutive_severe: int = 3,
        prefix_health_safety_stop_replans: int = 0,
        shutdown_event: Event | None = None,
    ) -> None:
        if rtc_queue_threshold < 0:
            raise ValueError("RTC queue threshold must be non-negative")
        policy_chunk_size = getattr(getattr(policy, "config", None), "chunk_size", None)
        if isinstance(policy_chunk_size, bool) or not isinstance(policy_chunk_size, int):
            policy_chunk_size = None
        if (
            rtc_timing_mode == "actual_consumed"
            and policy_chunk_size is not None
            and rtc_queue_threshold >= policy_chunk_size
        ):
            raise ValueError(
                "RTC queue threshold must be smaller than policy chunk_size: "
                f"queue_threshold={rtc_queue_threshold}, chunk_size={policy_chunk_size}"
            )
        if rtc_timing_mode not in ("legacy", "actual_consumed"):
            raise ValueError(f"Unsupported RTC timing mode: {rtc_timing_mode!r}")
        if rtc_timing_mode == "legacy" and guidance_delay_mode != "legacy_max":
            raise ValueError("legacy RTC timing requires guidance_delay_mode='legacy_max'")
        if enforce_guided_execution_window and rtc_timing_mode != "actual_consumed":
            raise ValueError("RTC guided execution window enforcement requires timing_mode='actual_consumed'")
        if rtc_timing_mode == "actual_consumed" and guidance_delay_mode not in (
            "fixed",
            "rolling_p95",
        ):
            raise ValueError(
                "actual_consumed RTC timing requires guidance_delay_mode='fixed' or 'rolling_p95'"
            )
        if (
            rtc_timing_mode == "actual_consumed"
            and fixed_guidance_delay_steps >= rtc_config.execution_horizon
        ):
            raise ValueError("RTC fixed guidance delay must be smaller than execution_horizon")
        if prefix_health_enabled and rtc_timing_mode != "actual_consumed":
            raise ValueError("RTC prefix health requires rtc_timing_mode='actual_consumed'")
        if (
            not math.isfinite(prefix_health_severe_residual_threshold)
            or prefix_health_severe_residual_threshold <= 0
        ):
            raise ValueError("RTC prefix health severe residual threshold must be finite and positive")
        if prefix_health_consecutive_severe < 1:
            raise ValueError("RTC prefix health consecutive severe count must be positive")
        if prefix_health_safety_stop_replans < 0:
            raise ValueError("RTC prefix health safety stop replans must be non-negative")

        self._policy = policy
        self._preprocessor = preprocessor
        self._postprocessor = postprocessor
        self._robot = robot_wrapper
        self._rtc_config = rtc_config
        self._hw_features = hw_features
        self._task = task
        self._fps = fps
        self._device = device or "cpu"
        self._use_torch_compile = use_torch_compile
        self._compile_warmup_inferences = compile_warmup_inferences
        self._rtc_queue_threshold = rtc_queue_threshold
        self._policy_chunk_size = policy_chunk_size
        self._rtc_timing_mode = rtc_timing_mode
        self._guidance_delay_mode = guidance_delay_mode
        self._fixed_guidance_delay_steps = fixed_guidance_delay_steps
        self._latency_warmup_inferences = latency_warmup_inferences
        self._latency_window_size = latency_window_size
        self._latency_percentile = latency_percentile
        self._delay_hysteresis_steps = delay_hysteresis_steps
        self._delay_change_confirmations = delay_change_confirmations
        self._timing_diagnostics = timing_diagnostics
        self._enforce_guided_execution_window = enforce_guided_execution_window
        self._prefix_health_enabled = prefix_health_enabled
        self._prefix_health_severe_residual_threshold = prefix_health_severe_residual_threshold
        self._prefix_health_consecutive_severe = prefix_health_consecutive_severe
        self._prefix_health_safety_stop_replans = prefix_health_safety_stop_replans

        self._action_queue: ActionQueue | None = None
        self._obs_holder: dict[str, Any] = {}
        self._obs_lock = Lock()
        self._observation_epoch = 0
        self._observation_sequence = 0
        self._inference_lock = RLock()
        self._policy_active = Event()
        self._compile_warmup_done = Event()
        self._shutdown_event = Event()
        self._rtc_error = Event()
        self._action_dispatch_lock = RLock()
        self._fatal_state_lock = Lock()
        self._fatal_error: BaseException | None = None
        self._global_shutdown_event = shutdown_event
        self._rtc_thread: Thread | None = None
        self._prefix_health_lock = Lock()
        self._prefix_health_replan_event = Event()
        self._prefix_health_stop_event = Event()
        self._reset_prefix_health_state()

        if not self._use_torch_compile:
            self._compile_warmup_done.set()
            logger.info("RTCInferenceEngine initialized (torch.compile disabled, no warmup needed)")
        else:
            logger.info(
                "RTCInferenceEngine initialized (torch.compile enabled, %d warmup inferences)",
                compile_warmup_inferences,
            )
        logger.info(
            "RTC timing mode=%s, guidance_delay_mode=%s, diagnostics=%s enforce_guided_execution_window=%s",
            self._rtc_timing_mode,
            self._guidance_delay_mode,
            self._timing_diagnostics,
            self._enforce_guided_execution_window,
        )
        if self._rtc_timing_mode == "actual_consumed" and self._policy_chunk_size is not None:
            nominal_replan_steps = self._policy_chunk_size - self._rtc_queue_threshold
            logger.info(
                "RTC cadence configuration: chunk_size=%d queue_threshold=%d "
                "nominal_replan_interval_steps=%d nominal_replan_hz=%.3f",
                self._policy_chunk_size,
                self._rtc_queue_threshold,
                nominal_replan_steps,
                self._fps / nominal_replan_steps,
            )
        if self._prefix_health_enabled:
            logger.info(
                "RTC prefix health enabled: severe_residual=%.3f consecutive=%d safety_stop_replans=%d",
                self._prefix_health_severe_residual_threshold,
                self._prefix_health_consecutive_severe,
                self._prefix_health_safety_stop_replans,
            )

        # Processor introspection for relative-action re-anchoring.
        self._relative_step = next(
            (s for s in preprocessor.steps if isinstance(s, RelativeActionsProcessorStep) and s.enabled),
            None,
        )
        self._normalizer_step = next(
            (s for s in preprocessor.steps if isinstance(s, NormalizerProcessorStep)),
            None,
        )
        if self._relative_step is not None:
            if self._relative_step.action_names is None:
                cfg_names = getattr(policy.config, "action_feature_names", None)
                if cfg_names:
                    self._relative_step.action_names = list(cfg_names)
                else:
                    self._relative_step.action_names = [
                        k for k in robot_wrapper.action_features if k.endswith(".pos")
                    ]
            logger.info("Relative actions enabled: RTC prefix will be re-anchored")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @property
    def ready(self) -> bool:
        """True once torch.compile warmup is complete (or immediately if compile is disabled)."""
        return self._compile_warmup_done.is_set()

    @property
    def failed(self) -> bool:
        """True if the RTC background thread exited due to an unrecoverable error."""
        return self._rtc_error.is_set()

    @property
    def action_queue(self) -> ActionQueue | None:
        """The shared action queue between the RTC thread and the main loop."""
        return self._action_queue

    def start(self) -> None:
        """Launch the RTC background thread."""
        self._action_queue = ActionQueue(self._rtc_config)
        with self._obs_lock:
            self._obs_holder = {
                "obs": None,
                "observation_epoch": self._observation_epoch,
                "observation_sequence": self._observation_sequence,
                "robot_type": self._robot.robot_type,
            }
        self._shutdown_event.clear()
        self._rtc_thread = Thread(
            target=self._rtc_loop,
            daemon=True,
            name="RTCInference",
        )
        self._rtc_thread.start()
        logger.info("RTC inference thread started")

    def stop(self) -> None:
        """Signal the RTC thread to stop and wait for it."""
        logger.info("Stopping RTC inference thread...")
        self._shutdown_event.set()
        self._policy_active.clear()
        thread = self._rtc_thread
        if thread is None:
            return
        if thread.is_alive():
            thread.join(timeout=_RTC_JOIN_TIMEOUT_S)
        if thread.is_alive():
            error = TimeoutError(f"RTC thread did not stop within {_RTC_JOIN_TIMEOUT_S:.1f}s")
            logger.error("%s", error)
            self._enter_fatal_state(error)
            return
        logger.info("RTC inference thread stopped")
        self._rtc_thread = None

    def pause(self) -> None:
        """Pause the RTC background thread."""
        logger.info("Pausing RTC inference thread")
        self._policy_active.clear()

    def resume(self) -> None:
        """Resume the RTC background thread."""
        logger.info("Resuming RTC inference thread")
        self._policy_active.set()

    def reset(self) -> None:
        """Reset the policy, processors, and action queue."""
        logger.info("Resetting RTC inference state (policy + processors + queue)")
        with self._inference_lock:
            with self._obs_lock:
                self._observation_epoch += 1
                self._obs_holder["obs"] = None
                self._obs_holder["observation_epoch"] = self._observation_epoch
            self._policy.reset()
            self._preprocessor.reset()
            self._postprocessor.reset()
            self._reset_prefix_health_state()
            if self._action_queue is not None:
                self._action_queue.clear()

    # ------------------------------------------------------------------
    # Action production (called from main thread)
    # ------------------------------------------------------------------

    def get_action(self, obs_frame: dict | None) -> torch.Tensor | None:
        """Pop the next action from the RTC queue (ignores ``obs_frame``)."""
        if (
            self._rtc_error.is_set()
            or self._prefix_health_stop_event.is_set()
            or self._prefix_health_replan_event.is_set()
        ):
            return None
        if self._action_queue is None:
            return None
        if self._enforce_guided_execution_window:
            snapshot = self._action_queue.snapshot()
            next_model_index = snapshot.next_model_action_index
            if next_model_index is not None and next_model_index >= self._rtc_config.execution_horizon:
                self._enter_fatal_state(
                    _RTCFatalError(
                        "RTC refused to dispatch an action outside the guided execution window: "
                        f"model_action_index={next_model_index}, "
                        f"execution_horizon={self._rtc_config.execution_horizon}"
                    )
                )
                return None
        if self._rtc_timing_mode == "actual_consumed" and self._timing_diagnostics:
            pop_result = self._action_queue.get_with_diagnostics()
            action = pop_result.action
            if action is not None:
                logger.debug(
                    "RTC action dequeue: source_chunk_generation=%s model_action_index=%s "
                    "total_consumed=%d queue_remaining=%d",
                    pop_result.source_chunk_generation,
                    pop_result.model_action_index,
                    pop_result.total_consumed,
                    pop_result.queue_size_after,
                )
        else:
            action = self._action_queue.get()
        if (
            self._rtc_error.is_set()
            or self._prefix_health_stop_event.is_set()
            or self._prefix_health_replan_event.is_set()
        ):
            return None
        return action

    @contextmanager
    def action_dispatch_guard(self) -> Iterator[bool]:
        """Prevent an action dispatch from racing a fatal-state transition."""
        with self._action_dispatch_lock:
            yield not (
                self._rtc_error.is_set()
                or self._prefix_health_stop_event.is_set()
                or self._prefix_health_replan_event.is_set()
            )

    def notify_observation(self, obs: dict) -> None:
        """Publish the latest observation for the RTC thread to consume."""
        with self._obs_lock:
            self._observation_sequence += 1
            self._obs_holder["obs"] = obs
            self._obs_holder["observation_epoch"] = self._observation_epoch
            self._obs_holder["observation_sequence"] = self._observation_sequence

    def notify_action_result(self, requested: dict, sent: dict | None, observation: dict) -> None:
        """Track robot-space safety residuals and request an unguided replan when persistent."""
        if not self._prefix_health_enabled:
            return

        residual = _max_action_residual(requested, sent)
        with self._obs_lock:
            trigger_observation_sequence = self._observation_sequence
        requested_revision: int | None = None
        request_stop = False
        stop_replan_count = 0
        with self._prefix_health_lock:
            self._prefix_health_total_action_results += 1
            if residual is None:
                self._prefix_health_uncomparable_action_results += 1
                self._prefix_health_current_consecutive_severe = 0
                return

            self._prefix_health_comparable_action_results += 1
            self._prefix_health_last_max_residual = residual
            self._prefix_health_max_residual = max(self._prefix_health_max_residual, residual)
            if residual < self._prefix_health_severe_residual_threshold:
                self._prefix_health_current_consecutive_severe = 0
                self._prefix_health_replans_since_healthy = 0
                return

            self._prefix_health_severe_action_results += 1
            self._prefix_health_current_consecutive_severe += 1
            if self._prefix_health_current_consecutive_severe < self._prefix_health_consecutive_severe:
                return

            self._prefix_health_current_consecutive_severe = 0
            self._prefix_health_feedback_revision += 1
            self._prefix_health_required_observation_sequence = max(
                self._prefix_health_required_observation_sequence,
                trigger_observation_sequence + 1,
            )
            self._prefix_health_replan_requests += 1
            self._prefix_health_replans_since_healthy += 1
            requested_revision = self._prefix_health_feedback_revision
            self._prefix_health_replan_event.set()
            if (
                self._prefix_health_safety_stop_replans > 0
                and self._prefix_health_replans_since_healthy >= self._prefix_health_safety_stop_replans
            ):
                self._prefix_health_stop_event.set()
                request_stop = True
                stop_replan_count = self._prefix_health_replans_since_healthy

        logger.warning(
            "RTC prefix health requested unguided replan: residual=%.3f revision=%d",
            residual,
            requested_revision,
        )
        if request_stop:
            logger.error(
                "RTC prefix health safety stop requested after %d replans without a healthy action",
                stop_replan_count,
            )
            self._enter_fatal_state(RuntimeError("RTC prefix health safety stop threshold reached"))

    def prefix_health_snapshot(self) -> RTCPrefixHealthSnapshot:
        """Return a lock-consistent snapshot of prefix-health state."""
        with self._prefix_health_lock:
            return RTCPrefixHealthSnapshot(
                enabled=self._prefix_health_enabled,
                total_action_results=self._prefix_health_total_action_results,
                comparable_action_results=self._prefix_health_comparable_action_results,
                uncomparable_action_results=self._prefix_health_uncomparable_action_results,
                severe_action_results=self._prefix_health_severe_action_results,
                consecutive_severe=self._prefix_health_current_consecutive_severe,
                feedback_revision=self._prefix_health_feedback_revision,
                acknowledged_revision=self._prefix_health_acknowledged_revision,
                replan_requested=(
                    self._prefix_health_feedback_revision > self._prefix_health_acknowledged_revision
                ),
                replan_requests=self._prefix_health_replan_requests,
                replans_since_healthy=self._prefix_health_replans_since_healthy,
                unguided_inferences=self._prefix_health_unguided_inferences,
                unguided_merges=self._prefix_health_unguided_merges,
                last_max_residual=self._prefix_health_last_max_residual,
                max_residual=self._prefix_health_max_residual,
                safety_stop_requested=self._prefix_health_stop_event.is_set(),
            )

    def _reset_prefix_health_state(self) -> None:
        with self._prefix_health_lock:
            self._prefix_health_total_action_results = 0
            self._prefix_health_comparable_action_results = 0
            self._prefix_health_uncomparable_action_results = 0
            self._prefix_health_severe_action_results = 0
            self._prefix_health_current_consecutive_severe = 0
            self._prefix_health_feedback_revision = 0
            self._prefix_health_acknowledged_revision = 0
            self._prefix_health_required_observation_sequence = 0
            self._prefix_health_replan_requests = 0
            self._prefix_health_replans_since_healthy = 0
            self._prefix_health_unguided_inferences = 0
            self._prefix_health_unguided_merges = 0
            self._prefix_health_last_max_residual = None
            self._prefix_health_max_residual = 0.0
            self._prefix_health_replan_event.clear()
            self._prefix_health_stop_event.clear()

    def _begin_prefix_health_inference(self, observation_sequence: int) -> tuple[int, bool, bool]:
        if not self._prefix_health_enabled:
            return 0, False, False
        with self._prefix_health_lock:
            revision = self._prefix_health_feedback_revision
            suppress_guidance = revision > self._prefix_health_acknowledged_revision
            waiting_for_observation = (
                suppress_guidance and observation_sequence < self._prefix_health_required_observation_sequence
            )
            if waiting_for_observation:
                return revision, False, True
            if suppress_guidance:
                self._prefix_health_unguided_inferences += 1
            return revision, suppress_guidance, False

    def _acknowledge_prefix_health_revision(self, revision: int) -> None:
        if not self._prefix_health_enabled or revision <= 0:
            return
        with self._prefix_health_lock:
            revision = min(revision, self._prefix_health_feedback_revision)
            if revision <= self._prefix_health_acknowledged_revision:
                return
            self._prefix_health_acknowledged_revision = revision
            self._prefix_health_unguided_merges += 1
            if self._prefix_health_acknowledged_revision >= self._prefix_health_feedback_revision:
                self._prefix_health_replan_event.clear()

        logger.info("RTC prefix health acknowledged unguided replan revision=%d", revision)

    def _prefix_health_revision_is_current(self, revision: int) -> bool:
        if not self._prefix_health_enabled:
            return True
        with self._prefix_health_lock:
            return revision == self._prefix_health_feedback_revision

    @property
    def fatal_error(self) -> BaseException | None:
        """The first exception that placed the RTC engine in a fatal state."""
        return self._fatal_error

    def _enter_fatal_state(self, error: BaseException) -> None:
        """Fail closed, clear queued actions, and stop the owning rollout."""
        with self._action_dispatch_lock, self._fatal_state_lock:
            if self._fatal_error is None:
                self._fatal_error = error
            self._rtc_error.set()
            self._policy_active.clear()
            self._shutdown_event.set()
            if self._action_queue is not None:
                self._action_queue.clear()
            if self._global_shutdown_event is not None:
                self._global_shutdown_event.set()

    # ------------------------------------------------------------------
    # RTC: background inference thread
    # ------------------------------------------------------------------

    def _rtc_loop(self) -> None:
        """Background thread that generates action chunks via RTC."""
        try:
            latency_tracker = LatencyTracker()
            guidance_estimator = (
                GuidanceDelayEstimator(
                    fps=self._fps,
                    mode=self._guidance_delay_mode,
                    fixed_delay_steps=self._fixed_guidance_delay_steps,
                    warmup_inferences=self._latency_warmup_inferences,
                    window_size=self._latency_window_size,
                    min_samples=min(5, self._latency_window_size),
                    percentile=self._latency_percentile,
                    hysteresis_steps=self._delay_hysteresis_steps,
                    change_confirmations=self._delay_change_confirmations,
                )
                if self._rtc_timing_mode == "actual_consumed"
                else None
            )
            time_per_action = 1.0 / self._fps
            policy_device = torch.device(self._device)

            warmup_required = max(1, self._compile_warmup_inferences) if self._use_torch_compile else 0
            inference_count = 0
            consecutive_errors = 0
            consecutive_merge_skip_clamps = 0
            consecutive_guidance_delay_clamps = 0
            previous_replan_total_consumed: int | None = None

            while not self._shutdown_event.is_set():
                if not self._policy_active.is_set():
                    time.sleep(_RTC_IDLE_SLEEP_S)
                    continue

                if self._prefix_health_stop_event.is_set():
                    raise RuntimeError("RTC prefix health safety stop threshold reached")

                queue = self._action_queue
                with self._obs_lock:
                    obs = self._obs_holder.get("obs")
                    observation_epoch = self._obs_holder.get("observation_epoch")
                    observation_sequence = self._obs_holder.get("observation_sequence", 0)
                if queue is None or obs is None:
                    time.sleep(_RTC_IDLE_SLEEP_S)
                    continue

                if queue.qsize() <= self._rtc_queue_threshold or self._prefix_health_replan_event.is_set():
                    try:
                        (
                            feedback_revision,
                            suppress_prefix_guidance,
                            waiting_for_feedback_observation,
                        ) = self._begin_prefix_health_inference(observation_sequence)
                        if waiting_for_feedback_observation:
                            time.sleep(_RTC_IDLE_SLEEP_S)
                            continue
                        inference_start = (
                            queue.snapshot() if self._rtc_timing_mode == "actual_consumed" else None
                        )
                        diagnostic_before = (
                            inference_start
                            if self._timing_diagnostics and inference_start is not None
                            else queue.snapshot()
                            if self._timing_diagnostics
                            else None
                        )
                        current_time = time.perf_counter()
                        if inference_start is not None:
                            idx_before = inference_start.next_action_index
                            prev_actions = inference_start.original_leftover
                            estimated_delay = guidance_estimator.estimate()
                            max_guidance_delay = self._rtc_config.execution_horizon - 1
                            if estimated_delay > max_guidance_delay:
                                consecutive_guidance_delay_clamps += 1
                                delay = max_guidance_delay
                                logger.error(
                                    "RTC guidance delay exceeded execution horizon and was clamped: "
                                    "estimated=%d clamped=%d count=%d",
                                    estimated_delay,
                                    delay,
                                    consecutive_guidance_delay_clamps,
                                )
                                if consecutive_guidance_delay_clamps >= 3:
                                    raise _RTCFatalError(
                                        "RTC guidance delay exceeded execution_horizon for "
                                        "3 consecutive inferences"
                                    )
                            else:
                                consecutive_guidance_delay_clamps = 0
                                delay = estimated_delay
                        else:
                            idx_before = queue.get_action_index()
                            prev_actions = queue.get_left_over()
                            latency = latency_tracker.max()
                            delay = math.ceil(latency / time_per_action) if latency else 0

                        if suppress_prefix_guidance:
                            prev_actions = None
                            logger.info(
                                "RTC prefix health suppressing guidance for revision=%d",
                                feedback_revision,
                            )

                        obs_batch = build_dataset_frame(self._hw_features, obs, prefix="observation")
                        obs_batch = prepare_observation_for_inference(
                            obs_batch, policy_device, self._task, self._robot.robot_type
                        )
                        obs_batch["task"] = [self._task]

                        with self._inference_lock:
                            with self._obs_lock:
                                observation_is_current = (
                                    observation_epoch == self._observation_epoch
                                    and self._obs_holder.get("obs") is not None
                                )
                            if not observation_is_current:
                                logger.info(
                                    "RTC discarded observation from a reset epoch: observed=%s current=%d",
                                    observation_epoch,
                                    self._observation_epoch,
                                )
                                continue

                            preprocessed = self._preprocessor(obs_batch)

                            if prev_actions is not None and self._relative_step is not None:
                                # Rebase against the raw cached state so the leftover tail stays in
                                # the training-time coordinate frame.
                                raw_state = self._relative_step.get_cached_state()
                                if raw_state is not None:
                                    prev_abs = (
                                        inference_start.processed_leftover
                                        if inference_start is not None
                                        else queue.get_processed_left_over()
                                    )
                                    if prev_abs is not None and prev_abs.numel() > 0:
                                        prev_actions = reanchor_relative_rtc_prefix(
                                            prev_actions_absolute=prev_abs,
                                            current_state=raw_state,
                                            relative_step=self._relative_step,
                                            normalizer_step=self._normalizer_step,
                                            policy_device=policy_device,
                                        )

                            if prev_actions is not None:
                                prev_actions = _normalize_prev_actions_length(
                                    prev_actions, target_steps=self._rtc_config.execution_horizon
                                )

                            actions = self._policy.predict_action_chunk(
                                preprocessed,
                                inference_delay=delay,
                                prev_chunk_left_over=prev_actions,
                            )

                            if self._shutdown_event.is_set():
                                logger.info("RTC discarded inference result requested during shutdown")
                                return

                            original = actions.squeeze(0).clone()
                            processed = self._postprocessor(actions).squeeze(0)
                        new_latency = time.perf_counter() - current_time
                        new_delay = math.ceil(new_latency / time_per_action)
                        diagnostic_after_inference = queue.snapshot() if self._timing_diagnostics else None

                        inference_count += 1
                        is_warmup = self._use_torch_compile and inference_count <= warmup_required
                        if (
                            is_warmup
                            and inference_count >= warmup_required
                            and not self._compile_warmup_done.is_set()
                        ):
                            self._compile_warmup_done.set()
                            logger.info("Compile warmup complete (%d inferences)", inference_count)
                        if is_warmup:
                            latency_tracker.reset()
                            if guidance_estimator is not None:
                                guidance_estimator.reset()
                        elif self._rtc_timing_mode == "legacy":
                            latency_tracker.add(new_latency)

                        merge_result = None
                        if self._shutdown_event.is_set():
                            logger.info("RTC discarded completed inference result during shutdown")
                            return
                        if self._prefix_health_stop_event.is_set():
                            raise _RTCFatalError("RTC prefix health safety stop occurred during inference")
                        if not self._prefix_health_revision_is_current(feedback_revision):
                            logger.warning(
                                "RTC discarded inference result after prefix-health revision changed: "
                                "started=%d current=%d",
                                feedback_revision,
                                self.prefix_health_snapshot().feedback_revision,
                            )
                            consecutive_errors = 0
                            continue
                        if inference_start is not None:
                            merge_result = queue.merge_actual_consumed(
                                original,
                                processed,
                                inference_start,
                                max_actual_consumed_steps=self._rtc_config.execution_horizon,
                            )
                            if merge_result.stale:
                                logger.info(
                                    "RTC discarded stale inference result: generation=%d->%d",
                                    merge_result.expected_generation,
                                    merge_result.observed_generation,
                                )
                                consecutive_errors = 0
                                continue

                            if merge_result.consumption_limit_exceeded:
                                raise _RTCFatalError(
                                    "RTC consumed actions reached execution_horizon during one inference: "
                                    f"actual_consumed_steps={merge_result.actual_consumed_steps}, "
                                    f"execution_horizon={self._rtc_config.execution_horizon}"
                                )

                            if suppress_prefix_guidance:
                                self._acknowledge_prefix_health_revision(feedback_revision)

                            if merge_result.skip_was_clamped:
                                consecutive_merge_skip_clamps += 1
                                if consecutive_merge_skip_clamps >= 3:
                                    raise _RTCFatalError(
                                        "RTC actual-consumed merge skip exceeded the action chunk "
                                        "for 3 consecutive inferences"
                                    )
                            else:
                                consecutive_merge_skip_clamps = 0

                            if not is_warmup:
                                guidance_estimator.observe(new_latency)
                        else:
                            queue.merge(original, processed, new_delay, idx_before)

                        if diagnostic_before is not None and diagnostic_after_inference is not None:
                            diagnostic_after_merge = queue.snapshot()
                            if merge_result is not None:
                                actual_consumed = merge_result.actual_consumed_steps
                                merge_skip = merge_result.merge_skip
                                merge_skip_clamped = int(merge_result.skip_was_clamped)
                                generation_before = merge_result.expected_generation
                                generation_after_inference = merge_result.observed_generation
                                generation_after_merge = merge_result.generation_after
                                index_before = inference_start.next_action_index
                                index_after_inference = index_before + actual_consumed
                                index_after_merge = diagnostic_after_merge.next_action_index
                                queue_before = inference_start.queue_size
                                queue_after_inference = max(0, queue_before - actual_consumed)
                                queue_after_merge = merge_result.queue_size_after
                                if previous_replan_total_consumed is None:
                                    replan_interval_steps = 0
                                else:
                                    replan_interval_steps = max(
                                        0,
                                        merge_result.current_total_consumed - previous_replan_total_consumed,
                                    )
                                previous_replan_total_consumed = merge_result.current_total_consumed
                            else:
                                actual_consumed = max(
                                    0,
                                    diagnostic_after_inference.total_consumed
                                    - diagnostic_before.total_consumed,
                                )
                                merge_skip = (
                                    max(0, min(new_delay, len(original), len(processed)))
                                    if self._rtc_config.enabled
                                    else 0
                                )
                                merge_skip_clamped = int(self._rtc_config.enabled and merge_skip != new_delay)
                                generation_before = diagnostic_before.generation
                                generation_after_inference = diagnostic_after_inference.generation
                                generation_after_merge = diagnostic_after_merge.generation
                                index_before = diagnostic_before.next_action_index
                                index_after_inference = diagnostic_after_inference.next_action_index
                                index_after_merge = diagnostic_after_merge.next_action_index
                                queue_before = diagnostic_before.queue_size
                                queue_after_inference = diagnostic_after_inference.queue_size
                                queue_after_merge = diagnostic_after_merge.queue_size
                                replan_interval_steps = 0
                            measured_control_fps = actual_consumed / new_latency if new_latency > 0.0 else 0.0
                            replan_hz = (
                                self._fps / replan_interval_steps if replan_interval_steps > 0 else 0.0
                            )
                            warmup_state = "compile_warmup" if is_warmup else "steady"
                            logger.info(
                                "RTC timing diagnostics: mode=%s latency_ms=%.3f "
                                "guidance_delay_estimate=%d wall_latency_steps=%d "
                                "actual_consumed_steps=%d merge_skip=%d "
                                "generation=%d->%d->%d index=%d->%d->%d "
                                "queue=%d->%d->%d measured_control_fps=%.2f "
                                "replan_interval_steps=%d replan_hz=%.3f "
                                "source_chunk_generation=%s next_model_action_index=%s "
                                "merge_skip_clamped=%d warmup_state=%s",
                                self._rtc_timing_mode,
                                new_latency * 1000.0,
                                delay,
                                new_delay,
                                actual_consumed,
                                merge_skip,
                                generation_before,
                                generation_after_inference,
                                generation_after_merge,
                                index_before,
                                index_after_inference,
                                index_after_merge,
                                queue_before,
                                queue_after_inference,
                                queue_after_merge,
                                measured_control_fps,
                                replan_interval_steps,
                                replan_hz,
                                diagnostic_after_merge.source_chunk_generation,
                                diagnostic_after_merge.next_model_action_index,
                                merge_skip_clamped,
                                warmup_state,
                            )

                        consecutive_errors = 0
                        logger.debug("RTC inference latency=%.2fs, queue=%d", new_latency, queue.qsize())

                    except _RTCFatalError:
                        raise
                    except Exception as e:
                        consecutive_errors += 1
                        logger.error(
                            "RTC inference error (%d/%d): %s",
                            consecutive_errors,
                            _RTC_MAX_CONSECUTIVE_ERRORS,
                            e,
                        )
                        logger.debug(traceback.format_exc())
                        if consecutive_errors >= _RTC_MAX_CONSECUTIVE_ERRORS:
                            # Persistent failure: stop retrying and propagate shutdown.
                            raise
                        time.sleep(_RTC_ERROR_RETRY_DELAY_S)
                else:
                    time.sleep(_RTC_IDLE_SLEEP_S)

        except Exception as e:
            logger.error("Fatal error in RTC thread: %s", e)
            logger.error(traceback.format_exc())
            self._enter_fatal_state(e)
            # Unblock any warmup waiters so the main loop doesn't spin forever
            self._compile_warmup_done.set()
