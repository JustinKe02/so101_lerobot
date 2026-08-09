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
from numbers import Real
from threading import Event, Lock, RLock, Thread
from typing import Any

import numpy as np
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
    TransitionKey,
    create_transition,
)
from lerobot.utils.feature_utils import build_dataset_frame

from ..realtime_executor import RealtimeExecutor
from ..robot_wrapper import ThreadSafeRobot
from ..speed_adapter import ROBOT_ACTION_COORDINATE_SPACE
from ..stall_guard import StallContactError, StallContactGuard
from ..time_axis import TimeAxisPlanner
from ..trajectory import (
    CommittedActionPrefill,
    DelayAlignedTrajectory,
    PrefillCapacityError,
    RealtimeTraceWriteError,
    RealtimeTraceWriter,
    TimedRecord,
)
from .base import InferenceEngine

logger = logging.getLogger(__name__)

_EXECUTOR_TIMESTAMP_TOLERANCE_S = 1e-6

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


@dataclass(frozen=True)
class RTCExecutorControlSnapshot:
    """Lock-consistent diagnostics from the independent executor heartbeat."""

    heartbeat_count: int
    dispatch_count: int
    hold_count: int
    queue_empty_count: int
    deadline_miss_count: int
    missed_periods: int
    last_scheduled_timestamp: float | None
    last_started_timestamp: float | None
    last_finished_timestamp: float | None
    last_lateness_s: float
    last_execution_s: float
    last_deadline_miss: bool
    last_queue_empty: bool
    last_command: tuple[float, ...] | None
    last_applied_command: dict[str, Any] | None


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
        rtc_inference_mode: str = "guided",
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
        dynamic_prefill_enabled: bool = False,
        image_capture_delay_s: float = 0.0,
        camera_capture_delay_s: dict[str, float] | None = None,
        state_observation_delay_s: float = 0.0,
        max_prefill_steps: int = 0,
        max_camera_skew_s: float = 0.05,
        prefill_overflow: str = "error",
        shutdown_event: Event | None = None,
        time_axis_planner: TimeAxisPlanner | None = None,
        trace: RealtimeTraceWriter | None = None,
        trajectory: DelayAlignedTrajectory | None = None,
        realtime_executor: RealtimeExecutor | None = None,
        ordered_action_keys: list[str] | None = None,
        robot_action_processor: Any | None = None,
        action_filter: Any | None = None,
        stall_guard: StallContactGuard | None = None,
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
        if rtc_inference_mode not in ("guided", "trained_prefix"):
            raise ValueError(f"Unsupported RTC inference mode: {rtc_inference_mode!r}")
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
        if dynamic_prefill_enabled and rtc_inference_mode != "trained_prefix":
            raise ValueError("RTC dynamic prefill requires inference_mode='trained_prefix'")
        if dynamic_prefill_enabled and rtc_timing_mode != "actual_consumed":
            raise ValueError("RTC dynamic prefill requires timing_mode='actual_consumed'")
        if prefill_overflow not in ("error", "truncate_oldest"):
            raise ValueError(f"Unsupported RTC prefill overflow mode: {prefill_overflow!r}")
        for name, value in (
            ("image_capture_delay_s", image_capture_delay_s),
            ("state_observation_delay_s", state_observation_delay_s),
            ("max_camera_skew_s", max_camera_skew_s),
        ):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"RTC {name} must be finite and non-negative")
        resolved_camera_delays: dict[str, float] = {}
        for camera_key, value in (camera_capture_delay_s or {}).items():
            if not isinstance(camera_key, str) or not camera_key.strip():
                raise ValueError("RTC camera capture delay keys must be non-empty strings")
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"RTC camera capture delay for {camera_key!r} must be numeric")
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError(
                    f"RTC camera capture delay for {camera_key!r} must be finite and non-negative"
                )
            resolved_camera_delays[camera_key] = float(value)
        if isinstance(max_prefill_steps, bool) or max_prefill_steps < 0:
            raise ValueError("RTC max_prefill_steps must be a non-negative integer")

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
        self._rtc_inference_mode = rtc_inference_mode
        self._rtc_training_max_delay = int(
            getattr(getattr(policy, "config", None), "rtc_training_max_delay", 0)
        )
        if self._rtc_inference_mode == "trained_prefix" and self._rtc_training_max_delay <= 0:
            raise ValueError("trained_prefix RTC mode requires policy.config.rtc_training_max_delay > 0")
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
        self._dynamic_prefill_enabled = dynamic_prefill_enabled
        self._image_capture_delay_s = float(image_capture_delay_s)
        self._camera_capture_delay_s = resolved_camera_delays
        self._state_observation_delay_s = float(state_observation_delay_s)
        self._max_prefill_steps = int(max_prefill_steps) or self._rtc_training_max_delay
        self._max_camera_skew_s = float(max_camera_skew_s)
        self._prefill_overflow = prefill_overflow
        if self._dynamic_prefill_enabled and self._max_prefill_steps > self._rtc_training_max_delay:
            raise ValueError(
                "RTC max prefill steps exceed checkpoint training capacity: "
                f"requested={self._max_prefill_steps}, capacity={self._rtc_training_max_delay}"
            )
        self._time_axis_planner = time_axis_planner
        self._trace = trace
        self._trajectory = trajectory
        self._realtime_executor = realtime_executor
        self._robot_action_processor = robot_action_processor
        self._action_filter = action_filter
        self._stall_guard = stall_guard
        self._realtime_executor_needs_reset = realtime_executor is not None
        self._executor_last_observation_timestamp: float | None = None
        self._action_keys = list(ordered_action_keys or ())
        if not self._action_keys:
            self._action_keys = [key for key in robot_wrapper.action_features if key.endswith(".pos")]
        if not self._action_keys:
            self._action_keys = list(robot_wrapper.action_features)

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
        self._executor_thread: Thread | None = None
        self._executor_state_lock = RLock()
        self._executor_cycle_lock = RLock()
        self._executor_postfix_not_before_heartbeat: int | None = None
        self._executor_applied_heartbeat_count = 0
        self._executor_metrics_lock = Lock()
        self._latest_dispatched_action: torch.Tensor | None = None
        self._executor_heartbeat_count = 0
        self._executor_dispatch_count = 0
        self._executor_hold_count = 0
        self._executor_queue_empty_count = 0
        self._executor_deadline_miss_count = 0
        self._executor_missed_periods = 0
        self._executor_last_scheduled_timestamp: float | None = None
        self._executor_last_started_timestamp: float | None = None
        self._executor_last_finished_timestamp: float | None = None
        self._executor_last_lateness_s = 0.0
        self._executor_last_execution_s = 0.0
        self._executor_last_deadline_miss = False
        self._executor_last_queue_empty = False
        self._executor_last_command: tuple[float, ...] | None = None
        self._executor_last_applied_command: dict[str, Any] | None = None
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
        logger.info("RTC inference mode=%s", self._rtc_inference_mode)
        if self._dynamic_prefill_enabled:
            logger.info(
                "RTC dynamic prefill enabled: max_steps=%d image_delay_ms=%.3f "
                "state_delay_ms=%.3f overflow=%s",
                self._max_prefill_steps,
                self._image_capture_delay_s * 1000.0,
                self._state_observation_delay_s * 1000.0,
                self._prefill_overflow,
            )
        if self._time_axis_planner is not None:
            logger.info("RTC time-axis planner enabled")
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

    @property
    def owns_action_dispatch(self) -> bool:
        """The realtime executor sends commands from an independent heartbeat."""

        return self._realtime_executor is not None

    def executor_control_snapshot(self) -> RTCExecutorControlSnapshot:
        with self._executor_metrics_lock:
            return RTCExecutorControlSnapshot(
                heartbeat_count=self._executor_heartbeat_count,
                dispatch_count=self._executor_dispatch_count,
                hold_count=self._executor_hold_count,
                queue_empty_count=self._executor_queue_empty_count,
                deadline_miss_count=self._executor_deadline_miss_count,
                missed_periods=self._executor_missed_periods,
                last_scheduled_timestamp=self._executor_last_scheduled_timestamp,
                last_started_timestamp=self._executor_last_started_timestamp,
                last_finished_timestamp=self._executor_last_finished_timestamp,
                last_lateness_s=self._executor_last_lateness_s,
                last_execution_s=self._executor_last_execution_s,
                last_deadline_miss=self._executor_last_deadline_miss,
                last_queue_empty=self._executor_last_queue_empty,
                last_command=self._executor_last_command,
                last_applied_command=(
                    None
                    if self._executor_last_applied_command is None
                    else dict(self._executor_last_applied_command)
                ),
            )

    def start(self) -> None:
        """Launch inference and, when enabled, fixed-heartbeat control threads."""
        self._action_queue = ActionQueue(self._rtc_config)
        with self._obs_lock:
            self._obs_holder = {
                "obs": None,
                "observation_epoch": self._observation_epoch,
                "observation_sequence": self._observation_sequence,
                "robot_type": self._robot.robot_type,
                "observation_timing": None,
                "control_observation": None,
                "control_state": None,
                "control_state_timestamp": None,
            }
        self._shutdown_event.clear()
        self._reset_executor_control_metrics()
        self._executor_postfix_not_before_heartbeat = None
        self._executor_applied_heartbeat_count = 0
        self._rtc_thread = Thread(
            target=self._rtc_loop,
            daemon=True,
            name="RTCInference",
        )
        self._rtc_thread.start()
        logger.info("RTC inference thread started")
        if self._realtime_executor is not None:
            self._executor_thread = Thread(
                target=self._executor_control_loop,
                daemon=True,
                name="RTCExecutorControl",
            )
            self._executor_thread.start()
            logger.info(
                "RTC independent executor thread started (heartbeat=%.6fs)",
                self._realtime_executor.config.heartbeat_dt_s,
            )

    def stop(self) -> None:
        """Signal all RTC threads to stop and wait for them."""
        logger.info("Stopping RTC inference/control threads...")
        self._shutdown_event.set()
        self._policy_active.clear()
        alive_threads: list[str] = []
        for label, thread in (
            ("executor control", self._executor_thread),
            ("inference", self._rtc_thread),
        ):
            if thread is None:
                continue
            if thread.is_alive():
                thread.join(timeout=_RTC_JOIN_TIMEOUT_S)
            if thread.is_alive():
                alive_threads.append(label)
            else:
                logger.info("RTC %s thread stopped", label)
        if alive_threads:
            error = TimeoutError(
                "RTC threads did not stop within the join timeout: " + ", ".join(alive_threads)
            )
            logger.error("%s", error)
            self._enter_fatal_state(error)
        elif self.failed and self._trace is not None:
            mark_abnormal = getattr(self._trace, "mark_abnormal", None)
            if callable(mark_abnormal):
                mark_abnormal(self._fatal_error)
        if alive_threads:
            return
        self._rtc_thread = None
        self._executor_thread = None

    def pause(self) -> None:
        """Pause inference and wait for an in-flight executor dispatch."""
        logger.info("Pausing RTC inference/control threads")
        self._policy_active.clear()
        if self._realtime_executor is not None:
            with self._executor_cycle_lock, self._action_dispatch_lock:
                pass

    def resume(self) -> None:
        """Resume the RTC background thread."""
        logger.info("Resuming RTC inference thread")
        self._policy_active.set()

    def reset(self) -> None:
        """Reset the policy, processors, and action queue."""
        logger.info("Resetting RTC inference state (policy + processors + queue)")
        self.pause()
        with self._executor_cycle_lock, self._inference_lock:
            with self._obs_lock:
                self._observation_epoch += 1
                self._obs_holder["obs"] = None
                self._obs_holder["observation_timing"] = None
                self._obs_holder["executor_state"] = None
                self._obs_holder["executor_state_timestamp"] = None
                self._obs_holder["control_observation"] = None
                self._obs_holder["control_state"] = None
                self._obs_holder["control_state_timestamp"] = None
                self._obs_holder["observation_epoch"] = self._observation_epoch
            self._policy.reset()
            self._preprocessor.reset()
            self._postprocessor.reset()
            if self._trajectory is not None:
                self._trajectory.reset()
            if self._realtime_executor is not None:
                with self._executor_state_lock:
                    self._realtime_executor.clear_waypoints()
                    self._realtime_executor_needs_reset = True
                    self._executor_last_observation_timestamp = None
                    self._executor_postfix_not_before_heartbeat = None
                    self._executor_applied_heartbeat_count = 0
                    self._latest_dispatched_action = None
                    if self._action_filter is not None:
                        self._action_filter.reset()
                    if self._stall_guard is not None:
                        self._stall_guard.reset()
                    self._reset_executor_control_metrics()
            self._reset_prefix_health_state()
            if self._action_queue is not None:
                self._action_queue.clear()

    # ------------------------------------------------------------------
    # Action production (called from main thread)
    # ------------------------------------------------------------------

    def get_action(self, obs_frame: dict | None) -> torch.Tensor | None:
        """Return the next action, or the latest independently dispatched command."""
        if (
            self._rtc_error.is_set()
            or self._prefix_health_stop_event.is_set()
            or self._prefix_health_replan_event.is_set()
        ):
            return None
        if self._realtime_executor is not None and self._executor_thread is not None:
            with self._executor_state_lock:
                return (
                    None if self._latest_dispatched_action is None else self._latest_dispatched_action.clone()
                )
        action, _ = self._dequeue_action()
        if action is not None and self._realtime_executor is not None:
            try:
                action = self._smooth_realtime_action(action)
            except Exception as exc:
                self._enter_fatal_state(_RTCFatalError(f"Realtime executor failed: {exc}"))
                return None
        return action

    def _dequeue_action(self) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if self._action_queue is None:
            return None, None
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
                return None, None
        pop_result = self._action_queue.get_with_diagnostics()
        action = pop_result.action
        if action is not None and self._rtc_timing_mode == "actual_consumed" and self._timing_diagnostics:
            logger.debug(
                "RTC action dequeue: source_chunk_generation=%s model_action_index=%s "
                "total_consumed=%d queue_remaining=%d",
                pop_result.source_chunk_generation,
                pop_result.model_action_index,
                pop_result.total_consumed,
                pop_result.queue_size_after,
            )
        if (
            self._rtc_error.is_set()
            or self._prefix_health_stop_event.is_set()
            or self._prefix_health_replan_event.is_set()
        ):
            return None, pop_result.processed_leftover
        return action, pop_result.processed_leftover

    def _smooth_realtime_action(self, action: torch.Tensor) -> torch.Tensor:
        executor = self._realtime_executor
        queue = self._action_queue
        if executor is None or queue is None:
            return action

        with self._obs_lock:
            state = self._obs_holder.get("executor_state")
            state_timestamp = self._obs_holder.get("executor_state_timestamp")
        state_tensor = None
        if isinstance(state, torch.Tensor) and state.numel() == len(self._action_keys):
            state_tensor = state.detach().to(device="cpu", dtype=torch.float64).reshape(-1)

        now = self._monotonic_seconds()
        if self._realtime_executor_needs_reset or not executor.initialized:
            initial = (
                state_tensor.numpy()
                if state_tensor is not None
                else action.detach().to(device="cpu", dtype=torch.float64).numpy()
            )
            observation_stamp = (
                float(state_timestamp)
                if isinstance(state_timestamp, Real) and math.isfinite(float(state_timestamp))
                else now
            )
            observation_stamp = min(observation_stamp, now)
            executor.reset(
                initial,
                timestamp=now,
                observation_timestamp=observation_stamp,
            )
            self._executor_last_observation_timestamp = observation_stamp
            self._realtime_executor_needs_reset = False

        tick_timestamp = executor.next_heartbeat_timestamp
        if tick_timestamp is None:
            raise RuntimeError("realtime executor has no heartbeat timestamp after reset")
        snapshot = queue.snapshot()
        current = action.detach().to(device="cpu", dtype=torch.float64).reshape(1, -1)
        future = snapshot.processed_leftover
        if future is not None and len(future) > 0:
            future = future.detach().to(device="cpu", dtype=torch.float64)
            waypoints = torch.cat((current, future), dim=0)
        else:
            waypoints = current
        waypoint_timestamps = (
            tick_timestamp
            + torch.arange(len(waypoints), dtype=torch.float64) * executor.config.heartbeat_dt_s
        )
        executor.replace_waypoints_from(
            tick_timestamp,
            waypoint_timestamps.numpy(),
            waypoints.numpy(),
        )

        observation = None
        observation_stamp = None
        if (
            state_tensor is not None
            and isinstance(state_timestamp, Real)
            and math.isfinite(float(state_timestamp))
        ):
            candidate_stamp = min(float(state_timestamp), tick_timestamp)
            if (
                self._executor_last_observation_timestamp is None
                or candidate_stamp >= self._executor_last_observation_timestamp
            ):
                observation = state_tensor.numpy()
                observation_stamp = candidate_stamp
                self._executor_last_observation_timestamp = candidate_stamp

        command = executor.heartbeat(
            observation,
            observation_timestamp=observation_stamp,
        )
        if self._trace is not None:
            self._trace.write(
                "smooth_execution",
                heartbeat_timestamp=executor.last_heartbeat_timestamp,
                model_target=action,
                reference=executor.last_reference,
                command=command,
                lookahead_steps=len(waypoints),
                queue_remaining=snapshot.queue_size,
                underrun=executor.last_tick_was_underrun,
                underrun_count=executor.underrun_count,
                observed_state=observation,
                observed_state_timestamp=observation_stamp,
            )
        return torch.as_tensor(command, dtype=action.dtype, device=action.device)

    def _reset_executor_control_metrics(self) -> None:
        with self._executor_metrics_lock:
            self._executor_heartbeat_count = 0
            self._executor_dispatch_count = 0
            self._executor_hold_count = 0
            self._executor_queue_empty_count = 0
            self._executor_deadline_miss_count = 0
            self._executor_missed_periods = 0
            self._executor_last_scheduled_timestamp = None
            self._executor_last_started_timestamp = None
            self._executor_last_finished_timestamp = None
            self._executor_last_lateness_s = 0.0
            self._executor_last_execution_s = 0.0
            self._executor_last_deadline_miss = False
            self._executor_last_queue_empty = False
            self._executor_last_command = None
            self._executor_last_applied_command = None

    def _executor_control_loop(self) -> None:
        executor = self._realtime_executor
        if executor is None:
            return
        heartbeat_dt = executor.config.heartbeat_dt_s
        deadline_tolerance = max(0.001, heartbeat_dt * 0.1)
        schedule_origin: float | None = None
        schedule_tick = 0
        try:
            while not self._shutdown_event.is_set():
                if not self._policy_active.is_set():
                    self._policy_active.wait(timeout=_RTC_IDLE_SLEEP_S)
                    schedule_origin = None
                    schedule_tick = 0
                    continue

                now = self._monotonic_seconds()
                if schedule_origin is None:
                    schedule_origin = now
                    schedule_tick = 0
                next_deadline = schedule_origin + schedule_tick * heartbeat_dt
                wait_s = next_deadline - now
                if wait_s > 0.0:
                    self._shutdown_event.wait(timeout=min(wait_s, _RTC_IDLE_SLEEP_S))
                    continue

                scheduled = next_deadline
                missed_periods = self._run_executor_control_cycle(
                    scheduled,
                    heartbeat_dt=heartbeat_dt,
                    deadline_tolerance=deadline_tolerance,
                )
                if missed_periods:
                    schedule_origin = scheduled + (missed_periods + 1) * heartbeat_dt
                    schedule_tick = 0
                else:
                    schedule_tick += 1
        except BaseException as exc:
            logger.error("RTC executor control thread failed:\n%s", traceback.format_exc())
            self._enter_fatal_state(_RTCFatalError(f"Realtime executor control failed: {exc}"))

    def _run_executor_control_cycle(
        self,
        scheduled: float,
        *,
        heartbeat_dt: float,
        deadline_tolerance: float,
    ) -> int:
        with self._executor_cycle_lock:
            started = self._monotonic_seconds()
            outcome = self._run_executor_control_tick(scheduled)
            finished = self._monotonic_seconds()
            nominal_next = scheduled + heartbeat_dt
            missed_periods = 0
            if finished > nominal_next:
                missed_periods = int(math.floor((finished - nominal_next) / heartbeat_dt)) + 1
            lateness = max(0.0, started - scheduled)
            execution_s = max(0.0, finished - started)
            deadline_miss = lateness > deadline_tolerance or execution_s > heartbeat_dt or missed_periods > 0
            self._record_executor_control_outcome(
                outcome,
                scheduled=scheduled,
                started=started,
                finished=finished,
                lateness=lateness,
                execution_s=execution_s,
                deadline_miss=deadline_miss,
                missed_periods=missed_periods,
            )
            return missed_periods

    def _run_executor_control_tick(self, scheduled_timestamp: float) -> dict[str, Any]:
        executor = self._realtime_executor
        queue = self._action_queue
        if executor is None or queue is None:
            return {"heartbeat": False, "dispatched": False, "reason": "not_initialized"}

        with self._action_dispatch_lock:
            if not self._executor_dispatch_allowed():
                return {"heartbeat": False, "dispatched": False, "reason": "inactive_or_failed"}
            with self._executor_state_lock:
                with self._obs_lock:
                    raw_observation = self._obs_holder.get("control_observation")
                    state = self._obs_holder.get("control_state")
                    state_timestamp = self._obs_holder.get("control_state_timestamp")
                    if state is None:
                        state = self._obs_holder.get("executor_state")
                        state_timestamp = self._obs_holder.get("executor_state_timestamp")
                raw_observation = dict(raw_observation) if isinstance(raw_observation, dict) else {}
                state_tensor = None
                if isinstance(state, torch.Tensor) and state.numel() == len(self._action_keys):
                    state_tensor = state.detach().to(device="cpu", dtype=torch.float64).reshape(-1)

                if self._realtime_executor_needs_reset or not executor.initialized:
                    if state_tensor is None:
                        return {"heartbeat": False, "dispatched": False, "reason": "no_control_state"}
                    observation_stamp = (
                        float(state_timestamp)
                        if isinstance(state_timestamp, Real) and math.isfinite(float(state_timestamp))
                        else scheduled_timestamp
                    )
                    observation_stamp = min(observation_stamp, scheduled_timestamp)
                    executor.reset(
                        state_tensor.numpy(),
                        timestamp=scheduled_timestamp,
                        observation_timestamp=observation_stamp,
                    )
                    self._executor_last_observation_timestamp = observation_stamp
                    self._realtime_executor_needs_reset = False
                else:
                    next_heartbeat = executor.next_heartbeat_timestamp
                    if next_heartbeat is None:
                        raise RuntimeError("realtime executor has no next heartbeat")
                    tolerance = max(
                        _EXECUTOR_TIMESTAMP_TOLERANCE_S,
                        executor.config.heartbeat_dt_s * 1e-6,
                    )
                    if scheduled_timestamp > next_heartbeat + tolerance:
                        executor.rebase_next_heartbeat(scheduled_timestamp)
                    elif scheduled_timestamp + tolerance < next_heartbeat:
                        raise RuntimeError(
                            "executor control schedule moved backwards: "
                            f"scheduled={scheduled_timestamp:.9f}, executor={next_heartbeat:.9f}"
                        )

                completed_heartbeats = self._executor_applied_heartbeat_count
                barrier = self._executor_postfix_not_before_heartbeat
                barrier_ready = barrier is None or completed_heartbeats >= barrier
                if barrier is not None and barrier_ready:
                    self._executor_postfix_not_before_heartbeat = None
                allow_queue = self.ready and barrier_ready
                if not self.ready:
                    executor.clear_waypoints()
                action = None
                future = None
                if allow_queue:
                    action, future = self._dequeue_action()

                tick_timestamp = executor.next_heartbeat_timestamp
                if tick_timestamp is None:
                    raise RuntimeError("realtime executor lost its heartbeat timestamp")
                observation = None
                observation_stamp = None
                if state_tensor is not None and isinstance(state_timestamp, Real):
                    candidate_stamp = min(float(state_timestamp), tick_timestamp)
                    if math.isfinite(candidate_stamp) and (
                        self._executor_last_observation_timestamp is None
                        or candidate_stamp >= self._executor_last_observation_timestamp
                    ):
                        observation = state_tensor.numpy()
                        observation_stamp = candidate_stamp
                        self._executor_last_observation_timestamp = candidate_stamp

                lookahead_steps = 0
                if action is not None:
                    current = action.detach().to(device="cpu", dtype=torch.float64).reshape(1, -1)
                    if future is not None and len(future) > 0:
                        future = future.detach().to(device="cpu", dtype=torch.float64)
                        waypoints = torch.cat((current, future), dim=0)
                    else:
                        waypoints = current
                    waypoint_timestamps = (
                        tick_timestamp
                        + torch.arange(len(waypoints), dtype=torch.float64) * executor.config.heartbeat_dt_s
                    )
                    command = executor.control_step(
                        tick_timestamp,
                        waypoint_timestamps.numpy(),
                        waypoints.numpy(),
                        observation,
                        observation_timestamp=observation_stamp,
                    )
                    lookahead_steps = len(waypoints)
                else:
                    command = executor.heartbeat(
                        observation,
                        observation_timestamp=observation_stamp,
                    )

                command_tensor = torch.as_tensor(command, dtype=torch.float32)
                command_dict = {key: float(command[index]) for index, key in enumerate(self._action_keys)}
                processed = (
                    command_dict
                    if self._robot_action_processor is None
                    else self._robot_action_processor((command_dict, raw_observation))
                )
                if not isinstance(processed, dict):
                    raise TypeError("robot_action_processor must return an action dictionary")
                pre_filter = dict(processed)
                if self._action_filter is not None:
                    processed = self._action_filter.apply(processed, raw_observation)
                if not self._executor_dispatch_allowed():
                    return {
                        "heartbeat": True,
                        "dispatched": False,
                        "reason": "paused_or_failed_before_send",
                        "queue_empty": action is None,
                        "underrun": executor.last_tick_was_underrun,
                        "command": command,
                    }

                sent = self._robot.send_action(processed)
                self._executor_applied_heartbeat_count += 1
                self._latest_dispatched_action = command_tensor
                self.notify_action_result(processed, sent, raw_observation)
                if self._trace is not None:
                    self._trace.write(
                        "smooth_execution",
                        heartbeat_timestamp=executor.last_heartbeat_timestamp,
                        model_target=action,
                        reference=executor.last_reference,
                        command=command,
                        applied_command=sent,
                        lookahead_steps=lookahead_steps,
                        queue_remaining=queue.qsize(),
                        queue_empty=action is None,
                        underrun=executor.last_tick_was_underrun,
                        underrun_count=executor.underrun_count,
                        observed_state=observation,
                        observed_state_timestamp=observation_stamp,
                    )
                    self._trace.write(
                        "dispatch",
                        raw_model_action=command_dict,
                        pre_filter_target=pre_filter,
                        filtered_target=processed,
                        applied_command=sent,
                        observed_state=raw_observation,
                        dispatch_owner="rtc_executor_control",
                    )
                if self._stall_guard is not None:
                    try:
                        self._stall_guard.observe(processed, sent)
                    except StallContactError:
                        self._hold_after_stall(processed, raw_observation)
                        raise
                return {
                    "heartbeat": True,
                    "dispatched": True,
                    "queue_empty": action is None,
                    "underrun": executor.last_tick_was_underrun,
                    "command": command,
                    "applied_command": sent,
                    "queue_remaining": queue.qsize(),
                }

    def _executor_dispatch_allowed(self) -> bool:
        return self._policy_active.is_set() and not (
            self._shutdown_event.is_set()
            or self._rtc_error.is_set()
            or self._prefix_health_replan_event.is_set()
            or self._prefix_health_stop_event.is_set()
        )

    def _hold_after_stall(self, requested: dict[str, Any], observation: dict[str, Any]) -> None:
        hold = {
            key: float(observation[key])
            for key in requested
            if isinstance(observation.get(key), (int, float))
        }
        if not hold:
            logger.error("No present positions available for realtime-executor stall hold")
            return
        try:
            self._robot.send_action(hold)
            if self._trace is not None:
                self._trace.write("stall_hold", requested=requested, applied_command=hold)
        except Exception:
            logger.exception("Failed to command hold after realtime-executor stall")

    def _record_executor_control_outcome(
        self,
        outcome: dict[str, Any],
        *,
        scheduled: float,
        started: float,
        finished: float,
        lateness: float,
        execution_s: float,
        deadline_miss: bool,
        missed_periods: int,
    ) -> None:
        heartbeat = outcome.get("heartbeat") is True
        dispatched = outcome.get("dispatched") is True
        queue_empty = outcome.get("queue_empty") is True
        underrun = outcome.get("underrun") is True
        command = outcome.get("command")
        command_tuple = (
            None if command is None else tuple(float(value) for value in np.asarray(command).reshape(-1))
        )
        applied = outcome.get("applied_command")
        applied_mapping = dict(applied) if isinstance(applied, dict) else None
        with self._executor_metrics_lock:
            self._executor_heartbeat_count += int(heartbeat)
            self._executor_dispatch_count += int(dispatched)
            self._executor_hold_count += int(underrun)
            self._executor_queue_empty_count += int(queue_empty)
            self._executor_deadline_miss_count += int(deadline_miss)
            self._executor_missed_periods += missed_periods
            self._executor_last_scheduled_timestamp = scheduled
            self._executor_last_started_timestamp = started
            self._executor_last_finished_timestamp = finished
            self._executor_last_lateness_s = lateness
            self._executor_last_execution_s = execution_s
            self._executor_last_deadline_miss = deadline_miss
            self._executor_last_queue_empty = queue_empty
            self._executor_last_command = command_tuple
            self._executor_last_applied_command = applied_mapping
            heartbeat_count = self._executor_heartbeat_count
            deadline_miss_count = self._executor_deadline_miss_count
        if self._trace is not None:
            self._trace.write(
                "executor_control",
                scheduled_timestamp=scheduled,
                started_timestamp=started,
                finished_timestamp=finished,
                lateness_s=lateness,
                execution_s=execution_s,
                deadline_miss=deadline_miss,
                deadline_miss_count=deadline_miss_count,
                missed_periods=missed_periods,
                heartbeat_count=heartbeat_count,
                heartbeat=heartbeat,
                dispatched=dispatched,
                queue_empty=queue_empty,
                underrun=underrun,
                command=command,
                applied_command=applied,
                reason=outcome.get("reason"),
            )

    @contextmanager
    def action_dispatch_guard(self) -> Iterator[bool]:
        """Prevent an action dispatch from racing a fatal-state transition."""
        with self._action_dispatch_lock:
            yield not (
                self._rtc_error.is_set()
                or self._prefix_health_stop_event.is_set()
                or self._prefix_health_replan_event.is_set()
            )

    @staticmethod
    def _monotonic_seconds() -> float:
        monotonic_ns = getattr(time, "monotonic_ns", None)
        if callable(monotonic_ns):
            return monotonic_ns() * 1e-9
        return time.perf_counter()

    @staticmethod
    def _numeric_values(record: dict) -> dict[str, float]:
        return {
            key: float(value)
            for key, value in record.items()
            if isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(float(value))
        }

    def _observation_timing(self) -> dict[str, Any]:
        timing = getattr(self._robot, "last_observation_timing", None)
        if callable(timing):
            timing = timing()
        if not isinstance(timing, dict):
            now = self._monotonic_seconds()
            return {
                "clock": "monotonic",
                "state_timestamp": now,
                "camera_timestamps": {},
                "observation_timestamp": now,
            }
        return dict(timing)

    def _image_capture_timing(self, timing: dict[str, Any]) -> tuple[float, float]:
        """Return host capture stamp and the effective calibrated delay to the oldest image."""

        camera_timestamps = timing.get("camera_timestamps", {})
        finite_camera_timestamps: dict[str, float] = {}
        if isinstance(camera_timestamps, dict):
            finite_camera_timestamps = {
                str(key): float(value)
                for key, value in camera_timestamps.items()
                if isinstance(value, Real) and math.isfinite(float(value))
            }
        if finite_camera_timestamps:
            if self._camera_capture_delay_s:
                missing = set(self._camera_capture_delay_s) - set(finite_camera_timestamps)
                unexpected = set(finite_camera_timestamps) - set(self._camera_capture_delay_s)
                if missing or unexpected:
                    raise _RTCFatalError(
                        "RTC camera timing layout differs from calibrated artifact: "
                        f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
                    )
                physical_timestamps = {
                    key: timestamp - self._camera_capture_delay_s[key]
                    for key, timestamp in finite_camera_timestamps.items()
                }
                capture_timestamp = min(finite_camera_timestamps.values())
                physical_anchor = min(physical_timestamps.values())
                effective_delay_s = capture_timestamp - physical_anchor
                skew_values = list(physical_timestamps.values())
            else:
                capture_timestamp = min(finite_camera_timestamps.values())
                effective_delay_s = self._image_capture_delay_s
                skew_values = list(finite_camera_timestamps.values())
            skew = max(skew_values) - min(skew_values)
            if self._max_camera_skew_s > 0 and skew > self._max_camera_skew_s:
                raise _RTCFatalError(
                    "RTC camera timestamp skew exceeds configured limit: "
                    f"skew_ms={skew * 1000.0:.3f}, limit_ms={self._max_camera_skew_s * 1000.0:.3f}"
                )
            # Align the shared proprioceptive input with the oldest image so no
            # camera is conditioned on a state from its future.
            return capture_timestamp, effective_delay_s
        if self._camera_capture_delay_s:
            raise _RTCFatalError("RTC calibrated sensor timing requires finite timestamps for every camera")
        fallback = timing.get("observation_timestamp")
        if isinstance(fallback, Real) and math.isfinite(float(fallback)):
            return float(fallback), self._image_capture_delay_s
        return self._monotonic_seconds(), self._image_capture_delay_s

    def _tensor_actions_as_mappings(self, actions: torch.Tensor | None) -> list[dict[str, float]]:
        if actions is None:
            return []
        rows = actions.detach().to(device="cpu", dtype=torch.float32)
        if rows.ndim != 2:
            raise ValueError(f"Expected action queue tensor [T,A], got {tuple(rows.shape)}")
        if rows.shape[1] < len(self._action_keys):
            raise ValueError(
                "Action queue has fewer dimensions than robot action keys: "
                f"shape={tuple(rows.shape)}, keys={len(self._action_keys)}"
            )
        return [{key: float(row[index]) for index, key in enumerate(self._action_keys)} for row in rows]

    def _executor_preview_as_timed_records(
        self,
        *,
        request_timestamp: float,
        predicted_completion_timestamp: float,
    ) -> list[TimedRecord]:
        executor = self._realtime_executor
        if executor is None:
            raise RuntimeError("executor preview requested without a realtime executor")
        with self._executor_state_lock:
            if not executor.initialized:
                raise _RTCFatalError(
                    "RTC dynamic prefill cannot preview commands before the realtime executor is initialized"
                )
            preview = executor.preview_through(predicted_completion_timestamp)
        tolerance = max(
            _EXECUTOR_TIMESTAMP_TOLERANCE_S,
            executor.config.heartbeat_dt_s * 1e-6,
        )
        return [
            TimedRecord(
                item.timestamp,
                {key: float(item.command[index]) for index, key in enumerate(self._action_keys)},
                "monotonic",
            )
            for item in preview
            if item.timestamp + tolerance >= request_timestamp
        ]

    def _prefill_actions_to_policy_space(
        self,
        prefill: CommittedActionPrefill,
        *,
        policy_device: torch.device,
    ) -> torch.Tensor:
        absolute = torch.tensor(
            [[float(action[key]) for key in self._action_keys] for action in prefill.actions],
            dtype=torch.float32,
        )
        if self._relative_step is not None:
            raw_state = self._relative_step.get_cached_state()
            if raw_state is None:
                raise RuntimeError("Dynamic RTC prefill requires a cached aligned observation state")
            return reanchor_relative_rtc_prefix(
                prev_actions_absolute=absolute,
                current_state=raw_state,
                relative_step=self._relative_step,
                normalizer_step=self._normalizer_step,
                policy_device=policy_device,
            )
        transition = create_transition(action=absolute)
        if self._normalizer_step is not None:
            transition = self._normalizer_step(transition)
        return transition[TransitionKey.ACTION].to(policy_device)

    def notify_control_observation(self, obs: dict) -> None:
        """Publish the latest raw state for independent command dispatch."""

        if self._realtime_executor is None:
            return
        timing = self._observation_timing()
        state_timestamp = timing.get("state_timestamp")
        if not isinstance(state_timestamp, Real) or not math.isfinite(float(state_timestamp)):
            state_timestamp = self._monotonic_seconds()
        physical_state_timestamp = float(state_timestamp) - self._state_observation_delay_s
        state_values = self._numeric_values(obs)
        control_state = None
        if all(key in state_values for key in self._action_keys):
            control_state = torch.tensor(
                [state_values[key] for key in self._action_keys],
                dtype=torch.float64,
            )
        with self._obs_lock:
            self._obs_holder["control_observation"] = dict(obs)
            self._obs_holder["control_state"] = control_state
            self._obs_holder["control_state_timestamp"] = physical_state_timestamp

    def notify_observation(self, obs: dict) -> None:
        """Publish the latest observation for the RTC thread to consume."""
        if self._realtime_executor is not None:
            with self._obs_lock:
                has_control_observation = self._obs_holder.get("control_observation") is not None
            if not has_control_observation:
                self.notify_control_observation(obs)
        needs_timing = (
            self._trajectory is not None
            or self._dynamic_prefill_enabled
            or self._trace is not None
            or self._realtime_executor is not None
        )
        timing = self._observation_timing() if needs_timing else {}
        physical_state_timestamp = None
        if needs_timing:
            state_timestamp = timing.get("state_timestamp")
            if not isinstance(state_timestamp, Real) or not math.isfinite(float(state_timestamp)):
                state_timestamp = self._monotonic_seconds()
            physical_state_timestamp = float(state_timestamp) - self._state_observation_delay_s
        state_values = self._numeric_values(obs)
        executor_state = None
        if self._realtime_executor is not None and all(key in state_values for key in self._action_keys):
            executor_state = torch.tensor(
                [state_values[key] for key in self._action_keys],
                dtype=torch.float64,
            )
        if self._trajectory is not None:
            if physical_state_timestamp is None:
                raise RuntimeError("trajectory observation timing was not initialized")
            self._trajectory.append_state(state_values, timestamp=physical_state_timestamp)
            if self._trace is not None:
                self._trace.write(
                    "observation",
                    observed_state=state_values,
                    observation_timing=timing,
                    physical_state_timestamp=physical_state_timestamp,
                )
        with self._obs_lock:
            self._observation_sequence += 1
            self._obs_holder["obs"] = obs
            self._obs_holder["observation_epoch"] = self._observation_epoch
            self._obs_holder["observation_sequence"] = self._observation_sequence
            self._obs_holder["observation_timing"] = timing
            self._obs_holder["executor_state"] = executor_state
            self._obs_holder["executor_state_timestamp"] = physical_state_timestamp

    def notify_action_result(self, requested: dict, sent: dict | None, observation: dict) -> None:
        """Track robot-space safety residuals and request an unguided replan when persistent."""
        if self._trajectory is not None:
            self._trajectory.append_action(
                self._numeric_values(sent if sent is not None else requested),
                timestamp=self._monotonic_seconds(),
            )
        if self._trace is not None:
            self._trace.write(
                "dispatch_result",
                requested=requested,
                sent=sent,
                observed=observation,
                aligned_future_actions=(
                    None
                    if self._trajectory is None
                    else self._trajectory.future_action_trajectory(
                        self._monotonic_seconds(),
                        1.0 / max(float(self._fps), 1.0),
                        self._rtc_config.execution_horizon,
                    )
                ),
            )
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
        if self._trace is not None:
            mark_abnormal = getattr(self._trace, "mark_abnormal", None)
            if callable(mark_abnormal):
                mark_abnormal(error)
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
                    observation_timing = self._obs_holder.get("observation_timing")
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
                        executor_heartbeat_start: int | None = None
                        if self._realtime_executor is not None:
                            with self._executor_state_lock:
                                executor_heartbeat_start = self._executor_applied_heartbeat_count
                        diagnostic_before = (
                            inference_start
                            if self._timing_diagnostics and inference_start is not None
                            else queue.snapshot()
                            if self._timing_diagnostics
                            else None
                        )
                        current_time = self._monotonic_seconds()
                        inference_started_monotonic = current_time
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

                        if (
                            self._rtc_inference_mode == "trained_prefix"
                            and delay > self._rtc_training_max_delay
                        ):
                            raise _RTCFatalError(
                                "RTC delay exceeds checkpoint trained-prefix capacity: "
                                f"delay={delay}, capacity={self._rtc_training_max_delay}"
                            )

                        if suppress_prefix_guidance:
                            prev_actions = None
                            logger.info(
                                "RTC prefix health suppressing guidance for revision=%d",
                                feedback_revision,
                            )

                        dynamic_prefill: CommittedActionPrefill | None = None
                        dynamic_anchor_steps: int | None = None
                        obs_for_inference = obs
                        if self._dynamic_prefill_enabled:
                            if self._trajectory is None:
                                raise _RTCFatalError("RTC dynamic prefill requires a trajectory timeline")
                            timing = observation_timing if isinstance(observation_timing, dict) else {}
                            capture_timestamp, effective_image_delay_s = self._image_capture_timing(timing)
                            aligned_timestamp = capture_timestamp - effective_image_delay_s
                            aligned_state = self._trajectory.estimate_state(aligned_timestamp)
                            if aligned_state is not None:
                                obs_for_inference = dict(obs)
                                for key, value in aligned_state.items():
                                    if key in obs_for_inference and isinstance(obs_for_inference[key], Real):
                                        obs_for_inference[key] = value

                            executor_prefill_available = (
                                self._realtime_executor is not None and self._realtime_executor.initialized
                            )
                            queued_prefill_available = (
                                inference_start is not None
                                and inference_start.processed_leftover is not None
                                and len(inference_start.processed_leftover) > 0
                            )
                            if (
                                not suppress_prefix_guidance
                                and inference_start is not None
                                and (executor_prefill_available or queued_prefill_available)
                            ):
                                predicted_completion = current_time + delay * time_per_action
                                overflow = "raise" if self._prefill_overflow == "error" else "truncate"
                                future_action_queue = (
                                    self._executor_preview_as_timed_records(
                                        request_timestamp=current_time,
                                        predicted_completion_timestamp=predicted_completion,
                                    )
                                    if self._realtime_executor is not None
                                    else self._tensor_actions_as_mappings(inference_start.processed_leftover)
                                )
                                try:
                                    dynamic_prefill = self._trajectory.build_committed_prefill(
                                        image_capture_timestamp=capture_timestamp,
                                        calibrated_image_delay_s=effective_image_delay_s,
                                        request_timestamp=current_time,
                                        predicted_completion_timestamp=predicted_completion,
                                        dt=time_per_action,
                                        future_action_queue=future_action_queue,
                                        max_prefill_steps=self._max_prefill_steps,
                                        overflow=overflow,
                                    )
                                except PrefillCapacityError as exc:
                                    raise _RTCFatalError(str(exc)) from exc
                                if dynamic_prefill.truncated:
                                    raise _RTCFatalError(
                                        "RTC dynamic prefill truncation is diagnostic-only and cannot be dispatched"
                                    )
                                if len(dynamic_prefill) == 0:
                                    dynamic_prefill = None
                                else:
                                    dynamic_anchor_steps = delay - 1

                        obs_batch = build_dataset_frame(
                            self._hw_features, obs_for_inference, prefix="observation"
                        )
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

                            if dynamic_prefill is not None:
                                prev_actions = self._prefill_actions_to_policy_space(
                                    dynamic_prefill,
                                    policy_device=policy_device,
                                )
                            elif prev_actions is not None and self._relative_step is not None:
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

                            model_prefill_len = len(dynamic_prefill) if dynamic_prefill is not None else delay
                            action_kwargs = {
                                "inference_delay": model_prefill_len,
                                "prev_chunk_left_over": prev_actions,
                            }
                            if self._rtc_inference_mode == "trained_prefix":
                                action_kwargs["rtc_mode"] = self._rtc_inference_mode
                            actions = self._policy.predict_action_chunk(preprocessed, **action_kwargs)

                            if self._shutdown_event.is_set():
                                logger.info("RTC discarded inference result requested during shutdown")
                                return

                            raw_model_chunk = actions.squeeze(0).clone()
                            raw_original = (
                                raw_model_chunk[model_prefill_len:].clone()
                                if dynamic_prefill is not None
                                else raw_model_chunk
                            )
                            if raw_original.numel() == 0:
                                raise _RTCFatalError(
                                    "RTC dynamic prefill consumed the entire model action chunk"
                                )
                            original = raw_original
                            raw_processed = self._postprocessor(raw_original.unsqueeze(0)).squeeze(0)
                            processed = raw_processed
                            plan_result = None
                            if self._time_axis_planner is not None:
                                committed_prefix_steps = (
                                    0
                                    if dynamic_prefill is not None
                                    else model_prefill_len
                                    if self._rtc_inference_mode == "trained_prefix"
                                    and prev_actions is not None
                                    else 0
                                )
                                planner_coordinate_space = getattr(
                                    self._time_axis_planner,
                                    "feature_coordinate_space",
                                    None,
                                )
                                # First and recovery inferences may have no prefix; the
                                # planner contract, not prefix presence, selects coordinates.
                                plan_in_robot_space = dynamic_prefill is not None
                                if planner_coordinate_space is not None:
                                    plan_in_robot_space = (
                                        planner_coordinate_space == ROBOT_ACTION_COORDINATE_SPACE
                                    )
                                if plan_in_robot_space:
                                    plan_result = self._time_axis_planner.plan(
                                        raw_processed,
                                        committed_prefix_steps=committed_prefix_steps,
                                    )
                                    processed = torch.as_tensor(
                                        plan_result.actions,
                                        dtype=raw_processed.dtype,
                                        device=raw_processed.device,
                                    )
                                else:
                                    # Legacy/guided RTC keeps the historical
                                    # policy-space planning contract so its
                                    # original and processed queues stay paired.
                                    plan_result = self._time_axis_planner.plan(
                                        raw_original,
                                        committed_prefix_steps=committed_prefix_steps,
                                    )
                                    original = torch.as_tensor(
                                        plan_result.actions,
                                        dtype=raw_original.dtype,
                                        device=raw_original.device,
                                    )
                                    processed = self._postprocessor(original.unsqueeze(0)).squeeze(0)
                            if self._trace is not None:
                                self._trace.write(
                                    "inference_chunk",
                                    inference_started_at=inference_started_monotonic,
                                    inference_finished_at=self._monotonic_seconds(),
                                    raw_model_chunk=raw_model_chunk,
                                    trained_prefix=prev_actions,
                                    raw_robot_chunk=raw_processed,
                                    planned_chunk=processed,
                                    planner_segment_durations=(
                                        None if plan_result is None else plan_result.segment_durations
                                    ),
                                    planner_reference_segment_durations=(
                                        None
                                        if plan_result is None
                                        else plan_result.reference_segment_durations
                                    ),
                                    planner_speed_factors=(
                                        None if plan_result is None else plan_result.speed_factors
                                    ),
                                    planner_feature_coordinate_space=(
                                        None
                                        if self._time_axis_planner is None
                                        else getattr(
                                            self._time_axis_planner,
                                            "feature_coordinate_space",
                                            None,
                                        )
                                    ),
                                    planner_fallback=(
                                        None if plan_result is None else plan_result.used_fallback
                                    ),
                                    planner_reason=(None if plan_result is None else plan_result.reason),
                                    inference_delay=delay,
                                    model_prefill_len=model_prefill_len,
                                    dynamic_prefill=dynamic_prefill,
                                    dynamic_anchor_steps=dynamic_anchor_steps,
                                )
                        new_latency = self._monotonic_seconds() - current_time
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
                        anchor_merge_result = None
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
                            if dynamic_prefill is not None:
                                if dynamic_anchor_steps is None:
                                    raise _RTCFatalError("RTC dynamic prefill has no queue anchor")
                                if self._realtime_executor is not None:
                                    if executor_heartbeat_start is None:
                                        raise _RTCFatalError(
                                            "RTC executor heartbeat baseline is missing for dynamic merge"
                                        )
                                    with self._executor_state_lock:
                                        executor_heartbeats_now = self._executor_applied_heartbeat_count
                                        external_consumed = max(
                                            0,
                                            executor_heartbeats_now - executor_heartbeat_start,
                                        )
                                        committed_steps = dynamic_anchor_steps + 1
                                        anchor_merge_result = queue.merge_postfix_after_external_anchor(
                                            original,
                                            processed,
                                            inference_start,
                                            external_consumed_steps=external_consumed,
                                            committed_steps=committed_steps,
                                            max_actual_consumed_steps=self._rtc_config.execution_horizon,
                                        )
                                        if anchor_merge_result.merged:
                                            self._executor_postfix_not_before_heartbeat = (
                                                executor_heartbeat_start + committed_steps
                                            )
                                else:
                                    anchor_merge_result = queue.merge_postfix_at_anchor(
                                        original,
                                        processed,
                                        inference_start,
                                        anchor_steps_after_start=dynamic_anchor_steps,
                                        max_actual_consumed_steps=self._rtc_config.execution_horizon,
                                    )
                                active_merge_result = anchor_merge_result
                            else:
                                merge_result = queue.merge_actual_consumed(
                                    original,
                                    processed,
                                    inference_start,
                                    max_actual_consumed_steps=self._rtc_config.execution_horizon,
                                )
                                active_merge_result = merge_result
                            if active_merge_result.stale:
                                logger.info(
                                    "RTC discarded stale inference result: generation=%d->%d",
                                    active_merge_result.expected_generation,
                                    active_merge_result.observed_generation,
                                )
                                consecutive_errors = 0
                                continue

                            if active_merge_result.consumption_limit_exceeded:
                                raise _RTCFatalError(
                                    "RTC consumed actions reached execution_horizon during one inference: "
                                    f"actual_consumed_steps={active_merge_result.actual_consumed_steps}, "
                                    f"execution_horizon={self._rtc_config.execution_horizon}"
                                )
                            if (
                                anchor_merge_result is not None
                                and anchor_merge_result.insufficient_committed_actions
                            ):
                                raise _RTCFatalError(
                                    "RTC dynamic anchor could not be merged: committed queue/postfix is no longer available"
                                )
                            if not active_merge_result.merged:
                                raise _RTCFatalError("RTC inference result was not merged")

                            if suppress_prefix_guidance:
                                self._acknowledge_prefix_health_revision(feedback_revision)

                            if merge_result is not None and merge_result.skip_was_clamped:
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
                            if anchor_merge_result is not None:
                                actual_consumed = anchor_merge_result.actual_consumed_steps
                                merge_skip = anchor_merge_result.postfix_skip
                                merge_skip_clamped = 0
                                generation_before = anchor_merge_result.expected_generation
                                generation_after_inference = anchor_merge_result.observed_generation
                                generation_after_merge = anchor_merge_result.generation_after
                                index_before = inference_start.next_action_index
                                index_after_inference = index_before + actual_consumed
                                index_after_merge = diagnostic_after_merge.next_action_index
                                queue_before = inference_start.queue_size
                                queue_after_inference = max(0, queue_before - actual_consumed)
                                queue_after_merge = anchor_merge_result.queue_size_after
                                current_total_consumed = inference_start.total_consumed + actual_consumed
                                if previous_replan_total_consumed is None:
                                    replan_interval_steps = 0
                                else:
                                    replan_interval_steps = max(
                                        0,
                                        current_total_consumed - previous_replan_total_consumed,
                                    )
                                previous_replan_total_consumed = current_total_consumed
                            elif merge_result is not None:
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
                            if self._trace is not None:
                                self._trace.write(
                                    "queue_merge",
                                    actual_consumed_steps=actual_consumed,
                                    merge_skip=merge_skip,
                                    generation_before=generation_before,
                                    generation_after_inference=generation_after_inference,
                                    generation_after_merge=generation_after_merge,
                                    index_before=index_before,
                                    index_after_inference=index_after_inference,
                                    index_after_merge=index_after_merge,
                                    queue_before=queue_before,
                                    queue_after_inference=queue_after_inference,
                                    queue_after_merge=queue_after_merge,
                                    latency_ms=new_latency * 1000.0,
                                    anchor_preserved_steps=(
                                        None
                                        if anchor_merge_result is None
                                        else anchor_merge_result.preserved_steps
                                    ),
                                    anchor_postfix_skip=(
                                        None
                                        if anchor_merge_result is None
                                        else anchor_merge_result.postfix_skip
                                    ),
                                )

                        consecutive_errors = 0
                        logger.debug("RTC inference latency=%.2fs, queue=%d", new_latency, queue.qsize())

                    except _RTCFatalError:
                        raise
                    except RealtimeTraceWriteError as exc:
                        raise _RTCFatalError("RTC deployment trace write failed") from exc
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
