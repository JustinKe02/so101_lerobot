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

"""Inference engine configs and factory.

Selection is explicit via ``--inference.type=sync|rtc``.  Adding a new
backend requires registering its config subclass and dispatching it in
:func:`create_inference_engine`.
"""

from __future__ import annotations

import abc
import logging
import math
from dataclasses import dataclass, field
from enum import StrEnum
from threading import Event

import draccus

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.processor import PolicyProcessorPipeline

from ..robot_wrapper import ThreadSafeRobot
from .base import InferenceEngine
from .rtc import RTCInferenceEngine
from .sync import SyncInferenceEngine

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------------


class RTCTimingMode(StrEnum):
    LEGACY = "legacy"
    ACTUAL_CONSUMED = "actual_consumed"


class RTCGuidanceDelayMode(StrEnum):
    LEGACY_MAX = "legacy_max"
    FIXED = "fixed"
    ROLLING_P95 = "rolling_p95"


@dataclass
class InferenceEngineConfig(draccus.ChoiceRegistry, abc.ABC):
    """Abstract base for inference backend configuration.

    Use ``--inference.type=<name>`` on the CLI to select a backend.
    """

    @property
    def type(self) -> str:
        return self.get_choice_name(self.__class__)


@InferenceEngineConfig.register_subclass("sync")
@dataclass
class SyncInferenceConfig(InferenceEngineConfig):
    """Inline synchronous inference (one policy call per control tick).

    Guarded execution is opt-in so existing sync rollouts retain their full
    policy action queue. ``max_actions_per_chunk`` bounds the open-loop
    prefix, while ``replan_on_clamp`` invalidates the remaining queue after
    robot-side safety handling materially changes a requested target.
    """

    max_actions_per_chunk: int | None = None
    replan_on_clamp: bool = False
    clamp_replan_threshold: float = 5.0

    def __post_init__(self) -> None:
        if self.max_actions_per_chunk is not None and (
            isinstance(self.max_actions_per_chunk, bool)
            or not isinstance(self.max_actions_per_chunk, int)
            or self.max_actions_per_chunk < 1
        ):
            raise ValueError("Sync max_actions_per_chunk must be a positive integer when set")
        if not math.isfinite(self.clamp_replan_threshold) or self.clamp_replan_threshold < 0:
            raise ValueError("Sync clamp_replan_threshold must be finite and non-negative")


@InferenceEngineConfig.register_subclass("rtc")
@dataclass
class RTCInferenceConfig(InferenceEngineConfig):
    """Real-Time Chunking: async policy inference in a background thread."""

    # Eagerly constructed so draccus exposes nested fields directly on the CLI
    # (e.g. ``--inference.rtc.execution_horizon=...``).
    rtc: RTCConfig = field(default_factory=RTCConfig)
    queue_threshold: int = 30
    timing_mode: RTCTimingMode = RTCTimingMode.LEGACY
    guidance_delay_mode: RTCGuidanceDelayMode = RTCGuidanceDelayMode.LEGACY_MAX
    fixed_guidance_delay_steps: int = 5
    latency_warmup_inferences: int = 5
    latency_window_size: int = 32
    latency_percentile: float = 0.95
    delay_hysteresis_steps: float = 0.25
    delay_change_confirmations: int = 3
    timing_diagnostics: bool = False
    enforce_guided_execution_window: bool = False
    prefix_health_enabled: bool = False
    prefix_health_severe_residual_threshold: float = 5.0
    prefix_health_consecutive_severe: int = 3
    prefix_health_safety_stop_replans: int = 0

    def __post_init__(self) -> None:
        self.timing_mode = RTCTimingMode(self.timing_mode)
        self.guidance_delay_mode = RTCGuidanceDelayMode(self.guidance_delay_mode)
        if self.queue_threshold < 0:
            raise ValueError("RTC queue_threshold must be non-negative")
        if self.fixed_guidance_delay_steps < 0:
            raise ValueError("RTC fixed_guidance_delay_steps must be non-negative")
        if self.latency_warmup_inferences < 0:
            raise ValueError("RTC latency_warmup_inferences must be non-negative")
        if self.latency_window_size < 1:
            raise ValueError("RTC latency_window_size must be positive")
        if not 0.0 < self.latency_percentile <= 1.0:
            raise ValueError("RTC latency_percentile must be in (0, 1]")
        if self.delay_hysteresis_steps < 0:
            raise ValueError("RTC delay_hysteresis_steps must be non-negative")
        if self.delay_change_confirmations < 1:
            raise ValueError("RTC delay_change_confirmations must be positive")
        if (
            not math.isfinite(self.prefix_health_severe_residual_threshold)
            or self.prefix_health_severe_residual_threshold <= 0
        ):
            raise ValueError("RTC prefix_health_severe_residual_threshold must be finite and positive")
        if self.prefix_health_consecutive_severe < 1:
            raise ValueError("RTC prefix_health_consecutive_severe must be positive")
        if self.prefix_health_safety_stop_replans < 0:
            raise ValueError("RTC prefix_health_safety_stop_replans must be non-negative")

        if self.timing_mode == RTCTimingMode.LEGACY:
            if self.enforce_guided_execution_window:
                raise ValueError(
                    "RTC guided execution window enforcement requires timing_mode='actual_consumed'"
                )
            if self.prefix_health_enabled:
                raise ValueError("RTC prefix health requires timing_mode='actual_consumed'")
            if self.guidance_delay_mode != RTCGuidanceDelayMode.LEGACY_MAX:
                raise ValueError("legacy timing requires guidance_delay_mode='legacy_max'")
            return

        if self.guidance_delay_mode == RTCGuidanceDelayMode.LEGACY_MAX:
            raise ValueError("actual_consumed timing requires guidance_delay_mode='fixed' or 'rolling_p95'")
        if self.rtc.execution_horizon <= 0:
            raise ValueError("RTC execution_horizon must be positive")
        if self.fixed_guidance_delay_steps >= self.rtc.execution_horizon:
            raise ValueError(
                "RTC fixed_guidance_delay_steps must be smaller than execution_horizon "
                "to preserve a non-empty guidance transition"
            )
        if self.guidance_delay_mode == RTCGuidanceDelayMode.ROLLING_P95 and self.latency_window_size < 5:
            raise ValueError("RTC rolling_p95 latency_window_size must be at least 5")

    def validate_policy_chunk_size(self, chunk_size: object) -> None:
        """Validate the queue watermark once a concrete policy config is available."""
        if self.timing_mode == RTCTimingMode.LEGACY:
            return
        if isinstance(chunk_size, bool) or not isinstance(chunk_size, int):
            return
        if self.queue_threshold >= chunk_size:
            raise ValueError(
                "RTC queue_threshold must be smaller than policy chunk_size: "
                f"queue_threshold={self.queue_threshold}, chunk_size={chunk_size}"
            )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_inference_engine(
    config: InferenceEngineConfig,
    *,
    policy: PreTrainedPolicy,
    preprocessor: PolicyProcessorPipeline,
    postprocessor: PolicyProcessorPipeline,
    robot_wrapper: ThreadSafeRobot,
    hw_features: dict,
    dataset_features: dict,
    ordered_action_keys: list[str],
    task: str,
    fps: float,
    device: str | None,
    use_torch_compile: bool = False,
    compile_warmup_inferences: int = 2,
    shutdown_event: Event | None = None,
) -> InferenceEngine:
    """Instantiate the appropriate inference engine from a config object."""
    logger.info("Creating inference engine: %s", config.type)
    if isinstance(config, SyncInferenceConfig):
        return SyncInferenceEngine(
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            dataset_features=dataset_features,
            ordered_action_keys=ordered_action_keys,
            task=task,
            device=device,
            robot_type=robot_wrapper.robot_type,
            max_actions_per_chunk=config.max_actions_per_chunk,
            replan_on_clamp=config.replan_on_clamp,
            clamp_replan_threshold=config.clamp_replan_threshold,
        )
    if isinstance(config, RTCInferenceConfig):
        config.validate_policy_chunk_size(getattr(policy.config, "chunk_size", None))
        return RTCInferenceEngine(
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            robot_wrapper=robot_wrapper,
            rtc_config=config.rtc,
            hw_features=hw_features,
            task=task,
            fps=fps,
            device=device,
            use_torch_compile=use_torch_compile,
            compile_warmup_inferences=compile_warmup_inferences,
            rtc_queue_threshold=config.queue_threshold,
            rtc_timing_mode=config.timing_mode.value,
            guidance_delay_mode=config.guidance_delay_mode.value,
            fixed_guidance_delay_steps=config.fixed_guidance_delay_steps,
            latency_warmup_inferences=config.latency_warmup_inferences,
            latency_window_size=config.latency_window_size,
            latency_percentile=config.latency_percentile,
            delay_hysteresis_steps=config.delay_hysteresis_steps,
            delay_change_confirmations=config.delay_change_confirmations,
            timing_diagnostics=config.timing_diagnostics,
            enforce_guided_execution_window=config.enforce_guided_execution_window,
            prefix_health_enabled=config.prefix_health_enabled,
            prefix_health_severe_residual_threshold=config.prefix_health_severe_residual_threshold,
            prefix_health_consecutive_severe=config.prefix_health_consecutive_severe,
            prefix_health_safety_stop_replans=config.prefix_health_safety_stop_replans,
            shutdown_event=shutdown_event,
        )
    raise ValueError(f"Unknown inference engine type: {type(config).__name__}")
