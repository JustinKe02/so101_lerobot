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

"""Configuration dataclasses for the rollout deployment engine."""

from __future__ import annotations

import abc
import logging
import math
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

import draccus

from lerobot.configs import PreTrainedConfig, parser
from lerobot.configs.dataset import DatasetRecordConfig
from lerobot.robots.config import RobotConfig
from lerobot.teleoperators.config import TeleoperatorConfig
from lerobot.utils.device_utils import auto_select_torch_device, is_torch_device_available

from .inference import InferenceEngineConfig, RTCInferenceConfig, SyncInferenceConfig
from .pi05_parity import PI05ParityReportArtifactConfig
from .speed_adapter import SpeedAdapterRuntimeConfig
from .time_axis import TimeAxisPlannerConfig

logger = logging.getLogger(__name__)


class PI05PrefixBackend(StrEnum):
    AUTO = "auto"
    PYTORCH = "pytorch"
    TENSORRT = "tensorrt"


class PI05ActionBackend(StrEnum):
    PYTORCH = "pytorch"
    TENSORRT = "tensorrt"
    TRITON = "triton"


@dataclass
class ActionOutputFilterConfig:
    """Second-order output limiter applied to policy actions before dispatch.

    Disabled by default so existing rollouts keep byte-identical behavior.
    Caps are per-joint, in dataset action units per second (and per second
    squared); keys may be bare joint names (``shoulder_pan``) or full action
    keys (``shoulder_pan.pos``).  See ``rollout/action_filter.py`` for the
    limiter equations and defaults provenance.
    """

    enabled: bool = False
    max_velocity: dict[str, float] = field(default_factory=dict)
    max_acceleration: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for label, limits in (
            ("max_velocity", self.max_velocity),
            ("max_acceleration", self.max_acceleration),
        ):
            for key, value in limits.items():
                if not isinstance(value, (int, float)) or isinstance(value, bool):
                    raise ValueError(f"action_filter.{label}['{key}'] must be a number, got {value!r}")
                if not float(value) > 0 or value != value or value in (float("inf"), float("-inf")):
                    raise ValueError(f"action_filter.{label}['{key}'] must be finite and positive")

    def resolved_max_velocity(self) -> dict[str, float]:
        from .action_filter import DEFAULT_MAX_VELOCITY

        return {**DEFAULT_MAX_VELOCITY, **{k: float(v) for k, v in self.max_velocity.items()}}

    def resolved_max_acceleration(self) -> dict[str, float]:
        from .action_filter import DEFAULT_MAX_ACCELERATION

        return {**DEFAULT_MAX_ACCELERATION, **{k: float(v) for k, v in self.max_acceleration.items()}}


@dataclass
class ActuatorCalibrationArtifactConfig:
    """Pinned offline actuator artifact used by the realtime executor."""

    enabled: bool = False
    path: str | None = None
    sha256: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("realtime_executor.actuator_calibration.enabled must be boolean")
        if self.path is not None and (not isinstance(self.path, str) or not self.path.strip()):
            raise ValueError("realtime_executor.actuator_calibration.path must be a non-empty string")
        if self.sha256 is not None and (
            not isinstance(self.sha256, str)
            or len(self.sha256) != 64
            or any(character not in "0123456789abcdefABCDEF" for character in self.sha256)
        ):
            raise ValueError(
                "realtime_executor.actuator_calibration.sha256 must be 64 hexadecimal characters"
            )
        if self.enabled and self.path is None:
            raise ValueError(
                "realtime_executor.actuator_calibration.path is required when calibration is enabled"
            )
        if self.enabled and self.sha256 is None:
            raise ValueError(
                "realtime_executor.actuator_calibration.sha256 is required when calibration is enabled"
            )


def _validate_executor_parameter_vector(
    value: float | list[float],
    *,
    name: str,
    positive: bool,
) -> None:
    values = value if isinstance(value, list) else [value]
    if not values:
        raise ValueError(f"realtime_executor.{name} must not be an empty list")
    for item in values:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ValueError(f"realtime_executor.{name} values must be numbers")
        number = float(item)
        valid = math.isfinite(number) and (number > 0.0 if positive else number >= 0.0)
        if not valid:
            requirement = "positive" if positive else "non-negative"
            raise ValueError(f"realtime_executor.{name} values must be finite and {requirement}")


@dataclass
class SmoothExecutorConfig:
    """Fixed-heartbeat local trajectory tracking for Realtime-VLA V2."""

    enabled: bool = False
    actuator_calibration: ActuatorCalibrationArtifactConfig = field(
        default_factory=ActuatorCalibrationArtifactConfig
    )
    actuator_tau_s: float | list[float] = 0.15
    command_delay_s: float | list[float] = 0.0
    enable_forward_tracking: bool = True
    forward_lead_s: float | list[float] | None = None
    forward_feedback_gain: float = 0.0
    savgol_window_length: int = 1
    savgol_polyorder: int = 2
    max_waypoints: int = 4096
    max_command_history: int = 512
    max_velocity: dict[str, float] = field(default_factory=dict)
    max_acceleration: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.actuator_calibration, ActuatorCalibrationArtifactConfig):
            raise ValueError(
                "realtime_executor.actuator_calibration must be an ActuatorCalibrationArtifactConfig"
            )
        if self.actuator_calibration.enabled and not self.enabled:
            raise ValueError("realtime_executor.enabled must be true when actuator calibration is enabled")
        _validate_executor_parameter_vector(
            self.actuator_tau_s,
            name="actuator_tau_s",
            positive=True,
        )
        _validate_executor_parameter_vector(
            self.command_delay_s,
            name="command_delay_s",
            positive=False,
        )
        if self.forward_lead_s is not None:
            _validate_executor_parameter_vector(
                self.forward_lead_s,
                name="forward_lead_s",
                positive=False,
            )
        if not math.isfinite(float(self.forward_feedback_gain)) or float(self.forward_feedback_gain) < 0:
            raise ValueError("realtime_executor.forward_feedback_gain must be finite and non-negative")
        if (
            isinstance(self.savgol_window_length, bool)
            or not isinstance(self.savgol_window_length, int)
            or self.savgol_window_length < 1
            or self.savgol_window_length % 2 == 0
        ):
            raise ValueError("realtime_executor.savgol_window_length must be odd and positive")
        if (
            isinstance(self.savgol_polyorder, bool)
            or not isinstance(self.savgol_polyorder, int)
            or self.savgol_polyorder < 0
            or (self.savgol_window_length > 1 and self.savgol_polyorder >= self.savgol_window_length)
        ):
            raise ValueError(
                "realtime_executor.savgol_polyorder must be non-negative and smaller than the window"
            )
        for name in ("max_waypoints", "max_command_history"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 2:
                raise ValueError(f"realtime_executor.{name} must be an integer of at least two")
        ActionOutputFilterConfig(
            max_velocity=self.max_velocity,
            max_acceleration=self.max_acceleration,
        )

    def resolved_max_velocity(self) -> dict[str, float]:
        from .action_filter import DEFAULT_MAX_VELOCITY

        return {**DEFAULT_MAX_VELOCITY, **{key: float(value) for key, value in self.max_velocity.items()}}

    def resolved_max_acceleration(self) -> dict[str, float]:
        from .action_filter import DEFAULT_MAX_ACCELERATION

        return {
            **DEFAULT_MAX_ACCELERATION,
            **{key: float(value) for key, value in self.max_acceleration.items()},
        }


@dataclass
class RealtimeTraceConfig:
    """Unified JSONL trace for model, queue, controller, and robot timing."""

    enabled: bool = False
    path: str | None = None
    max_history: int = 512

    def __post_init__(self) -> None:
        if self.enabled and not self.path:
            raise ValueError("trace.path is required when trace.enabled=true")
        if self.max_history < 2:
            raise ValueError("trace.max_history must be at least 2")


# ---------------------------------------------------------------------------
# Strategy configs (polymorphic dispatch via draccus ChoiceRegistry)
# ---------------------------------------------------------------------------


@dataclass
class RolloutStrategyConfig(draccus.ChoiceRegistry, abc.ABC):
    """Abstract base for rollout strategy configurations.

    Use ``--strategy.type=<name>`` on the CLI to select a strategy.
    """

    @property
    def type(self) -> str:
        return self.get_choice_name(self.__class__)


@RolloutStrategyConfig.register_subclass("base")
@dataclass
class BaseStrategyConfig(RolloutStrategyConfig):
    """Autonomous rollout with no data recording."""

    pass


@RolloutStrategyConfig.register_subclass("sentry")
@dataclass
class SentryStrategyConfig(RolloutStrategyConfig):
    """Continuous autonomous rollout with always-on recording.

    Episode duration is derived from camera resolution, FPS, and
    ``target_video_file_size_mb`` so that each saved episode produces a
    video file that has crossed the target size.  This aligns episode
    boundaries with the dataset's video file chunking, so each
    ``push_to_hub`` call uploads complete video files rather than
    re-uploading a growing file that hasn't crossed the chunk boundary.
    """

    upload_every_n_episodes: int = 5
    # Target video file size in MB for episode rotation.  Episodes are
    # saved once the estimated video duration would exceed this limit.
    # Defaults to DEFAULT_VIDEO_FILE_SIZE_IN_MB when set to None.
    target_video_file_size_mb: int | None = None


@RolloutStrategyConfig.register_subclass("highlight")
@dataclass
class HighlightStrategyConfig(RolloutStrategyConfig):
    """Autonomous rollout with on-demand recording via ring buffer.

    A memory-bounded ring buffer continuously captures telemetry.  When
    the user presses the save key, the buffer contents are flushed to
    the dataset and live recording continues until the key is pressed
    again.
    """

    ring_buffer_seconds: float = 10.0
    ring_buffer_max_memory_mb: int = 1024
    save_key: str = "s"
    push_key: str = "h"


@dataclass
class DAggerKeyboardConfig:
    """Keyboard key bindings for DAgger controls.

    Keys are specified as single characters (e.g. ``"c"``, ``"h"``) or
    special key names (``"space"``).
    """

    pause_resume: str = "space"
    correction: str = "tab"
    upload: str = "enter"


@dataclass
class DAggerPedalConfig:
    """Foot pedal configuration for DAgger controls.

    Pedal codes are evdev key code strings (e.g. ``"KEY_A"``).
    """

    device_path: str = "/dev/input/by-id/usb-PCsensor_FootSwitch-event-kbd"
    pause_resume: str = "KEY_A"
    correction: str = "KEY_B"
    upload: str = "KEY_C"


@RolloutStrategyConfig.register_subclass("episodic")
@dataclass
class EpisodicStrategyConfig(RolloutStrategyConfig):
    """Episode-oriented recording that mirrors the behavior of ``lerobot-record``.

    Records ``dataset.num_episodes`` episodes of maximum ``dataset.episode_time_s`` each.
    After each episode, runs ``dataset.reset_time_s`` seconds of reset time.

    Keyboard controls:
        Right arrow  — end current episode or reset phase early
        Left arrow   — discard current episode and re-record
        Escape       — stop recording session

    In between episodes:
    - if there is no teleop leader, the robot is held at its initial joint positions captured at startup.
    - else, the robot is moved smoothly to the position of the teleop leader.
    """

    # This only applies if there are no teleop leaders specified.
    # When True (default), moves the robot back to the joint positions captured at startup.
    # Otherwise, leave the robot in its current position.
    reset_to_initial_position: bool = True

    # Whether to turn on or off the leader -> follower smooth handover behavior.
    # When False, fallback to follower -> leader handover.
    # Note that leader -> follower handover is only supported when the leader has `send_feedback` capability.
    smooth_leader_to_follower_handover: bool = True


@RolloutStrategyConfig.register_subclass("dagger")
@dataclass
class DAggerStrategyConfig(RolloutStrategyConfig):
    """Human-in-the-loop data collection (DAgger / RaC).

    Alternates between autonomous policy execution and human intervention.
    Intervention frames are tagged with ``intervention=True``.

    Input is controlled via either a keyboard or foot pedal, selected by
    ``input_device``.  Each device exposes three actions:

    1. **pause_resume** — toggle policy execution on/off.
    2. **correction** — toggle human correction recording.
    3. **upload** — push dataset to hub on demand (corrections-only mode).

    When ``record_autonomous=False`` (default) only human-correction windows
    are recorded — each correction becomes its own episode.  Set to ``True``
    to record both autonomous and correction frames with size-based episode
    rotation (same as Sentry) and background uploading.  ``push_to_hub`` is
    blocked while a correction is in progress.
    """

    # Number of correction episodes to collect (corrections-only mode).
    # When None, falls back to ``--dataset.num_episodes``.
    num_episodes: int | None = None
    record_autonomous: bool = False
    upload_every_n_episodes: int = 5
    # Target video file size in MB for episode rotation (record_autonomous
    # mode only).  Defaults to DEFAULT_VIDEO_FILE_SIZE_IN_MB when None.
    target_video_file_size_mb: int | None = None
    input_device: str = "keyboard"
    keyboard: DAggerKeyboardConfig = field(default_factory=DAggerKeyboardConfig)
    pedal: DAggerPedalConfig = field(default_factory=DAggerPedalConfig)

    def __post_init__(self):
        if self.input_device not in ("keyboard", "pedal"):
            raise ValueError(f"DAgger input_device must be 'keyboard' or 'pedal', got '{self.input_device}'")


# ---------------------------------------------------------------------------
# Top-level rollout config
# ---------------------------------------------------------------------------


@dataclass
class RolloutConfig:
    """Top-level configuration for the ``lerobot-rollout`` CLI.

    Combines hardware, policy, strategy, and runtime settings.  The
    ``__post_init__`` method performs fail-fast validation to reject
    invalid flag combinations early.
    """

    # Hardware
    robot: RobotConfig | None = None
    teleop: TeleoperatorConfig | None = None

    # Policy (loaded from --policy.path via __post_init__)
    policy: PreTrainedConfig | None = None

    # Strategy (polymorphic: --strategy.type=base|sentry|highlight|dagger)
    strategy: RolloutStrategyConfig = field(default_factory=BaseStrategyConfig)

    # Inference backend (polymorphic: --inference.type=sync|rtc)
    inference: InferenceEngineConfig = field(default_factory=SyncInferenceConfig)

    # Dataset (required for sentry, highlight, dagger; None for base)
    dataset: DatasetRecordConfig | None = None

    # Runtime
    fps: float = 30.0
    duration: float = 0.0  # 0 = infinite (24/7 mode)
    # Validate all configured software artifacts/backends, then exit before any
    # robot, camera, teleoperator, or serial object is constructed.
    preflight_only: bool = False
    # Upgrade software-only preflight to the complete paper-ready V2 contract.
    # This mode fails closed unless every measured calibration/provenance
    # artifact is configured, and it exercises the planner, executor, RTC
    # engine, and trace lifecycle without constructing hardware.
    preflight_require_paper_ready: bool = False
    interpolation_multiplier: int = 1
    device: str | None = None
    task: str = ""
    # None preserves the historical non-deterministic rollout behavior. Set an
    # explicit value for reproducible policy-noise schedules and backend A/Bs.
    seed: int | None = None
    display_data: bool = False
    # Visualization backend used when display_data is True: "rerun" or "foxglove".
    display_mode: str = "rerun"
    # For "rerun": IP of a remote server to send to. For "foxglove": interface to bind the WebSocket
    # server to (127.0.0.1 for local only, 0.0.0.0 for all interfaces).
    display_ip: str | None = None
    # For "rerun": port of the remote server. For "foxglove": port to bind the WebSocket server to.
    display_port: int | None = None
    # Whether to display compressed (JPEG) images instead of raw frames
    display_compressed_images: bool = False
    # Use vocal synthesis to read events
    play_sounds: bool = True
    resume: bool = False
    # Rename map for mapping robot/dataset observation keys to policy keys
    rename_map: dict[str, str] = field(default_factory=dict)

    # Hardware teardown
    # When True (default), smoothly interpolate the robot back to the joint
    # positions captured at startup before disconnecting.  Set to False to
    # leave the robot in its final achieved pose at shutdown.
    return_to_initial_position: bool = True

    # Torch compile
    use_torch_compile: bool = False
    torch_compile_backend: str = "inductor"
    torch_compile_mode: str = "default"
    compile_warmup_inferences: int = 2

    # Second-order output limiter on the policy action stream (off by default).
    action_filter: ActionOutputFilterConfig = field(default_factory=ActionOutputFilterConfig)

    # Fixed-heartbeat smoothing/tracking executor used by the complete V2 path.
    realtime_executor: SmoothExecutorConfig = field(default_factory=SmoothExecutorConfig)

    # Paper-level Realtime-VLA V2 timing and trajectory instrumentation.
    time_axis_planner: TimeAxisPlannerConfig = field(default_factory=TimeAxisPlannerConfig)
    speed_adapter: SpeedAdapterRuntimeConfig = field(default_factory=SpeedAdapterRuntimeConfig)
    trace: RealtimeTraceConfig = field(default_factory=RealtimeTraceConfig)

    # Stall/contact guard on the dispatched action stream.  If the robot's
    # relative-motion safety clamp (max_relative_target) rewrites the commanded
    # goal for this many consecutive dispatches, the arm is persistently
    # failing to track its targets (obstruction or unintended contact); the
    # rollout re-targets the present position and aborts instead of grinding
    # into the obstacle.  0 disables the guard.
    stall_guard_ticks: int = 5
    # Minimum |requested - applied| (normalized joint units) that counts as a
    # clamp rewrite; deviations at or below this are float pass-through noise.
    stall_guard_tolerance: float = 1e-3

    # PI0.5 hybrid inference: TensorRT prefix/KV cache + PyTorch denoise/RTC.
    pi05_prefix_backend: PI05PrefixBackend = PI05PrefixBackend.AUTO
    pi05_action_backend: PI05ActionBackend = PI05ActionBackend.PYTORCH
    pi05_tensorrt_prefix_engine: str | None = None
    pi05_tensorrt_release_torch_prefix: bool = True
    pi05_tensorrt_action_engine: str | None = None
    # Full-model realtime-vla-v2 Triton runtime. These artifacts are separate
    # from TensorRT prefix engines and are never used as a fallback.
    pi05_triton_export_weights: str | None = None
    pi05_triton_weights_sha256: str | None = None
    pi05_triton_model_config: str | None = None
    pi05_triton_tokenizer_path: str | None = None
    pi05_triton_camera_keys: list[str] = field(default_factory=list)
    pi05_triton_prompt_capacity: int = 64
    pi05_triton_tokenizer_max_length: int = 200
    pi05_triton_min_free_cuda_gib: float = 18.0
    pi05_triton_parity_report: PI05ParityReportArtifactConfig = field(
        default_factory=PI05ParityReportArtifactConfig
    )

    @property
    def use_pi05_tensorrt_prefix(self) -> bool:
        """Whether the resolved prefix backend explicitly selects TensorRT."""
        return self.pi05_prefix_backend == PI05PrefixBackend.TENSORRT or (
            self.pi05_prefix_backend == PI05PrefixBackend.AUTO
            and self.pi05_tensorrt_prefix_engine is not None
        )

    def resolved_pi05_triton_camera_keys(self) -> tuple[str, str]:
        """Return the captured two-camera order for the full Triton model."""
        keys = self.pi05_triton_camera_keys or list(getattr(self.policy, "image_features", {}))
        if len(keys) != 2 or len(set(keys)) != 2 or any(not isinstance(key, str) or not key for key in keys):
            raise ValueError("pi05_action_backend='triton' requires exactly two unique camera keys")
        return keys[0], keys[1]

    def resolved_pi05_triton_model_config(self) -> str:
        """Resolve the source checkpoint config tied to the Triton export."""
        if self.pi05_triton_model_config:
            return self.pi05_triton_model_config
        pretrained_path = getattr(self.policy, "pretrained_path", None)
        if not pretrained_path:
            raise ValueError("Triton model config cannot be resolved without policy.pretrained_path")
        return str(Path(pretrained_path).expanduser() / "config.json")

    def __post_init__(self):
        """Validate config invariants and load the policy config from ``--policy.path``."""
        if not isinstance(self.preflight_only, bool):
            raise ValueError("preflight_only must be boolean")
        if not isinstance(self.preflight_require_paper_ready, bool):
            raise ValueError("preflight_require_paper_ready must be boolean")
        if self.preflight_require_paper_ready and not self.preflight_only:
            raise ValueError("preflight_require_paper_ready requires preflight_only=true")
        if not math.isfinite(float(self.fps)) or float(self.fps) <= 0:
            raise ValueError("fps must be finite and positive")
        if (
            isinstance(self.interpolation_multiplier, bool)
            or not isinstance(self.interpolation_multiplier, int)
            or self.interpolation_multiplier < 1
        ):
            raise ValueError("interpolation_multiplier must be a positive integer")

        # --- Strategy-specific validation ---
        if isinstance(self.strategy, DAggerStrategyConfig) and self.teleop is None:
            raise ValueError("DAgger strategy requires --teleop.type to be set")

        # TODO(Steven): DAgger shouldn't require a dataset (user may want to just rollout+intervene without recording), but for now we require it to simplify the implementation.
        needs_dataset = isinstance(
            self.strategy,
            (
                SentryStrategyConfig,
                HighlightStrategyConfig,
                DAggerStrategyConfig,
                EpisodicStrategyConfig,
            ),
        )
        if needs_dataset and (self.dataset is None or not self.dataset.repo_id):
            raise ValueError(f"{self.strategy.type} strategy requires --dataset.repo_id to be set")

        if isinstance(self.strategy, BaseStrategyConfig) and self.dataset is not None:
            raise ValueError(
                "Base strategy does not record data. Use sentry, highlight, or dagger for recording."
            )

        # Sentry MUST use streaming encoding to avoid disk I/O blocking the control loop
        if (
            isinstance(self.strategy, SentryStrategyConfig)
            and self.dataset is not None
            and not self.dataset.streaming_encoding
        ):
            logger.warning("Sentry mode forces streaming_encoding=True")
            self.dataset.streaming_encoding = True

        # Highlight writes frames while the policy is still running, so streaming is mandatory.
        if (
            isinstance(self.strategy, HighlightStrategyConfig)
            and self.dataset is not None
            and not self.dataset.streaming_encoding
        ):
            logger.warning("Highlight mode forces streaming_encoding=True")
            self.dataset.streaming_encoding = True

        # DAgger: streaming is mandatory only when the autonomous phase is also recorded.
        if isinstance(self.strategy, DAggerStrategyConfig) and self.dataset is not None:
            if self.strategy.record_autonomous and not self.dataset.streaming_encoding:
                logger.warning("DAgger with record_autonomous=True forces streaming_encoding=True")
                self.dataset.streaming_encoding = True
            elif not self.strategy.record_autonomous and not self.dataset.streaming_encoding:
                logger.info(
                    "Streaming encoding is disabled for DAgger corrections-only mode. "
                    "Consider enabling it for faster episode saving: "
                    "--dataset.streaming_encoding=true --dataset.encoder_threads=2"
                )

        # DAgger: resolve num_episodes from dataset config when not explicitly set.
        if isinstance(self.strategy, DAggerStrategyConfig) and self.strategy.num_episodes is None:
            if self.dataset is not None:
                self.strategy.num_episodes = self.dataset.num_episodes
                logger.info(
                    "DAgger num_episodes not set — using --dataset.num_episodes=%d",
                    self.strategy.num_episodes,
                )
            else:
                raise ValueError(
                    "DAgger num_episodes must be set either via --strategy.num_episodes or --dataset.num_episodes"
                )

        # --- Policy loading ---
        if self.robot is None:
            raise ValueError("--robot.type is required for rollout")

        policy_path = parser.get_path_arg("policy")
        if policy_path:
            cli_overrides = parser.get_cli_overrides("policy")
            self.policy = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=cli_overrides)
            self.policy.pretrained_path = policy_path
        if self.policy is None:
            raise ValueError("--policy.path is required for rollout")

        if isinstance(self.inference, RTCInferenceConfig):
            self.inference.validate_policy_config(self.policy)
            if self.inference.dynamic_prefill_enabled and self.inference.guidance_delay_mode.value == "fixed":
                training_capacity = int(getattr(self.policy, "rtc_training_max_delay", 0))
                effective_capacity = self.inference.max_prefill_steps or training_capacity
                minimum_prefix_steps = (
                    self.inference.fixed_guidance_delay_steps
                    + math.floor(self.inference.image_capture_delay_s * self.fps + 1e-12)
                    + 1
                )
                if effective_capacity < minimum_prefix_steps:
                    raise ValueError(
                        "RTC prefix capacity cannot cover the inclusive completion anchor plus "
                        "configured image delay: "
                        f"minimum={minimum_prefix_steps}, capacity={effective_capacity}"
                    )

        if (
            isinstance(self.inference, RTCInferenceConfig)
            and self.inference.timing_mode.value == "actual_consumed"
            and self.interpolation_multiplier != 1
        ):
            raise ValueError("RTC actual_consumed timing currently requires --interpolation_multiplier=1")

        if self.speed_adapter.enabled:
            if not isinstance(self.inference, RTCInferenceConfig):
                raise ValueError("speed_adapter requires --inference.type=rtc")
            if (
                self.speed_adapter.feature_coordinate_space == "robot_action_units"
                and not self.inference.dynamic_prefill_enabled
            ):
                raise ValueError(
                    "robot_action_units speed_adapter requires RTC dynamic_prefill_enabled=true "
                    "so planning runs in postprocessed robot coordinates"
                )
            if not self.time_axis_planner.enabled:
                raise ValueError("speed_adapter requires time_axis_planner.enabled=true")
            if not math.isclose(
                self.time_axis_planner.dt_ref,
                1.0 / self.fps,
                rel_tol=1e-6,
                abs_tol=1e-9,
            ):
                raise ValueError("time_axis_planner.dt_ref must equal 1/fps with speed_adapter")

        if self.realtime_executor.enabled:
            if not isinstance(self.inference, RTCInferenceConfig):
                raise ValueError("realtime_executor requires --inference.type=rtc")
            if self.interpolation_multiplier != 1:
                raise ValueError("realtime_executor requires --interpolation_multiplier=1")
            if self.action_filter.enabled:
                raise ValueError(
                    "realtime_executor already enforces velocity/acceleration limits; "
                    "disable action_filter to avoid double filtering"
                )
            if self.time_axis_planner.enabled and not math.isclose(
                self.time_axis_planner.dt_ref,
                1.0 / self.fps,
                rel_tol=1e-6,
                abs_tol=1e-9,
            ):
                raise ValueError("time_axis_planner.dt_ref must equal 1/fps with realtime_executor")

        if self.stall_guard_ticks < 0:
            raise ValueError("stall_guard_ticks must be >= 0 (0 disables the guard)")
        if not math.isfinite(self.stall_guard_tolerance) or self.stall_guard_tolerance < 0:
            raise ValueError("stall_guard_tolerance must be finite and >= 0")

        # --- Task resolution ---
        # When any --dataset.* flag is passed, draccus creates a DatasetRecordConfig with single_task="".
        # If the user set the task via the top-level --task flag, propagate it so that all
        # downstream consumers (inference engine, dataset frame builders) see it.
        if self.dataset is not None and not self.dataset.single_task and self.task:
            logger.info("Propagating top-level task '%s' to dataset config", self.task)
            self.dataset.single_task = self.task
        elif self.dataset is not None and self.dataset.single_task and not self.task:
            logger.info("Propagating dataset single_task '%s' to top-level task", self.dataset.single_task)
            self.task = self.dataset.single_task

        # --- Device resolution ---
        # Resolve device from the policy config when not explicitly set so all
        # components (policy.to, preprocessor, inference engine) use the same
        # device string instead of inconsistent fallbacks.
        if self.device is None or not is_torch_device_available(self.device):
            resolved = self.policy.device
            if resolved:
                self.device = resolved
                logger.info("Resolved device from policy config: %s", self.device)
            else:
                self.device = auto_select_torch_device().type
                logger.info("No policy config to resolve device from; auto-selected device: %s", self.device)

        self.pi05_prefix_backend = PI05PrefixBackend(self.pi05_prefix_backend)
        self.pi05_action_backend = PI05ActionBackend(self.pi05_action_backend)

        if (
            self.pi05_prefix_backend == PI05PrefixBackend.PYTORCH
            and self.pi05_tensorrt_prefix_engine is not None
        ):
            logger.warning("Ignoring PI0.5 prefix engine because pi05_prefix_backend='pytorch' was selected")

        if (
            self.pi05_prefix_backend == PI05PrefixBackend.TENSORRT
            and self.pi05_tensorrt_prefix_engine is None
        ):
            raise ValueError("pi05_prefix_backend='tensorrt' requires --pi05_tensorrt_prefix_engine")

        if self.use_pi05_tensorrt_prefix:
            if self.policy.type != "pi05":
                raise ValueError("--pi05_tensorrt_prefix_engine can only be used with a pi05 policy")
            if not self.device.startswith("cuda"):
                raise ValueError("--pi05_tensorrt_prefix_engine requires a CUDA device")
            if self.use_torch_compile:
                raise ValueError(
                    "PI0.5 TensorRT prefix is not compatible with whole-sample torch.compile; "
                    "leave --use_torch_compile=false"
                )

        if (
            self.pi05_action_backend == PI05ActionBackend.PYTORCH
            and self.pi05_tensorrt_action_engine is not None
        ):
            logger.warning("Ignoring PI0.5 action engine because pi05_action_backend='pytorch' was selected")
        if self.pi05_action_backend == PI05ActionBackend.TENSORRT:
            if self.policy.type != "pi05":
                raise ValueError("pi05_action_backend='tensorrt' requires a pi05 policy")
            if self.pi05_tensorrt_action_engine is None:
                raise ValueError("pi05_action_backend='tensorrt' requires --pi05_tensorrt_action_engine")
            raise ValueError("PI0.5 TensorRT action backend is not implemented yet")
        if self.pi05_action_backend == PI05ActionBackend.TRITON:
            if self.policy.type != "pi05":
                raise ValueError("pi05_action_backend='triton' requires a pi05 policy")
            if not isinstance(self.inference, RTCInferenceConfig):
                raise ValueError("pi05_action_backend='triton' requires --inference.type=rtc")
            if self.inference.mode.value != "trained_prefix":
                raise ValueError("pi05_action_backend='triton' requires RTC mode='trained_prefix'")
            if self.inference.timing_mode.value != "actual_consumed":
                raise ValueError("pi05_action_backend='triton' requires RTC timing_mode='actual_consumed'")
            if not self.device.startswith("cuda"):
                raise ValueError("pi05_action_backend='triton' requires a CUDA device")
            if self.use_torch_compile:
                raise ValueError(
                    "pi05_action_backend='triton' owns its CUDA Graph; disable use_torch_compile"
                )
            if getattr(self.policy, "use_peft", False):
                raise ValueError("pi05_action_backend='triton' does not support PEFT checkpoints")
            if self.pi05_prefix_backend == PI05PrefixBackend.TENSORRT or self.pi05_tensorrt_prefix_engine:
                raise ValueError(
                    "full PI0.5 Triton inference cannot be combined with a TensorRT prefix engine"
                )
            if self.pi05_tensorrt_action_engine is not None:
                raise ValueError("full PI0.5 Triton inference does not use pi05_tensorrt_action_engine")

            if not self.pi05_triton_export_weights:
                raise ValueError("pi05_action_backend='triton' requires --pi05_triton_export_weights")
            export_path = Path(self.pi05_triton_export_weights).expanduser()
            if not export_path.is_file():
                raise FileNotFoundError(f"PI0.5 Triton export weights not found: {export_path}")
            if (
                not isinstance(self.pi05_triton_weights_sha256, str)
                or len(self.pi05_triton_weights_sha256) != 64
                or any(
                    character not in "0123456789abcdefABCDEF" for character in self.pi05_triton_weights_sha256
                )
            ):
                raise ValueError("pi05_action_backend='triton' requires a 64-hex weights SHA-256")
            model_config_path = Path(self.resolved_pi05_triton_model_config()).expanduser()
            if not model_config_path.is_file():
                raise FileNotFoundError(f"PI0.5 Triton source model config not found: {model_config_path}")
            if not self.pi05_triton_tokenizer_path or not self.pi05_triton_tokenizer_path.strip():
                raise ValueError("pi05_action_backend='triton' requires --pi05_triton_tokenizer_path")
            if not isinstance(self.pi05_triton_parity_report, PI05ParityReportArtifactConfig):
                raise ValueError("pi05_triton_parity_report must be a PI05ParityReportArtifactConfig")
            if not self.pi05_triton_parity_report.enabled:
                raise ValueError("pi05_action_backend='triton' requires a checksum-pinned parity report")

            camera_keys = self.resolved_pi05_triton_camera_keys()
            expected_camera_keys = tuple(getattr(self.policy, "image_features", {}))
            if expected_camera_keys and camera_keys != expected_camera_keys:
                raise ValueError(
                    "pi05_triton_camera_keys must exactly match policy image feature order: "
                    f"configured={camera_keys}, policy={expected_camera_keys}"
                )
            action_feature = getattr(self.policy, "action_feature", None)
            state_feature = getattr(self.policy, "robot_state_feature", None)
            if tuple(getattr(action_feature, "shape", ())) != (6,):
                raise ValueError("pi05_action_backend='triton' requires a 6-dimensional action feature")
            if tuple(getattr(state_feature, "shape", ())) != (6,):
                raise ValueError("pi05_action_backend='triton' requires a 6-dimensional state feature")
            if (
                isinstance(self.pi05_triton_prompt_capacity, bool)
                or not isinstance(self.pi05_triton_prompt_capacity, int)
                or self.pi05_triton_prompt_capacity <= 0
            ):
                raise ValueError("pi05_triton_prompt_capacity must be a positive integer")
            if (
                isinstance(self.pi05_triton_tokenizer_max_length, bool)
                or not isinstance(self.pi05_triton_tokenizer_max_length, int)
                or self.pi05_triton_tokenizer_max_length < self.pi05_triton_prompt_capacity
            ):
                raise ValueError(
                    "pi05_triton_tokenizer_max_length must be an integer no smaller than prompt capacity"
                )
            if (
                isinstance(self.pi05_triton_min_free_cuda_gib, bool)
                or not isinstance(self.pi05_triton_min_free_cuda_gib, (int, float))
                or not math.isfinite(float(self.pi05_triton_min_free_cuda_gib))
                or self.pi05_triton_min_free_cuda_gib < 0
            ):
                raise ValueError("pi05_triton_min_free_cuda_gib must be finite and non-negative")

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        return ["policy"]
