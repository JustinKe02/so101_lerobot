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

"""Rollout context: shared state created once before strategy dispatch.

Grouped into five topical sub-contexts — :class:`RuntimeContext`,
:class:`HardwareContext`, :class:`PolicyContext`, :class:`ProcessorContext`,
and :class:`DatasetContext` — assembled into :class:`RolloutContext`.
"""

from __future__ import annotations

import logging
import math
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from threading import Event

import numpy as np
import torch

from lerobot.configs import FeatureType
from lerobot.datasets import (
    LeRobotDataset,
    aggregate_pipeline_dataset_features,
    create_initial_features,
)
from lerobot.policies import get_policy_class, make_pre_post_processors
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.processor import (
    PolicyProcessorPipeline,
    RobotAction,
    RobotObservation,
    RobotProcessorPipeline,
    make_default_processors,
    rename_stats,
)
from lerobot.processor.relative_action_processor import RelativeActionsProcessorStep
from lerobot.robots import make_robot_from_config
from lerobot.teleoperators import Teleoperator, make_teleoperator_from_config
from lerobot.utils.feature_utils import combine_feature_dicts, hw_to_dataset_features

from .action_filter import ActionDictOutputFilter, resolve_joint_limit
from .actuator_calibration import (
    ActuatorCalibrationArtifact,
    load_actuator_calibration_artifact,
    normalize_action_joint_names,
)
from .configs import BaseStrategyConfig, DAggerStrategyConfig, PI05ActionBackend, RolloutConfig
from .inference import (
    InferenceEngine,
    RTCInferenceConfig,
    SyncInferenceConfig,
    create_inference_engine,
)
from .joint_constraints import JointConstraintArtifact, load_joint_constraint_artifact
from .pi05_parity import load_and_validate_pi05_parity_report
from .realtime_executor import RealtimeExecutor, RealtimeExecutorConfig
from .robot_wrapper import ThreadSafeRobot
from .sensor_timing_calibration import (
    SensorTimingCalibrationArtifact,
    load_sensor_timing_calibration_artifact,
)
from .speed_adapter import SpeedAdapter, load_runtime_speed_adapter
from .stall_guard import StallContactGuard
from .time_axis import TimeAxisPlanner
from .trajectory import DelayAlignedTrajectory, RealtimeTraceWriter

logger = logging.getLogger(__name__)


@dataclass
class _HardwareBuildState:
    """Hardware acquired while assembling a rollout context."""

    robot: object | None = None
    teleop: Teleoperator | None = None
    trace: RealtimeTraceWriter | None = None


def _rollback_hardware_build(state: _HardwareBuildState, reason: BaseException) -> None:
    """Best-effort rollback that never masks the context-build error."""
    for label, resource in (("teleoperator", state.teleop), ("robot", state.robot)):
        if resource is None:
            continue
        try:
            if getattr(resource, "is_connected", False):
                logger.info("Disconnecting %s after rollout context build failure...", label)
                resource.disconnect()
        except BaseException:
            logger.exception("Failed to disconnect %s during rollout context rollback", label)
    if state.trace is not None:
        try:
            state.trace.mark_abnormal(reason)
            state.trace.close()
        except BaseException:
            logger.exception("Failed to finalize trace during rollout context rollback")


def _resolve_action_key_order(
    policy_action_names: list[str] | None, dataset_action_names: list[str]
) -> list[str]:
    """Choose action name ordering for mapping policy tensor outputs to robot action dicts."""
    if not policy_action_names:
        return dataset_action_names
    policy_action_names = list(policy_action_names)
    if len(policy_action_names) != len(dataset_action_names):
        logger.warning(
            "policy.action_feature_names length (%d) != dataset action dim (%d); using dataset order",
            len(policy_action_names),
            len(dataset_action_names),
        )
        return dataset_action_names
    if set(dataset_action_names) != set(policy_action_names):
        logger.warning("policy.action_feature_names keys don't match dataset; using dataset order")
        return dataset_action_names
    return policy_action_names


def _configured_policy_action_dim(policy_config: object) -> int | None:
    action_names = getattr(policy_config, "action_feature_names", None)
    if action_names:
        return len(action_names)
    output_features = getattr(policy_config, "output_features", None)
    if not isinstance(output_features, dict):
        return None
    action_feature = output_features.get("action")
    shape = getattr(action_feature, "shape", None)
    if isinstance(shape, (list, tuple)) and len(shape) == 1 and isinstance(shape[0], int):
        return shape[0]
    return None


def _inject_pi05_parity_trace_provenance(
    config_snapshot: dict,
    validated_report: object | None,
) -> None:
    if validated_report is not None:
        config_snapshot["resolved_pi05_triton_parity_report"] = validated_report.audit_snapshot()


def _resolve_sensor_timing_calibration(cfg: RolloutConfig) -> SensorTimingCalibrationArtifact | None:
    if not isinstance(cfg.inference, RTCInferenceConfig):
        return None
    timing_config = cfg.inference.sensor_timing_calibration
    if not timing_config.enabled:
        return None
    if timing_config.path is None or timing_config.sha256 is None:
        raise ValueError("enabled sensor timing calibration requires artifact path and SHA-256")
    configured_cameras = tuple(getattr(cfg.robot, "cameras", {}))
    artifact = load_sensor_timing_calibration_artifact(
        timing_config.path,
        expected_sha256=timing_config.sha256,
        camera_keys=configured_cameras or None,
    )
    cfg.inference.camera_capture_delay_s = artifact.camera_delay_by_key
    cfg.inference.image_capture_delay_s = artifact.conservative_image_capture_delay_s
    cfg.inference.state_observation_delay_s = artifact.state_observation_delay_s
    cfg.inference.max_camera_skew_s = artifact.max_camera_skew_s
    cfg.inference.validate_policy_config(cfg.policy)
    if cfg.inference.guidance_delay_mode.value == "fixed":
        training_capacity = int(getattr(cfg.policy, "rtc_training_max_delay", 0))
        effective_capacity = cfg.inference.max_prefill_steps or training_capacity
        minimum_prefix_steps = (
            cfg.inference.fixed_guidance_delay_steps
            + math.floor(cfg.inference.image_capture_delay_s * cfg.fps + 1e-12)
            + 1
        )
        if effective_capacity < minimum_prefix_steps:
            raise ValueError(
                "RTC prefix capacity cannot cover the calibrated image delay: "
                f"minimum={minimum_prefix_steps}, capacity={effective_capacity}"
            )
    return artifact


def _resolve_joint_constraint_artifact(cfg: RolloutConfig) -> JointConstraintArtifact | None:
    planner_config = getattr(cfg, "time_axis_planner", None)
    if planner_config is None:
        return None
    constraint_config = planner_config.joint_constraints
    if not constraint_config.enabled:
        return None
    if constraint_config.path is None or constraint_config.sha256 is None:
        raise ValueError("enabled joint constraints require artifact path and SHA-256")
    policy_action_names = getattr(cfg.policy, "action_feature_names", None)
    expected_joint_names = normalize_action_joint_names(policy_action_names) if policy_action_names else None
    artifact = load_joint_constraint_artifact(
        constraint_config.path,
        expected_sha256=constraint_config.sha256,
        joint_names=expected_joint_names,
    )
    planner_config.max_velocity = list(artifact.max_velocity)
    planner_config.max_acceleration = list(artifact.max_acceleration)
    if cfg.realtime_executor.enabled:
        cfg.realtime_executor.max_velocity = dict(
            zip(artifact.joint_names, artifact.max_velocity, strict=True)
        )
        cfg.realtime_executor.max_acceleration = dict(
            zip(artifact.joint_names, artifact.max_acceleration, strict=True)
        )
    return artifact


def _paper_ready_preflight_gaps(cfg: RolloutConfig) -> tuple[str, ...]:
    """Return requirements that prevent an honest paper-ready V2 preflight."""

    gaps: list[str] = []
    if not isinstance(cfg.inference, RTCInferenceConfig):
        gaps.append("RTC inference")
    else:
        if cfg.inference.mode.value != "trained_prefix":
            gaps.append("trained-prefix RTC inference")
        if cfg.inference.timing_mode.value != "actual_consumed":
            gaps.append("actual-consumed RTC timing")
        if not cfg.inference.dynamic_prefill_enabled:
            gaps.append("dynamic action-prefix prefill")
        if not cfg.inference.sensor_timing_calibration.enabled:
            gaps.append("measured sensor timing calibration artifact")
        if not cfg.inference.timing_diagnostics:
            gaps.append("RTC timing diagnostics")

    planner = cfg.time_axis_planner
    if not planner.enabled:
        gaps.append("time-axis planner")
    if not planner.joint_constraints.enabled:
        gaps.append("measured joint velocity/acceleration constraints artifact")

    executor = cfg.realtime_executor
    if not executor.enabled:
        gaps.append("fixed-heartbeat realtime executor")
    if not executor.actuator_calibration.enabled:
        gaps.append("measured actuator calibration artifact")

    speed_adapter = cfg.speed_adapter
    if not speed_adapter.enabled or not speed_adapter.require_trained_checkpoint:
        gaps.append("trained human-labeled speed adapter checkpoint")

    if not cfg.trace.enabled:
        gaps.append("schema-v2 deployment trace")
    if cfg.pi05_action_backend != PI05ActionBackend.TRITON:
        gaps.append("full PI0.5 Triton action backend")
    if not cfg.pi05_triton_parity_report.enabled:
        gaps.append("checksum-pinned PyTorch/Triton parity evidence")
    if cfg.seed is None:
        gaps.append("deterministic rollout seed")
    if bool(getattr(cfg.policy, "freeze_vision_encoder", True)):
        gaps.append("fully unfrozen vision encoder checkpoint")
    if bool(getattr(cfg.policy, "train_expert_only", True)):
        gaps.append("fully unfrozen non-expert policy checkpoint")
    return tuple(gaps)


def _raise_for_paper_ready_preflight_gaps(cfg: RolloutConfig) -> None:
    gaps = _paper_ready_preflight_gaps(cfg)
    if gaps:
        formatted = "\n".join(f"  - {gap}" for gap in gaps)
        raise ValueError(
            "Paper-ready Realtime-VLA V2 preflight cannot run because required measured "
            f"artifacts or runtime components are missing:\n{formatted}"
        )


def _build_realtime_executor(
    cfg: RolloutConfig,
    ordered_action_keys: list[str],
    actuator_calibration: ActuatorCalibrationArtifact | None,
) -> RealtimeExecutor | None:
    if not cfg.realtime_executor.enabled:
        return None
    executor_velocity = cfg.realtime_executor.resolved_max_velocity()
    executor_acceleration = cfg.realtime_executor.resolved_max_acceleration()
    if actuator_calibration is not None:
        actuator_tau_s = actuator_calibration.tau_s
        command_delay_s = actuator_calibration.command_delay_s
    else:
        actuator_tau_s = cfg.realtime_executor.actuator_tau_s
        command_delay_s = cfg.realtime_executor.command_delay_s
    return RealtimeExecutor(
        RealtimeExecutorConfig(
            action_dim=len(ordered_action_keys),
            heartbeat_dt_s=1.0 / cfg.fps,
            max_velocity=[resolve_joint_limit(key, executor_velocity) for key in ordered_action_keys],
            max_acceleration=[resolve_joint_limit(key, executor_acceleration) for key in ordered_action_keys],
            actuator_tau_s=actuator_tau_s,
            command_delay_s=command_delay_s,
            enable_forward_tracking=cfg.realtime_executor.enable_forward_tracking,
            forward_lead_s=cfg.realtime_executor.forward_lead_s,
            forward_feedback_gain=cfg.realtime_executor.forward_feedback_gain,
            savgol_window_length=cfg.realtime_executor.savgol_window_length,
            savgol_polyorder=cfg.realtime_executor.savgol_polyorder,
            max_waypoints=cfg.realtime_executor.max_waypoints,
            max_command_history=cfg.realtime_executor.max_command_history,
        )
    )


class _SoftwarePreflightRobot:
    """Hardware-free robot surface used only to construct the RTC engine."""

    name = "realtime-vla-v2-software-preflight"
    robot_type = "software_preflight"
    observation_features: dict = {}
    is_connected = False

    def __init__(self, action_keys: list[str]) -> None:
        self.action_features = dict.fromkeys(action_keys, float)

    def get_observation(self) -> dict:
        raise AssertionError("paper-ready software preflight must not read robot hardware")

    def send_action(self, action: dict) -> dict:
        raise AssertionError("paper-ready software preflight must not command robot hardware")


def _run_paper_ready_component_preflight(
    cfg: RolloutConfig,
    *,
    policy: PreTrainedPolicy,
    shutdown_event: Event,
    sensor_timing_calibration: SensorTimingCalibrationArtifact | None,
    joint_constraint_artifact: JointConstraintArtifact | None,
    actuator_calibration: ActuatorCalibrationArtifact | None,
    runtime_speed_adapter: SpeedAdapter | None,
) -> tuple[str, ...]:
    """Construct and lifecycle-smoke every non-hardware V2 runtime component."""

    missing_loaded = [
        name
        for name, artifact in (
            ("sensor timing calibration", sensor_timing_calibration),
            ("joint constraints", joint_constraint_artifact),
            ("actuator calibration", actuator_calibration),
            ("speed adapter", runtime_speed_adapter),
        )
        if artifact is None
    ]
    if missing_loaded:
        raise RuntimeError(
            "Paper-ready preflight configured artifacts but did not load: " + ", ".join(missing_loaded)
        )

    action_keys = list(getattr(cfg.policy, "action_feature_names", ()) or ())
    action_dim = _configured_policy_action_dim(cfg.policy)
    if action_dim is None or action_dim < 1 or len(action_keys) != action_dim:
        raise ValueError(
            "Paper-ready preflight requires a complete ordered policy.action_feature_names layout"
        )
    joint_names = normalize_action_joint_names(action_keys)
    actuator_calibration.validate_layout(action_dim=action_dim, joint_names=joint_names)
    joint_constraint_artifact.validate_layout(joint_names)
    if runtime_speed_adapter.config.action_dim != action_dim:
        raise ValueError(
            "Paper-ready speed adapter action dimension does not match the policy: "
            f"adapter={runtime_speed_adapter.config.action_dim}, policy={action_dim}"
        )

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        dataset_stats=None,
        preprocessor_overrides={
            "device_processor": {"device": cfg.device},
            "rename_observations_processor": {"rename_map": cfg.rename_map},
        },
    )
    _, robot_action_processor, _ = make_default_processors()

    planner = TimeAxisPlanner(cfg.time_axis_planner, speed_adapter=runtime_speed_adapter)
    waypoint_count = max(3, min(cfg.time_axis_planner.horizon + 1, 8))
    synthetic_actions = np.zeros((waypoint_count, action_dim), dtype=np.float32)
    state_dim = runtime_speed_adapter.config.state_dim
    phase_dim = runtime_speed_adapter.config.phase_dim
    plan = planner.plan(
        synthetic_actions,
        state_embeddings=(np.zeros((waypoint_count, state_dim), dtype=np.float32) if state_dim else None),
        phase_embeddings=(np.zeros((waypoint_count, phase_dim), dtype=np.float32) if phase_dim else None),
    )
    if plan.used_fallback:
        raise RuntimeError(f"Paper-ready time-axis planner smoke fell back: {plan.reason}")
    if plan.actions.shape != synthetic_actions.shape or not np.isfinite(plan.actions).all():
        raise RuntimeError("Paper-ready time-axis planner returned an invalid trajectory")

    executor = _build_realtime_executor(cfg, action_keys, actuator_calibration)
    if executor is None:
        raise RuntimeError("Paper-ready realtime executor was not constructed")
    initial = np.zeros(action_dim, dtype=np.float64)
    start = 1.0
    dt = executor.config.heartbeat_dt_s
    executor.reset(initial, timestamp=start, observation_timestamp=start)
    waypoint_timestamps = start + np.arange(3, dtype=np.float64) * dt
    command = executor.control_step(
        start,
        waypoint_timestamps,
        np.zeros((3, action_dim), dtype=np.float64),
        initial,
        observation_timestamp=start,
    )
    preview = executor.preview(2)
    if command.shape != (action_dim,) or len(preview) != 2 or not np.isfinite(command).all():
        raise RuntimeError("Paper-ready realtime executor smoke returned invalid commands")

    trajectory = DelayAlignedTrajectory(cfg.trace.max_history)
    robot_wrapper = ThreadSafeRobot(_SoftwarePreflightRobot(action_keys))
    with tempfile.TemporaryDirectory(prefix="lerobot-v2-preflight-") as temporary_directory:
        trace = RealtimeTraceWriter(
            Path(temporary_directory) / "trace.jsonl",
            config_snapshot={"preflight_require_paper_ready": True},
        )
        engine = None
        try:
            engine = create_inference_engine(
                cfg.inference,
                policy=policy,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                robot_wrapper=robot_wrapper,
                hw_features={},
                dataset_features={},
                ordered_action_keys=action_keys,
                task=cfg.dataset.single_task if cfg.dataset else cfg.task,
                fps=cfg.fps,
                device=cfg.device,
                use_torch_compile=cfg.use_torch_compile,
                compile_warmup_inferences=cfg.compile_warmup_inferences,
                shutdown_event=shutdown_event,
                time_axis_planner=planner,
                trace=trace,
                trajectory=trajectory,
                realtime_executor=executor,
                robot_action_processor=robot_action_processor,
                action_filter=None,
                stall_guard=None,
            )
            if not engine.owns_action_dispatch:
                raise RuntimeError("Paper-ready RTC engine did not assign dispatch ownership to executor")
            engine.start()
            engine.stop()
            if engine.failed:
                raise RuntimeError("Paper-ready RTC engine lifecycle smoke entered a fatal state")
            trace.close()
            if not trace.closed:
                raise RuntimeError("Paper-ready trace lifecycle smoke did not write a terminal record")
        finally:
            if engine is not None:
                engine.stop()
            trace.close()

    return (
        "policy_processors",
        "time_axis_planner",
        "realtime_executor",
        "rtc_engine_lifecycle",
        "trace_lifecycle",
    )


# ---------------------------------------------------------------------------
# Sub-contexts
# ---------------------------------------------------------------------------


@dataclass
class RuntimeContext:
    """Runtime knobs shared with every strategy."""

    cfg: RolloutConfig
    shutdown_event: Event


@dataclass
class HardwareContext:
    """Connected hardware.

    The raw robot is available via ``robot_wrapper.inner`` when needed
    (e.g. for disconnect); strategies should otherwise go through the
    thread-safe wrapper.

    ``initial_position`` stores the robot's joint positions at connect
    time.  Strategies use it to return the robot to a safe pose before
    shutting down.
    """

    robot_wrapper: ThreadSafeRobot
    teleop: Teleoperator | None
    initial_position: dict | None = None
    # Set by the rollout entry point when teardown follows an error.  Hardware
    # teardown then skips return-to-initial and keeps motor torque enabled so
    # the arm holds its pose instead of collapsing from an unknown position.
    abnormal_shutdown: bool = False


@dataclass
class PolicyContext:
    """Loaded policy and its inference engine."""

    policy: PreTrainedPolicy
    preprocessor: PolicyProcessorPipeline
    postprocessor: PolicyProcessorPipeline
    inference: InferenceEngine
    # Optional second-order output limiter applied to the policy action
    # stream in ``send_next_action`` (never to teleop corrections).
    action_filter: ActionDictOutputFilter | None = None
    # Optional stall/contact guard watching the same dispatch stream.
    stall_guard: StallContactGuard | None = None
    time_axis_planner: TimeAxisPlanner | None = None
    trace: RealtimeTraceWriter | None = None
    trajectory: DelayAlignedTrajectory | None = None
    realtime_executor: RealtimeExecutor | None = None


@dataclass
class ProcessorContext:
    """Robot-side pipelines (run outside the policy)."""

    teleop_action_processor: RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction]
    robot_action_processor: RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction]
    robot_observation_processor: RobotProcessorPipeline[RobotObservation, RobotObservation]


@dataclass
class DatasetContext:
    """Dataset and feature bookkeeping."""

    dataset: LeRobotDataset | None
    dataset_features: dict = field(default_factory=dict)
    hw_features: dict = field(default_factory=dict)
    ordered_action_keys: list[str] = field(default_factory=list)


@dataclass
class RolloutContext:
    """Bundle of sub-contexts passed to every rollout strategy.

    Built once by :func:`build_rollout_context` before strategy dispatch.
    """

    runtime: RuntimeContext
    hardware: HardwareContext
    policy: PolicyContext
    processors: ProcessorContext
    data: DatasetContext


@dataclass(frozen=True)
class RolloutPreflightResult:
    """Successful software-only validation completed before hardware construction."""

    policy_type: str
    device: str
    action_backend: str
    trained_prefix_max: int | None
    warmed_prefix_lengths: tuple[int, ...]
    validated_artifacts: tuple[str, ...]
    paper_ready_requested: bool = False
    component_checks: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------


def build_rollout_context(
    cfg: RolloutConfig,
    shutdown_event: Event,
    teleop_action_processor: RobotProcessorPipeline | None = None,
    robot_action_processor: RobotProcessorPipeline | None = None,
    robot_observation_processor: RobotProcessorPipeline | None = None,
) -> RolloutContext | RolloutPreflightResult:
    """Build a rollout context and release connected hardware on failure."""
    hardware_state = _HardwareBuildState()
    try:
        return _build_rollout_context(
            cfg,
            shutdown_event,
            teleop_action_processor=teleop_action_processor,
            robot_action_processor=robot_action_processor,
            robot_observation_processor=robot_observation_processor,
            hardware_state=hardware_state,
        )
    except BaseException as exc:
        _rollback_hardware_build(hardware_state, exc)
        raise


def _build_rollout_context(
    cfg: RolloutConfig,
    shutdown_event: Event,
    teleop_action_processor: RobotProcessorPipeline | None = None,
    robot_action_processor: RobotProcessorPipeline | None = None,
    robot_observation_processor: RobotProcessorPipeline | None = None,
    *,
    hardware_state: _HardwareBuildState,
) -> RolloutContext | RolloutPreflightResult:
    """Wire up policy, processors, hardware, dataset, and inference engine.

    The order is policy-first / hardware-last so a bad ``--policy.path``
    fails fast without touching the robot.
    """
    is_rtc = isinstance(cfg.inference, RTCInferenceConfig)
    paper_ready_preflight = bool(getattr(cfg, "preflight_require_paper_ready", False))
    if paper_ready_preflight:
        _raise_for_paper_ready_preflight_gaps(cfg)
    validated_pi05_parity_report = None
    if getattr(cfg, "pi05_action_backend", PI05ActionBackend.PYTORCH) == PI05ActionBackend.TRITON:
        validated_pi05_parity_report = load_and_validate_pi05_parity_report(
            cfg.pi05_triton_parity_report,
            checkpoint_path=cfg.policy.pretrained_path,
            triton_weights_path=cfg.pi05_triton_export_weights,
            triton_weights_sha256=cfg.pi05_triton_weights_sha256,
            triton_model_config_path=cfg.resolved_pi05_triton_model_config(),
            training_max_delay=cfg.policy.rtc_training_max_delay,
        )
        logger.info(
            "Validated checksum-pinned PI0.5 Triton parity before CUDA and hardware: "
            "report_sha256=%s, prefixes=%s, max_abs_errors=%s",
            validated_pi05_parity_report.sha256,
            validated_pi05_parity_report.prefixes,
            validated_pi05_parity_report.max_abs_errors,
        )
    sensor_timing_calibration = _resolve_sensor_timing_calibration(cfg)
    joint_constraint_artifact = _resolve_joint_constraint_artifact(cfg)

    actuator_calibration: ActuatorCalibrationArtifact | None = None
    realtime_executor_config = getattr(cfg, "realtime_executor", None)
    if (
        realtime_executor_config is not None
        and realtime_executor_config.enabled
        and realtime_executor_config.actuator_calibration.enabled
    ):
        calibration_config = realtime_executor_config.actuator_calibration
        if calibration_config.path is None or calibration_config.sha256 is None:
            raise ValueError("enabled actuator calibration requires both artifact path and SHA-256")
        policy_action_names = getattr(cfg.policy, "action_feature_names", None)
        expected_joint_names = (
            normalize_action_joint_names(policy_action_names) if policy_action_names else None
        )
        actuator_calibration = load_actuator_calibration_artifact(
            calibration_config.path,
            expected_sha256=calibration_config.sha256,
            action_dim=len(expected_joint_names) if expected_joint_names is not None else None,
            joint_names=expected_joint_names,
        )
    runtime_speed_adapter: SpeedAdapter | None = None
    speed_adapter_metadata: dict | None = None
    speed_adapter_config = getattr(cfg, "speed_adapter", None)
    loaded_speed_adapter = (
        load_runtime_speed_adapter(
            speed_adapter_config,
            expected_action_dim=_configured_policy_action_dim(cfg.policy),
        )
        if speed_adapter_config is not None
        else None
    )
    if loaded_speed_adapter is not None:
        runtime_speed_adapter, speed_adapter_metadata = loaded_speed_adapter
        logger.info(
            "Validated speed adapter checkpoint before hardware connection: path=%s, "
            "action_dim=%d, beta=[%.6f, %.6f]",
            speed_adapter_config.checkpoint,
            runtime_speed_adapter.config.action_dim,
            runtime_speed_adapter.config.beta_min,
            runtime_speed_adapter.config.beta_max,
        )

    # --- 1. Policy (heavy I/O, but no hardware yet) -------------------
    task_str = cfg.dataset.single_task if cfg.dataset else getattr(cfg, "task", "")
    policy_config = cfg.policy

    if hasattr(policy_config, "compile_model"):
        policy_config.compile_model = cfg.use_torch_compile

    if policy_config.type == "vqbet" and cfg.device == "mps":
        raise NotImplementedError(
            "Current implementation of VQBeT does not support `mps` backend. "
            "Please use `cpu` or `cuda` backend."
        )

    if getattr(cfg, "pi05_action_backend", PI05ActionBackend.PYTORCH) == PI05ActionBackend.TRITON:
        from lerobot.policies.pi05.realtime_vla_v2_triton import (
            PI05RealtimeVLATritonBackend,
            PI05RealtimeVLATritonConfig,
            PI05RealtimeVLATritonPolicyAdapter,
        )

        camera_keys = cfg.resolved_pi05_triton_camera_keys()
        logger.info(
            "Loading and warming full PI0.5 realtime-vla-v2 Triton backend before hardware: %s",
            cfg.pi05_triton_export_weights,
        )
        triton_config = PI05RealtimeVLATritonConfig(
            prompt=task_str,
            tokenizer_path=cfg.pi05_triton_tokenizer_path,
            camera_keys=camera_keys,
            weights_path=cfg.pi05_triton_export_weights,
            model_config_path=cfg.resolved_pi05_triton_model_config(),
            weights_sha256=cfg.pi05_triton_weights_sha256,
            device=cfg.device,
            image_value_range="zero_one",
            tokenizer_max_length=cfg.pi05_triton_tokenizer_max_length,
            prompt_capacity=cfg.pi05_triton_prompt_capacity,
            noise_seed=cfg.seed,
            min_free_cuda_gib=cfg.pi05_triton_min_free_cuda_gib,
        )
        triton_backend = PI05RealtimeVLATritonBackend.from_export(triton_config)
        policy = PI05RealtimeVLATritonPolicyAdapter(
            triton_backend,
            policy_config,
            camera_keys=camera_keys,
            task=task_str,
        )
        logger.info(
            "Full PI0.5 Triton backend ready: cameras=%s, chunk=50, action_dim=6, prefix_capacity=%d",
            camera_keys,
            triton_backend.trained_prefix_max,
        )
    else:
        logger.info("Loading policy from '%s'...", cfg.policy.pretrained_path)
        policy_class = get_policy_class(policy_config.type)
        if policy_config.use_peft:
            from peft import PeftConfig, PeftModel

            peft_path = policy_config.pretrained_path
            peft_config = PeftConfig.from_pretrained(peft_path)
            policy = policy_class.from_pretrained(
                pretrained_name_or_path=peft_config.base_model_name_or_path, config=policy_config
            )
            policy = PeftModel.from_pretrained(policy, peft_path, config=peft_config)
        else:
            policy = policy_class.from_pretrained(policy_config.pretrained_path, config=policy_config)

    if is_rtc:
        if cfg.inference.mode.value == "trained_prefix":
            training_capacity = getattr(policy_config, "rtc_training_max_delay", 0)
            if training_capacity <= 0:
                raise ValueError(
                    "trained_prefix RTC mode requires policy.rtc_training_max_delay > 0; "
                    "the loaded checkpoint was not trained with action-prefix conditioning"
                )
        policy.config.rtc_config = cfg.inference.rtc
        if hasattr(policy, "init_rtc_processor"):
            policy.init_rtc_processor()

    policy = policy.to(cfg.device)
    policy.eval()

    if cfg.use_pi05_tensorrt_prefix:
        from lerobot.policies.pi05.tensorrt_prefix import enable_pi05_tensorrt_prefix

        backend = enable_pi05_tensorrt_prefix(
            policy,
            cfg.pi05_tensorrt_prefix_engine,
            device=cfg.device,
            release_torch_prefix=cfg.pi05_tensorrt_release_torch_prefix,
        )
        logger.info(
            "TensorRT PI0.5 prefix enabled: engine=%s, cameras=%d, cache_layers=%d",
            cfg.pi05_tensorrt_prefix_engine,
            backend.num_cameras,
            backend.num_layers,
        )

    logger.info("Policy loaded: type=%s, device=%s", policy_config.type, cfg.device)

    if cfg.use_torch_compile and policy.type not in ("pi0", "pi05"):
        try:
            if hasattr(torch, "compile"):
                compile_kwargs = {
                    "backend": cfg.torch_compile_backend,
                    "mode": cfg.torch_compile_mode,
                    "options": {"triton.cudagraphs": False},
                }
                policy.predict_action_chunk = torch.compile(policy.predict_action_chunk, **compile_kwargs)
                logger.info("torch.compile applied to predict_action_chunk")
        except Exception as e:
            logger.warning("Failed to apply torch.compile: %s", e)

    if getattr(cfg, "preflight_only", False):
        trained_prefix_max: int | None = None
        warmed_prefix_lengths: tuple[int, ...] = ()
        if cfg.pi05_action_backend == PI05ActionBackend.TRITON:
            trained_prefix_max = int(triton_backend.trained_prefix_max)
            warmed_prefix_lengths = tuple(triton_backend.warmed_prefill_lengths)
            expected_warmup = tuple(range(trained_prefix_max + 1))
            if warmed_prefix_lengths != expected_warmup:
                raise RuntimeError(
                    "Triton preflight did not warm every trained prefix length: "
                    f"warmed={warmed_prefix_lengths}, expected={expected_warmup}"
                )
        validated_artifacts = tuple(
            name
            for name, present in (
                ("sensor_timing", sensor_timing_calibration is not None),
                ("joint_constraints", joint_constraint_artifact is not None),
                ("actuator_calibration", actuator_calibration is not None),
                ("speed_adapter", runtime_speed_adapter is not None),
                ("triton_parity", validated_pi05_parity_report is not None),
            )
            if present
        )
        component_checks = ()
        if paper_ready_preflight:
            component_checks = _run_paper_ready_component_preflight(
                cfg,
                policy=policy,
                shutdown_event=shutdown_event,
                sensor_timing_calibration=sensor_timing_calibration,
                joint_constraint_artifact=joint_constraint_artifact,
                actuator_calibration=actuator_calibration,
                runtime_speed_adapter=runtime_speed_adapter,
            )
        logger.info(
            "Software-only rollout preflight passed: policy=%s backend=%s artifacts=%s "
            "prefixes=%s paper_ready=%s component_checks=%s",
            policy_config.type,
            cfg.pi05_action_backend.value,
            validated_artifacts,
            warmed_prefix_lengths,
            paper_ready_preflight,
            component_checks,
        )
        return RolloutPreflightResult(
            policy_type=policy_config.type,
            device=str(cfg.device),
            action_backend=cfg.pi05_action_backend.value,
            trained_prefix_max=trained_prefix_max,
            warmed_prefix_lengths=warmed_prefix_lengths,
            validated_artifacts=validated_artifacts,
            paper_ready_requested=paper_ready_preflight,
            component_checks=component_checks,
        )

    # --- 2. Robot-side processors (user-supplied or defaults) --------
    if (
        teleop_action_processor is None
        or robot_action_processor is None
        or robot_observation_processor is None
    ):
        _t, _r, _o = make_default_processors()
        teleop_action_processor = teleop_action_processor or _t
        robot_action_processor = robot_action_processor or _r
        robot_observation_processor = robot_observation_processor or _o

    # --- 3. Hardware (heaviest side-effect, deferred) -----------------
    logger.info("Connecting robot (%s)...", cfg.robot.type if cfg.robot else "?")
    robot = make_robot_from_config(cfg.robot)
    hardware_state.robot = robot
    robot.connect()
    logger.info("Robot connected: %s", robot.name)

    # Store the initial joint positions so we can return to a safe pose on shutdown.
    initial_obs = robot.get_observation()
    initial_position = {k: v for k, v in initial_obs.items() if k.endswith(".pos")}
    logger.info("Captured initial robot position (%d keys)", len(initial_position))

    robot_wrapper = ThreadSafeRobot(robot)

    teleop = None
    if cfg.teleop is not None:
        logger.info("Connecting teleoperator (%s)...", cfg.teleop.type if cfg.teleop else "?")
        teleop = make_teleoperator_from_config(cfg.teleop)
        hardware_state.teleop = teleop
        teleop.connect()
        logger.info("Teleoperator connected")

    # TODO(Steven): once Teleoperator motor-control methods are standardised
    # (``enable_torque`` / ``disable_torque`` / ``write_goal_positions``), gate
    # the DAgger strategy on their presence here and fail fast with a helpful
    # message instead of relying on the operator to pre-align the leader by
    # hand.  See :func:`DAggerStrategy._apply_transition` for the matching
    # disabled call sites.
    # if isinstance(cfg.strategy, DAggerStrategyConfig) and teleop is not None:
    #     required_teleop_methods = ("enable_torque", "disable_torque", "write_goal_positions")
    #     missing = [m for m in required_teleop_methods if not callable(getattr(teleop, m, None))]
    #     if missing:
    #         teleop.disconnect()
    #         raise ValueError(
    #             f"DAgger strategy requires a teleoperator with motor control methods "
    #             f"{required_teleop_methods}. '{type(teleop).__name__}' is missing: {missing}"
    #         )

    # --- 4. Features + action-key reconciliation ---------------------
    # TODO(Steven):Only ``.pos`` joint features are routed to the policy as state and as the
    # action target; velocity and torque channels (when present) are kept in
    # the raw observation but excluded from the policy-facing tensors.
    all_obs_features = robot.observation_features
    # ``observation_features`` values are either a tuple (camera shape) or the
    # ``float`` type itself used as a sentinel for scalar motor features —
    # see ``dict[str, type | tuple]`` annotation on ``Robot.observation_features``.
    observation_features_hw = {
        k: v
        for k, v in all_obs_features.items()
        if isinstance(v, tuple) or (v is float and k.endswith(".pos"))
    }
    action_features_hw = {k: v for k, v in robot.action_features.items() if k.endswith(".pos")}

    # The action side is always needed: sync inference reads action names from
    # ``dataset_features[ACTION]`` to map policy tensors back to robot actions.
    action_dataset_features = aggregate_pipeline_dataset_features(
        pipeline=teleop_action_processor,
        initial_features=create_initial_features(action=action_features_hw),
        use_videos=cfg.dataset.video if cfg.dataset else True,
    )
    # Observation-side aggregation is needed because of build_dataset_frame
    observation_dataset_features = aggregate_pipeline_dataset_features(
        pipeline=robot_observation_processor,
        initial_features=create_initial_features(observation=observation_features_hw),
        use_videos=cfg.dataset.video if cfg.dataset else True,
    )
    dataset_features = combine_feature_dicts(action_dataset_features, observation_dataset_features)
    hw_features = hw_to_dataset_features(observation_features_hw, "observation")
    raw_action_keys = list(action_features_hw.keys())
    policy_action_names = getattr(policy_config, "action_feature_names", None)
    ordered_action_keys = _resolve_action_key_order(
        list(policy_action_names) if policy_action_names else None,
        raw_action_keys,
    )
    if runtime_speed_adapter is not None and runtime_speed_adapter.config.action_dim != len(
        ordered_action_keys
    ):
        raise ValueError(
            "speed adapter action dimension does not match the resolved robot action layout: "
            f"checkpoint={runtime_speed_adapter.config.action_dim}, robot={len(ordered_action_keys)}"
        )

    # Validate visual features if no rename_map is active
    rename_map = cfg.rename_map
    if not rename_map:
        expected_visuals = {
            k for k, v in policy_config.input_features.items() if v.type == FeatureType.VISUAL
        }
        provided_visuals = {
            f"observation.images.{k}" for k, v in robot.observation_features.items() if isinstance(v, tuple)
        }
        policy_subset = expected_visuals.issubset(provided_visuals)
        hw_subset = provided_visuals.issubset(expected_visuals)
        if not (policy_subset or hw_subset):
            raise ValueError(
                f"Visual feature mismatch between policy and robot hardware.\n"
                f"Policy expects: {expected_visuals}\n"
                f"Robot provides: {provided_visuals}\n"
                f"Use --rename_map to map camera names, e.g. "
                f"""--rename_map='{{"observation.images.top": "observation.images.cam0"}}'"""
            )

    # --- 5. Dataset -------------
    dataset = None
    if cfg.dataset is not None and not isinstance(cfg.strategy, BaseStrategyConfig):
        logger.info("Setting up dataset (repo_id=%s)...", cfg.dataset.repo_id)
        if cfg.resume:
            dataset = LeRobotDataset.resume(
                cfg.dataset.repo_id,
                root=cfg.dataset.root,
                batch_encoding_size=cfg.dataset.video_encoding_batch_size,
                rgb_encoder=cfg.dataset.rgb_encoder,
                depth_encoder=cfg.dataset.depth_encoder,
                streaming_encoding=cfg.dataset.streaming_encoding,
                encoder_queue_maxsize=cfg.dataset.encoder_queue_maxsize,
                encoder_threads=cfg.dataset.encoder_threads,
                image_writer_processes=cfg.dataset.num_image_writer_processes,
                image_writer_threads=cfg.dataset.num_image_writer_threads_per_camera
                * len(robot.cameras if hasattr(robot, "cameras") else []),
            )
        else:
            if isinstance(cfg.strategy, DAggerStrategyConfig):
                dataset_features["intervention"] = {
                    "dtype": "bool",
                    "shape": (1,),
                    "names": None,
                }

            repo_name = cfg.dataset.repo_id.split("/", 1)[-1]
            if not repo_name.startswith("rollout_"):
                raise ValueError(
                    "Dataset names for rollout must start with 'rollout_'. "
                    "Use --dataset.repo_id=<user>/rollout_<name> for policy deployment datasets."
                )
            cfg.dataset.stamp_repo_id()
            target_video_mb = getattr(cfg.strategy, "target_video_file_size_mb", None)
            dataset = LeRobotDataset.create(
                cfg.dataset.repo_id,
                cfg.dataset.fps,
                root=cfg.dataset.root,
                robot_type=robot.name,
                features=dataset_features,
                use_videos=cfg.dataset.video,
                image_writer_processes=cfg.dataset.num_image_writer_processes,
                image_writer_threads=cfg.dataset.num_image_writer_threads_per_camera
                * len(robot.cameras if hasattr(robot, "cameras") else []),
                batch_encoding_size=cfg.dataset.video_encoding_batch_size,
                rgb_encoder=cfg.dataset.rgb_encoder,
                depth_encoder=cfg.dataset.depth_encoder,
                streaming_encoding=cfg.dataset.streaming_encoding,
                encoder_queue_maxsize=cfg.dataset.encoder_queue_maxsize,
                encoder_threads=cfg.dataset.encoder_threads,
                video_files_size_in_mb=target_video_mb,
            )

    if dataset is not None:
        logger.info("Dataset ready: %s (%d existing episodes)", dataset.repo_id, dataset.num_episodes)

    # --- 6. Policy pre/post processors (needs dataset stats if any) ---
    dataset_stats = None
    if dataset is not None:
        dataset_stats = rename_stats(
            dataset.meta.stats,
            cfg.rename_map,
        )

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_config,
        pretrained_path=cfg.policy.pretrained_path,
        dataset_stats=dataset_stats,
        preprocessor_overrides={
            "device_processor": {"device": cfg.device},
            "rename_observations_processor": {"rename_map": cfg.rename_map},
        },
    )

    if isinstance(cfg.inference, SyncInferenceConfig) and any(
        isinstance(step, RelativeActionsProcessorStep) and step.enabled
        for step in getattr(preprocessor, "steps", ())
    ):
        raise NotImplementedError(
            "SyncInferenceEngine does not support policies with relative actions for now."
            "Use --inference.type=rtc or remove relative action processor steps from the policy pipeline."
        )

    # --- 7. Inference strategy (needs policy + pre/post + hardware) --
    logger.info(
        "Creating inference engine (type=%s)...",
        cfg.inference.type if hasattr(cfg.inference, "type") else "sync",
    )
    if actuator_calibration is not None:
        actuator_calibration.validate_layout(
            action_dim=len(ordered_action_keys),
            joint_names=normalize_action_joint_names(ordered_action_keys),
        )
    if joint_constraint_artifact is not None:
        joint_constraint_artifact.validate_layout(normalize_action_joint_names(ordered_action_keys))
    time_axis_planner = (
        TimeAxisPlanner(cfg.time_axis_planner, speed_adapter=runtime_speed_adapter)
        if cfg.time_axis_planner.enabled
        else None
    )
    trace_config_snapshot = asdict(cfg)
    _inject_pi05_parity_trace_provenance(
        trace_config_snapshot,
        validated_pi05_parity_report,
    )
    if actuator_calibration is not None:
        trace_config_snapshot["realtime_executor"]["resolved_actuator_calibration"] = (
            actuator_calibration.audit_snapshot()
        )
    if sensor_timing_calibration is not None:
        trace_config_snapshot["inference"]["resolved_sensor_timing_calibration"] = (
            sensor_timing_calibration.audit_snapshot()
        )
    if joint_constraint_artifact is not None:
        trace_config_snapshot["time_axis_planner"]["resolved_joint_constraint_provenance"] = (
            joint_constraint_artifact.audit_snapshot()
        )
    if speed_adapter_metadata is not None:
        trace_config_snapshot["speed_adapter"]["resolved_checkpoint"] = {
            key: speed_adapter_metadata[key]
            for key in (
                "format",
                "format_version",
                "architecture",
                "feature_contract",
                "output_contract",
                "provenance",
                "weights",
            )
        }
    trace = (
        RealtimeTraceWriter(cfg.trace.path, config_snapshot=trace_config_snapshot)
        if cfg.trace.enabled
        else None
    )
    hardware_state.trace = trace
    trajectory = DelayAlignedTrajectory(cfg.trace.max_history)
    realtime_executor = _build_realtime_executor(cfg, ordered_action_keys, actuator_calibration)
    if realtime_executor is not None:
        logger.info(
            "Realtime smooth executor enabled: heartbeat_dt=%.6f, savgol=%d/%d, tau_s=%s, command_delay_s=%s",
            1.0 / cfg.fps,
            cfg.realtime_executor.savgol_window_length,
            cfg.realtime_executor.savgol_polyorder,
            realtime_executor.config.actuator_tau_s.tolist(),
            realtime_executor.config.command_delay_s.tolist(),
        )
        if actuator_calibration is not None:
            logger.info(
                "Validated actuator calibration: artifact_sha256=%s, source_sha256=%s, "
                "joint_names=%s, fit_boundary_status=%s",
                actuator_calibration.artifact_sha256,
                actuator_calibration.source_sha256,
                list(actuator_calibration.joint_names),
                actuator_calibration.audit_snapshot()["fit_boundary_status"],
            )
        else:
            logger.info(
                "Realtime executor is using manually configured actuator parameters; "
                "they are not marked as SO-101 measurements"
            )

    # Dispatch guards are built before the inference engine because an enabled
    # realtime executor owns the fixed-heartbeat hardware dispatch path.
    action_filter = None
    if cfg.action_filter.enabled:
        filter_dt = (
            1.0 / cfg.fps if realtime_executor is not None else 1.0 / (cfg.fps * cfg.interpolation_multiplier)
        )
        max_velocity = cfg.action_filter.resolved_max_velocity()
        max_acceleration = cfg.action_filter.resolved_max_acceleration()
        action_filter = ActionDictOutputFilter(
            action_keys=ordered_action_keys,
            max_velocity=max_velocity,
            max_acceleration=max_acceleration,
            dt=filter_dt,
        )
        logger.info(
            "Action output filter enabled: joints=%d, dt=%.6f, max_velocity=%s, max_acceleration=%s",
            len(ordered_action_keys),
            filter_dt,
            {key: resolve_joint_limit(key, max_velocity) for key in ordered_action_keys},
            {key: resolve_joint_limit(key, max_acceleration) for key in ordered_action_keys},
        )

    stall_guard = None
    if cfg.stall_guard_ticks > 0:
        stall_guard = StallContactGuard(cfg.stall_guard_ticks, cfg.stall_guard_tolerance)
        logger.info(
            "Stall/contact guard enabled: ticks=%d, tolerance=%.4f",
            cfg.stall_guard_ticks,
            cfg.stall_guard_tolerance,
        )

    inference_strategy = create_inference_engine(
        cfg.inference,
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        robot_wrapper=robot_wrapper,
        hw_features=hw_features,
        dataset_features=dataset_features,
        ordered_action_keys=ordered_action_keys,
        task=task_str,
        fps=cfg.fps,
        device=cfg.device,
        use_torch_compile=cfg.use_torch_compile,
        compile_warmup_inferences=cfg.compile_warmup_inferences,
        shutdown_event=shutdown_event,
        time_axis_planner=time_axis_planner,
        trace=trace,
        trajectory=trajectory,
        realtime_executor=realtime_executor,
        robot_action_processor=robot_action_processor,
        action_filter=action_filter,
        stall_guard=stall_guard,
    )

    # --- 9. Assemble ---------------------------------------------------
    logger.info("Rollout context assembled successfully")
    return RolloutContext(
        runtime=RuntimeContext(cfg=cfg, shutdown_event=shutdown_event),
        hardware=HardwareContext(
            robot_wrapper=robot_wrapper, teleop=teleop, initial_position=initial_position
        ),
        policy=PolicyContext(
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            inference=inference_strategy,
            action_filter=action_filter,
            stall_guard=stall_guard,
            time_axis_planner=time_axis_planner,
            trace=trace,
            trajectory=trajectory,
            realtime_executor=realtime_executor,
        ),
        processors=ProcessorContext(
            teleop_action_processor=teleop_action_processor,
            robot_action_processor=robot_action_processor,
            robot_observation_processor=robot_observation_processor,
        ),
        data=DatasetContext(
            dataset=dataset,
            dataset_features=dataset_features,
            hw_features=hw_features,
            ordered_action_keys=ordered_action_keys,
        ),
    )
