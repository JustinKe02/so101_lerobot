#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

from __future__ import annotations

from pathlib import Path
from threading import Event
from types import SimpleNamespace
from unittest.mock import patch

import draccus
import pytest

from lerobot.policies.pi05.realtime_vla_v2_triton import PI05RealtimeVLATritonBackend
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.rollout.configs import (
    ActuatorCalibrationArtifactConfig,
    PI05ActionBackend,
    RealtimeTraceConfig,
    RolloutConfig,
    SmoothExecutorConfig,
)
from lerobot.rollout.context import RolloutPreflightResult, build_rollout_context
from lerobot.rollout.inference.factory import RTCInferenceConfig
from lerobot.rollout.joint_constraints import JointConstraintArtifactConfig
from lerobot.rollout.pi05_parity import PI05ParityReportArtifactConfig
from lerobot.rollout.sensor_timing_calibration import SensorTimingCalibrationArtifactConfig
from lerobot.rollout.speed_adapter import SpeedAdapterRuntimeConfig
from lerobot.rollout.time_axis import TimeAxisPlannerConfig


def _policy_config(checkpoint: Path) -> SimpleNamespace:
    visual = SimpleNamespace(shape=(3, 480, 640))
    return SimpleNamespace(
        type="pi05",
        device="cuda",
        chunk_size=50,
        rtc_training_max_delay=6,
        pretrained_path=str(checkpoint),
        use_peft=False,
        image_features={"observation.images.top": visual, "observation.images.wrist": visual},
        action_feature=SimpleNamespace(shape=(6,)),
        robot_state_feature=SimpleNamespace(shape=(6,)),
        action_feature_names=[
            "shoulder_pan.pos",
            "shoulder_lift.pos",
            "elbow_flex.pos",
            "wrist_flex.pos",
            "wrist_roll.pos",
            "gripper.pos",
        ],
        freeze_vision_encoder=False,
        train_expert_only=False,
    )


def _validated_parity() -> SimpleNamespace:
    return SimpleNamespace(
        sha256="b" * 64,
        prefixes=tuple(range(7)),
        max_abs_errors=(0.0,) * 7,
    )


def _triton_rollout_config(tmp_path: Path, **overrides) -> RolloutConfig:
    checkpoint = tmp_path / "pretrained_model"
    checkpoint.mkdir(exist_ok=True)
    (checkpoint / "config.json").write_text("{}", encoding="utf-8")
    weights = tmp_path / "pi05-triton.pkl"
    weights.write_bytes(b"not-loaded-by-static-validation")
    values = {
        "robot": SimpleNamespace(type="mock"),
        "policy": _policy_config(checkpoint),
        "device": "cuda",
        "task": "pick up cube",
        "inference": RTCInferenceConfig(
            rtc=RTCConfig(enabled=True, execution_horizon=10),
            mode="trained_prefix",
            timing_mode="actual_consumed",
            guidance_delay_mode="fixed",
            fixed_guidance_delay_steps=5,
            dynamic_prefill_enabled=True,
            max_prefill_steps=6,
        ),
        "pi05_prefix_backend": "pytorch",
        "pi05_action_backend": "triton",
        "pi05_triton_export_weights": str(weights),
        "pi05_triton_weights_sha256": "a" * 64,
        "pi05_triton_tokenizer_path": "unused-tokenizer",
        "pi05_triton_camera_keys": ["observation.images.top", "observation.images.wrist"],
        "pi05_triton_min_free_cuda_gib": 0.0,
        "pi05_triton_parity_report": PI05ParityReportArtifactConfig(
            enabled=True,
            path=str(tmp_path / "parity-report.json"),
            sha256="b" * 64,
        ),
    }
    values.update(overrides)
    with patch("lerobot.rollout.configs.parser.get_path_arg", return_value=None):
        return RolloutConfig(**values)


def test_rollout_draccus_decodes_distinct_triton_action_backend_fields() -> None:
    raw = {
        "preflight_only": True,
        "preflight_require_paper_ready": False,
        "pi05_action_backend": "triton",
        "pi05_triton_export_weights": "/generated/pi05.pkl",
        "pi05_triton_weights_sha256": "a" * 64,
        "pi05_triton_tokenizer_path": "google/paligemma-3b-pt-224",
        "pi05_triton_camera_keys": ["observation.images.top", "observation.images.wrist"],
        "pi05_triton_min_free_cuda_gib": 20.0,
        "pi05_triton_parity_report": {
            "enabled": True,
            "path": "/generated/parity-report.json",
            "sha256": "b" * 64,
        },
    }
    with patch.object(RolloutConfig, "__post_init__", lambda self: None):
        config = draccus.decode(RolloutConfig, raw)

    assert config.pi05_action_backend is PI05ActionBackend.TRITON
    assert config.preflight_only is True
    assert config.preflight_require_paper_ready is False
    assert config.pi05_triton_export_weights == "/generated/pi05.pkl"
    assert config.pi05_triton_min_free_cuda_gib == pytest.approx(20.0)
    assert config.pi05_triton_parity_report.enabled is True
    assert config.pi05_tensorrt_action_engine is None


def test_paper_ready_preflight_requires_software_only_mode(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="requires preflight_only=true"):
        _triton_rollout_config(
            tmp_path,
            preflight_only=False,
            preflight_require_paper_ready=True,
        )


def test_paper_ready_preflight_lists_missing_real_artifacts_before_cuda_or_hardware(
    tmp_path: Path,
) -> None:
    inference = RTCInferenceConfig(
        rtc=RTCConfig(enabled=True, execution_horizon=20),
        mode="trained_prefix",
        timing_mode="actual_consumed",
        guidance_delay_mode="rolling_p95",
        timing_diagnostics=True,
        dynamic_prefill_enabled=True,
        max_prefill_steps=6,
    )
    config = _triton_rollout_config(
        tmp_path,
        preflight_only=True,
        preflight_require_paper_ready=True,
        seed=1000,
        inference=inference,
        time_axis_planner=TimeAxisPlannerConfig(enabled=True, dt_ref=1.0 / 30.0),
        realtime_executor=SmoothExecutorConfig(enabled=True),
        trace=RealtimeTraceConfig(enabled=True, path=str(tmp_path / "trace.jsonl")),
    )

    with (
        patch("lerobot.rollout.context.load_and_validate_pi05_parity_report") as load_parity,
        patch.object(PI05RealtimeVLATritonBackend, "from_export") as load_backend,
        patch("lerobot.rollout.context.make_robot_from_config") as make_robot,
        pytest.raises(ValueError, match="Paper-ready Realtime-VLA V2 preflight") as exc_info,
    ):
        build_rollout_context(config, Event())

    message = str(exc_info.value)
    assert "measured sensor timing calibration artifact" in message
    assert "measured joint velocity/acceleration constraints artifact" in message
    assert "measured actuator calibration artifact" in message
    assert "trained human-labeled speed adapter checkpoint" in message
    assert message.count("\n  - ") == 4
    load_parity.assert_not_called()
    load_backend.assert_not_called()
    make_robot.assert_not_called()


def test_paper_ready_preflight_runs_component_lifecycle_without_hardware(tmp_path: Path) -> None:
    sensor_config = SensorTimingCalibrationArtifactConfig(
        enabled=True,
        path=str(tmp_path / "sensor.json"),
        sha256="c" * 64,
    )
    joint_config = JointConstraintArtifactConfig(
        enabled=True,
        path=str(tmp_path / "joints.json"),
        sha256="d" * 64,
    )
    actuator_config = ActuatorCalibrationArtifactConfig(
        enabled=True,
        path=str(tmp_path / "actuator.json"),
        sha256="e" * 64,
    )
    inference = RTCInferenceConfig(
        rtc=RTCConfig(enabled=True, execution_horizon=20),
        mode="trained_prefix",
        timing_mode="actual_consumed",
        guidance_delay_mode="rolling_p95",
        timing_diagnostics=True,
        dynamic_prefill_enabled=True,
        sensor_timing_calibration=sensor_config,
        max_prefill_steps=6,
    )
    config = _triton_rollout_config(
        tmp_path,
        preflight_only=True,
        preflight_require_paper_ready=True,
        seed=1000,
        inference=inference,
        time_axis_planner=TimeAxisPlannerConfig(
            enabled=True,
            dt_ref=1.0 / 30.0,
            joint_constraints=joint_config,
        ),
        realtime_executor=SmoothExecutorConfig(
            enabled=True,
            actuator_calibration=actuator_config,
        ),
        speed_adapter=SpeedAdapterRuntimeConfig(
            enabled=True,
            checkpoint=str(tmp_path / "speed-adapter"),
        ),
        trace=RealtimeTraceConfig(enabled=True, path=str(tmp_path / "trace.jsonl")),
    )
    fake_backend = SimpleNamespace(
        config=SimpleNamespace(camera_keys=("observation.images.top", "observation.images.wrist")),
        trained_prefix_max=6,
        warmed_prefill_lengths=tuple(range(7)),
    )
    fake_speed_adapter = SimpleNamespace(config=SimpleNamespace(action_dim=6, beta_min=0.25, beta_max=2.0))
    component_checks = (
        "policy_processors",
        "time_axis_planner",
        "realtime_executor",
        "rtc_engine_lifecycle",
        "trace_lifecycle",
    )

    with (
        patch(
            "lerobot.rollout.context.load_and_validate_pi05_parity_report",
            return_value=_validated_parity(),
        ),
        patch("lerobot.rollout.context._resolve_sensor_timing_calibration", return_value=object()),
        patch("lerobot.rollout.context._resolve_joint_constraint_artifact", return_value=object()),
        patch("lerobot.rollout.context.load_actuator_calibration_artifact", return_value=object()),
        patch(
            "lerobot.rollout.context.load_runtime_speed_adapter",
            return_value=(fake_speed_adapter, {}),
        ),
        patch.object(PI05RealtimeVLATritonBackend, "from_export", return_value=fake_backend),
        patch(
            "lerobot.rollout.context._run_paper_ready_component_preflight",
            return_value=component_checks,
        ) as run_components,
        patch("lerobot.rollout.context.make_robot_from_config") as make_robot,
    ):
        result = build_rollout_context(config, Event())

    assert isinstance(result, RolloutPreflightResult)
    assert result.paper_ready_requested is True
    assert result.component_checks == component_checks
    assert result.validated_artifacts == (
        "sensor_timing",
        "joint_constraints",
        "actuator_calibration",
        "speed_adapter",
        "triton_parity",
    )
    run_components.assert_called_once()
    make_robot.assert_not_called()


def test_triton_rollout_static_validation_requires_existing_export(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Triton export weights not found"):
        _triton_rollout_config(
            tmp_path,
            pi05_triton_export_weights=str(tmp_path / "missing.pkl"),
        )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"inference": RTCInferenceConfig()}, "requires RTC mode='trained_prefix'"),
        ({"pi05_triton_weights_sha256": None}, "64-hex weights SHA-256"),
        ({"pi05_triton_tokenizer_path": None}, "requires --pi05_triton_tokenizer_path"),
        (
            {"pi05_triton_parity_report": PI05ParityReportArtifactConfig()},
            "requires a checksum-pinned parity report",
        ),
        ({"use_torch_compile": True}, "owns its CUDA Graph"),
        (
            {"pi05_triton_camera_keys": ["observation.images.wrist", "observation.images.top"]},
            "must exactly match policy image feature order",
        ),
        ({"pi05_tensorrt_prefix_engine": "prefix.plan"}, "cannot be combined"),
    ],
)
def test_invalid_triton_rollout_combinations_fail_closed(
    tmp_path: Path, overrides: dict, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _triton_rollout_config(tmp_path, **overrides)


def test_triton_backend_failure_occurs_before_robot_construction(tmp_path: Path) -> None:
    config = _triton_rollout_config(tmp_path)
    events: list[str] = []

    def fail_backend(_config):
        events.append("triton-backend")
        raise RuntimeError("synthetic Triton startup failure")

    def construct_robot(_config):
        events.append("robot")
        raise AssertionError("robot construction must not happen")

    with (
        patch(
            "lerobot.rollout.context.load_and_validate_pi05_parity_report",
            return_value=_validated_parity(),
        ),
        patch.object(PI05RealtimeVLATritonBackend, "from_export", side_effect=fail_backend),
        patch("lerobot.rollout.context.make_robot_from_config", side_effect=construct_robot),
        pytest.raises(RuntimeError, match="synthetic Triton startup failure"),
    ):
        build_rollout_context(config, Event())

    assert events == ["triton-backend"]


def test_triton_backend_is_ready_before_robot_and_skips_pytorch_policy_load(tmp_path: Path) -> None:
    config = _triton_rollout_config(tmp_path)
    events: list[str] = []
    fake_backend = SimpleNamespace(
        config=SimpleNamespace(camera_keys=("observation.images.top", "observation.images.wrist")),
        trained_prefix_max=6,
    )

    def ready_backend(_config):
        events.append("triton-ready")
        return fake_backend

    def construct_robot(_config):
        events.append("robot")
        raise RuntimeError("stop after proving build order")

    with (
        patch(
            "lerobot.rollout.context.load_and_validate_pi05_parity_report",
            return_value=_validated_parity(),
        ),
        patch.object(PI05RealtimeVLATritonBackend, "from_export", side_effect=ready_backend),
        patch(
            "lerobot.rollout.context.get_policy_class", side_effect=AssertionError("must not load PyTorch")
        ),
        patch("lerobot.rollout.context.make_robot_from_config", side_effect=construct_robot),
        pytest.raises(RuntimeError, match="stop after proving build order"),
    ):
        build_rollout_context(config, Event())

    assert events == ["triton-ready", "robot"]


def test_triton_software_preflight_warms_every_prefix_and_never_constructs_hardware(
    tmp_path: Path,
) -> None:
    config = _triton_rollout_config(tmp_path, preflight_only=True)
    events: list[str] = []
    fake_backend = SimpleNamespace(
        config=SimpleNamespace(camera_keys=("observation.images.top", "observation.images.wrist")),
        trained_prefix_max=6,
        warmed_prefill_lengths=tuple(range(7)),
    )

    def ready_backend(triton_config):
        assert triton_config.weights_sha256 == "a" * 64
        events.append("triton-ready")
        return fake_backend

    with (
        patch(
            "lerobot.rollout.context.load_and_validate_pi05_parity_report",
            return_value=_validated_parity(),
        ),
        patch.object(PI05RealtimeVLATritonBackend, "from_export", side_effect=ready_backend),
        patch(
            "lerobot.rollout.context.get_policy_class", side_effect=AssertionError("must not load PyTorch")
        ),
        patch("lerobot.rollout.context.make_default_processors") as make_processors,
        patch("lerobot.rollout.context.make_robot_from_config") as make_robot,
    ):
        result = build_rollout_context(config, Event())

    assert isinstance(result, RolloutPreflightResult)
    assert result.action_backend == "triton"
    assert result.trained_prefix_max == 6
    assert result.warmed_prefix_lengths == tuple(range(7))
    assert "triton_parity" in result.validated_artifacts
    assert events == ["triton-ready"]
    make_processors.assert_not_called()
    make_robot.assert_not_called()


def test_triton_software_preflight_rejects_incomplete_prefix_warmup_before_hardware(
    tmp_path: Path,
) -> None:
    config = _triton_rollout_config(tmp_path, preflight_only=True)
    fake_backend = SimpleNamespace(
        config=SimpleNamespace(camera_keys=("observation.images.top", "observation.images.wrist")),
        trained_prefix_max=6,
        warmed_prefill_lengths=(0, 1),
    )

    with (
        patch(
            "lerobot.rollout.context.load_and_validate_pi05_parity_report",
            return_value=_validated_parity(),
        ),
        patch.object(PI05RealtimeVLATritonBackend, "from_export", return_value=fake_backend),
        patch("lerobot.rollout.context.make_robot_from_config") as make_robot,
        pytest.raises(RuntimeError, match="did not warm every trained prefix length"),
    ):
        build_rollout_context(config, Event())

    make_robot.assert_not_called()


@pytest.mark.parametrize("duration", [0.0, 60.0])
def test_rollout_preflight_cli_exits_normally_without_strategy_or_visualization(duration: float) -> None:
    from lerobot.scripts import lerobot_rollout as rollout_module

    cfg = SimpleNamespace(
        seed=None,
        preflight_only=True,
        display_data=True,
        strategy=SimpleNamespace(type="base"),
        duration=duration,
    )
    result = RolloutPreflightResult(
        policy_type="pi05",
        device="cuda",
        action_backend="triton",
        trained_prefix_max=6,
        warmed_prefix_lengths=tuple(range(7)),
        validated_artifacts=(
            "sensor_timing",
            "joint_constraints",
            "actuator_calibration",
            "speed_adapter",
        ),
    )

    with (
        patch.object(rollout_module, "init_logging"),
        patch.object(
            rollout_module,
            "ProcessSignalHandler",
            return_value=SimpleNamespace(shutdown_event=Event()),
        ),
        patch.object(rollout_module, "build_rollout_context", return_value=result) as build_context,
        patch.object(rollout_module, "create_strategy") as create_strategy,
        patch.object(rollout_module, "init_visualization") as init_visualization,
        patch.object(rollout_module, "shutdown_visualization") as shutdown_visualization,
    ):
        rollout_module.rollout.__wrapped__(cfg)

    build_context.assert_called_once()
    create_strategy.assert_not_called()
    init_visualization.assert_not_called()
    shutdown_visualization.assert_not_called()


def test_triton_backend_allows_non_dynamic_trained_prefix_ablation(tmp_path: Path) -> None:
    inference = RTCInferenceConfig(
        rtc=RTCConfig(enabled=True, execution_horizon=10),
        mode="trained_prefix",
        timing_mode="actual_consumed",
        guidance_delay_mode="fixed",
        fixed_guidance_delay_steps=5,
        dynamic_prefill_enabled=False,
    )

    config = _triton_rollout_config(tmp_path, inference=inference)

    assert config.pi05_action_backend is PI05ActionBackend.TRITON
    assert config.inference.dynamic_prefill_enabled is False


def test_triton_backend_rejects_legacy_trained_prefix(tmp_path: Path) -> None:
    inference = RTCInferenceConfig(
        rtc=RTCConfig(enabled=True, execution_horizon=10),
        mode="trained_prefix",
        timing_mode="legacy",
        guidance_delay_mode="legacy_max",
    )

    with pytest.raises(ValueError, match="requires RTC timing_mode='actual_consumed'"):
        _triton_rollout_config(tmp_path, inference=inference)
