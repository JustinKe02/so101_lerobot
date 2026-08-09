from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import draccus
import pytest

from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.rollout import context as rollout_context
from lerobot.rollout.configs import SmoothExecutorConfig
from lerobot.rollout.inference.factory import (
    RTCGuidanceDelayMode,
    RTCInferenceConfig,
    RTCInferenceMode,
    RTCTimingMode,
)
from lerobot.rollout.inference.rtc import RTCInferenceEngine
from lerobot.rollout.joint_constraints import (
    JointConstraintArtifactConfig,
    JointConstraintError,
    load_joint_constraint_artifact,
)
from lerobot.rollout.sensor_timing_calibration import (
    SensorTimingCalibrationArtifactConfig,
    SensorTimingCalibrationError,
    load_sensor_timing_calibration_artifact,
)
from lerobot.rollout.time_axis import TimeAxisPlannerConfig
from lerobot.rollout.trajectory import RealtimeTraceWriter

CAMERAS = ("top", "wrist")
JOINTS = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)
ACTION_KEYS = tuple(f"{name}.pos" for name in JOINTS)


def _write_json(path: Path, document: dict) -> str:
    payload = json.dumps(document, separators=(",", ":")).encode()
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def _sensor_document() -> dict:
    return {
        "schema_version": 1,
        "artifact_type": "lerobot.sensor_timing_calibration",
        "clock_domain": "monotonic",
        "camera_keys": list(CAMERAS),
        "camera_capture_delay_s": {"top": 0.020, "wrist": 0.030},
        "state_observation_delay_s": 0.010,
        "max_camera_skew_s": 0.015,
        "method": "paired monotonic LED and bus reference",
        "quantile": 0.99,
        "source": {
            "path": "/offline/sensor.jsonl",
            "format": "jsonl",
            "sha256": "11" * 32,
            "sample_count": 500,
        },
    }


def _joint_document() -> dict:
    return {
        "schema_version": 1,
        "artifact_type": "lerobot.so101_joint_constraints",
        "coordinate_space": "robot_action_units",
        "joint_names": list(JOINTS),
        "max_velocity": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
        "max_acceleration": [100.0, 200.0, 300.0, 400.0, 500.0, 600.0],
        "method": "within-episode finite-difference P95",
        "quantile": 0.95,
        "safety_factor": 1.0,
        "source": {
            "path": "/offline/joints.jsonl",
            "format": "jsonl",
            "sha256": "22" * 32,
            "sample_count": 600,
            "velocity_sample_count": 590,
            "acceleration_sample_count": 580,
        },
    }


def test_strict_loaders_validate_hash_schema_and_layout(tmp_path: Path) -> None:
    sensor_path = tmp_path / "sensor.json"
    sensor_sha = _write_json(sensor_path, _sensor_document())
    sensor = load_sensor_timing_calibration_artifact(
        sensor_path, expected_sha256=sensor_sha, camera_keys=CAMERAS
    )
    assert sensor.camera_delay_by_key == {"top": 0.02, "wrist": 0.03}
    assert sensor.audit_snapshot()["artifact_sha256"] == sensor_sha
    with pytest.raises(SensorTimingCalibrationError, match="SHA-256 mismatch"):
        load_sensor_timing_calibration_artifact(sensor_path, expected_sha256="00" * 32)
    with pytest.raises(SensorTimingCalibrationError, match="camera order mismatch"):
        load_sensor_timing_calibration_artifact(
            sensor_path, expected_sha256=sensor_sha, camera_keys=list(reversed(CAMERAS))
        )

    joint_path = tmp_path / "joint.json"
    joint_sha = _write_json(joint_path, _joint_document())
    joint = load_joint_constraint_artifact(joint_path, expected_sha256=joint_sha, joint_names=JOINTS)
    assert joint.max_velocity == (10.0, 20.0, 30.0, 40.0, 50.0, 60.0)
    with pytest.raises(JointConstraintError, match="order mismatch"):
        load_joint_constraint_artifact(
            joint_path, expected_sha256=joint_sha, joint_names=list(reversed(JOINTS))
        )


@pytest.mark.parametrize(
    "config_type",
    [SensorTimingCalibrationArtifactConfig, JointConstraintArtifactConfig],
)
def test_enabled_artifact_config_requires_path_and_sha(config_type) -> None:
    with pytest.raises(ValueError, match="path is required"):
        config_type(enabled=True)
    with pytest.raises(ValueError, match="sha256 is required"):
        config_type(enabled=True, path="artifact.json")


def _rtc_config(sensor: SensorTimingCalibrationArtifactConfig) -> RTCInferenceConfig:
    return RTCInferenceConfig(
        rtc=RTCConfig(enabled=True, execution_horizon=20),
        mode=RTCInferenceMode.TRAINED_PREFIX,
        timing_mode=RTCTimingMode.ACTUAL_CONSUMED,
        guidance_delay_mode=RTCGuidanceDelayMode.FIXED,
        fixed_guidance_delay_steps=2,
        dynamic_prefill_enabled=True,
        max_prefill_steps=6,
        sensor_timing_calibration=sensor,
    )


def test_context_resolvers_apply_artifact_values_before_runtime_build(tmp_path: Path) -> None:
    sensor_path = tmp_path / "sensor.json"
    sensor_sha = _write_json(sensor_path, _sensor_document())
    joint_path = tmp_path / "joint.json"
    joint_sha = _write_json(joint_path, _joint_document())
    cfg = SimpleNamespace(
        inference=_rtc_config(
            SensorTimingCalibrationArtifactConfig(enabled=True, path=str(sensor_path), sha256=sensor_sha)
        ),
        time_axis_planner=TimeAxisPlannerConfig(
            enabled=True,
            joint_constraints=JointConstraintArtifactConfig(
                enabled=True, path=str(joint_path), sha256=joint_sha
            ),
        ),
        realtime_executor=SmoothExecutorConfig(enabled=True),
        policy=SimpleNamespace(
            type="pi05",
            chunk_size=50,
            rtc_training_max_delay=6,
            action_feature_names=list(ACTION_KEYS),
        ),
        robot=SimpleNamespace(cameras={key: object() for key in CAMERAS}),
        fps=30.0,
    )

    sensor = rollout_context._resolve_sensor_timing_calibration(cfg)
    constraints = rollout_context._resolve_joint_constraint_artifact(cfg)

    assert sensor is not None and constraints is not None
    assert cfg.inference.camera_capture_delay_s == {"top": 0.02, "wrist": 0.03}
    assert cfg.inference.image_capture_delay_s == pytest.approx(0.03)
    assert cfg.inference.state_observation_delay_s == pytest.approx(0.01)
    assert cfg.time_axis_planner.max_velocity == list(constraints.max_velocity)
    assert cfg.realtime_executor.max_acceleration["gripper"] == 600.0


def test_bad_sensor_artifact_fails_before_policy_or_hardware_build(tmp_path: Path) -> None:
    sensor_path = tmp_path / "sensor.json"
    _write_json(sensor_path, _sensor_document())
    cfg = SimpleNamespace(
        inference=_rtc_config(
            SensorTimingCalibrationArtifactConfig(enabled=True, path=str(sensor_path), sha256="00" * 32)
        ),
        policy=SimpleNamespace(
            type="pi05",
            chunk_size=50,
            rtc_training_max_delay=6,
            action_feature_names=list(ACTION_KEYS),
        ),
        robot=SimpleNamespace(cameras={key: object() for key in CAMERAS}),
        fps=30.0,
    )

    with (
        patch.object(rollout_context, "get_policy_class") as get_policy_class,
        patch.object(rollout_context, "make_robot_from_config") as make_robot,
        pytest.raises(SensorTimingCalibrationError, match="SHA-256 mismatch"),
    ):
        rollout_context._build_rollout_context(
            cfg,
            SimpleNamespace(),
            hardware_state=rollout_context._HardwareBuildState(),
        )

    get_policy_class.assert_not_called()
    make_robot.assert_not_called()


def test_bad_joint_artifact_fails_before_policy_or_hardware_build(tmp_path: Path) -> None:
    joint_path = tmp_path / "joint.json"
    _write_json(joint_path, _joint_document())
    cfg = SimpleNamespace(
        inference=SimpleNamespace(),
        time_axis_planner=TimeAxisPlannerConfig(
            enabled=True,
            joint_constraints=JointConstraintArtifactConfig(
                enabled=True, path=str(joint_path), sha256="00" * 32
            ),
        ),
        policy=SimpleNamespace(action_feature_names=list(ACTION_KEYS)),
    )

    with (
        patch.object(rollout_context, "get_policy_class") as get_policy_class,
        patch.object(rollout_context, "make_robot_from_config") as make_robot,
        pytest.raises(JointConstraintError, match="SHA-256 mismatch"),
    ):
        rollout_context._build_rollout_context(
            cfg,
            SimpleNamespace(),
            hardware_state=rollout_context._HardwareBuildState(),
        )

    get_policy_class.assert_not_called()
    make_robot.assert_not_called()


def test_per_camera_delay_aligns_to_oldest_physical_image() -> None:
    engine = RTCInferenceEngine.__new__(RTCInferenceEngine)
    engine._camera_capture_delay_s = {"top": 0.020, "wrist": 0.040}
    engine._image_capture_delay_s = 0.040
    engine._max_camera_skew_s = 0.050

    capture, effective_delay = engine._image_capture_timing(
        {"camera_timestamps": {"top": 10.000, "wrist": 10.010}}
    )

    assert capture == pytest.approx(10.000)
    assert capture - effective_delay == pytest.approx(9.970)

    with pytest.raises(RuntimeError, match="requires finite timestamps"):
        engine._image_capture_timing({"camera_timestamps": {}})


def test_resolved_provenance_round_trips_through_session_start(tmp_path: Path) -> None:
    sensor_path = tmp_path / "sensor.json"
    sensor_sha = _write_json(sensor_path, _sensor_document())
    joint_path = tmp_path / "joint.json"
    joint_sha = _write_json(joint_path, _joint_document())
    sensor = load_sensor_timing_calibration_artifact(sensor_path, expected_sha256=sensor_sha)
    joint = load_joint_constraint_artifact(joint_path, expected_sha256=joint_sha)
    trace_path = tmp_path / "trace.jsonl"
    trace = RealtimeTraceWriter(
        trace_path,
        config_snapshot={
            "inference": {"resolved_sensor_timing_calibration": sensor.audit_snapshot()},
            "time_axis_planner": {"resolved_joint_constraint_provenance": joint.audit_snapshot()},
        },
    )
    trace.close()

    session = json.loads(trace_path.read_text(encoding="utf-8").splitlines()[0])
    assert (
        session["config_snapshot"]["inference"]["resolved_sensor_timing_calibration"]["artifact_sha256"]
        == sensor_sha
    )
    assert (
        session["config_snapshot"]["time_axis_planner"]["resolved_joint_constraint_provenance"][
            "artifact_sha256"
        ]
        == joint_sha
    )


def test_config_is_json_serializable_with_disabled_artifacts() -> None:
    config = {
        "inference": asdict(RTCInferenceConfig()),
        "time_axis_planner": asdict(TimeAxisPlannerConfig()),
    }
    json.dumps(config)


def test_draccus_decodes_nested_artifact_configs() -> None:
    inference = draccus.decode(
        RTCInferenceConfig,
        {
            "rtc": {"enabled": True, "execution_horizon": 20},
            "mode": "trained_prefix",
            "timing_mode": "actual_consumed",
            "guidance_delay_mode": "fixed",
            "fixed_guidance_delay_steps": 2,
            "dynamic_prefill_enabled": True,
            "max_prefill_steps": 6,
            "sensor_timing_calibration": {
                "enabled": True,
                "path": "sensor.json",
                "sha256": "11" * 32,
            },
        },
    )
    planner = draccus.decode(
        TimeAxisPlannerConfig,
        {
            "enabled": True,
            "joint_constraints": {
                "enabled": True,
                "path": "joints.json",
                "sha256": "22" * 32,
            },
        },
    )

    assert isinstance(inference.sensor_timing_calibration, SensorTimingCalibrationArtifactConfig)
    assert isinstance(planner.joint_constraints, JointConstraintArtifactConfig)
