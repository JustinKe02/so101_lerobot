from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from lerobot.rollout import context as rollout_context
from lerobot.rollout.actuator_calibration import (
    ActuatorCalibrationError,
    load_actuator_calibration_artifact,
    normalize_action_joint_names,
)
from lerobot.rollout.configs import (
    ActuatorCalibrationArtifactConfig,
    SmoothExecutorConfig,
)
from lerobot.rollout.trajectory import RealtimeTraceWriter

JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]
ACTION_KEYS = [f"{name}.pos" for name in JOINT_NAMES]


def _artifact_document() -> dict:
    return {
        "schema_version": 1,
        "artifact_type": "lerobot.so101_actuator_calibration",
        "clock_domain": "monotonic",
        "joint_names": JOINT_NAMES,
        "command_delay_s": [0.01, 0.02, 0.03, 0.04, 0.05, 0.06],
        "tau_s": [0.08, 0.09, 0.10, 0.11, 0.12, 0.13],
        "model": {
            "type": "pure_delay_first_order",
            "equation": "dx/dt = (command(t - delay) - x) / tau",
            "command_interpolation": "zero_order_hold",
            "residual": "one_step_prediction",
        },
        "fit_diagnostics": [
            {
                "joint_name": name,
                "residual_rmse": 0.01 + index * 0.001,
                "residual_mae": 0.008,
                "residual_rss": 0.1,
                "r_squared": 0.99,
                "sample_count": 180,
                "command_span": 25.0,
                "optimizer_nfev": 20,
                "optimizer_status": 1,
                "at_delay_bound": index == 0,
                "at_tau_bound": False,
            }
            for index, name in enumerate(JOINT_NAMES)
        ],
        "fit_config": {
            "max_delay_s": 0.30,
            "min_tau_s": 0.01,
            "max_tau_s": 1.0,
            "min_fit_samples": 30,
            "min_command_span": 0.001,
        },
        "source": {
            "path": "/offline/trace.jsonl",
            "format": "jsonl",
            "sha256": "ab" * 32,
            "sample_count": 200,
        },
    }


def _write_artifact(path: Path, document: dict | None = None) -> str:
    payload = json.dumps(document or _artifact_document(), separators=(",", ":")).encode()
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def test_strict_loader_returns_immutable_parameters_and_audit_snapshot(tmp_path: Path) -> None:
    path = tmp_path / "actuator.json"
    sha256 = _write_artifact(path)

    artifact = load_actuator_calibration_artifact(
        path,
        expected_sha256=sha256,
        action_dim=6,
        joint_names=normalize_action_joint_names(ACTION_KEYS),
    )

    np.testing.assert_allclose(artifact.command_delay_s, [0.01, 0.02, 0.03, 0.04, 0.05, 0.06])
    np.testing.assert_allclose(artifact.tau_s, [0.08, 0.09, 0.10, 0.11, 0.12, 0.13])
    assert artifact.command_delay_s.flags.writeable is False
    assert artifact.tau_s.flags.writeable is False
    audit = artifact.audit_snapshot()
    assert audit["artifact_sha256"] == sha256
    assert audit["source_sha256"] == "ab" * 32
    assert audit["fit_boundary_status"][0] == {
        "joint_name": "shoulder_pan",
        "sample_count": 180,
        "residual_rmse": 0.01,
        "at_delay_bound": True,
        "at_tau_bound": False,
    }


def test_calibration_audit_is_serialized_in_session_start(tmp_path: Path) -> None:
    artifact_path = tmp_path / "actuator.json"
    sha256 = _write_artifact(artifact_path)
    artifact = load_actuator_calibration_artifact(
        artifact_path,
        expected_sha256=sha256,
        action_dim=6,
        joint_names=JOINT_NAMES,
    )
    trace_path = tmp_path / "session.jsonl"
    snapshot = {
        "realtime_executor": {
            "resolved_actuator_calibration": artifact.audit_snapshot(),
        }
    }

    trace = RealtimeTraceWriter(trace_path, session_id="calibration-test", config_snapshot=snapshot)
    trace.close()

    session = json.loads(trace_path.read_text(encoding="utf-8").splitlines()[0])
    audit = session["config_snapshot"]["realtime_executor"]["resolved_actuator_calibration"]
    assert audit["artifact_sha256"] == sha256
    assert audit["source_sha256"] == "ab" * 32
    assert audit["fit_boundary_status"][0]["at_delay_bound"] is True


def test_loader_rejects_missing_artifact_and_hash_mismatch(tmp_path: Path) -> None:
    missing = tmp_path / "missing.json"
    with pytest.raises(ActuatorCalibrationError, match="does not exist"):
        load_actuator_calibration_artifact(missing, expected_sha256="00" * 32)

    path = tmp_path / "actuator.json"
    _write_artifact(path)
    with pytest.raises(ActuatorCalibrationError, match="SHA-256 mismatch"):
        load_actuator_calibration_artifact(path, expected_sha256="00" * 32)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda document: document.update(schema_version=2), "schema_version"),
        (lambda document: document.update(clock_domain="unix"), "clock_domain"),
        (lambda document: document["source"].update(sha256="bad"), "source.sha256"),
        (lambda document: document.update(extra_field=True), "fields do not match schema"),
    ],
)
def test_loader_rejects_invalid_schema(tmp_path: Path, mutation, message: str) -> None:
    document = _artifact_document()
    mutation(document)
    path = tmp_path / "invalid.json"
    sha256 = _write_artifact(path, document)

    with pytest.raises(ActuatorCalibrationError, match=message):
        load_actuator_calibration_artifact(path, expected_sha256=sha256)


def test_loader_rejects_action_dimension_and_joint_order_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "actuator.json"
    sha256 = _write_artifact(path)

    with pytest.raises(ActuatorCalibrationError, match="action_dim mismatch"):
        load_actuator_calibration_artifact(
            path,
            expected_sha256=sha256,
            action_dim=5,
            joint_names=JOINT_NAMES[:5],
        )
    with pytest.raises(ActuatorCalibrationError, match="joint order mismatch"):
        load_actuator_calibration_artifact(
            path,
            expected_sha256=sha256,
            action_dim=6,
            joint_names=[JOINT_NAMES[1], JOINT_NAMES[0], *JOINT_NAMES[2:]],
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"enabled": True}, "path is required"),
        ({"enabled": True, "path": "artifact.json"}, "sha256 is required"),
        (
            {"enabled": True, "path": "artifact.json", "sha256": "not-a-sha"},
            "64 hexadecimal",
        ),
    ],
)
def test_enabled_calibration_config_requires_path_and_sha256(kwargs: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        ActuatorCalibrationArtifactConfig(**kwargs)


def test_calibration_cannot_be_enabled_without_realtime_executor() -> None:
    calibration = ActuatorCalibrationArtifactConfig(
        enabled=True,
        path="artifact.json",
        sha256="00" * 32,
    )
    with pytest.raises(ValueError, match="realtime_executor.enabled"):
        SmoothExecutorConfig(enabled=False, actuator_calibration=calibration)


def test_artifact_is_loaded_before_policy_or_hardware_build(tmp_path: Path) -> None:
    path = tmp_path / "invalid.json"
    path.write_text("{}", encoding="utf-8")
    calibration = SimpleNamespace(enabled=True, path=str(path), sha256="00" * 32)
    cfg = SimpleNamespace(
        inference=SimpleNamespace(),
        realtime_executor=SimpleNamespace(enabled=True, actuator_calibration=calibration),
        policy=SimpleNamespace(action_feature_names=ACTION_KEYS),
    )

    with (
        patch.object(rollout_context, "get_policy_class") as get_policy_class,
        patch.object(rollout_context, "make_robot_from_config") as make_robot,
        pytest.raises(ActuatorCalibrationError, match="SHA-256 mismatch"),
    ):
        rollout_context._build_rollout_context(
            cfg,
            SimpleNamespace(),
            hardware_state=rollout_context._HardwareBuildState(),
        )

    get_policy_class.assert_not_called()
    make_robot.assert_not_called()


def test_target_runtime_config_does_not_claim_unmeasured_calibration() -> None:
    config_path = Path(__file__).parents[2] / "pi05_realtime_vla_v2_40ep_full_rtc15.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))

    assert config["realtime_executor"]["actuator_calibration"] == {"enabled": False}
    assert config["realtime_executor"]["forward_lead_s"] is None
