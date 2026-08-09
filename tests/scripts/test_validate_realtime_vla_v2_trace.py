from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from lerobot.scripts.lerobot_validate_realtime_vla_v2_trace import (
    AcceptanceThresholds,
    TraceValidationError,
    load_trace_sessions,
    main,
    validate_trace,
)


def _paper_ready_config() -> dict:
    return {
        "pi05_action_backend": "triton",
        "pi05_triton_weights_sha256": "66" * 32,
        "pi05_triton_parity_report": {
            "enabled": True,
            "path": "/offline/pi05-triton-parity.json",
            "sha256": "77" * 32,
        },
        "resolved_pi05_triton_parity_report": {
            "path": "/offline/pi05-triton-parity.json",
            "sha256": "77" * 32,
            "checkpoint_path": "/offline/pi05-rtc6-checkpoint",
            "checkpoint_config_sha256": "88" * 32,
            "checkpoint_model_sha256": "99" * 32,
            "triton_weights_sha256": "66" * 32,
            "triton_model_config_sha256": "88" * 32,
            "training_max_delay": 6,
            "thresholds": {"atol": 0.08, "rtol": 0.02},
            "prefixes": list(range(7)),
            "max_abs_errors": [0.01] * 7,
            "passed": True,
        },
        "policy": {
            "pretrained_path": "/offline/pi05-rtc6-checkpoint",
            "rtc_training_max_delay": 6,
        },
        "speed_adapter": {
            "enabled": True,
            "resolved_checkpoint": {
                "output_contract": {"beta_min": 0.5, "beta_max": 2.0},
                "provenance": {"trained_from_collected_throttle_data": True},
                "weights": {"sha256": "11" * 32},
            },
        },
        "realtime_executor": {
            "enabled": True,
            "actuator_calibration": {
                "enabled": True,
                "path": "/offline/actuator.json",
                "sha256": "22" * 32,
            },
            "resolved_actuator_calibration": {
                "artifact_sha256": "22" * 32,
                "source_sha256": "33" * 32,
                "source_sample_count": 500,
                "fit_boundary_status": [
                    {
                        "joint_name": name,
                        "at_delay_bound": False,
                        "at_tau_bound": False,
                    }
                    for name in (
                        "shoulder_pan",
                        "shoulder_lift",
                        "elbow_flex",
                        "wrist_flex",
                        "wrist_roll",
                        "gripper",
                    )
                ],
            },
        },
        "inference": {
            "type": "rtc",
            "mode": "trained_prefix",
            "dynamic_prefill_enabled": True,
            "max_prefill_steps": 6,
            "sensor_timing_calibration": {
                "enabled": True,
                "path": "/offline/sensor.json",
                "sha256": "44" * 32,
            },
            "camera_capture_delay_s": {"top": 0.02, "wrist": 0.03},
            "image_capture_delay_s": 0.03,
            "state_observation_delay_s": 0.01,
            "max_camera_skew_s": 0.02,
            "resolved_sensor_timing_calibration": {
                "artifact_sha256": "44" * 32,
                "source_sha256": "44" * 32,
                "method": "offline monotonic cross-correlation",
                "sample_count": 600,
                "camera_keys": ["top", "wrist"],
                "camera_capture_delay_s": {"top": 0.02, "wrist": 0.03},
                "image_capture_delay_s": 0.03,
                "state_observation_delay_s": 0.01,
                "max_camera_skew_s": 0.02,
            },
        },
        "time_axis_planner": {
            "enabled": True,
            "dt_min": 0.02,
            "dt_max": 0.10,
            "joint_constraints": {
                "enabled": True,
                "path": "/offline/joints.json",
                "sha256": "55" * 32,
            },
            "max_velocity": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            "max_acceleration": [2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
            "resolved_joint_constraint_provenance": {
                "artifact_sha256": "55" * 32,
                "source_sha256": "55" * 32,
                "method": "offline differentiated joint trace quantiles",
                "sample_count": 600,
                "coordinate_space": "robot_action_units",
                "joint_names": [
                    "shoulder_pan",
                    "shoulder_lift",
                    "elbow_flex",
                    "wrist_flex",
                    "wrist_roll",
                    "gripper",
                ],
                "max_velocity": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                "max_acceleration": [2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
            },
        },
        "trace": {"enabled": True, "path": "/offline/trace.jsonl"},
    }


def _record(sequence: int, event: str, monotonic_timestamp: float, **fields) -> dict:
    return {
        "event": event,
        "session_id": "session-a",
        "sequence": sequence,
        **fields,
        "timestamp": monotonic_timestamp,
        "clock_domain": "monotonic",
        "monotonic_timestamp": monotonic_timestamp,
        "monotonic_clock_domain": "monotonic",
        "wall_timestamp": 1_800_000_000.0 + monotonic_timestamp,
        "wall_clock_domain": "unix",
    }


def _valid_records(config: dict | None = None) -> list[dict]:
    return [
        _record(
            0,
            "session_start",
            10.0,
            schema_version=2,
            config_snapshot=config or _paper_ready_config(),
        ),
        _record(
            1,
            "smooth_execution",
            10.1,
            heartbeat_timestamp=10.1,
            applied_command=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            underrun=False,
            underrun_count=0,
        ),
        _record(
            2,
            "inference_chunk",
            10.2,
            inference_started_at=10.05,
            inference_finished_at=10.17,
            planner_fallback=False,
            planner_segment_durations=[0.04, 0.05],
            planner_reference_segment_durations=[0.04, 0.05],
            planner_speed_factors=[1.0, 1.2],
        ),
        _record(
            3,
            "queue_merge",
            10.21,
            generation_before=0,
            generation_after_inference=0,
            generation_after_merge=1,
            queue_before=10,
            queue_after_inference=6,
            queue_after_merge=20,
            actual_consumed_steps=4,
            merge_skip=4,
        ),
        _record(
            4,
            "smooth_execution",
            10.3,
            heartbeat_timestamp=10.3,
            applied_command=[0.1, -0.1, 0.0, 0.0, 0.0, 0.0],
            underrun=False,
            underrun_count=0,
        ),
        _record(
            5,
            "inference_chunk",
            10.4,
            inference_started_at=10.25,
            inference_finished_at=10.35,
            planner_fallback=False,
            planner_segment_durations=[0.03, 0.04],
            planner_reference_segment_durations=[0.03, 0.04],
            planner_speed_factors=[1.4, 1.6],
        ),
        _record(
            6,
            "queue_merge",
            10.41,
            generation_before=1,
            generation_after_inference=1,
            generation_after_merge=2,
            queue_before=12,
            queue_after_inference=9,
            queue_after_merge=20,
            actual_consumed_steps=3,
            merge_skip=3,
        ),
        _record(
            7,
            "smooth_execution",
            10.5,
            heartbeat_timestamp=10.5,
            applied_command=[0.15, -0.15, 0.0, 0.0, 0.0, 0.0],
            underrun=False,
            underrun_count=0,
        ),
        _record(
            8,
            "session_end",
            10.6,
            status="completed",
            reason=None,
            records_before_end=8,
        ),
    ]


def _write_trace(path: Path, records: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(record, separators=(",", ":")) + "\n" for record in records),
        encoding="utf-8",
    )


def _resequence(records: list[dict]) -> list[dict]:
    records.sort(key=lambda record: record["monotonic_timestamp"])
    for sequence, record in enumerate(records):
        record["sequence"] = sequence
    if records[-1]["event"] == "session_end":
        records[-1]["records_before_end"] = records[-1]["sequence"]
    return records


def _strict_thresholds(*, require_paper_ready: bool = True) -> AcceptanceThresholds:
    return AcceptanceThresholds(
        min_inference_count=2,
        max_inference_p95_ms=125.0,
        max_inference_max_ms=125.0,
        max_planner_fallbacks=0,
        min_queue_merges=2,
        max_queue_stale=0,
        max_queue_underflows=0,
        max_smooth_underruns=0,
        max_command_velocity=1.0,
        max_command_acceleration=2.0,
        max_boundary_jump=0.11,
        require_paper_ready=require_paper_ready,
    )


def test_paper_ready_trace_passes_and_reports_requested_metrics(tmp_path: Path) -> None:
    trace_path = tmp_path / "trace.jsonl"
    _write_trace(trace_path, _valid_records())

    report = validate_trace(trace_path, thresholds=_strict_thresholds())

    assert report["overall_pass"] is True
    session = report["sessions"][0]
    assert session["paper_ready"]["pass"] is True
    assert session["metrics"]["inference_latency_ms"]["p50"] == pytest.approx(110.0)
    assert session["metrics"]["inference_latency_ms"]["p95"] == pytest.approx(119.0)
    assert session["metrics"]["planner"]["fallback_count"] == 0
    assert session["metrics"]["planner"]["speed_beta"]["max"] == 1.6
    assert session["metrics"]["queue"] == {
        "merge_count": 2,
        "stale_count": 0,
        "underflow_count": 0,
        "merge_anomaly_count": 0,
        "min_queue_after_merge": 20,
    }
    assert session["metrics"]["smooth_executor"]["underrun_count"] == 0
    assert session["metrics"]["commands"]["velocity_abs"]["max"] == pytest.approx(0.5)
    assert session["metrics"]["commands"]["acceleration_abs"]["max"] == pytest.approx(1.25)
    assert session["metrics"]["commands"]["boundary_jump_abs"]["max"] == pytest.approx(0.1)
    assert session["metrics"]["commands"]["source"] == "smooth_execution.applied_command"


@pytest.mark.parametrize(
    ("terminal_status", "expected_issue"),
    [
        (None, "trace session is incomplete because session_end is missing"),
        ("abnormal", "trace session ended with non-success status 'abnormal'"),
    ],
)
def test_paper_ready_rejects_incomplete_or_abnormal_session(
    terminal_status: str | None,
    expected_issue: str,
    tmp_path: Path,
) -> None:
    records = _valid_records()
    if terminal_status is None:
        records.pop()
    else:
        records[-1].update(status=terminal_status, reason="executor fatal")
    trace_path = tmp_path / "not-completed.jsonl"
    _write_trace(trace_path, records)

    report = validate_trace(
        trace_path,
        thresholds=AcceptanceThresholds(require_paper_ready=True),
    )

    session = report["sessions"][0]
    assert report["overall_pass"] is False
    assert expected_issue in session["paper_ready"]["issues"]
    complete_check = next(check for check in session["checks"] if check["name"] == "complete_trace_session")
    assert complete_check["pass"] is False


def test_dispatch_applied_command_takes_priority_over_smooth_command(tmp_path: Path) -> None:
    records = _valid_records()
    for record in records:
        if record["event"] == "smooth_execution":
            record.pop("applied_command")
            record["command"] = [100.0] * 6
    records.extend(
        [
            _record(-1, "dispatch", 10.11, applied_command=[0.0] * 6),
            _record(-1, "dispatch", 10.31, applied_command=[0.1, -0.1, 0.0, 0.0, 0.0, 0.0]),
            _record(-1, "dispatch", 10.51, applied_command=[0.15, -0.15, 0.0, 0.0, 0.0, 0.0]),
        ]
    )
    trace_path = tmp_path / "dispatch-priority.jsonl"
    _write_trace(trace_path, _resequence(records))

    report = validate_trace(trace_path, thresholds=_strict_thresholds())

    assert report["overall_pass"] is True
    commands = report["sessions"][0]["metrics"]["commands"]
    assert commands["source"] == "dispatch.applied_command"
    assert commands["velocity_abs"]["max"] == pytest.approx(0.5)


def test_calibrated_command_limits_apply_without_cli_thresholds(tmp_path: Path) -> None:
    records = _valid_records()
    records[4]["applied_command"][5] = 0.25
    trace_path = tmp_path / "calibrated-command-violation.jsonl"
    _write_trace(trace_path, records)

    report = validate_trace(
        trace_path,
        thresholds=AcceptanceThresholds(require_paper_ready=True),
    )

    assert report["overall_pass"] is False
    session = report["sessions"][0]
    failed = {check["name"] for check in session["checks"] if not check["pass"]}
    assert {
        "calibrated_command_velocity",
        "calibrated_command_acceleration",
        "calibrated_chunk_boundary",
    } <= failed
    assert not {"command_velocity", "command_acceleration", "chunk_boundary_jump"} & {
        check["name"] for check in session["checks"]
    }
    calibrated = session["metrics"]["commands"]["calibrated_constraints"]
    assert calibrated["velocity_violation_count"] > 0
    assert calibrated["acceleration_violation_count"] > 0
    assert calibrated["boundary_violation_count"] > 0
    assert calibrated["per_joint"]["gripper"]["max_abs_velocity"] == pytest.approx(1.25)


@pytest.mark.parametrize(
    ("mutation", "expected_issue"),
    [
        (
            lambda config: config.update(pi05_action_backend="pytorch"),
            "pi05_action_backend is not 'triton'",
        ),
        (
            lambda config: config["resolved_pi05_triton_parity_report"].update(
                path="/offline/tampered-parity.json"
            ),
            "configured and resolved PI0.5 Triton parity report paths differ",
        ),
        (
            lambda config: config["resolved_pi05_triton_parity_report"].update(sha256="aa" * 32),
            "configured and resolved PI0.5 Triton parity report SHA-256 values differ",
        ),
        (
            lambda config: config["resolved_pi05_triton_parity_report"].update(prefixes=list(range(6))),
            "resolved PI0.5 parity prefixes do not cover 0..training_max_delay exactly",
        ),
        (
            lambda config: config["pi05_triton_parity_report"].pop("path"),
            "configured PI0.5 Triton parity report path is missing",
        ),
        (
            lambda config: config["policy"].pop("rtc_training_max_delay"),
            "policy rtc_training_max_delay is missing or invalid",
        ),
        (
            lambda config: config.pop("resolved_pi05_triton_parity_report"),
            "resolved PI0.5 Triton parity provenance is missing",
        ),
    ],
)
def test_paper_ready_rejects_tampered_pi05_triton_parity_provenance(
    mutation,
    expected_issue: str,
    tmp_path: Path,
) -> None:
    config = copy.deepcopy(_paper_ready_config())
    mutation(config)
    trace_path = tmp_path / "tampered-parity.jsonl"
    _write_trace(trace_path, _valid_records(config))

    report = validate_trace(
        trace_path,
        thresholds=AcceptanceThresholds(require_paper_ready=True),
    )

    assert report["overall_pass"] is False
    assert expected_issue in report["sessions"][0]["paper_ready"]["issues"]


def test_threshold_and_paper_ready_failures_are_reported_not_hidden(tmp_path: Path) -> None:
    config = _paper_ready_config()
    config["speed_adapter"] = {"enabled": False}
    records = _valid_records(config)
    records[2]["planner_fallback"] = True
    trace_path = tmp_path / "trace.jsonl"
    _write_trace(trace_path, records)

    thresholds = AcceptanceThresholds(
        max_inference_p95_ms=50.0,
        max_planner_fallbacks=0,
        require_paper_ready=True,
    )
    report = validate_trace(trace_path, thresholds=thresholds)

    assert report["overall_pass"] is False
    session = report["sessions"][0]
    failed = {check["name"] for check in session["checks"] if not check["pass"]}
    assert {"inference_latency_p95_ms", "planner_fallbacks", "paper_ready_runtime"} <= failed
    assert "speed_adapter.enabled is not true" in session["paper_ready"]["issues"]
    assert "speed adapter resolved checkpoint provenance is missing" in session["paper_ready"]["issues"]


def test_paper_ready_rejects_unproven_sensor_timing_and_joint_constraints(tmp_path: Path) -> None:
    config = _paper_ready_config()
    del config["inference"]["resolved_sensor_timing_calibration"]
    del config["time_axis_planner"]["resolved_joint_constraint_provenance"]
    trace_path = tmp_path / "unproven.jsonl"
    _write_trace(trace_path, _valid_records(config))

    report = validate_trace(
        trace_path,
        thresholds=AcceptanceThresholds(require_paper_ready=True),
    )

    assert report["overall_pass"] is False
    issues = report["sessions"][0]["paper_ready"]["issues"]
    assert "resolved sensor timing calibration provenance is missing" in issues
    assert "resolved joint velocity/acceleration constraint provenance is missing" in issues


@pytest.mark.parametrize(
    ("mutation", "expected_issue"),
    [
        (
            lambda config: config["inference"]["sensor_timing_calibration"].update(sha256="99" * 32),
            "sensor timing configured and resolved SHA-256 values differ",
        ),
        (
            lambda config: config["inference"].update(state_observation_delay_s=0.5),
            "sensor timing resolved state_observation_delay_s differs from RTC runtime value",
        ),
        (
            lambda config: config["time_axis_planner"]["joint_constraints"].update(sha256="99" * 32),
            "joint constraint configured and resolved SHA-256 values differ",
        ),
        (
            lambda config: config["time_axis_planner"].update(max_velocity=[9.0, 9.0, 9.0, 9.0, 9.0, 9.0]),
            "resolved joint velocity constraints differ from QP runtime values",
        ),
    ],
)
def test_paper_ready_rejects_calibration_hash_or_runtime_value_mismatch(
    tmp_path: Path, mutation, expected_issue: str
) -> None:
    config = copy.deepcopy(_paper_ready_config())
    mutation(config)
    trace_path = tmp_path / "calibration-mismatch.jsonl"
    _write_trace(trace_path, _valid_records(config))

    report = validate_trace(
        trace_path,
        thresholds=AcceptanceThresholds(require_paper_ready=True),
    )

    assert report["overall_pass"] is False
    assert expected_issue in report["sessions"][0]["paper_ready"]["issues"]


@pytest.mark.parametrize("corruption", ["sequence", "monotonic"])
def test_strict_timeline_validation_rejects_corruption(corruption: str, tmp_path: Path) -> None:
    records = _valid_records()
    if corruption == "sequence":
        records[1]["sequence"] = 2
    else:
        records[1]["timestamp"] = 10.0
        records[1]["monotonic_timestamp"] = 10.0
    trace_path = tmp_path / "bad.jsonl"
    _write_trace(trace_path, records)

    with pytest.raises(TraceValidationError, match="sequence is not contiguous|strictly increasing"):
        load_trace_sessions(trace_path)


def test_cli_writes_json_and_returns_pass_fail_exit_code(tmp_path: Path) -> None:
    trace_path = tmp_path / "trace.jsonl"
    report_path = tmp_path / "report.json"
    _write_trace(trace_path, _valid_records())

    exit_code = main(
        [
            "--input",
            str(trace_path),
            "--output",
            str(report_path),
            "--max-inference-p95-ms",
            "125",
            "--min-queue-merges",
            "2",
            "--max-command-velocity",
            "1",
            "--require-paper-ready",
        ]
    )

    assert exit_code == 0
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["report_type"] == "lerobot.realtime_vla_v2_trace_acceptance"
    assert report["overall_pass"] is True
