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

"""Offline acceptance analysis for RealtimeTraceWriter schema-v2 JSONL.

The module intentionally imports no rollout, policy, torch, camera, robot, or
motor code. It only reads an existing JSONL trace and writes a JSON report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from bisect import bisect_right
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

TRACE_SCHEMA_VERSION = 2
REPORT_SCHEMA_VERSION = 1
REPORT_TYPE = "lerobot.realtime_vla_v2_trace_acceptance"
_SHA256_HEX = frozenset("0123456789abcdefABCDEF")
_BASE_RECORD_FIELDS = {
    "event",
    "session_id",
    "sequence",
    "timestamp",
    "clock_domain",
    "monotonic_timestamp",
    "monotonic_clock_domain",
    "wall_timestamp",
    "wall_clock_domain",
}


class TraceValidationError(ValueError):
    """Raised when the trace itself violates the schema-v2 timeline contract."""


@dataclass(frozen=True)
class AcceptanceThresholds:
    """Pass/fail thresholds; ``None`` disables an optional numeric limit."""

    min_inference_count: int = 1
    max_inference_p95_ms: float | None = None
    max_inference_max_ms: float | None = None
    max_planner_fallbacks: int = 0
    min_speed_beta: float | None = None
    max_speed_beta: float | None = None
    min_reference_duration_s: float | None = None
    max_reference_duration_s: float | None = None
    min_queue_merges: int = 0
    max_queue_stale: int = 0
    max_queue_underflows: int = 0
    max_smooth_underruns: int = 0
    max_command_velocity: float | None = None
    max_command_acceleration: float | None = None
    max_boundary_jump: float | None = None
    require_paper_ready: bool = False

    def __post_init__(self) -> None:
        for name in (
            "min_inference_count",
            "max_planner_fallbacks",
            "min_queue_merges",
            "max_queue_stale",
            "max_queue_underflows",
            "max_smooth_underruns",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        for name in (
            "max_inference_p95_ms",
            "max_inference_max_ms",
            "min_speed_beta",
            "max_speed_beta",
            "min_reference_duration_s",
            "max_reference_duration_s",
            "max_command_velocity",
            "max_command_acceleration",
            "max_boundary_jump",
        ):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool) or not math.isfinite(float(value)) or float(value) < 0.0
            ):
                raise ValueError(f"{name} must be finite and non-negative when set")
        for lower_name, upper_name in (
            ("min_speed_beta", "max_speed_beta"),
            ("min_reference_duration_s", "max_reference_duration_s"),
        ):
            lower = getattr(self, lower_name)
            upper = getattr(self, upper_name)
            if lower is not None and upper is not None and lower > upper:
                raise ValueError(f"{lower_name} must be <= {upper_name}")


@dataclass(frozen=True)
class TraceSession:
    session_id: str
    records: tuple[Mapping[str, Any], ...]
    config_snapshot: Mapping[str, Any]
    terminal_status: str | None
    terminal_reason: str | None


@dataclass(frozen=True)
class _CommandSample:
    timestamp: float
    command: tuple[float, ...]
    action_keys: tuple[str, ...] | None


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise TraceValidationError(f"duplicate JSON key is not allowed: {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise TraceValidationError(f"non-standard JSON numeric constant is not allowed: {value}")


def _finite_number(value: Any, *, context: str, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TraceValidationError(f"{context} must be a finite JSON number")
    result = float(value)
    if not math.isfinite(result) or (nonnegative and result < 0.0):
        suffix = " and non-negative" if nonnegative else ""
        raise TraceValidationError(f"{context} must be finite{suffix}")
    return result


def _nonnegative_integer(value: Any, *, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise TraceValidationError(f"{context} must be a non-negative integer")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_record(path: Path, line_number: int, line: str) -> Mapping[str, Any]:
    try:
        record = json.loads(
            line,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except json.JSONDecodeError as exc:
        raise TraceValidationError(
            f"{path}:{line_number}: invalid JSON at column {exc.colno}: {exc.msg}"
        ) from exc
    if not isinstance(record, Mapping):
        raise TraceValidationError(f"{path}:{line_number}: trace record must be a JSON object")
    missing = sorted(_BASE_RECORD_FIELDS - set(record))
    if missing:
        raise TraceValidationError(f"{path}:{line_number}: missing base fields: {missing!r}")
    event = record["event"]
    session_id = record["session_id"]
    if not isinstance(event, str) or not event:
        raise TraceValidationError(f"{path}:{line_number}: event must be a non-empty string")
    if not isinstance(session_id, str) or not session_id:
        raise TraceValidationError(f"{path}:{line_number}: session_id must be a non-empty string")
    _nonnegative_integer(record["sequence"], context=f"{path}:{line_number}: sequence")
    timestamp = _finite_number(
        record["timestamp"], context=f"{path}:{line_number}: timestamp", nonnegative=True
    )
    monotonic_timestamp = _finite_number(
        record["monotonic_timestamp"],
        context=f"{path}:{line_number}: monotonic_timestamp",
        nonnegative=True,
    )
    wall_timestamp = _finite_number(
        record["wall_timestamp"],
        context=f"{path}:{line_number}: wall_timestamp",
        nonnegative=True,
    )
    if record["monotonic_clock_domain"] != "monotonic":
        raise TraceValidationError(f"{path}:{line_number}: monotonic_clock_domain must be 'monotonic'")
    if record["wall_clock_domain"] != "unix":
        raise TraceValidationError(f"{path}:{line_number}: wall_clock_domain must be 'unix'")
    clock_domain = record["clock_domain"]
    if clock_domain not in {"monotonic", "unix"}:
        raise TraceValidationError(f"{path}:{line_number}: clock_domain must be 'monotonic' or 'unix'")
    expected_timestamp = monotonic_timestamp if clock_domain == "monotonic" else wall_timestamp
    if timestamp != expected_timestamp:
        raise TraceValidationError(
            f"{path}:{line_number}: timestamp does not match the declared clock domain"
        )
    return record


def load_trace_sessions(path: str | Path) -> tuple[TraceSession, ...]:
    """Strictly parse and validate sequence/timeline invariants for every session."""

    trace_path = Path(path).expanduser().resolve()
    if not trace_path.is_file():
        raise TraceValidationError(f"trace does not exist or is not a file: {trace_path}")
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    with trace_path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            record = _parse_record(trace_path, line_number, line)
            grouped[record["session_id"]].append(record)
    if not grouped:
        raise TraceValidationError("trace contains no records")

    sessions: list[TraceSession] = []
    for session_id, records in grouped.items():
        starts = [record for record in records if record["event"] == "session_start"]
        if len(starts) != 1:
            raise TraceValidationError(
                f"session {session_id!r} must contain exactly one session_start, got {len(starts)}"
            )
        if records[0] is not starts[0] or starts[0]["sequence"] != 0:
            raise TraceValidationError(f"session {session_id!r} must begin with session_start sequence 0")
        if starts[0].get("schema_version") != TRACE_SCHEMA_VERSION:
            raise TraceValidationError(
                f"session {session_id!r} uses unsupported trace schema_version "
                f"{starts[0].get('schema_version')!r}; expected {TRACE_SCHEMA_VERSION}"
            )
        if "config_snapshot" not in starts[0]:
            raise TraceValidationError(f"session {session_id!r} session_start must contain config_snapshot")
        raw_config_snapshot = starts[0]["config_snapshot"]
        if raw_config_snapshot is not None and not isinstance(raw_config_snapshot, Mapping):
            raise TraceValidationError(
                f"session {session_id!r} session_start.config_snapshot must be an object or null"
            )
        config_snapshot = raw_config_snapshot or {}
        ends = [record for record in records if record["event"] == "session_end"]
        if len(ends) > 1:
            raise TraceValidationError(
                f"session {session_id!r} must contain at most one session_end, got {len(ends)}"
            )
        terminal_status: str | None = None
        terminal_reason: str | None = None
        if ends:
            end = ends[0]
            if records[-1] is not end:
                raise TraceValidationError(f"session {session_id!r} session_end must be the final record")
            terminal_status = end.get("status")
            if terminal_status not in {"completed", "abnormal"}:
                raise TraceValidationError(
                    f"session {session_id!r} session_end.status must be 'completed' or 'abnormal'"
                )
            records_before_end = end.get("records_before_end")
            if (
                isinstance(records_before_end, bool)
                or not isinstance(records_before_end, int)
                or records_before_end != end["sequence"]
            ):
                raise TraceValidationError(
                    f"session {session_id!r} session_end.records_before_end must equal its sequence"
                )
            terminal_reason_value = end.get("reason")
            if terminal_reason_value is not None and (
                not isinstance(terminal_reason_value, str) or not terminal_reason_value.strip()
            ):
                raise TraceValidationError(
                    f"session {session_id!r} session_end.reason must be a non-empty string when set"
                )
            terminal_reason = terminal_reason_value
        previous_timestamp: float | None = None
        for expected_sequence, record in enumerate(records):
            if record["sequence"] != expected_sequence:
                raise TraceValidationError(
                    f"session {session_id!r} sequence is not contiguous: "
                    f"expected {expected_sequence}, got {record['sequence']}"
                )
            current_timestamp = float(record["monotonic_timestamp"])
            if previous_timestamp is not None and current_timestamp <= previous_timestamp:
                raise TraceValidationError(
                    f"session {session_id!r} monotonic timestamps must be strictly increasing: "
                    f"sequence {expected_sequence - 1}={previous_timestamp}, "
                    f"sequence {expected_sequence}={current_timestamp}"
                )
            previous_timestamp = current_timestamp
        sessions.append(
            TraceSession(
                session_id=session_id,
                records=tuple(records),
                config_snapshot=config_snapshot,
                terminal_status=terminal_status,
                terminal_reason=terminal_reason,
            )
        )
    return tuple(sessions)


def _percentile(values: Sequence[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _summary(values: Sequence[float]) -> dict[str, float | int | None]:
    return {
        "count": len(values),
        "min": min(values) if values else None,
        "p50": _percentile(values, 0.50),
        "p95": _percentile(values, 0.95),
        "max": max(values) if values else None,
        "mean": sum(values) / len(values) if values else None,
    }


def _event_values(
    records: Sequence[Mapping[str, Any]],
    *,
    event: str,
    field: str,
    errors: list[str],
    positive: bool = False,
) -> list[float]:
    values: list[float] = []
    for record in records:
        if record["event"] != event or record.get(field) is None:
            continue
        raw = record[field]
        if not isinstance(raw, list):
            errors.append(f"{event}.{field} at sequence {record['sequence']} must be an array")
            continue
        for index, item in enumerate(raw):
            try:
                value = _finite_number(
                    item,
                    context=f"{event}.{field}[{index}]",
                    nonnegative=positive,
                )
            except TraceValidationError as exc:
                errors.append(str(exc))
                continue
            if positive and value <= 0.0:
                errors.append(f"{event}.{field}[{index}] must be positive")
                continue
            values.append(value)
    return values


def _nested_mapping(root: Mapping[str, Any], *keys: str) -> Mapping[str, Any] | None:
    value: Any = root
    for key in keys:
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    return value if isinstance(value, Mapping) else None


def _configured_bounds(
    config: Mapping[str, Any], thresholds: AcceptanceThresholds
) -> tuple[float | None, float | None, float | None, float | None]:
    beta_min = thresholds.min_speed_beta
    beta_max = thresholds.max_speed_beta
    output_contract = _nested_mapping(
        config,
        "speed_adapter",
        "resolved_checkpoint",
        "output_contract",
    )
    if output_contract is not None:
        configured_min = output_contract.get("beta_min")
        configured_max = output_contract.get("beta_max")
        if (
            beta_min is None
            and isinstance(configured_min, (int, float))
            and not isinstance(configured_min, bool)
            and math.isfinite(float(configured_min))
            and float(configured_min) >= 0.0
        ):
            beta_min = float(configured_min)
        if (
            beta_max is None
            and isinstance(configured_max, (int, float))
            and not isinstance(configured_max, bool)
            and math.isfinite(float(configured_max))
            and float(configured_max) >= 0.0
        ):
            beta_max = float(configured_max)

    duration_min = thresholds.min_reference_duration_s
    duration_max = thresholds.max_reference_duration_s
    planner_config = _nested_mapping(config, "time_axis_planner")
    if planner_config is not None:
        configured_min = planner_config.get("dt_min")
        configured_max = planner_config.get("dt_max")
        if (
            duration_min is None
            and isinstance(configured_min, (int, float))
            and not isinstance(configured_min, bool)
            and math.isfinite(float(configured_min))
            and float(configured_min) >= 0.0
        ):
            duration_min = float(configured_min)
        if (
            duration_max is None
            and isinstance(configured_max, (int, float))
            and not isinstance(configured_max, bool)
            and math.isfinite(float(configured_max))
            and float(configured_max) >= 0.0
        ):
            duration_max = float(configured_max)
    return beta_min, beta_max, duration_min, duration_max


def _valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str) and len(value) == 64 and all(character in _SHA256_HEX for character in value)
    )


def _same_finite_number(left: Any, right: Any) -> bool:
    return (
        isinstance(left, (int, float))
        and not isinstance(left, bool)
        and isinstance(right, (int, float))
        and not isinstance(right, bool)
        and math.isfinite(float(left))
        and math.isfinite(float(right))
        and math.isclose(float(left), float(right), rel_tol=1e-12, abs_tol=1e-12)
    )


def _same_numeric_sequence(left: Any, right: Any) -> bool:
    return (
        isinstance(left, list)
        and isinstance(right, list)
        and len(left) == len(right)
        and all(_same_finite_number(a, b) for a, b in zip(left, right, strict=True))
    )


def _same_numeric_mapping(left: Any, right: Any) -> bool:
    return (
        isinstance(left, Mapping)
        and isinstance(right, Mapping)
        and set(left) == set(right)
        and all(_same_finite_number(left[key], right[key]) for key in left)
    )


def _pi05_triton_parity_issues(config: Mapping[str, Any]) -> list[str]:
    issues: list[str] = []
    if config.get("pi05_action_backend") != "triton":
        issues.append("pi05_action_backend is not 'triton'")

    configured = _nested_mapping(config, "pi05_triton_parity_report")
    if configured is None or configured.get("enabled") is not True:
        issues.append("pi05_triton_parity_report.enabled is not true")
    configured_sha = configured.get("sha256") if configured is not None else None
    if not _valid_sha256(configured_sha):
        issues.append("configured PI0.5 Triton parity report SHA-256 is missing or invalid")
    configured_path = configured.get("path") if configured is not None else None
    if not isinstance(configured_path, str) or not configured_path:
        issues.append("configured PI0.5 Triton parity report path is missing")

    resolved = _nested_mapping(config, "resolved_pi05_triton_parity_report")
    if resolved is None:
        issues.append("resolved PI0.5 Triton parity provenance is missing")
        return issues

    resolved_sha = resolved.get("sha256")
    if not _valid_sha256(resolved_sha):
        issues.append("resolved PI0.5 Triton parity report SHA-256 is missing or invalid")
    elif _valid_sha256(configured_sha) and str(configured_sha).lower() != str(resolved_sha).lower():
        issues.append("configured and resolved PI0.5 Triton parity report SHA-256 values differ")
    if not isinstance(resolved.get("path"), str) or not resolved["path"]:
        issues.append("resolved PI0.5 Triton parity report path is missing")
    elif isinstance(configured_path, str) and configured_path != resolved["path"]:
        issues.append("configured and resolved PI0.5 Triton parity report paths differ")
    if resolved.get("passed") is not True:
        issues.append("resolved PI0.5 Triton parity report did not pass")

    for field, label in (
        ("checkpoint_config_sha256", "checkpoint config"),
        ("checkpoint_model_sha256", "checkpoint model"),
        ("triton_weights_sha256", "Triton export weights"),
        ("triton_model_config_sha256", "Triton model config"),
    ):
        if not _valid_sha256(resolved.get(field)):
            issues.append(f"resolved PI0.5 parity {label} SHA-256 is missing or invalid")

    configured_weights_sha = config.get("pi05_triton_weights_sha256")
    if not _valid_sha256(configured_weights_sha):
        issues.append("configured Triton export weights SHA-256 is missing or invalid")
    elif (
        _valid_sha256(resolved.get("triton_weights_sha256"))
        and str(configured_weights_sha).lower() != str(resolved["triton_weights_sha256"]).lower()
    ):
        issues.append("configured and resolved Triton export weights SHA-256 values differ")
    if (
        _valid_sha256(resolved.get("checkpoint_config_sha256"))
        and _valid_sha256(resolved.get("triton_model_config_sha256"))
        and str(resolved["checkpoint_config_sha256"]).lower()
        != str(resolved["triton_model_config_sha256"]).lower()
    ):
        issues.append("checkpoint and Triton model config SHA-256 values differ")

    policy = _nested_mapping(config, "policy")
    configured_checkpoint = None
    configured_training_max_delay = None
    if policy is not None:
        configured_checkpoint = policy.get("pretrained_path", policy.get("path"))
        configured_training_max_delay = policy.get("rtc_training_max_delay")
    if not isinstance(configured_checkpoint, str) or not configured_checkpoint:
        issues.append("rollout policy checkpoint path is missing")
    if (
        isinstance(configured_training_max_delay, bool)
        or not isinstance(configured_training_max_delay, int)
        or configured_training_max_delay < 1
    ):
        issues.append("policy rtc_training_max_delay is missing or invalid")
        configured_training_max_delay = None
    resolved_checkpoint = resolved.get("checkpoint_path")
    if not isinstance(resolved_checkpoint, str) or not resolved_checkpoint:
        issues.append("resolved PI0.5 parity checkpoint path is missing")
    elif isinstance(configured_checkpoint, str) and configured_checkpoint != resolved_checkpoint:
        issues.append("rollout policy and resolved PI0.5 parity checkpoint paths differ")

    training_max_delay = resolved.get("training_max_delay")
    if (
        isinstance(training_max_delay, bool)
        or not isinstance(training_max_delay, int)
        or training_max_delay < 1
    ):
        issues.append("resolved PI0.5 parity training_max_delay is invalid")
        training_max_delay = None
    elif configured_training_max_delay is not None and configured_training_max_delay != training_max_delay:
        issues.append("policy and resolved PI0.5 parity training_max_delay values differ")

    prefixes = resolved.get("prefixes")
    expected_prefixes = list(range(training_max_delay + 1)) if training_max_delay is not None else None
    if expected_prefixes is None or prefixes != expected_prefixes:
        issues.append("resolved PI0.5 parity prefixes do not cover 0..training_max_delay exactly")
    max_abs_errors = resolved.get("max_abs_errors")
    if not (
        isinstance(max_abs_errors, list)
        and expected_prefixes is not None
        and len(max_abs_errors) == len(expected_prefixes)
        and all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            and float(value) >= 0.0
            for value in max_abs_errors
        )
    ):
        issues.append("resolved PI0.5 parity max_abs_errors are missing or invalid")
    thresholds = resolved.get("thresholds")
    if not (
        isinstance(thresholds, Mapping)
        and set(thresholds) == {"atol", "rtol"}
        and all(
            isinstance(thresholds[key], (int, float))
            and not isinstance(thresholds[key], bool)
            and math.isfinite(float(thresholds[key]))
            and float(thresholds[key]) >= 0.0
            for key in ("atol", "rtol")
        )
    ):
        issues.append("resolved PI0.5 parity thresholds are missing or invalid")
    return issues


def _paper_ready_issues(
    config: Mapping[str, Any],
    *,
    terminal_status: str | None,
    command_source: str | None,
    calibrated_command_metrics: Mapping[str, Any],
    beta_count: int,
    reference_duration_count: int,
    planner_count: int,
    smooth_count: int,
    queue_merge_count: int,
) -> list[str]:
    issues = _pi05_triton_parity_issues(config)
    if terminal_status is None:
        issues.append("trace session is incomplete because session_end is missing")
    elif terminal_status != "completed":
        issues.append(f"trace session ended with non-success status {terminal_status!r}")
    if command_source not in {
        "dispatch.applied_command",
        "smooth_execution.applied_command",
        "executor_control.applied_command",
    }:
        issues.append("trace contains no validated actual applied-command stream")
    if calibrated_command_metrics.get("available") is not True:
        issues.append("calibrated per-joint command dynamics could not be evaluated")
    else:
        if calibrated_command_metrics.get("boundary_sample_count", 0) < 1:
            issues.append("no applied-command transition spans a chunk boundary")
        if calibrated_command_metrics.get("velocity_violation_count") != 0:
            issues.append("actual applied commands exceed calibrated per-joint velocity limits")
        if calibrated_command_metrics.get("acceleration_violation_count") != 0:
            issues.append("actual applied commands exceed calibrated per-joint acceleration limits")
        if calibrated_command_metrics.get("boundary_violation_count") != 0:
            issues.append("actual applied commands exceed the velocity-derived chunk-boundary limits")
    speed = _nested_mapping(config, "speed_adapter")
    if speed is None or speed.get("enabled") is not True:
        issues.append("speed_adapter.enabled is not true")
    resolved_speed = _nested_mapping(config, "speed_adapter", "resolved_checkpoint")
    if resolved_speed is None:
        issues.append("speed adapter resolved checkpoint provenance is missing")
    else:
        provenance = resolved_speed.get("provenance")
        if (
            not isinstance(provenance, Mapping)
            or provenance.get("trained_from_collected_throttle_data") is not True
        ):
            issues.append("speed adapter is not proven trained from collected throttle data")
        weights = resolved_speed.get("weights")
        if not isinstance(weights, Mapping) or not _valid_sha256(weights.get("sha256")):
            issues.append("speed adapter resolved weights SHA-256 is missing or invalid")
        output_contract = resolved_speed.get("output_contract")
        if not isinstance(output_contract, Mapping):
            issues.append("speed adapter output contract is missing")
        else:
            beta_min = output_contract.get("beta_min")
            beta_max = output_contract.get("beta_max")
            if not (
                isinstance(beta_min, (int, float))
                and not isinstance(beta_min, bool)
                and math.isfinite(float(beta_min))
                and float(beta_min) > 0.0
                and isinstance(beta_max, (int, float))
                and not isinstance(beta_max, bool)
                and math.isfinite(float(beta_max))
                and float(beta_max) > float(beta_min)
            ):
                issues.append("speed adapter beta bounds are missing or invalid")

    executor = _nested_mapping(config, "realtime_executor")
    if executor is None or executor.get("enabled") is not True:
        issues.append("realtime_executor.enabled is not true")
    calibration = _nested_mapping(config, "realtime_executor", "actuator_calibration")
    if calibration is None or calibration.get("enabled") is not True:
        issues.append("actuator_calibration.enabled is not true")
    resolved_calibration = _nested_mapping(
        config,
        "realtime_executor",
        "resolved_actuator_calibration",
    )
    if resolved_calibration is None:
        issues.append("resolved actuator calibration provenance is missing")
    else:
        if not _valid_sha256(resolved_calibration.get("artifact_sha256")):
            issues.append("actuator calibration artifact SHA-256 is missing or invalid")
        if not _valid_sha256(resolved_calibration.get("source_sha256")):
            issues.append("actuator calibration source SHA-256 is missing or invalid")
        boundary = resolved_calibration.get("fit_boundary_status")
        if not isinstance(boundary, list) or not boundary:
            issues.append("actuator calibration fit boundary audit is missing")
        if not (
            isinstance(resolved_calibration.get("source_sample_count"), int)
            and not isinstance(resolved_calibration.get("source_sample_count"), bool)
            and resolved_calibration["source_sample_count"] > 0
        ):
            issues.append("actuator calibration source sample count is missing or invalid")
        if (
            isinstance(calibration, Mapping)
            and _valid_sha256(calibration.get("sha256"))
            and calibration.get("sha256", "").lower()
            != str(resolved_calibration.get("artifact_sha256", "")).lower()
        ):
            issues.append("actuator calibration configured and resolved SHA-256 values differ")

    inference = _nested_mapping(config, "inference")
    if (
        inference is None
        or inference.get("type") != "rtc"
        or inference.get("dynamic_prefill_enabled") is not True
    ):
        issues.append("RTC dynamic prefill is not enabled")
    sensor_timing = (
        inference.get("resolved_sensor_timing_calibration") if isinstance(inference, Mapping) else None
    )
    sensor_config = inference.get("sensor_timing_calibration") if isinstance(inference, Mapping) else None
    if not isinstance(sensor_config, Mapping) or sensor_config.get("enabled") is not True:
        issues.append("sensor_timing_calibration.enabled is not true")
    configured_sensor_sha = sensor_config.get("sha256") if isinstance(sensor_config, Mapping) else None
    if not _valid_sha256(configured_sensor_sha):
        issues.append("sensor timing configured artifact SHA-256 is missing or invalid")
    if not isinstance(sensor_timing, Mapping):
        issues.append("resolved sensor timing calibration provenance is missing")
    else:
        resolved_sensor_sha = sensor_timing.get("artifact_sha256")
        if not _valid_sha256(resolved_sensor_sha):
            issues.append("sensor timing resolved artifact SHA-256 is missing or invalid")
        elif (
            _valid_sha256(configured_sensor_sha)
            and str(configured_sensor_sha).lower() != str(resolved_sensor_sha).lower()
        ):
            issues.append("sensor timing configured and resolved SHA-256 values differ")
        if not _valid_sha256(sensor_timing.get("source_sha256")):
            issues.append("sensor timing calibration source SHA-256 is missing or invalid")
        if not isinstance(sensor_timing.get("method"), str) or not sensor_timing["method"].strip():
            issues.append("sensor timing calibration method is missing")
        if not (
            isinstance(sensor_timing.get("sample_count"), int)
            and not isinstance(sensor_timing.get("sample_count"), bool)
            and sensor_timing["sample_count"] > 0
        ):
            issues.append("sensor timing calibration sample count is missing or invalid")
        camera_keys = sensor_timing.get("camera_keys")
        resolved_camera_delays = sensor_timing.get("camera_capture_delay_s")
        if not (
            isinstance(camera_keys, list)
            and camera_keys
            and all(isinstance(key, str) and key for key in camera_keys)
            and len(set(camera_keys)) == len(camera_keys)
            and isinstance(resolved_camera_delays, Mapping)
            and set(resolved_camera_delays) == set(camera_keys)
            and all(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(float(value))
                and float(value) >= 0.0
                for value in resolved_camera_delays.values()
            )
        ):
            issues.append("resolved per-camera sensor delays are invalid")
        elif not _same_numeric_mapping(inference.get("camera_capture_delay_s"), resolved_camera_delays):
            issues.append("sensor timing resolved camera delays differ from RTC runtime values")
        for runtime_key in (
            "image_capture_delay_s",
            "state_observation_delay_s",
            "max_camera_skew_s",
        ):
            if not _same_finite_number(inference.get(runtime_key), sensor_timing.get(runtime_key)):
                issues.append(f"sensor timing resolved {runtime_key} differs from RTC runtime value")
        if not (
            isinstance(sensor_timing.get("max_camera_skew_s"), (int, float))
            and not isinstance(sensor_timing.get("max_camera_skew_s"), bool)
            and math.isfinite(float(sensor_timing["max_camera_skew_s"]))
            and float(sensor_timing["max_camera_skew_s"]) > 0.0
        ):
            issues.append("sensor timing max_camera_skew_s must be positive")
    planner = _nested_mapping(config, "time_axis_planner")
    if planner is None or planner.get("enabled") is not True:
        issues.append("time_axis_planner.enabled is not true")
    constraints = (
        planner.get("resolved_joint_constraint_provenance") if isinstance(planner, Mapping) else None
    )
    constraint_config = planner.get("joint_constraints") if isinstance(planner, Mapping) else None
    if not isinstance(constraint_config, Mapping) or constraint_config.get("enabled") is not True:
        issues.append("time_axis_planner.joint_constraints.enabled is not true")
    configured_constraint_sha = (
        constraint_config.get("sha256") if isinstance(constraint_config, Mapping) else None
    )
    if not _valid_sha256(configured_constraint_sha):
        issues.append("joint constraint configured artifact SHA-256 is missing or invalid")
    if not isinstance(constraints, Mapping):
        issues.append("resolved joint velocity/acceleration constraint provenance is missing")
    else:
        resolved_constraint_sha = constraints.get("artifact_sha256")
        if not _valid_sha256(resolved_constraint_sha):
            issues.append("joint constraint resolved artifact SHA-256 is missing or invalid")
        elif (
            _valid_sha256(configured_constraint_sha)
            and str(configured_constraint_sha).lower() != str(resolved_constraint_sha).lower()
        ):
            issues.append("joint constraint configured and resolved SHA-256 values differ")
        if not _valid_sha256(constraints.get("source_sha256")):
            issues.append("joint constraint source SHA-256 is missing or invalid")
        if not isinstance(constraints.get("method"), str) or not constraints["method"].strip():
            issues.append("joint constraint calibration method is missing")
        if not (
            isinstance(constraints.get("sample_count"), int)
            and not isinstance(constraints.get("sample_count"), bool)
            and constraints["sample_count"] > 0
        ):
            issues.append("joint constraint sample count is missing or invalid")
        joint_names = constraints.get("joint_names")
        velocity = constraints.get("max_velocity")
        acceleration = constraints.get("max_acceleration")
        if not (
            isinstance(joint_names, list)
            and joint_names
            and all(isinstance(name, str) and name for name in joint_names)
            and len(set(joint_names)) == len(joint_names)
            and isinstance(velocity, list)
            and isinstance(acceleration, list)
            and len(velocity) == len(joint_names)
            and len(acceleration) == len(joint_names)
            and all(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(float(value))
                and float(value) > 0.0
                for value in [*velocity, *acceleration]
            )
        ):
            issues.append("resolved per-joint velocity/acceleration constraints are invalid")
        else:
            if constraints.get("coordinate_space") != "robot_action_units":
                issues.append("joint constraints are not in robot_action_units")
            if not _same_numeric_sequence(planner.get("max_velocity"), velocity):
                issues.append("resolved joint velocity constraints differ from QP runtime values")
            if not _same_numeric_sequence(planner.get("max_acceleration"), acceleration):
                issues.append("resolved joint acceleration constraints differ from QP runtime values")
    trace = _nested_mapping(config, "trace")
    if trace is None or trace.get("enabled") is not True:
        issues.append("trace.enabled is not true")

    if beta_count == 0:
        issues.append("trace contains no speed beta diagnostics")
    if reference_duration_count == 0:
        issues.append("trace contains no planner reference-duration diagnostics")
    if planner_count == 0:
        issues.append("trace contains no planner outcomes")
    if smooth_count == 0:
        issues.append("trace contains no smooth executor heartbeat events")
    if queue_merge_count == 0:
        issues.append("trace contains no queue merge events")
    return issues


def _command_vector(value: Any) -> tuple[tuple[float, ...], tuple[str, ...] | None] | None:
    action_keys: tuple[str, ...] | None = None
    if isinstance(value, Mapping):
        action_keys = tuple(sorted(value))
        items = [value[key] for key in action_keys]
    elif isinstance(value, list):
        items = value
    else:
        return None
    if not items:
        return None
    result: list[float] = []
    for item in items:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            return None
        number = float(item)
        if not math.isfinite(number):
            return None
        result.append(number)
    return tuple(result), action_keys


def _command_samples(
    records: Sequence[Mapping[str, Any]], errors: list[str]
) -> tuple[list[_CommandSample], str | None]:
    source_specs = (
        ("dispatch", "applied_command", "monotonic_timestamp", "dispatch.applied_command"),
        (
            "smooth_execution",
            "applied_command",
            "heartbeat_timestamp",
            "smooth_execution.applied_command",
        ),
        (
            "executor_control",
            "applied_command",
            "scheduled_timestamp",
            "executor_control.applied_command",
        ),
        ("smooth_execution", "command", "heartbeat_timestamp", "smooth_execution.command"),
    )
    samples: list[_CommandSample] = []
    source: str | None = None
    for event, field, timestamp_field, candidate_source in source_specs:
        candidates = [
            record for record in records if record["event"] == event and record.get(field) is not None
        ]
        if not candidates:
            continue
        source = candidate_source
        for record in candidates:
            parsed = _command_vector(record.get(field))
            timestamp_raw = record.get(timestamp_field)
            if (
                parsed is None
                or isinstance(timestamp_raw, bool)
                or not isinstance(timestamp_raw, (int, float))
                or not math.isfinite(float(timestamp_raw))
            ):
                errors.append(
                    f"{candidate_source} sequence {record['sequence']} has invalid command/timestamp"
                )
                continue
            vector, action_keys = parsed
            samples.append(_CommandSample(float(timestamp_raw), vector, action_keys))
        break

    for index in range(1, len(samples)):
        if samples[index].timestamp <= samples[index - 1].timestamp:
            errors.append(f"{source} timestamps must be strictly increasing")
            break
        if len(samples[index].command) != len(samples[0].command):
            errors.append(f"{source} command dimensions are inconsistent")
            break
        if samples[index].action_keys != samples[0].action_keys:
            errors.append(f"{source} action key layouts are inconsistent")
            break
    return samples, source


def _command_dynamics(
    samples: Sequence[_CommandSample], boundary_timestamps: Sequence[float]
) -> dict[str, Any]:
    velocity_samples: list[tuple[float, tuple[float, ...]]] = []
    adjacent_jumps: list[float] = []
    for previous, current in zip(samples, samples[1:], strict=False):
        dt = current.timestamp - previous.timestamp
        if dt <= 0.0 or len(previous.command) != len(current.command):
            continue
        delta = tuple(right - left for left, right in zip(previous.command, current.command, strict=True))
        adjacent_jumps.append(max(abs(value) for value in delta))
        velocity = tuple(value / dt for value in delta)
        velocity_samples.append(((previous.timestamp + current.timestamp) * 0.5, velocity))
    accelerations: list[float] = []
    for previous, current in zip(velocity_samples, velocity_samples[1:], strict=False):
        dt = current[0] - previous[0]
        if dt <= 0.0:
            continue
        accelerations.extend(
            abs((right - left) / dt) for left, right in zip(previous[1], current[1], strict=True)
        )
    velocities = [abs(value) for _, vector in velocity_samples for value in vector]

    sample_times = [sample.timestamp for sample in samples]
    boundary_jumps: list[float] = []
    for boundary in boundary_timestamps:
        right_index = bisect_right(sample_times, boundary)
        if right_index == 0 or right_index >= len(samples):
            continue
        before = samples[right_index - 1].command
        after = samples[right_index].command
        if len(before) == len(after):
            boundary_jumps.append(max(abs(right - left) for left, right in zip(before, after, strict=True)))
    return {
        "sample_count": len(samples),
        "action_dim": len(samples[0].command) if samples else None,
        "action_keys": list(samples[0].action_keys) if samples and samples[0].action_keys else None,
        "velocity_abs": _summary(velocities),
        "acceleration_abs": _summary(accelerations),
        "adjacent_jump_abs": _summary(adjacent_jumps),
        "boundary_jump_abs": _summary(boundary_jumps),
    }


def _calibrated_command_metrics(
    config: Mapping[str, Any],
    samples: Sequence[_CommandSample],
    boundary_timestamps: Sequence[float],
) -> dict[str, Any]:
    unavailable: dict[str, Any] = {
        "available": False,
        "reason": None,
        "velocity_violation_count": 0,
        "acceleration_violation_count": 0,
        "boundary_violation_count": 0,
        "boundary_rule": "abs(delta_q[j]) <= max_velocity[j] * actual_elapsed_s",
    }
    planner = _nested_mapping(config, "time_axis_planner")
    constraints = planner.get("resolved_joint_constraint_provenance") if planner is not None else None
    if not isinstance(constraints, Mapping):
        unavailable["reason"] = "resolved joint constraints are missing"
        return unavailable
    joint_names = constraints.get("joint_names")
    velocity_limits = constraints.get("max_velocity")
    acceleration_limits = constraints.get("max_acceleration")
    if not (
        isinstance(joint_names, list)
        and joint_names
        and len(set(joint_names)) == len(joint_names)
        and all(isinstance(name, str) and name for name in joint_names)
        and isinstance(velocity_limits, list)
        and isinstance(acceleration_limits, list)
        and len(velocity_limits) == len(joint_names)
        and len(acceleration_limits) == len(joint_names)
        and all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            and float(value) > 0.0
            for value in [*velocity_limits, *acceleration_limits]
        )
    ):
        unavailable["reason"] = "resolved joint constraints are invalid"
        return unavailable
    if len(samples) < 3:
        unavailable["reason"] = "at least three applied command samples are required"
        return unavailable

    first_keys = samples[0].action_keys
    if first_keys is None:
        if len(samples[0].command) != len(joint_names):
            unavailable["reason"] = "applied command dimension does not match joint constraints"
            return unavailable
        source_indices = tuple(range(len(joint_names)))
    else:
        normalized_keys = tuple(key.removesuffix(".pos") for key in first_keys)
        if len(set(normalized_keys)) != len(normalized_keys) or set(normalized_keys) != set(joint_names):
            unavailable["reason"] = "applied command keys do not match calibrated joint names"
            return unavailable
        source_indices = tuple(normalized_keys.index(name) for name in joint_names)

    ordered_commands = [tuple(sample.command[index] for index in source_indices) for sample in samples]
    velocity_vectors: list[tuple[float, tuple[float, ...]]] = []
    velocity_violation_count = 0
    velocity_values: list[list[float]] = [[] for _ in joint_names]
    for previous_sample, current_sample, previous, current in zip(
        samples[:-1],
        samples[1:],
        ordered_commands[:-1],
        ordered_commands[1:],
        strict=True,
    ):
        dt = current_sample.timestamp - previous_sample.timestamp
        if dt <= 0.0:
            unavailable["reason"] = "applied command timestamps are not strictly increasing"
            return unavailable
        velocity = tuple((right - left) / dt for left, right in zip(previous, current, strict=True))
        velocity_vectors.append(((previous_sample.timestamp + current_sample.timestamp) * 0.5, velocity))
        for index, value in enumerate(velocity):
            absolute = abs(value)
            velocity_values[index].append(absolute)
            if absolute > float(velocity_limits[index]) * (1.0 + 1e-9) + 1e-12:
                velocity_violation_count += 1

    acceleration_violation_count = 0
    acceleration_values: list[list[float]] = [[] for _ in joint_names]
    for previous, current in zip(velocity_vectors, velocity_vectors[1:], strict=False):
        dt = current[0] - previous[0]
        if dt <= 0.0:
            unavailable["reason"] = "derived velocity timestamps are not strictly increasing"
            return unavailable
        for index, (left, right) in enumerate(zip(previous[1], current[1], strict=True)):
            absolute = abs((right - left) / dt)
            acceleration_values[index].append(absolute)
            if absolute > float(acceleration_limits[index]) * (1.0 + 1e-9) + 1e-12:
                acceleration_violation_count += 1

    sample_times = [sample.timestamp for sample in samples]
    boundary_violation_count = 0
    boundary_ratio_values: list[list[float]] = [[] for _ in joint_names]
    boundary_sample_count = 0
    for boundary in boundary_timestamps:
        right_index = bisect_right(sample_times, boundary)
        if right_index == 0 or right_index >= len(samples):
            continue
        dt = samples[right_index].timestamp - samples[right_index - 1].timestamp
        if dt <= 0.0:
            continue
        boundary_sample_count += 1
        before = ordered_commands[right_index - 1]
        after = ordered_commands[right_index]
        for index, (left, right) in enumerate(zip(before, after, strict=True)):
            allowed_jump = float(velocity_limits[index]) * dt
            ratio = abs(right - left) / allowed_jump
            boundary_ratio_values[index].append(ratio)
            if ratio > 1.0 + 1e-9:
                boundary_violation_count += 1

    return {
        "available": True,
        "reason": None,
        "joint_names": list(joint_names),
        "velocity_limits": [float(value) for value in velocity_limits],
        "acceleration_limits": [float(value) for value in acceleration_limits],
        "velocity_sample_count": len(velocity_vectors),
        "acceleration_sample_count": max(0, len(velocity_vectors) - 1),
        "boundary_sample_count": boundary_sample_count,
        "velocity_violation_count": velocity_violation_count,
        "acceleration_violation_count": acceleration_violation_count,
        "boundary_violation_count": boundary_violation_count,
        "boundary_rule": unavailable["boundary_rule"],
        "per_joint": {
            name: {
                "max_abs_velocity": max(velocity_values[index], default=None),
                "max_velocity_limit": float(velocity_limits[index]),
                "max_abs_acceleration": max(acceleration_values[index], default=None),
                "max_acceleration_limit": float(acceleration_limits[index]),
                "max_boundary_velocity_ratio": max(boundary_ratio_values[index], default=None),
            }
            for index, name in enumerate(joint_names)
        },
    }


def _check(
    checks: list[dict[str, Any]],
    *,
    name: str,
    passed: bool,
    value: Any,
    limit: Any,
    detail: str,
) -> None:
    checks.append(
        {
            "name": name,
            "pass": bool(passed),
            "value": value,
            "limit": limit,
            "detail": detail,
        }
    )


def evaluate_session(session: TraceSession, thresholds: AcceptanceThresholds) -> dict[str, Any]:
    """Compute metrics and pass/fail checks for one validated trace session."""

    errors: list[str] = []
    inference_latencies_ms: list[float] = []
    inference_records = [record for record in session.records if record["event"] == "inference_chunk"]
    for record in inference_records:
        try:
            start = _finite_number(
                record.get("inference_started_at"),
                context=f"inference_chunk sequence {record['sequence']} start",
                nonnegative=True,
            )
            finish = _finite_number(
                record.get("inference_finished_at"),
                context=f"inference_chunk sequence {record['sequence']} finish",
                nonnegative=True,
            )
        except TraceValidationError as exc:
            errors.append(str(exc))
            continue
        if finish < start:
            errors.append(f"inference_chunk sequence {record['sequence']} finishes before it starts")
            continue
        if finish > float(record["monotonic_timestamp"]) + 1e-9:
            errors.append(
                f"inference_chunk sequence {record['sequence']} finish is later than its trace timestamp"
            )
            continue
        inference_latencies_ms.append((finish - start) * 1000.0)

    planner_records = [record for record in inference_records if record.get("planner_fallback") is not None]
    fallback_count = 0
    for record in planner_records:
        fallback = record.get("planner_fallback")
        if not isinstance(fallback, bool):
            errors.append(f"inference_chunk sequence {record['sequence']} planner_fallback must be boolean")
        elif fallback:
            fallback_count += 1
    speed_beta = _event_values(
        inference_records,
        event="inference_chunk",
        field="planner_speed_factors",
        errors=errors,
        positive=True,
    )
    reference_durations = _event_values(
        inference_records,
        event="inference_chunk",
        field="planner_reference_segment_durations",
        errors=errors,
        positive=True,
    )
    segment_durations = _event_values(
        inference_records,
        event="inference_chunk",
        field="planner_segment_durations",
        errors=errors,
        positive=True,
    )

    queue_records = [record for record in session.records if record["event"] == "queue_merge"]
    queue_stale_count = sum(
        record.get("stale") is True
        or record.get("generation_after_inference") != record.get("generation_before")
        for record in queue_records
    ) + sum(record["event"] == "queue_stale" for record in session.records)
    queue_underflow_count = sum(
        record.get("underflow") is True or record.get("queue_after_inference") == 0
        for record in queue_records
    ) + sum(record["event"] == "queue_underflow" for record in session.records)
    merge_anomalies = 0
    for record in queue_records:
        before = record.get("generation_before")
        after_inference = record.get("generation_after_inference")
        after_merge = record.get("generation_after_merge")
        if not all(
            isinstance(value, int) and not isinstance(value, bool)
            for value in (before, after_inference, after_merge)
        ):
            errors.append(f"queue_merge sequence {record['sequence']} has invalid generations")
            merge_anomalies += 1
        elif after_inference != before or after_merge != before + 1:
            merge_anomalies += 1

    smooth_records = [record for record in session.records if record["event"] == "smooth_execution"]
    smooth_underruns = 0
    for record in smooth_records:
        underrun = record.get("underrun")
        if not isinstance(underrun, bool):
            errors.append(f"smooth_execution sequence {record['sequence']} underrun must be boolean")
        elif underrun:
            smooth_underruns += 1
    max_underrun_count = max(
        (
            int(record["underrun_count"])
            for record in smooth_records
            if isinstance(record.get("underrun_count"), int)
            and not isinstance(record.get("underrun_count"), bool)
        ),
        default=0,
    )

    command_samples, command_source = _command_samples(session.records, errors)
    boundary_timestamps = [float(record["monotonic_timestamp"]) for record in queue_records]
    command_metrics = _command_dynamics(command_samples, boundary_timestamps)
    command_metrics["source"] = command_source
    calibrated_command_metrics = _calibrated_command_metrics(
        session.config_snapshot,
        command_samples,
        boundary_timestamps,
    )
    command_metrics["calibrated_constraints"] = calibrated_command_metrics

    beta_min, beta_max, duration_min, duration_max = _configured_bounds(session.config_snapshot, thresholds)
    beta_violations = sum(
        (beta_min is not None and value < beta_min - 1e-12)
        or (beta_max is not None and value > beta_max + 1e-12)
        for value in speed_beta
    )
    reference_duration_violations = sum(
        (duration_min is not None and value < duration_min - 1e-12)
        or (duration_max is not None and value > duration_max + 1e-12)
        for value in reference_durations
    )
    segment_duration_violations = sum(
        (duration_min is not None and value < duration_min - 1e-12)
        or (duration_max is not None and value > duration_max + 1e-12)
        for value in segment_durations
    )

    paper_issues = _paper_ready_issues(
        session.config_snapshot,
        terminal_status=session.terminal_status,
        command_source=command_source,
        calibrated_command_metrics=calibrated_command_metrics,
        beta_count=len(speed_beta),
        reference_duration_count=len(reference_durations),
        planner_count=len(planner_records),
        smooth_count=len(smooth_records),
        queue_merge_count=len(queue_records),
    )
    latency_summary = _summary(inference_latencies_ms)
    planner_metrics = {
        "evaluated_count": len(planner_records),
        "fallback_count": fallback_count,
        "fallback_rate": fallback_count / len(planner_records) if planner_records else None,
        "speed_beta": _summary(speed_beta),
        "speed_beta_bounds": {"min": beta_min, "max": beta_max},
        "speed_beta_bound_violations": beta_violations,
        "reference_duration_s": _summary(reference_durations),
        "segment_duration_s": _summary(segment_durations),
        "duration_bounds_s": {"min": duration_min, "max": duration_max},
        "reference_duration_bound_violations": reference_duration_violations,
        "segment_duration_bound_violations": segment_duration_violations,
    }
    queue_metrics = {
        "merge_count": len(queue_records),
        "stale_count": queue_stale_count,
        "underflow_count": queue_underflow_count,
        "merge_anomaly_count": merge_anomalies,
        "min_queue_after_merge": min(
            (
                record["queue_after_merge"]
                for record in queue_records
                if isinstance(record.get("queue_after_merge"), int)
            ),
            default=None,
        ),
    }
    smooth_metrics = {
        "event_count": len(smooth_records),
        "underrun_count": smooth_underruns,
        "max_reported_underrun_count": max_underrun_count,
    }

    checks: list[dict[str, Any]] = []
    _check(
        checks,
        name="event_fields_valid",
        passed=not errors,
        value=len(errors),
        limit=0,
        detail="event-specific numeric and shape validation",
    )
    _check(
        checks,
        name="minimum_inference_count",
        passed=len(inference_latencies_ms) >= thresholds.min_inference_count,
        value=len(inference_latencies_ms),
        limit={"min": thresholds.min_inference_count},
        detail="valid inference latency samples",
    )
    if thresholds.max_inference_p95_ms is not None:
        value = latency_summary["p95"]
        _check(
            checks,
            name="inference_latency_p95_ms",
            passed=value is not None and value <= thresholds.max_inference_p95_ms,
            value=value,
            limit={"max": thresholds.max_inference_p95_ms},
            detail="p95 model inference latency",
        )
    if thresholds.max_inference_max_ms is not None:
        value = latency_summary["max"]
        _check(
            checks,
            name="inference_latency_max_ms",
            passed=value is not None and value <= thresholds.max_inference_max_ms,
            value=value,
            limit={"max": thresholds.max_inference_max_ms},
            detail="maximum model inference latency",
        )
    _check(
        checks,
        name="planner_fallbacks",
        passed=fallback_count <= thresholds.max_planner_fallbacks,
        value=fallback_count,
        limit={"max": thresholds.max_planner_fallbacks},
        detail="time-axis planner fallbacks",
    )
    _check(
        checks,
        name="speed_beta_bounds",
        passed=beta_violations == 0,
        value=beta_violations,
        limit=planner_metrics["speed_beta_bounds"],
        detail="speed adapter beta factors outside configured/CLI bounds",
    )
    _check(
        checks,
        name="reference_duration_bounds",
        passed=reference_duration_violations == 0 and segment_duration_violations == 0,
        value={
            "reference": reference_duration_violations,
            "optimized": segment_duration_violations,
        },
        limit=planner_metrics["duration_bounds_s"],
        detail="reference and optimized segment periods outside bounds",
    )
    _check(
        checks,
        name="minimum_queue_merges",
        passed=len(queue_records) >= thresholds.min_queue_merges,
        value=len(queue_records),
        limit={"min": thresholds.min_queue_merges},
        detail="queue merge events",
    )
    _check(
        checks,
        name="queue_stale",
        passed=queue_stale_count <= thresholds.max_queue_stale,
        value=queue_stale_count,
        limit={"max": thresholds.max_queue_stale},
        detail="explicit stale events or generation changes before merge",
    )
    _check(
        checks,
        name="queue_underflow",
        passed=queue_underflow_count <= thresholds.max_queue_underflows,
        value=queue_underflow_count,
        limit={"max": thresholds.max_queue_underflows},
        detail="explicit underflow events or queue exhausted during inference",
    )
    _check(
        checks,
        name="queue_merge_invariants",
        passed=merge_anomalies == 0,
        value=merge_anomalies,
        limit=0,
        detail="generation remains stable during inference and increments once on merge",
    )
    _check(
        checks,
        name="smooth_executor_underruns",
        passed=smooth_underruns <= thresholds.max_smooth_underruns,
        value=smooth_underruns,
        limit={"max": thresholds.max_smooth_underruns},
        detail="fixed-heartbeat executor underrun ticks",
    )
    for check_name, metric_name, threshold in (
        ("command_velocity", "velocity_abs", thresholds.max_command_velocity),
        ("command_acceleration", "acceleration_abs", thresholds.max_command_acceleration),
        ("chunk_boundary_jump", "boundary_jump_abs", thresholds.max_boundary_jump),
    ):
        if threshold is None:
            continue
        value = command_metrics[metric_name]["max"]
        _check(
            checks,
            name=check_name,
            passed=value is not None and value <= threshold,
            value=value,
            limit={"max": threshold},
            detail=f"maximum absolute {metric_name}",
        )
    if thresholds.require_paper_ready:
        _check(
            checks,
            name="complete_trace_session",
            passed=session.terminal_status == "completed",
            value=session.terminal_status,
            limit={"required": "completed"},
            detail="session_end exists and records successful completion",
        )
        _check(
            checks,
            name="calibrated_command_velocity",
            passed=(
                calibrated_command_metrics.get("available") is True
                and calibrated_command_metrics.get("velocity_violation_count") == 0
            ),
            value=calibrated_command_metrics.get("velocity_violation_count"),
            limit={"per_joint": calibrated_command_metrics.get("velocity_limits")},
            detail="actual applied-command velocities against resolved per-joint calibration",
        )
        _check(
            checks,
            name="calibrated_command_acceleration",
            passed=(
                calibrated_command_metrics.get("available") is True
                and calibrated_command_metrics.get("acceleration_violation_count") == 0
            ),
            value=calibrated_command_metrics.get("acceleration_violation_count"),
            limit={"per_joint": calibrated_command_metrics.get("acceleration_limits")},
            detail="actual applied-command accelerations against resolved per-joint calibration",
        )
        _check(
            checks,
            name="calibrated_chunk_boundary",
            passed=(
                calibrated_command_metrics.get("available") is True
                and calibrated_command_metrics.get("boundary_sample_count", 0) > 0
                and calibrated_command_metrics.get("boundary_violation_count") == 0
            ),
            value={
                "evaluated": calibrated_command_metrics.get("boundary_sample_count"),
                "violations": calibrated_command_metrics.get("boundary_violation_count"),
            },
            limit={"rule": calibrated_command_metrics.get("boundary_rule")},
            detail="chunk-boundary jumps against velocity times actual elapsed time",
        )
        _check(
            checks,
            name="paper_ready_runtime",
            passed=not paper_issues,
            value={"issues": paper_issues},
            limit={"required": True},
            detail="all V2 runtime components enabled with resolved provenance",
        )

    return {
        "session_id": session.session_id,
        "pass": all(check["pass"] for check in checks),
        "record_count": len(session.records),
        "first_monotonic_timestamp": session.records[0]["monotonic_timestamp"],
        "last_monotonic_timestamp": session.records[-1]["monotonic_timestamp"],
        "terminal": {
            "status": session.terminal_status,
            "reason": session.terminal_reason,
        },
        "validation_errors": errors,
        "checks": checks,
        "paper_ready": {
            "required": thresholds.require_paper_ready,
            "pass": not paper_issues,
            "issues": paper_issues,
        },
        "metrics": {
            "inference_latency_ms": latency_summary,
            "planner": planner_metrics,
            "queue": queue_metrics,
            "smooth_executor": smooth_metrics,
            "commands": command_metrics,
            "event_counts": dict(
                sorted(
                    {
                        event: sum(record["event"] == event for record in session.records)
                        for event in {record["event"] for record in session.records}
                    }.items()
                )
            ),
        },
    }


def validate_trace(
    path: str | Path,
    *,
    thresholds: AcceptanceThresholds | None = None,
) -> dict[str, Any]:
    """Validate all sessions and return one JSON-safe acceptance report."""

    trace_path = Path(path).expanduser().resolve()
    active_thresholds = thresholds or AcceptanceThresholds()
    sessions = load_trace_sessions(trace_path)
    session_reports = [evaluate_session(session, active_thresholds) for session in sessions]
    record_count = sum(len(session.records) for session in sessions)
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "report_type": REPORT_TYPE,
        "overall_pass": all(report["pass"] for report in session_reports),
        "source": {
            "path": str(trace_path),
            "sha256": _sha256_file(trace_path),
            "record_count": record_count,
            "session_count": len(sessions),
        },
        "thresholds": asdict(active_thresholds),
        "validation_errors": [],
        "sessions": session_reports,
    }


def _failure_report(
    path: Path,
    thresholds: AcceptanceThresholds,
    error: str,
) -> dict[str, Any]:
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "report_type": REPORT_TYPE,
        "overall_pass": False,
        "source": {
            "path": str(path.expanduser().resolve()),
            "sha256": _sha256_file(path) if path.is_file() else None,
            "record_count": None,
            "session_count": None,
        },
        "thresholds": asdict(thresholds),
        "validation_errors": [error],
        "sessions": [],
    }


def _write_report(report: Mapping[str, Any], output_path: Path | None) -> None:
    payload = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if output_path is None:
        print(payload, end="")
        return
    destination = output_path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
        os.replace(temporary, destination)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline pass/fail acceptance analysis for RealtimeTraceWriter schema-v2 JSONL."
    )
    parser.add_argument("--input", type=Path, required=True, help="RealtimeTraceWriter JSONL")
    parser.add_argument("--output", type=Path, help="JSON report; stdout when omitted")
    parser.add_argument("--min-inference-count", type=int, default=1)
    parser.add_argument("--max-inference-p95-ms", type=float)
    parser.add_argument("--max-inference-max-ms", type=float)
    parser.add_argument("--max-planner-fallbacks", type=int, default=0)
    parser.add_argument("--min-speed-beta", type=float)
    parser.add_argument("--max-speed-beta", type=float)
    parser.add_argument("--min-reference-duration-s", type=float)
    parser.add_argument("--max-reference-duration-s", type=float)
    parser.add_argument("--min-queue-merges", type=int, default=0)
    parser.add_argument("--max-queue-stale", type=int, default=0)
    parser.add_argument("--max-queue-underflows", type=int, default=0)
    parser.add_argument("--max-smooth-underruns", type=int, default=0)
    parser.add_argument("--max-command-velocity", type=float)
    parser.add_argument("--max-command-acceleration", type=float)
    parser.add_argument("--max-boundary-jump", type=float)
    parser.add_argument("--require-paper-ready", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        thresholds = AcceptanceThresholds(
            min_inference_count=args.min_inference_count,
            max_inference_p95_ms=args.max_inference_p95_ms,
            max_inference_max_ms=args.max_inference_max_ms,
            max_planner_fallbacks=args.max_planner_fallbacks,
            min_speed_beta=args.min_speed_beta,
            max_speed_beta=args.max_speed_beta,
            min_reference_duration_s=args.min_reference_duration_s,
            max_reference_duration_s=args.max_reference_duration_s,
            min_queue_merges=args.min_queue_merges,
            max_queue_stale=args.max_queue_stale,
            max_queue_underflows=args.max_queue_underflows,
            max_smooth_underruns=args.max_smooth_underruns,
            max_command_velocity=args.max_command_velocity,
            max_command_acceleration=args.max_command_acceleration,
            max_boundary_jump=args.max_boundary_jump,
            require_paper_ready=args.require_paper_ready,
        )
    except ValueError as exc:
        raise SystemExit(f"invalid acceptance threshold: {exc}") from exc

    try:
        report = validate_trace(args.input, thresholds=thresholds)
    except (OSError, TraceValidationError) as exc:
        report = _failure_report(args.input, thresholds, str(exc))
        _write_report(report, args.output)
        return 2
    _write_report(report, args.output)
    return 0 if report["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
