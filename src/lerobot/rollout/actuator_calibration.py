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

"""Strict offline loader for SO-101 actuator-calibration artifacts."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

SCHEMA_VERSION = 1
ARTIFACT_TYPE = "lerobot.so101_actuator_calibration"
MONOTONIC_CLOCK_DOMAIN = "monotonic"
_SHA256_PATTERN = re.compile(r"[0-9a-fA-F]{64}")
_TOP_LEVEL_FIELDS = {
    "schema_version",
    "artifact_type",
    "clock_domain",
    "joint_names",
    "command_delay_s",
    "tau_s",
    "model",
    "fit_diagnostics",
    "fit_config",
    "source",
}
_MODEL_FIELDS = {"type", "equation", "command_interpolation", "residual"}
_FIT_CONFIG_FIELDS = {
    "max_delay_s",
    "min_tau_s",
    "max_tau_s",
    "min_fit_samples",
    "min_command_span",
}
_SOURCE_FIELDS = {"path", "format", "sha256", "sample_count"}
_DIAGNOSTIC_FIELDS = {
    "joint_name",
    "residual_rmse",
    "residual_mae",
    "residual_rss",
    "r_squared",
    "sample_count",
    "command_span",
    "optimizer_nfev",
    "optimizer_status",
    "at_delay_bound",
    "at_tau_bound",
}


class ActuatorCalibrationError(ValueError):
    """Raised when a calibration artifact is missing, corrupt, or incompatible."""


@dataclass(frozen=True)
class JointFitAudit:
    """Runtime-relevant audit fields for one fitted joint."""

    joint_name: str
    sample_count: int
    residual_rmse: float
    at_delay_bound: bool
    at_tau_bound: bool


@dataclass(frozen=True)
class ActuatorCalibrationArtifact:
    """Validated immutable parameters and provenance from an artifact."""

    path: Path
    artifact_sha256: str
    source_path: str
    source_sha256: str
    source_sample_count: int
    joint_names: tuple[str, ...]
    command_delay_s: NDArray[np.float64]
    tau_s: NDArray[np.float64]
    fit_audit: tuple[JointFitAudit, ...]

    def __post_init__(self) -> None:
        delay = np.asarray(self.command_delay_s, dtype=np.float64).reshape(-1).copy()
        tau = np.asarray(self.tau_s, dtype=np.float64).reshape(-1).copy()
        delay.setflags(write=False)
        tau.setflags(write=False)
        object.__setattr__(self, "command_delay_s", delay)
        object.__setattr__(self, "tau_s", tau)

    @property
    def action_dim(self) -> int:
        return len(self.joint_names)

    def validate_layout(self, *, action_dim: int, joint_names: Sequence[str]) -> None:
        """Require exact runtime dimension and joint order, with no reordering."""

        if isinstance(action_dim, bool) or not isinstance(action_dim, int) or action_dim < 1:
            raise ActuatorCalibrationError("runtime action_dim must be a positive integer")
        expected_names = _validated_joint_names(joint_names, context="runtime joint_names")
        if self.action_dim != action_dim:
            raise ActuatorCalibrationError(
                f"actuator calibration action_dim mismatch: artifact={self.action_dim}, runtime={action_dim}"
            )
        if len(expected_names) != action_dim:
            raise ActuatorCalibrationError(
                f"runtime joint_names has {len(expected_names)} entries for action_dim={action_dim}"
            )
        if self.joint_names != expected_names:
            raise ActuatorCalibrationError(
                "actuator calibration joint order mismatch: "
                f"artifact={list(self.joint_names)!r}, runtime={list(expected_names)!r}"
            )

    def audit_snapshot(self) -> dict[str, Any]:
        """Return JSON-safe provenance for session_start and startup logs."""

        return {
            "schema_version": SCHEMA_VERSION,
            "artifact_path": str(self.path),
            "artifact_sha256": self.artifact_sha256,
            "source_path": self.source_path,
            "source_sha256": self.source_sha256,
            "source_sample_count": self.source_sample_count,
            "joint_names": list(self.joint_names),
            "command_delay_s": self.command_delay_s.tolist(),
            "tau_s": self.tau_s.tolist(),
            "fit_boundary_status": [
                {
                    "joint_name": fit.joint_name,
                    "sample_count": fit.sample_count,
                    "residual_rmse": fit.residual_rmse,
                    "at_delay_bound": fit.at_delay_bound,
                    "at_tau_bound": fit.at_tau_bound,
                }
                for fit in self.fit_audit
            ],
        }


def normalize_action_joint_names(action_keys: Sequence[str]) -> tuple[str, ...]:
    """Map ordered LeRobot ``*.pos`` action keys to artifact joint names."""

    normalized: list[str] = []
    for index, key in enumerate(action_keys):
        if not isinstance(key, str) or not key.strip():
            raise ActuatorCalibrationError(f"action key {index} must be a non-empty string")
        clean = key.strip()
        normalized.append(clean.removesuffix(".pos"))
    return _validated_joint_names(normalized, context="normalized action joint_names")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _validated_sha256(value: Any, *, context: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ActuatorCalibrationError(f"{context} must be exactly 64 hexadecimal characters")
    return value.lower()


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ActuatorCalibrationError(f"duplicate JSON key is not allowed: {key!r}")
        result[key] = value
    return result


def _mapping(value: Any, *, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ActuatorCalibrationError(f"{context} must be a JSON object")
    return value


def _exact_fields(value: Mapping[str, Any], expected: set[str], *, context: str) -> None:
    actual = set(value)
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if missing or unknown:
        raise ActuatorCalibrationError(
            f"{context} fields do not match schema; missing={missing!r}, unknown={unknown!r}"
        )


def _finite_number(
    value: Any,
    *,
    context: str,
    minimum: float | None = None,
    strict_minimum: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ActuatorCalibrationError(f"{context} must be a finite JSON number")
    result = float(value)
    if not math.isfinite(result):
        raise ActuatorCalibrationError(f"{context} must be finite")
    if minimum is not None:
        invalid = result <= minimum if strict_minimum else result < minimum
        if invalid:
            comparison = ">" if strict_minimum else ">="
            raise ActuatorCalibrationError(f"{context} must be {comparison} {minimum}")
    return result


def _integer(value: Any, *, context: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ActuatorCalibrationError(f"{context} must be an integer >= {minimum}")
    return value


def _validated_joint_names(value: Any, *, context: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ActuatorCalibrationError(f"{context} must be an array of strings")
    names = tuple(value)
    if not names:
        raise ActuatorCalibrationError(f"{context} must not be empty")
    if any(not isinstance(name, str) or not name.strip() or name != name.strip() for name in names):
        raise ActuatorCalibrationError(f"{context} entries must be non-empty, whitespace-trimmed strings")
    if len(set(names)) != len(names):
        raise ActuatorCalibrationError(f"{context} entries must be unique")
    return names


def _numeric_vector(
    value: Any,
    *,
    context: str,
    size: int,
    minimum: float,
    strict_minimum: bool,
) -> NDArray[np.float64]:
    if not isinstance(value, list) or len(value) != size:
        raise ActuatorCalibrationError(f"{context} must be a JSON array with exactly {size} values")
    result = np.asarray(
        [
            _finite_number(
                item,
                context=f"{context}[{index}]",
                minimum=minimum,
                strict_minimum=strict_minimum,
            )
            for index, item in enumerate(value)
        ],
        dtype=np.float64,
    )
    result.setflags(write=False)
    return result


def _validate_model(value: Any) -> None:
    model = _mapping(value, context="model")
    _exact_fields(model, _MODEL_FIELDS, context="model")
    expected = {
        "type": "pure_delay_first_order",
        "equation": "dx/dt = (command(t - delay) - x) / tau",
        "command_interpolation": "zero_order_hold",
        "residual": "one_step_prediction",
    }
    if dict(model) != expected:
        raise ActuatorCalibrationError("model definition is incompatible with the runtime estimator")


def _validate_fit_config(value: Any) -> None:
    config = _mapping(value, context="fit_config")
    _exact_fields(config, _FIT_CONFIG_FIELDS, context="fit_config")
    _finite_number(config["max_delay_s"], context="fit_config.max_delay_s", minimum=0.0)
    min_tau = _finite_number(
        config["min_tau_s"], context="fit_config.min_tau_s", minimum=0.0, strict_minimum=True
    )
    max_tau = _finite_number(
        config["max_tau_s"], context="fit_config.max_tau_s", minimum=0.0, strict_minimum=True
    )
    if max_tau <= min_tau:
        raise ActuatorCalibrationError("fit_config.max_tau_s must exceed min_tau_s")
    _integer(config["min_fit_samples"], context="fit_config.min_fit_samples", minimum=3)
    _finite_number(
        config["min_command_span"],
        context="fit_config.min_command_span",
        minimum=0.0,
        strict_minimum=True,
    )


def _validate_source(value: Any) -> tuple[str, str, int]:
    source = _mapping(value, context="source")
    _exact_fields(source, _SOURCE_FIELDS, context="source")
    source_path = source["path"]
    if not isinstance(source_path, str) or not source_path:
        raise ActuatorCalibrationError("source.path must be a non-empty string")
    if source["format"] not in {"jsonl", "csv"}:
        raise ActuatorCalibrationError("source.format must be 'jsonl' or 'csv'")
    source_sha256 = _validated_sha256(source["sha256"], context="source.sha256")
    source_sample_count = _integer(source["sample_count"], context="source.sample_count", minimum=3)
    return source_path, source_sha256, source_sample_count


def _validate_fit_diagnostics(
    value: Any,
    *,
    joint_names: tuple[str, ...],
    source_sample_count: int,
) -> tuple[JointFitAudit, ...]:
    if not isinstance(value, list) or len(value) != len(joint_names):
        raise ActuatorCalibrationError(f"fit_diagnostics must contain exactly {len(joint_names)} entries")
    result: list[JointFitAudit] = []
    for index, (raw, expected_name) in enumerate(zip(value, joint_names, strict=True)):
        context = f"fit_diagnostics[{index}]"
        diagnostic = _mapping(raw, context=context)
        _exact_fields(diagnostic, _DIAGNOSTIC_FIELDS, context=context)
        if diagnostic["joint_name"] != expected_name:
            raise ActuatorCalibrationError(
                f"{context}.joint_name must be {expected_name!r} to preserve joint order"
            )
        rmse = _finite_number(diagnostic["residual_rmse"], context=f"{context}.residual_rmse", minimum=0.0)
        _finite_number(diagnostic["residual_mae"], context=f"{context}.residual_mae", minimum=0.0)
        _finite_number(diagnostic["residual_rss"], context=f"{context}.residual_rss", minimum=0.0)
        r_squared = diagnostic["r_squared"]
        if r_squared is not None:
            _finite_number(r_squared, context=f"{context}.r_squared")
        sample_count = _integer(diagnostic["sample_count"], context=f"{context}.sample_count", minimum=1)
        if sample_count > source_sample_count:
            raise ActuatorCalibrationError(f"{context}.sample_count cannot exceed source.sample_count")
        _finite_number(
            diagnostic["command_span"],
            context=f"{context}.command_span",
            minimum=0.0,
            strict_minimum=True,
        )
        _integer(diagnostic["optimizer_nfev"], context=f"{context}.optimizer_nfev", minimum=1)
        _integer(diagnostic["optimizer_status"], context=f"{context}.optimizer_status", minimum=-1)
        for field in ("at_delay_bound", "at_tau_bound"):
            if not isinstance(diagnostic[field], bool):
                raise ActuatorCalibrationError(f"{context}.{field} must be boolean")
        result.append(
            JointFitAudit(
                joint_name=expected_name,
                sample_count=sample_count,
                residual_rmse=rmse,
                at_delay_bound=diagnostic["at_delay_bound"],
                at_tau_bound=diagnostic["at_tau_bound"],
            )
        )
    return tuple(result)


def load_actuator_calibration_artifact(
    path: str | Path,
    *,
    expected_sha256: str,
    action_dim: int | None = None,
    joint_names: Sequence[str] | None = None,
) -> ActuatorCalibrationArtifact:
    """Load an artifact and fail closed on checksum, schema, or layout errors."""

    expected_hash = _validated_sha256(expected_sha256, context="expected artifact sha256")
    artifact_path = Path(path).expanduser().resolve()
    if not artifact_path.is_file():
        raise ActuatorCalibrationError(f"actuator calibration artifact does not exist: {artifact_path}")
    try:
        payload = artifact_path.read_bytes()
    except OSError as exc:
        raise ActuatorCalibrationError(
            f"cannot read actuator calibration artifact: {artifact_path}: {exc}"
        ) from exc
    actual_hash = _sha256_bytes(payload)
    if actual_hash != expected_hash:
        raise ActuatorCalibrationError(
            "actuator calibration artifact SHA-256 mismatch: "
            f"expected={expected_hash}, actual={actual_hash}, path={artifact_path}"
        )
    try:
        document = json.loads(payload, object_pairs_hook=_reject_duplicate_keys)
    except UnicodeDecodeError as exc:
        raise ActuatorCalibrationError("actuator calibration artifact must be UTF-8 JSON") from exc
    except json.JSONDecodeError as exc:
        raise ActuatorCalibrationError(
            f"invalid actuator calibration JSON at line {exc.lineno}, column {exc.colno}"
        ) from exc

    root = _mapping(document, context="artifact")
    _exact_fields(root, _TOP_LEVEL_FIELDS, context="artifact")
    if isinstance(root["schema_version"], bool) or root["schema_version"] != SCHEMA_VERSION:
        raise ActuatorCalibrationError(
            f"unsupported actuator calibration schema_version: {root['schema_version']!r}; "
            f"expected {SCHEMA_VERSION}"
        )
    if root["artifact_type"] != ARTIFACT_TYPE:
        raise ActuatorCalibrationError(
            f"artifact_type must be {ARTIFACT_TYPE!r}, got {root['artifact_type']!r}"
        )
    if root["clock_domain"] != MONOTONIC_CLOCK_DOMAIN:
        raise ActuatorCalibrationError("artifact clock_domain must be 'monotonic'")

    artifact_joint_names = _validated_joint_names(root["joint_names"], context="joint_names")
    if len(artifact_joint_names) != 6:
        raise ActuatorCalibrationError(
            f"SO-101 actuator calibration must contain exactly 6 joints, got {len(artifact_joint_names)}"
        )
    command_delay_s = _numeric_vector(
        root["command_delay_s"],
        context="command_delay_s",
        size=len(artifact_joint_names),
        minimum=0.0,
        strict_minimum=False,
    )
    tau_s = _numeric_vector(
        root["tau_s"],
        context="tau_s",
        size=len(artifact_joint_names),
        minimum=0.0,
        strict_minimum=True,
    )
    _validate_model(root["model"])
    _validate_fit_config(root["fit_config"])
    source_path, source_sha256, source_sample_count = _validate_source(root["source"])
    fit_audit = _validate_fit_diagnostics(
        root["fit_diagnostics"],
        joint_names=artifact_joint_names,
        source_sample_count=source_sample_count,
    )

    artifact = ActuatorCalibrationArtifact(
        path=artifact_path,
        artifact_sha256=actual_hash,
        source_path=source_path,
        source_sha256=source_sha256,
        source_sample_count=source_sample_count,
        joint_names=artifact_joint_names,
        command_delay_s=command_delay_s,
        tau_s=tau_s,
        fit_audit=fit_audit,
    )
    if (action_dim is None) != (joint_names is None):
        raise ActuatorCalibrationError("action_dim and joint_names must be provided together")
    if action_dim is not None and joint_names is not None:
        artifact.validate_layout(action_dim=action_dim, joint_names=joint_names)
    return artifact
