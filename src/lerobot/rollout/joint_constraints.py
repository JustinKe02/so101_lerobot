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

"""Strict loader for checksum-pinned joint velocity/acceleration constraints."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
ARTIFACT_TYPE = "lerobot.so101_joint_constraints"
COORDINATE_SPACE = "robot_action_units"
_SHA256_PATTERN = re.compile(r"[0-9a-fA-F]{64}")
_TOP_LEVEL_FIELDS = {
    "schema_version",
    "artifact_type",
    "coordinate_space",
    "joint_names",
    "max_velocity",
    "max_acceleration",
    "method",
    "quantile",
    "safety_factor",
    "source",
}
_SOURCE_FIELDS = {
    "path",
    "format",
    "sha256",
    "sample_count",
    "velocity_sample_count",
    "acceleration_sample_count",
}


class JointConstraintError(ValueError):
    """Raised when a joint constraint artifact is invalid or incompatible."""


@dataclass
class JointConstraintArtifactConfig:
    """Pinned offline constraint artifact used by the QP and executor."""

    enabled: bool = False
    path: str | None = None
    sha256: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("time_axis_planner.joint_constraints.enabled must be boolean")
        if self.path is not None and (not isinstance(self.path, str) or not self.path.strip()):
            raise ValueError("time_axis_planner.joint_constraints.path must be a non-empty string")
        if self.sha256 is not None and not _valid_sha256(self.sha256):
            raise ValueError("time_axis_planner.joint_constraints.sha256 must be 64 hexadecimal characters")
        if self.enabled and self.path is None:
            raise ValueError("time_axis_planner.joint_constraints.path is required when enabled")
        if self.enabled and self.sha256 is None:
            raise ValueError("time_axis_planner.joint_constraints.sha256 is required when enabled")


@dataclass(frozen=True)
class JointConstraintArtifact:
    """Validated immutable constraints and provenance."""

    path: Path
    artifact_sha256: str
    source_path: str
    source_format: str
    source_sha256: str
    source_sample_count: int
    velocity_sample_count: int
    acceleration_sample_count: int
    joint_names: tuple[str, ...]
    max_velocity: tuple[float, ...]
    max_acceleration: tuple[float, ...]
    method: str
    quantile: float
    safety_factor: float

    def validate_layout(self, joint_names: Sequence[str]) -> None:
        expected = _validated_names(joint_names, context="runtime joint names")
        if self.joint_names != expected:
            raise JointConstraintError(
                "joint constraint order mismatch: "
                f"artifact={list(self.joint_names)!r}, runtime={list(expected)!r}"
            )

    def audit_snapshot(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "artifact_path": str(self.path),
            "artifact_sha256": self.artifact_sha256,
            "source_path": self.source_path,
            "source_format": self.source_format,
            "source_sha256": self.source_sha256,
            "sample_count": self.source_sample_count,
            "velocity_sample_count": self.velocity_sample_count,
            "acceleration_sample_count": self.acceleration_sample_count,
            "method": self.method,
            "quantile": self.quantile,
            "safety_factor": self.safety_factor,
            "coordinate_space": COORDINATE_SPACE,
            "joint_names": list(self.joint_names),
            "max_velocity": list(self.max_velocity),
            "max_acceleration": list(self.max_acceleration),
        }


def _valid_sha256(value: object) -> bool:
    return isinstance(value, str) and _SHA256_PATTERN.fullmatch(value) is not None


def _validated_names(value: object, *, context: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or not value:
        raise JointConstraintError(f"{context} must be a non-empty array")
    names = tuple(value)
    if any(not isinstance(name, str) or not name.strip() for name in names):
        raise JointConstraintError(f"{context} must contain non-empty strings")
    if len(set(names)) != len(names):
        raise JointConstraintError(f"{context} must not contain duplicates")
    return names


def _positive_vector(value: object, *, name: str, size: int) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) != size:
        raise JointConstraintError(f"{name} must contain exactly {size} values")
    result = []
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise JointConstraintError(f"{name}[{index}] must be numeric")
        number = float(item)
        if not math.isfinite(number) or number <= 0.0:
            raise JointConstraintError(f"{name}[{index}] must be finite and positive")
        result.append(number)
    return tuple(result)


def _read_document(path: Path, expected_sha256: str) -> tuple[Mapping[str, Any], str]:
    if not path.is_file():
        raise JointConstraintError(f"joint constraint artifact does not exist: {path}")
    if not _valid_sha256(expected_sha256):
        raise JointConstraintError("expected joint constraint SHA-256 is invalid")
    payload = path.read_bytes()
    actual_sha256 = hashlib.sha256(payload).hexdigest()
    if actual_sha256.lower() != expected_sha256.lower():
        raise JointConstraintError(
            "joint constraint artifact SHA-256 mismatch: "
            f"expected={expected_sha256.lower()}, actual={actual_sha256}"
        )
    try:
        document = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise JointConstraintError("joint constraint artifact is not valid UTF-8 JSON") from exc
    if not isinstance(document, Mapping):
        raise JointConstraintError("joint constraint artifact root must be an object")
    return document, actual_sha256


def load_joint_constraint_artifact(
    path: str | Path,
    *,
    expected_sha256: str,
    joint_names: Sequence[str] | None = None,
) -> JointConstraintArtifact:
    """Load and strictly validate a pinned joint constraint artifact."""

    artifact_path = Path(path).expanduser().resolve()
    document, artifact_sha256 = _read_document(artifact_path, expected_sha256)
    if set(document) != _TOP_LEVEL_FIELDS:
        raise JointConstraintError(
            "joint constraint artifact fields do not match schema: "
            f"expected={sorted(_TOP_LEVEL_FIELDS)}, actual={sorted(document)}"
        )
    if document["schema_version"] != SCHEMA_VERSION:
        raise JointConstraintError(
            f"unsupported joint constraint schema_version: {document['schema_version']!r}"
        )
    if document["artifact_type"] != ARTIFACT_TYPE:
        raise JointConstraintError(
            f"unexpected joint constraint artifact_type: {document['artifact_type']!r}"
        )
    if document["coordinate_space"] != COORDINATE_SPACE:
        raise JointConstraintError(f"joint constraint coordinate_space must be {COORDINATE_SPACE!r}")

    resolved_joint_names = _validated_names(document["joint_names"], context="joint_names")
    method = document["method"]
    if not isinstance(method, str) or not method.strip():
        raise JointConstraintError("joint constraint method must be a non-empty string")
    quantile = document["quantile"]
    if isinstance(quantile, bool) or not isinstance(quantile, (int, float)):
        raise JointConstraintError("joint constraint quantile must be numeric")
    quantile = float(quantile)
    if not math.isfinite(quantile) or not 0.0 < quantile <= 1.0:
        raise JointConstraintError("joint constraint quantile must be in (0, 1]")
    safety_factor = document["safety_factor"]
    if isinstance(safety_factor, bool) or not isinstance(safety_factor, (int, float)):
        raise JointConstraintError("joint constraint safety_factor must be numeric")
    safety_factor = float(safety_factor)
    if not math.isfinite(safety_factor) or safety_factor <= 0.0:
        raise JointConstraintError("joint constraint safety_factor must be finite and positive")

    source = document["source"]
    if not isinstance(source, Mapping) or set(source) != _SOURCE_FIELDS:
        raise JointConstraintError("joint constraint source fields do not match schema")
    if not isinstance(source["path"], str) or not source["path"].strip():
        raise JointConstraintError("joint constraint source.path must be non-empty")
    if not isinstance(source["format"], str) or not source["format"].strip():
        raise JointConstraintError("joint constraint source.format must be non-empty")
    if not _valid_sha256(source["sha256"]):
        raise JointConstraintError("joint constraint source.sha256 is invalid")
    counts = {}
    for key in ("sample_count", "velocity_sample_count", "acceleration_sample_count"):
        value = source[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise JointConstraintError(f"joint constraint source.{key} must be positive")
        counts[key] = value

    artifact = JointConstraintArtifact(
        path=artifact_path,
        artifact_sha256=artifact_sha256,
        source_path=source["path"],
        source_format=source["format"],
        source_sha256=source["sha256"].lower(),
        source_sample_count=counts["sample_count"],
        velocity_sample_count=counts["velocity_sample_count"],
        acceleration_sample_count=counts["acceleration_sample_count"],
        joint_names=resolved_joint_names,
        max_velocity=_positive_vector(
            document["max_velocity"], name="max_velocity", size=len(resolved_joint_names)
        ),
        max_acceleration=_positive_vector(
            document["max_acceleration"], name="max_acceleration", size=len(resolved_joint_names)
        ),
        method=method.strip(),
        quantile=quantile,
        safety_factor=safety_factor,
    )
    if joint_names is not None:
        artifact.validate_layout(joint_names)
    return artifact
