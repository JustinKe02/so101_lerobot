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

"""Strict loader for checksum-pinned sensor timing calibration artifacts."""

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
ARTIFACT_TYPE = "lerobot.sensor_timing_calibration"
MONOTONIC_CLOCK_DOMAIN = "monotonic"
_SHA256_PATTERN = re.compile(r"[0-9a-fA-F]{64}")
_TOP_LEVEL_FIELDS = {
    "schema_version",
    "artifact_type",
    "clock_domain",
    "camera_keys",
    "camera_capture_delay_s",
    "state_observation_delay_s",
    "max_camera_skew_s",
    "method",
    "quantile",
    "source",
}
_SOURCE_FIELDS = {"path", "format", "sha256", "sample_count"}


class SensorTimingCalibrationError(ValueError):
    """Raised when a sensor timing artifact is invalid or incompatible."""


@dataclass
class SensorTimingCalibrationArtifactConfig:
    """Pinned offline sensor timing artifact used by RTC dynamic prefill."""

    enabled: bool = False
    path: str | None = None
    sha256: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("inference.sensor_timing_calibration.enabled must be boolean")
        if self.path is not None and (not isinstance(self.path, str) or not self.path.strip()):
            raise ValueError("inference.sensor_timing_calibration.path must be a non-empty string")
        if self.sha256 is not None and not _valid_sha256(self.sha256):
            raise ValueError("inference.sensor_timing_calibration.sha256 must be 64 hexadecimal characters")
        if self.enabled and self.path is None:
            raise ValueError("inference.sensor_timing_calibration.path is required when enabled")
        if self.enabled and self.sha256 is None:
            raise ValueError("inference.sensor_timing_calibration.sha256 is required when enabled")


@dataclass(frozen=True)
class SensorTimingCalibrationArtifact:
    """Validated immutable sensor timing parameters and source provenance."""

    path: Path
    artifact_sha256: str
    source_path: str
    source_format: str
    source_sha256: str
    source_sample_count: int
    camera_keys: tuple[str, ...]
    camera_capture_delay_s: tuple[float, ...]
    state_observation_delay_s: float
    max_camera_skew_s: float
    method: str
    quantile: float

    @property
    def camera_delay_by_key(self) -> dict[str, float]:
        return dict(zip(self.camera_keys, self.camera_capture_delay_s, strict=True))

    @property
    def conservative_image_capture_delay_s(self) -> float:
        return max(self.camera_capture_delay_s)

    def validate_camera_layout(self, camera_keys: Sequence[str]) -> None:
        expected = _validated_names(camera_keys, context="runtime camera keys")
        if self.camera_keys != expected:
            raise SensorTimingCalibrationError(
                "sensor timing camera order mismatch: "
                f"artifact={list(self.camera_keys)!r}, runtime={list(expected)!r}"
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
            "method": self.method,
            "quantile": self.quantile,
            "camera_keys": list(self.camera_keys),
            "camera_capture_delay_s": self.camera_delay_by_key,
            "image_capture_delay_s": self.conservative_image_capture_delay_s,
            "state_observation_delay_s": self.state_observation_delay_s,
            "max_camera_skew_s": self.max_camera_skew_s,
        }


def _valid_sha256(value: object) -> bool:
    return isinstance(value, str) and _SHA256_PATTERN.fullmatch(value) is not None


def _validated_names(value: object, *, context: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or not value:
        raise SensorTimingCalibrationError(f"{context} must be a non-empty array")
    names = tuple(value)
    if any(not isinstance(name, str) or not name.strip() for name in names):
        raise SensorTimingCalibrationError(f"{context} must contain non-empty strings")
    if len(set(names)) != len(names):
        raise SensorTimingCalibrationError(f"{context} must not contain duplicates")
    return names


def _finite_nonnegative(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SensorTimingCalibrationError(f"{name} must be a number")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise SensorTimingCalibrationError(f"{name} must be finite and non-negative")
    return result


def _read_document(path: Path, expected_sha256: str) -> tuple[Mapping[str, Any], str]:
    if not path.is_file():
        raise SensorTimingCalibrationError(f"sensor timing artifact does not exist: {path}")
    if not _valid_sha256(expected_sha256):
        raise SensorTimingCalibrationError("expected sensor timing SHA-256 is invalid")
    payload = path.read_bytes()
    actual_sha256 = hashlib.sha256(payload).hexdigest()
    if actual_sha256.lower() != expected_sha256.lower():
        raise SensorTimingCalibrationError(
            "sensor timing artifact SHA-256 mismatch: "
            f"expected={expected_sha256.lower()}, actual={actual_sha256}"
        )
    try:
        document = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SensorTimingCalibrationError("sensor timing artifact is not valid UTF-8 JSON") from exc
    if not isinstance(document, Mapping):
        raise SensorTimingCalibrationError("sensor timing artifact root must be an object")
    return document, actual_sha256


def load_sensor_timing_calibration_artifact(
    path: str | Path,
    *,
    expected_sha256: str,
    camera_keys: Sequence[str] | None = None,
) -> SensorTimingCalibrationArtifact:
    """Load and strictly validate a pinned sensor timing artifact."""

    artifact_path = Path(path).expanduser().resolve()
    document, artifact_sha256 = _read_document(artifact_path, expected_sha256)
    if set(document) != _TOP_LEVEL_FIELDS:
        raise SensorTimingCalibrationError(
            "sensor timing artifact fields do not match schema: "
            f"expected={sorted(_TOP_LEVEL_FIELDS)}, actual={sorted(document)}"
        )
    if document["schema_version"] != SCHEMA_VERSION:
        raise SensorTimingCalibrationError(
            f"unsupported sensor timing schema_version: {document['schema_version']!r}"
        )
    if document["artifact_type"] != ARTIFACT_TYPE:
        raise SensorTimingCalibrationError(
            f"unexpected sensor timing artifact_type: {document['artifact_type']!r}"
        )
    if document["clock_domain"] != MONOTONIC_CLOCK_DOMAIN:
        raise SensorTimingCalibrationError("sensor timing clock_domain must be 'monotonic'")

    resolved_camera_keys = _validated_names(document["camera_keys"], context="camera_keys")
    delay_mapping = document["camera_capture_delay_s"]
    if not isinstance(delay_mapping, Mapping) or set(delay_mapping) != set(resolved_camera_keys):
        raise SensorTimingCalibrationError("camera_capture_delay_s keys must exactly match camera_keys")
    camera_delays = tuple(
        _finite_nonnegative(delay_mapping[key], name=f"camera_capture_delay_s[{key!r}]")
        for key in resolved_camera_keys
    )

    method = document["method"]
    if not isinstance(method, str) or not method.strip():
        raise SensorTimingCalibrationError("sensor timing method must be a non-empty string")
    quantile = document["quantile"]
    if isinstance(quantile, bool) or not isinstance(quantile, (int, float)):
        raise SensorTimingCalibrationError("sensor timing quantile must be numeric")
    quantile = float(quantile)
    if not math.isfinite(quantile) or not 0.0 < quantile <= 1.0:
        raise SensorTimingCalibrationError("sensor timing quantile must be in (0, 1]")

    source = document["source"]
    if not isinstance(source, Mapping) or set(source) != _SOURCE_FIELDS:
        raise SensorTimingCalibrationError("sensor timing source fields do not match schema")
    if not isinstance(source["path"], str) or not source["path"].strip():
        raise SensorTimingCalibrationError("sensor timing source.path must be non-empty")
    if not isinstance(source["format"], str) or not source["format"].strip():
        raise SensorTimingCalibrationError("sensor timing source.format must be non-empty")
    if not _valid_sha256(source["sha256"]):
        raise SensorTimingCalibrationError("sensor timing source.sha256 is invalid")
    sample_count = source["sample_count"]
    if isinstance(sample_count, bool) or not isinstance(sample_count, int) or sample_count < 1:
        raise SensorTimingCalibrationError("sensor timing source.sample_count must be positive")

    max_camera_skew_s = _finite_nonnegative(document["max_camera_skew_s"], name="max_camera_skew_s")
    if max_camera_skew_s <= 0.0:
        raise SensorTimingCalibrationError("max_camera_skew_s must be positive for fail-closed runtime use")
    artifact = SensorTimingCalibrationArtifact(
        path=artifact_path,
        artifact_sha256=artifact_sha256,
        source_path=source["path"],
        source_format=source["format"],
        source_sha256=source["sha256"].lower(),
        source_sample_count=sample_count,
        camera_keys=resolved_camera_keys,
        camera_capture_delay_s=camera_delays,
        state_observation_delay_s=_finite_nonnegative(
            document["state_observation_delay_s"], name="state_observation_delay_s"
        ),
        max_camera_skew_s=max_camera_skew_s,
        method=method.strip(),
        quantile=quantile,
    )
    if camera_keys is not None:
        artifact.validate_camera_layout(camera_keys)
    return artifact
