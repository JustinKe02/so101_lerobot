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

"""Checksum-pinned PI0.5 PyTorch/Triton parity artifacts.

The rollout loader in this module is deliberately CPU-only. It validates the
complete evidence chain before a CUDA backend or robot is constructed.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

PI05_PARITY_REPORT_TYPE = "lerobot.pi05_realtime_vla_v2_parity"
PI05_PARITY_REPORT_SCHEMA_VERSION = 2
PI05_PARITY_REQUEST_SCHEMA_VERSION = 1
_MAX_REPORT_BYTES = 4 * 1024 * 1024
_SHA256_HEX_LENGTH = 64


class PI05ParityReportError(RuntimeError):
    """Raised when parity evidence is missing, malformed, stale, or failed."""


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == _SHA256_HEX_LENGTH
        and all(character in "0123456789abcdefABCDEF" for character in value)
    )


@dataclass
class PI05ParityReportArtifactConfig:
    """Checksum-pinned aggregate parity report required by Triton rollout."""

    enabled: bool = False
    path: str | None = None
    sha256: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("pi05_triton_parity_report.enabled must be boolean")
        if self.path is not None and (not isinstance(self.path, str) or not self.path.strip()):
            raise ValueError("pi05_triton_parity_report.path must be a non-empty string")
        if self.sha256 is not None and not _is_sha256(self.sha256):
            raise ValueError("pi05_triton_parity_report.sha256 must be 64 hexadecimal characters")
        if self.enabled and self.path is None:
            raise ValueError("pi05_triton_parity_report.path is required when enabled")
        if self.enabled and self.sha256 is None:
            raise ValueError("pi05_triton_parity_report.sha256 is required when enabled")


@dataclass(frozen=True)
class ValidatedPI05ParityReport:
    """Small immutable deployment view of a fully validated parity report."""

    path: str
    sha256: str
    checkpoint_path: str
    checkpoint_config_sha256: str
    checkpoint_model_sha256: str
    triton_weights_sha256: str
    triton_model_config_sha256: str
    training_max_delay: int
    atol: float
    rtol: float
    prefixes: tuple[int, ...]
    max_abs_errors: tuple[float, ...]

    def audit_snapshot(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "sha256": self.sha256,
            "report_schema_version": PI05_PARITY_REPORT_SCHEMA_VERSION,
            "checkpoint_path": self.checkpoint_path,
            "checkpoint_config_sha256": self.checkpoint_config_sha256,
            "checkpoint_model_sha256": self.checkpoint_model_sha256,
            "triton_weights_sha256": self.triton_weights_sha256,
            "triton_model_config_sha256": self.triton_model_config_sha256,
            "training_max_delay": self.training_max_delay,
            "thresholds": {"atol": self.atol, "rtol": self.rtol},
            "prefixes": list(self.prefixes),
            "max_abs_errors": list(self.max_abs_errors),
            "passed": True,
        }


@dataclass(frozen=True)
class ValidatedPI05ParityRequest:
    """Strictly decoded request evidence used to bind parity outputs to inputs."""

    images: dict[str, np.ndarray]
    normalized_state: np.ndarray
    normalized_prefill_actions: np.ndarray
    prompt: str
    camera_keys: tuple[str, str]
    image_value_range: str
    fingerprint: str

    @property
    def prefill_length(self) -> int:
        return int(self.normalized_prefill_actions.shape[0])


def _array_digest(digest: Any, name: str, value: np.ndarray) -> None:
    contiguous = np.ascontiguousarray(value)
    digest.update(name.encode())
    digest.update(str(contiguous.shape).encode())
    digest.update(str(contiguous.dtype).encode())
    digest.update(contiguous.view(np.uint8))


def compute_pi05_parity_request_fingerprint(
    *,
    images: Mapping[str, np.ndarray],
    normalized_state: np.ndarray,
    normalized_prefill_actions: np.ndarray,
    prompt: str,
    camera_keys: tuple[str, str],
    image_value_range: str,
) -> str:
    """Hash the exact serialized request fields consumed by both backends."""
    digest = hashlib.sha256()
    digest.update(prompt.encode())
    digest.update(json.dumps(camera_keys).encode())
    digest.update(image_value_range.encode())
    for key in camera_keys:
        _array_digest(digest, key, images[key])
    _array_digest(digest, "normalized_state", normalized_state)
    _array_digest(digest, "normalized_prefill_actions", normalized_prefill_actions)
    return digest.hexdigest()


def file_sha256(path: str | Path) -> str:
    """Return a streaming SHA-256 digest without loading large weights into RAM."""
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Parity artifact file not found: {resolved}")
    digest = hashlib.sha256()
    with resolved.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _read_json(path: Path, *, label: str, size_limit: int = _MAX_REPORT_BYTES) -> dict[str, Any]:
    if not path.is_file():
        raise PI05ParityReportError(f"{label} not found: {path}")
    size = path.stat().st_size
    if size <= 0 or size > size_limit:
        raise PI05ParityReportError(f"{label} size must be in [1,{size_limit}] bytes, got {size}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_reject_duplicate_keys)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise PI05ParityReportError(f"Cannot parse {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise PI05ParityReportError(f"{label} root must be a JSON object")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], *, label: str) -> None:
    actual = set(value)
    if actual != expected:
        raise PI05ParityReportError(
            f"{label} keys mismatch: missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )


def _mapping(value: object, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise PI05ParityReportError(f"{label} must be a JSON object")
    return value


def _boolean(value: object, *, label: str) -> bool:
    if not isinstance(value, bool):
        raise PI05ParityReportError(f"{label} must be boolean")
    return value


def _integer(value: object, *, label: str, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise PI05ParityReportError(f"{label} must be an integer")
    if minimum is not None and value < minimum:
        raise PI05ParityReportError(f"{label} must be >= {minimum}, got {value}")
    return value


def _number(value: object, *, label: str, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PI05ParityReportError(f"{label} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise PI05ParityReportError(f"{label} must be finite")
    if minimum is not None and result < minimum:
        raise PI05ParityReportError(f"{label} must be >= {minimum}, got {result}")
    return result


def _string(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise PI05ParityReportError(f"{label} must be a non-empty string")
    return value


def _sha256(value: object, *, label: str) -> str:
    if not _is_sha256(value):
        raise PI05ParityReportError(f"{label} must be 64 hexadecimal characters")
    return str(value).lower()


def _canonical_path(value: object, *, label: str) -> Path:
    raw = _string(value, label=label)
    resolved = Path(raw).expanduser().resolve()
    if raw != str(resolved):
        raise PI05ParityReportError(f"{label} must be an absolute canonical path: {resolved}")
    return resolved


def _file_record(
    value: object,
    *,
    label: str,
    verify_file: bool,
    digest_cache: dict[Path, str],
) -> tuple[Path, str]:
    record = _mapping(value, label=label)
    _exact_keys(record, {"path", "sha256"}, label=label)
    path = _canonical_path(record["path"], label=f"{label}.path")
    expected_sha256 = _sha256(record["sha256"], label=f"{label}.sha256")
    if verify_file:
        actual_sha256 = digest_cache.setdefault(path, file_sha256(path))
        if actual_sha256 != expected_sha256:
            raise PI05ParityReportError(
                f"{label} SHA-256 mismatch: got {actual_sha256}, expected {expected_sha256}"
            )
    return path, expected_sha256


def _config_training_max_delay(path: Path, *, label: str) -> int:
    document = _read_json(path, label=label)
    return _integer(
        document.get("rtc_training_max_delay"),
        label=f"{label}.rtc_training_max_delay",
        minimum=1,
    )


def load_pi05_parity_request(
    path: str | Path, *, label: str = "PI0.5 parity request"
) -> ValidatedPI05ParityRequest:
    """Strictly decode and fingerprint a serialized request without pickle support."""
    path = Path(path).expanduser().resolve()
    expected_archive_keys = {
        "metadata",
        "camera_0",
        "camera_1",
        "normalized_state",
        "normalized_prefill_actions",
    }
    try:
        with np.load(path, allow_pickle=False) as archive:
            if (
                len(archive.files) != len(expected_archive_keys)
                or set(archive.files) != expected_archive_keys
            ):
                raise PI05ParityReportError(
                    f"{label} archive keys mismatch: expected={sorted(expected_archive_keys)}, "
                    f"got={sorted(archive.files)}"
                )
            metadata_raw = archive["metadata"]
            if metadata_raw.shape != () or not isinstance(metadata_raw.item(), str):
                raise PI05ParityReportError(f"{label} metadata must be a scalar JSON string")
            metadata = json.loads(
                metadata_raw.item(),
                object_pairs_hook=_reject_duplicate_keys,
            )
            camera_0 = np.asarray(archive["camera_0"])
            camera_1 = np.asarray(archive["camera_1"])
            state = np.asarray(archive["normalized_state"])
            prefix = np.asarray(archive["normalized_prefill_actions"])
    except PI05ParityReportError:
        raise
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        raise PI05ParityReportError(f"Cannot load {label} {path}: {exc}") from exc

    if not isinstance(metadata, dict):
        raise PI05ParityReportError(f"{label} metadata must be a JSON object")
    _exact_keys(
        metadata,
        {"schema", "prompt", "camera_keys", "image_value_range", "fingerprint"},
        label=f"{label} metadata",
    )
    schema = _integer(metadata["schema"], label=f"{label}.metadata.schema", minimum=1)
    if schema != PI05_PARITY_REQUEST_SCHEMA_VERSION:
        raise PI05ParityReportError(f"{label} has unsupported request schema {schema!r}")
    prompt = _string(metadata["prompt"], label=f"{label}.metadata.prompt")
    raw_camera_keys = metadata["camera_keys"]
    if not isinstance(raw_camera_keys, list) or len(raw_camera_keys) != 2:
        raise PI05ParityReportError(f"{label}.metadata.camera_keys must contain exactly two strings")
    if any(not isinstance(key, str) or not key for key in raw_camera_keys):
        raise PI05ParityReportError(f"{label}.metadata.camera_keys must contain exactly two strings")
    if raw_camera_keys[0] == raw_camera_keys[1]:
        raise PI05ParityReportError(f"{label}.metadata.camera_keys must be unique")
    camera_keys = (raw_camera_keys[0], raw_camera_keys[1])
    image_value_range = _string(metadata["image_value_range"], label=f"{label}.metadata.image_value_range")
    if image_value_range not in {"uint8", "zero_one", "minus_one_one"}:
        raise PI05ParityReportError(
            f"{label}.metadata.image_value_range is unsupported: {image_value_range!r}"
        )
    reported_fingerprint = _sha256(metadata["fingerprint"], label=f"{label}.metadata.fingerprint")

    images = {camera_keys[0]: camera_0, camera_keys[1]: camera_1}
    for key, image in images.items():
        if (
            image.ndim != 3
            or 3 not in (image.shape[0], image.shape[-1])
            or any(dimension <= 0 for dimension in image.shape)
        ):
            raise PI05ParityReportError(
                f"{label} camera {key!r} must be non-empty HWC or CHW with 3 channels, got {image.shape}"
            )
        try:
            finite = bool(np.isfinite(image).all())
        except TypeError as exc:
            raise PI05ParityReportError(f"{label} camera {key!r} must be numeric") from exc
        if not finite:
            raise PI05ParityReportError(f"{label} camera {key!r} contains NaN or infinity")
        if image_value_range == "uint8":
            if image.dtype != np.uint8:
                raise PI05ParityReportError(
                    f"{label} camera {key!r} must be uint8 for image_value_range='uint8'"
                )
        elif not np.issubdtype(image.dtype, np.floating):
            raise PI05ParityReportError(
                f"{label} camera {key!r} must be floating point for {image_value_range!r}"
            )
        else:
            lower_bound = 0.0 if image_value_range == "zero_one" else -1.0
            if float(image.min()) < lower_bound - 1e-5 or float(image.max()) > 1.0 + 1e-5:
                raise PI05ParityReportError(
                    f"{label} camera {key!r} values are outside {image_value_range!r}"
                )

    if state.dtype != np.float32 or state.shape != (6,):
        raise PI05ParityReportError(
            f"{label}.normalized_state must be float32 with shape (6,), got {state.dtype} {state.shape}"
        )
    if prefix.dtype != np.float32 or prefix.ndim != 2 or prefix.shape[1:] != (6,):
        raise PI05ParityReportError(
            f"{label}.normalized_prefill_actions must be float32 with shape [T,6], "
            f"got {prefix.dtype} {prefix.shape}"
        )
    if prefix.shape[0] >= 50:
        raise PI05ParityReportError(f"{label} prefill length must be smaller than 50")
    if not np.isfinite(state).all() or not np.isfinite(prefix).all():
        raise PI05ParityReportError(f"{label} normalized state or prefix contains NaN or infinity")

    actual_fingerprint = compute_pi05_parity_request_fingerprint(
        images=images,
        normalized_state=state,
        normalized_prefill_actions=prefix,
        prompt=prompt,
        camera_keys=camera_keys,
        image_value_range=image_value_range,
    )
    if actual_fingerprint != reported_fingerprint:
        raise PI05ParityReportError(
            f"{label} fingerprint mismatch: got {actual_fingerprint}, expected {reported_fingerprint}"
        )
    return ValidatedPI05ParityRequest(
        images=images,
        normalized_state=state,
        normalized_prefill_actions=prefix,
        prompt=prompt,
        camera_keys=camera_keys,
        image_value_range=image_value_range,
        fingerprint=actual_fingerprint,
    )


def _load_serialized_parity_result(
    path: Path, *, label: str
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            if len(archive.files) != 3 or set(archive.files) != {"metadata", "actions", "noise"}:
                raise PI05ParityReportError(
                    f"{label} archive keys must be metadata/actions/noise, got {sorted(archive.files)}"
                )
            metadata_raw = archive["metadata"]
            if metadata_raw.shape != () or not isinstance(metadata_raw.item(), str):
                raise PI05ParityReportError(f"{label} metadata must be a scalar JSON string")
            metadata = json.loads(
                metadata_raw.item(),
                object_pairs_hook=_reject_duplicate_keys,
            )
            actions = np.asarray(archive["actions"])
            noise = np.asarray(archive["noise"])
    except PI05ParityReportError:
        raise
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        raise PI05ParityReportError(f"Cannot load {label} {path}: {exc}") from exc
    if not isinstance(metadata, dict):
        raise PI05ParityReportError(f"{label} metadata must be a JSON object")
    _exact_keys(
        metadata,
        {"schema", "implementation", "request_fingerprint", "prefill_length", "seed"},
        label=f"{label} metadata",
    )
    schema = _integer(metadata["schema"], label=f"{label}.metadata.schema", minimum=1)
    if schema != 1:
        raise PI05ParityReportError(f"{label} has unsupported result schema {schema!r}")
    normalized_metadata = {
        "schema": schema,
        "implementation": _string(metadata["implementation"], label=f"{label}.metadata.implementation"),
        "request_fingerprint": _sha256(
            metadata["request_fingerprint"], label=f"{label}.metadata.request_fingerprint"
        ),
        "prefill_length": _integer(
            metadata["prefill_length"], label=f"{label}.metadata.prefill_length", minimum=0
        ),
        "seed": _integer(metadata["seed"], label=f"{label}.metadata.seed", minimum=0),
    }
    if normalized_metadata["prefill_length"] >= 50:
        raise PI05ParityReportError(f"{label} prefill length must be smaller than 50")
    if actions.dtype != np.float32 or noise.dtype != np.float32:
        raise PI05ParityReportError(
            f"{label} actions and noise must be float32, got {actions.dtype} and {noise.dtype}"
        )
    if actions.shape != (50, 6) or noise.shape != (50, 32):
        raise PI05ParityReportError(
            f"{label} shapes must be actions=(50, 6), noise=(50, 32); "
            f"got actions={actions.shape}, noise={noise.shape}"
        )
    if not np.isfinite(actions).all() or not np.isfinite(noise).all():
        raise PI05ParityReportError(f"{label} contains NaN or infinity")
    return normalized_metadata, actions, noise


def _recompute_prefix_comparison(
    *,
    request_path: Path,
    pytorch_path: Path,
    triton_path: Path,
    atol: float,
    rtol: float,
    label: str,
) -> dict[str, Any]:
    request = load_pi05_parity_request(request_path, label=f"{label}.request")
    pytorch_metadata, pytorch_actions, pytorch_noise = _load_serialized_parity_result(
        pytorch_path, label=f"{label}.pytorch_output"
    )
    triton_metadata, triton_actions, triton_noise = _load_serialized_parity_result(
        triton_path, label=f"{label}.triton_output"
    )
    if pytorch_metadata["implementation"] != "lerobot-pytorch":
        raise PI05ParityReportError(f"{label} PyTorch implementation must be 'lerobot-pytorch'")
    if triton_metadata["implementation"] != "realtime-vla-v2-triton":
        raise PI05ParityReportError(f"{label} Triton implementation must be 'realtime-vla-v2-triton'")
    for field in ("request_fingerprint", "prefill_length", "seed"):
        if pytorch_metadata[field] != triton_metadata[field]:
            raise PI05ParityReportError(f"{label} result metadata disagrees on {field}")
    if pytorch_metadata["request_fingerprint"] != request.fingerprint:
        raise PI05ParityReportError(f"{label} results are not bound to the checksum-pinned request")
    if pytorch_metadata["prefill_length"] != request.prefill_length:
        raise PI05ParityReportError(
            f"{label} result prefill length does not match the checksum-pinned request"
        )
    seed = pytorch_metadata["seed"]
    try:
        expected_noise = np.random.default_rng(seed).standard_normal((50, 32)).astype(np.float32)
    except (TypeError, ValueError) as exc:
        raise PI05ParityReportError(f"{label} has invalid fixed diffusion noise seed {seed}") from exc
    if not np.array_equal(pytorch_noise, expected_noise):
        raise PI05ParityReportError(
            f"{label} PyTorch noise does not match fixed diffusion noise reconstructed from seed {seed}"
        )
    if not np.array_equal(triton_noise, expected_noise):
        raise PI05ParityReportError(
            f"{label} Triton noise does not match fixed diffusion noise reconstructed from seed {seed}"
        )
    error = np.abs(triton_actions - pytorch_actions)
    return {
        "prefix_length": request.prefill_length,
        "seed": seed,
        "request_fingerprint": request.fingerprint,
        "passed": bool(np.allclose(triton_actions, pytorch_actions, atol=atol, rtol=rtol)),
        "max_abs_error": float(error.max(initial=0.0)),
        "mean_abs_error": float(error.mean()) if error.size else 0.0,
    }


def _validate_report_document(
    document: Mapping[str, Any],
    *,
    verify_files: bool,
    require_passed: bool,
) -> dict[str, Any]:
    _exact_keys(
        document,
        {
            "report_type",
            "schema_version",
            "passed",
            "training_max_delay",
            "thresholds",
            "source_checkpoint",
            "triton_export",
            "prefix_results",
        },
        label="parity report",
    )
    if document["report_type"] != PI05_PARITY_REPORT_TYPE:
        raise PI05ParityReportError(f"Unsupported parity report type: {document['report_type']!r}")
    if document["schema_version"] != PI05_PARITY_REPORT_SCHEMA_VERSION:
        raise PI05ParityReportError(f"Unsupported parity report schema: {document['schema_version']!r}")
    training_max_delay = _integer(document["training_max_delay"], label="training_max_delay", minimum=1)

    thresholds = _mapping(document["thresholds"], label="thresholds")
    _exact_keys(thresholds, {"atol", "rtol"}, label="thresholds")
    atol = _number(thresholds["atol"], label="thresholds.atol", minimum=0.0)
    rtol = _number(thresholds["rtol"], label="thresholds.rtol", minimum=0.0)

    digest_cache: dict[Path, str] = {}
    checkpoint = _mapping(document["source_checkpoint"], label="source_checkpoint")
    _exact_keys(checkpoint, {"path", "config", "model"}, label="source_checkpoint")
    checkpoint_path = _canonical_path(checkpoint["path"], label="source_checkpoint.path")
    if verify_files and not checkpoint_path.is_dir():
        raise PI05ParityReportError(f"source_checkpoint.path is not a directory: {checkpoint_path}")
    checkpoint_config_path, checkpoint_config_sha256 = _file_record(
        checkpoint["config"],
        label="source_checkpoint.config",
        verify_file=verify_files,
        digest_cache=digest_cache,
    )
    checkpoint_model_path, checkpoint_model_sha256 = _file_record(
        checkpoint["model"],
        label="source_checkpoint.model",
        verify_file=verify_files,
        digest_cache=digest_cache,
    )
    if checkpoint_config_path != checkpoint_path / "config.json":
        raise PI05ParityReportError("source_checkpoint.config must be <checkpoint>/config.json")
    if checkpoint_model_path != checkpoint_path / "model.safetensors":
        raise PI05ParityReportError("source_checkpoint.model must be <checkpoint>/model.safetensors")

    triton = _mapping(document["triton_export"], label="triton_export")
    _exact_keys(triton, {"weights", "model_config"}, label="triton_export")
    triton_weights_path, triton_weights_sha256 = _file_record(
        triton["weights"],
        label="triton_export.weights",
        verify_file=verify_files,
        digest_cache=digest_cache,
    )
    triton_model_config_path, triton_model_config_sha256 = _file_record(
        triton["model_config"],
        label="triton_export.model_config",
        verify_file=verify_files,
        digest_cache=digest_cache,
    )

    if (
        _config_training_max_delay(checkpoint_config_path, label="source checkpoint config")
        != training_max_delay
    ):
        raise PI05ParityReportError("source checkpoint config training delay does not match parity report")
    if (
        _config_training_max_delay(triton_model_config_path, label="Triton model config")
        != training_max_delay
    ):
        raise PI05ParityReportError("Triton model config training delay does not match parity report")

    prefix_results = document["prefix_results"]
    if not isinstance(prefix_results, list):
        raise PI05ParityReportError("prefix_results must be a JSON array")
    expected_prefixes = tuple(range(training_max_delay + 1))
    if len(prefix_results) != len(expected_prefixes):
        raise PI05ParityReportError(
            "prefix_results must cover every trained prefix exactly: "
            f"got {len(prefix_results)} entries, expected {len(expected_prefixes)}"
        )

    normalized_results: list[dict[str, Any]] = []
    for expected_prefix, raw_result in zip(expected_prefixes, prefix_results, strict=True):
        label = f"prefix_results[{expected_prefix}]"
        result = _mapping(raw_result, label=label)
        _exact_keys(
            result,
            {
                "prefix_length",
                "passed",
                "max_abs_error",
                "mean_abs_error",
                "shape",
                "seed",
                "request_fingerprint",
                "request",
                "pytorch_output",
                "triton_output",
            },
            label=label,
        )
        prefix_length = _integer(result["prefix_length"], label=f"{label}.prefix_length", minimum=0)
        if prefix_length != expected_prefix:
            raise PI05ParityReportError(
                "prefix_results must be sorted and cover 0..training_max_delay exactly: "
                f"entry {expected_prefix} contains prefix {prefix_length}"
            )
        passed = _boolean(result["passed"], label=f"{label}.passed")
        max_abs_error = _number(result["max_abs_error"], label=f"{label}.max_abs_error", minimum=0.0)
        mean_abs_error = _number(result["mean_abs_error"], label=f"{label}.mean_abs_error", minimum=0.0)
        shape = result["shape"]
        if shape != [50, 6]:
            raise PI05ParityReportError(f"{label}.shape must be [50, 6], got {shape!r}")
        seed = _integer(result["seed"], label=f"{label}.seed", minimum=0)
        request_fingerprint = _sha256(result["request_fingerprint"], label=f"{label}.request_fingerprint")
        request_path, request_sha256 = _file_record(
            result["request"],
            label=f"{label}.request",
            verify_file=verify_files,
            digest_cache=digest_cache,
        )
        pytorch_output_path, pytorch_output_sha256 = _file_record(
            result["pytorch_output"],
            label=f"{label}.pytorch_output",
            verify_file=verify_files,
            digest_cache=digest_cache,
        )
        triton_output_path, triton_output_sha256 = _file_record(
            result["triton_output"],
            label=f"{label}.triton_output",
            verify_file=verify_files,
            digest_cache=digest_cache,
        )
        if verify_files:
            recomputed = _recompute_prefix_comparison(
                request_path=request_path,
                pytorch_path=pytorch_output_path,
                triton_path=triton_output_path,
                atol=atol,
                rtol=rtol,
                label=label,
            )
            for field, reported in (
                ("prefix_length", prefix_length),
                ("seed", seed),
                ("request_fingerprint", request_fingerprint),
                ("passed", passed),
            ):
                if recomputed[field] != reported:
                    raise PI05ParityReportError(
                        f"{label}.{field} disagrees with recomputed NPZ evidence: "
                        f"reported={reported!r}, recomputed={recomputed[field]!r}"
                    )
            for field, reported in (
                ("max_abs_error", max_abs_error),
                ("mean_abs_error", mean_abs_error),
            ):
                if not math.isclose(recomputed[field], reported, rel_tol=0.0, abs_tol=1e-12):
                    raise PI05ParityReportError(
                        f"{label}.{field} disagrees with recomputed NPZ evidence: "
                        f"reported={reported!r}, recomputed={recomputed[field]!r}"
                    )
        normalized_results.append(
            {
                "prefix_length": prefix_length,
                "passed": passed,
                "max_abs_error": max_abs_error,
                "mean_abs_error": mean_abs_error,
                "shape": [50, 6],
                "seed": seed,
                "request_fingerprint": request_fingerprint,
                "request": {
                    "path": str(request_path),
                    "sha256": request_sha256,
                },
                "pytorch_output": {
                    "path": str(pytorch_output_path),
                    "sha256": pytorch_output_sha256,
                },
                "triton_output": {
                    "path": str(triton_output_path),
                    "sha256": triton_output_sha256,
                },
            }
        )

    all_prefixes_passed = all(result["passed"] for result in normalized_results)
    aggregate_passed = _boolean(document["passed"], label="passed")
    if aggregate_passed != all_prefixes_passed:
        raise PI05ParityReportError("aggregate passed must equal the conjunction of all prefix results")
    if require_passed and not aggregate_passed:
        raise PI05ParityReportError("parity report did not pass every trained prefix length")

    return {
        "report_type": PI05_PARITY_REPORT_TYPE,
        "schema_version": PI05_PARITY_REPORT_SCHEMA_VERSION,
        "passed": aggregate_passed,
        "training_max_delay": training_max_delay,
        "thresholds": {"atol": atol, "rtol": rtol},
        "source_checkpoint": {
            "path": str(checkpoint_path),
            "config": {"path": str(checkpoint_config_path), "sha256": checkpoint_config_sha256},
            "model": {"path": str(checkpoint_model_path), "sha256": checkpoint_model_sha256},
        },
        "triton_export": {
            "weights": {"path": str(triton_weights_path), "sha256": triton_weights_sha256},
            "model_config": {
                "path": str(triton_model_config_path),
                "sha256": triton_model_config_sha256,
            },
        },
        "prefix_results": normalized_results,
    }


def _record_for_file(path: str | Path, *, expected_sha256: str | None = None) -> dict[str, str]:
    resolved = Path(path).expanduser().resolve()
    actual_sha256 = file_sha256(resolved)
    if expected_sha256 is not None:
        expected = _sha256(expected_sha256, label=f"expected SHA-256 for {resolved}")
        if actual_sha256 != expected:
            raise PI05ParityReportError(
                f"Artifact SHA-256 mismatch for {resolved}: got {actual_sha256}, expected {expected}"
            )
    return {"path": str(resolved), "sha256": actual_sha256}


def build_pi05_parity_report(
    *,
    checkpoint_path: str | Path,
    triton_weights_path: str | Path,
    triton_weights_sha256: str,
    triton_model_config_path: str | Path,
    training_max_delay: int,
    atol: float,
    rtol: float,
    prefix_results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build one aggregate report from already-computed per-prefix comparisons."""
    checkpoint = Path(checkpoint_path).expanduser().resolve()
    if not checkpoint.is_dir():
        raise PI05ParityReportError(f"Source checkpoint directory not found: {checkpoint}")
    training_max_delay = _integer(training_max_delay, label="training_max_delay", minimum=1)
    atol = _number(atol, label="atol", minimum=0.0)
    rtol = _number(rtol, label="rtol", minimum=0.0)

    normalized_by_prefix: dict[int, dict[str, Any]] = {}
    expected_input_keys = {
        "prefix_length",
        "passed",
        "max_abs_error",
        "mean_abs_error",
        "shape",
        "seed",
        "request_fingerprint",
        "request_path",
        "pytorch_output_path",
        "triton_output_path",
    }
    for index, raw_result in enumerate(prefix_results):
        _exact_keys(raw_result, expected_input_keys, label=f"prefix result input {index}")
        prefix = _integer(raw_result["prefix_length"], label="prefix_length", minimum=0)
        if prefix in normalized_by_prefix:
            raise PI05ParityReportError(f"Duplicate prefix result: {prefix}")
        normalized_by_prefix[prefix] = {
            "prefix_length": prefix,
            "passed": _boolean(raw_result["passed"], label=f"prefix {prefix}.passed"),
            "max_abs_error": _number(
                raw_result["max_abs_error"], label=f"prefix {prefix}.max_abs_error", minimum=0.0
            ),
            "mean_abs_error": _number(
                raw_result["mean_abs_error"], label=f"prefix {prefix}.mean_abs_error", minimum=0.0
            ),
            "shape": list(raw_result["shape"]),
            "seed": _integer(raw_result["seed"], label=f"prefix {prefix}.seed", minimum=0),
            "request_fingerprint": _sha256(
                raw_result["request_fingerprint"], label=f"prefix {prefix}.request_fingerprint"
            ),
            "request": _record_for_file(raw_result["request_path"]),
            "pytorch_output": _record_for_file(raw_result["pytorch_output_path"]),
            "triton_output": _record_for_file(raw_result["triton_output_path"]),
        }

    expected_prefixes = set(range(training_max_delay + 1))
    actual_prefixes = set(normalized_by_prefix)
    if actual_prefixes != expected_prefixes:
        raise PI05ParityReportError(
            "Aggregate parity input must cover 0..training_max_delay exactly: "
            f"missing={sorted(expected_prefixes - actual_prefixes)}, "
            f"extra={sorted(actual_prefixes - expected_prefixes)}"
        )
    ordered_results = [normalized_by_prefix[prefix] for prefix in range(training_max_delay + 1)]
    report = {
        "report_type": PI05_PARITY_REPORT_TYPE,
        "schema_version": PI05_PARITY_REPORT_SCHEMA_VERSION,
        "passed": all(result["passed"] for result in ordered_results),
        "training_max_delay": training_max_delay,
        "thresholds": {"atol": atol, "rtol": rtol},
        "source_checkpoint": {
            "path": str(checkpoint),
            "config": _record_for_file(checkpoint / "config.json"),
            "model": _record_for_file(checkpoint / "model.safetensors"),
        },
        "triton_export": {
            "weights": _record_for_file(triton_weights_path, expected_sha256=triton_weights_sha256),
            "model_config": _record_for_file(triton_model_config_path),
        },
        "prefix_results": ordered_results,
    }
    return _validate_report_document(report, verify_files=False, require_passed=False)


def write_pi05_parity_report_atomic(path: str | Path, report: Mapping[str, Any]) -> str:
    """Atomically persist a canonical report and return its SHA-256."""
    normalized = _validate_report_document(report, verify_files=False, require_passed=False)
    output_path = Path(path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(normalized, indent=2, sort_keys=True) + "\n").encode("utf-8")
    temporary_path: Path | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
        )
        temporary_path = Path(temporary_name)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, output_path)
        temporary_path = None
        directory_descriptor = os.open(output_path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return file_sha256(output_path)


def load_and_validate_pi05_parity_report(
    config: PI05ParityReportArtifactConfig,
    *,
    checkpoint_path: str | Path,
    triton_weights_path: str | Path,
    triton_weights_sha256: str,
    triton_model_config_path: str | Path,
    training_max_delay: int,
) -> ValidatedPI05ParityReport:
    """Fail closed unless the pinned report and every referenced artifact match."""
    if not isinstance(config, PI05ParityReportArtifactConfig):
        raise PI05ParityReportError("pi05_triton_parity_report must be a PI05ParityReportArtifactConfig")
    if not config.enabled or config.path is None or config.sha256 is None:
        raise PI05ParityReportError("checksum-pinned PI0.5 Triton parity report is required")
    report_path = Path(config.path).expanduser().resolve()
    expected_report_sha256 = _sha256(config.sha256, label="pi05_triton_parity_report.sha256")
    actual_report_sha256 = file_sha256(report_path)
    if actual_report_sha256 != expected_report_sha256:
        raise PI05ParityReportError(
            "PI0.5 parity report SHA-256 mismatch: "
            f"got {actual_report_sha256}, expected {expected_report_sha256}"
        )
    document = _read_json(report_path, label="PI0.5 parity report")
    report = _validate_report_document(document, verify_files=True, require_passed=True)

    expected_checkpoint = Path(checkpoint_path).expanduser().resolve()
    expected_weights = Path(triton_weights_path).expanduser().resolve()
    expected_model_config = Path(triton_model_config_path).expanduser().resolve()
    expected_weights_sha256 = _sha256(triton_weights_sha256, label="configured Triton weights SHA-256")
    expected_training_max_delay = _integer(
        training_max_delay, label="configured training_max_delay", minimum=1
    )
    if Path(report["source_checkpoint"]["path"]) != expected_checkpoint:
        raise PI05ParityReportError(
            "Parity report source checkpoint does not match rollout policy checkpoint"
        )
    if Path(report["triton_export"]["weights"]["path"]) != expected_weights:
        raise PI05ParityReportError("Parity report Triton weights path does not match rollout config")
    if report["triton_export"]["weights"]["sha256"] != expected_weights_sha256:
        raise PI05ParityReportError("Parity report Triton weights SHA-256 does not match rollout config")
    if Path(report["triton_export"]["model_config"]["path"]) != expected_model_config:
        raise PI05ParityReportError("Parity report model config path does not match rollout config")
    if report["training_max_delay"] != expected_training_max_delay:
        raise PI05ParityReportError(
            "Parity report prefix coverage does not match policy.rtc_training_max_delay"
        )

    results = report["prefix_results"]
    return ValidatedPI05ParityReport(
        path=str(report_path),
        sha256=actual_report_sha256,
        checkpoint_path=report["source_checkpoint"]["path"],
        checkpoint_config_sha256=report["source_checkpoint"]["config"]["sha256"],
        checkpoint_model_sha256=report["source_checkpoint"]["model"]["sha256"],
        triton_weights_sha256=report["triton_export"]["weights"]["sha256"],
        triton_model_config_sha256=report["triton_export"]["model_config"]["sha256"],
        training_max_delay=report["training_max_delay"],
        atol=report["thresholds"]["atol"],
        rtol=report["thresholds"]["rtol"],
        prefixes=tuple(result["prefix_length"] for result in results),
        max_abs_errors=tuple(result["max_abs_error"] for result in results),
    )
