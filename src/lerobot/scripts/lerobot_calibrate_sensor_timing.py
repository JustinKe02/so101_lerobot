#!/usr/bin/env python

"""Build or validate an offline sensor timing calibration artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from lerobot.rollout.sensor_timing_calibration import (
    ARTIFACT_TYPE,
    MONOTONIC_CLOCK_DOMAIN,
    SCHEMA_VERSION,
    load_sensor_timing_calibration_artifact,
)


class SensorTimingFitError(ValueError):
    """Raised when offline sensor timing samples cannot produce an artifact."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _finite_nonnegative(value: object, *, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SensorTimingFitError(f"{context} must be numeric")
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise SensorTimingFitError(f"{context} must be finite and non-negative")
    return number


def load_sensor_timing_samples(
    path: str | Path,
    *,
    camera_keys: Sequence[str] | None = None,
) -> tuple[tuple[str, ...], np.ndarray, np.ndarray, np.ndarray]:
    """Load measured latency rows from a monotonic JSONL source."""

    source_path = Path(path).expanduser().resolve()
    if not source_path.is_file() or source_path.suffix.lower() not in {".jsonl", ".ndjson"}:
        raise SensorTimingFitError("sensor timing input must be an existing .jsonl or .ndjson file")
    expected_keys = tuple(camera_keys) if camera_keys is not None else None
    if expected_keys is not None and (
        not expected_keys
        or len(set(expected_keys)) != len(expected_keys)
        or any(not isinstance(key, str) or not key.strip() for key in expected_keys)
    ):
        raise SensorTimingFitError("camera_keys must be unique non-empty strings")

    camera_rows: list[list[float]] = []
    state_rows: list[float] = []
    skew_rows: list[float] = []
    with source_path.open(encoding="utf-8") as source:
        for line_number, raw_line in enumerate(source, start=1):
            if not raw_line.strip():
                continue
            context = f"{source_path}:{line_number}"
            try:
                row = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise SensorTimingFitError(f"{context}: invalid JSON") from exc
            if not isinstance(row, Mapping):
                raise SensorTimingFitError(f"{context}: row must be an object")
            if row.get("clock_domain") != MONOTONIC_CLOCK_DOMAIN:
                raise SensorTimingFitError(f"{context}: clock_domain must be 'monotonic'")
            delays = row.get("camera_capture_delay_s")
            if not isinstance(delays, Mapping) or not delays:
                raise SensorTimingFitError(f"{context}: camera_capture_delay_s must be an object")
            if expected_keys is None:
                expected_keys = tuple(delays)
                if any(not isinstance(key, str) or not key.strip() for key in expected_keys):
                    raise SensorTimingFitError(f"{context}: camera keys must be non-empty strings")
            if set(delays) != set(expected_keys):
                raise SensorTimingFitError(
                    f"{context}: camera keys differ from expected order {list(expected_keys)!r}"
                )
            camera_rows.append(
                [
                    _finite_nonnegative(delays[key], context=f"{context}: camera delay {key!r}")
                    for key in expected_keys
                ]
            )
            state_rows.append(
                _finite_nonnegative(
                    row.get("state_observation_delay_s"),
                    context=f"{context}: state_observation_delay_s",
                )
            )
            skew_rows.append(
                _finite_nonnegative(row.get("camera_skew_s"), context=f"{context}: camera_skew_s")
            )
    if expected_keys is None or len(camera_rows) < 3:
        raise SensorTimingFitError("sensor timing input requires at least three complete samples")
    return (
        expected_keys,
        np.asarray(camera_rows, dtype=np.float64),
        np.asarray(state_rows, dtype=np.float64),
        np.asarray(skew_rows, dtype=np.float64),
    )


def build_sensor_timing_artifact(
    input_path: str | Path,
    *,
    method: str,
    quantile: float = 0.99,
    camera_keys: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Aggregate measured latency samples with a conservative higher quantile."""

    if not isinstance(method, str) or not method.strip():
        raise SensorTimingFitError("method must be a non-empty measurement description")
    if not math.isfinite(float(quantile)) or not 0.0 < float(quantile) <= 1.0:
        raise SensorTimingFitError("quantile must be in (0, 1]")
    source_path = Path(input_path).expanduser().resolve()
    keys, camera_delays, state_delays, camera_skews = load_sensor_timing_samples(
        source_path, camera_keys=camera_keys
    )
    camera_q = np.quantile(camera_delays, quantile, axis=0, method="higher")
    state_q = float(np.quantile(state_delays, quantile, method="higher"))
    skew_q = float(np.quantile(camera_skews, quantile, method="higher"))
    if skew_q <= 0.0:
        raise SensorTimingFitError(
            "calibrated camera skew bound must be positive; collect non-degenerate timing samples"
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "clock_domain": MONOTONIC_CLOCK_DOMAIN,
        "camera_keys": list(keys),
        "camera_capture_delay_s": {key: float(value) for key, value in zip(keys, camera_q, strict=True)},
        "state_observation_delay_s": state_q,
        "max_camera_skew_s": skew_q,
        "method": method.strip(),
        "quantile": float(quantile),
        "source": {
            "path": str(source_path),
            "format": "jsonl",
            "sha256": _sha256(source_path),
            "sample_count": int(camera_delays.shape[0]),
        },
    }


def write_artifact(document: Mapping[str, Any], path: str | Path, *, overwrite: bool = False) -> str:
    destination = Path(path).expanduser().resolve()
    if destination.exists() and not overwrite:
        raise SensorTimingFitError(f"output already exists: {destination}; pass --overwrite")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=destination.parent, prefix=f".{destination.name}.", delete=False
        ) as output:
            temporary = Path(output.name)
            json.dump(document, output, indent=2, allow_nan=False)
            output.write("\n")
        os.replace(temporary, destination)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return _sha256(destination)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build", help="Build an artifact from measured JSONL samples")
    build.add_argument("--input", required=True)
    build.add_argument("--output", required=True)
    build.add_argument("--method", required=True)
    build.add_argument("--quantile", type=float, default=0.99)
    build.add_argument("--camera-keys", nargs="+")
    build.add_argument("--overwrite", action="store_true")
    validate = subparsers.add_parser("validate", help="Validate a pinned artifact")
    validate.add_argument("--artifact", required=True)
    validate.add_argument("--sha256", required=True)
    validate.add_argument("--camera-keys", nargs="+")
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "build":
        document = build_sensor_timing_artifact(
            args.input,
            method=args.method,
            quantile=args.quantile,
            camera_keys=args.camera_keys,
        )
        sha256 = write_artifact(document, args.output, overwrite=args.overwrite)
        load_sensor_timing_calibration_artifact(
            args.output, expected_sha256=sha256, camera_keys=document["camera_keys"]
        )
        print(f"wrote sensor timing artifact: {Path(args.output).expanduser().resolve()}")
        print(f"sha256: {sha256}")
        return
    artifact = load_sensor_timing_calibration_artifact(
        args.artifact, expected_sha256=args.sha256, camera_keys=args.camera_keys
    )
    print(json.dumps(artifact.audit_snapshot(), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
