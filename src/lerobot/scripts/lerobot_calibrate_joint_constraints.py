#!/usr/bin/env python

"""Build or validate checksum-pinned SO-101 joint dynamic constraints."""

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

from lerobot.rollout.joint_constraints import (
    ARTIFACT_TYPE,
    COORDINATE_SPACE,
    SCHEMA_VERSION,
    load_joint_constraint_artifact,
)

DEFAULT_SO101_JOINT_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)


class JointConstraintFitError(ValueError):
    """Raised when position traces cannot produce reliable dynamic limits."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_artifact(document: Mapping[str, Any], path: str | Path, *, overwrite: bool = False) -> str:
    destination = Path(path).expanduser().resolve()
    if destination.exists() and not overwrite:
        raise JointConstraintFitError(f"output already exists: {destination}; pass --overwrite")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            delete=False,
        ) as output:
            temporary = Path(output.name)
            json.dump(document, output, indent=2, allow_nan=False)
            output.write("\n")
        os.replace(temporary, destination)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return _sha256(destination)


def _finite(value: object, *, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise JointConstraintFitError(f"{context} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise JointConstraintFitError(f"{context} must be finite")
    return number


def _position_vector(value: object, names: tuple[str, ...], *, context: str) -> list[float]:
    if isinstance(value, Mapping):
        if set(value) != set(names):
            raise JointConstraintFitError(f"{context} keys must exactly match joint_names")
        return [_finite(value[name], context=f"{context}.{name}") for name in names]
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) != len(names):
        raise JointConstraintFitError(f"{context} must contain exactly {len(names)} values")
    return [_finite(item, context=f"{context}[{index}]") for index, item in enumerate(value)]


def load_position_trace(
    path: str | Path,
    *,
    joint_names: Sequence[str] = DEFAULT_SO101_JOINT_NAMES,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load monotonic positions, retaining episode IDs to avoid boundary derivatives."""

    source_path = Path(path).expanduser().resolve()
    if not source_path.is_file() or source_path.suffix.lower() not in {".jsonl", ".ndjson"}:
        raise JointConstraintFitError("joint constraint input must be an existing JSONL file")
    names = tuple(joint_names)
    if not names or len(set(names)) != len(names) or any(not name for name in names):
        raise JointConstraintFitError("joint_names must be unique non-empty strings")
    timestamps: list[float] = []
    positions: list[list[float]] = []
    episodes: list[int | str] = []
    with source_path.open(encoding="utf-8") as source:
        for line_number, raw_line in enumerate(source, start=1):
            if not raw_line.strip():
                continue
            context = f"{source_path}:{line_number}"
            try:
                row = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise JointConstraintFitError(f"{context}: invalid JSON") from exc
            if not isinstance(row, Mapping):
                raise JointConstraintFitError(f"{context}: row must be an object")
            if row.get("clock_domain") != "monotonic":
                raise JointConstraintFitError(f"{context}: clock_domain must be 'monotonic'")
            timestamp = row.get("monotonic_timestamp", row.get("timestamp"))
            timestamps.append(_finite(timestamp, context=f"{context}: timestamp"))
            positions.append(_position_vector(row.get("observed"), names, context=f"{context}: observed"))
            episode = row.get("episode_index", 0)
            if isinstance(episode, bool) or not isinstance(episode, (int, str)):
                raise JointConstraintFitError(f"{context}: episode_index must be an integer or string")
            episodes.append(episode)
    if len(timestamps) < 4:
        raise JointConstraintFitError("joint constraint input requires at least four samples")
    return (
        np.asarray(timestamps, dtype=np.float64),
        np.asarray(positions, dtype=np.float64),
        np.asarray(episodes, dtype=object),
    )


def derive_dynamics(
    timestamps: np.ndarray,
    positions: np.ndarray,
    episodes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Differentiate within episode boundaries using non-uniform timestamps."""

    velocities: list[np.ndarray] = []
    velocity_times: list[float] = []
    velocity_episodes: list[object] = []
    for index in range(1, len(timestamps)):
        if episodes[index] != episodes[index - 1]:
            continue
        dt = float(timestamps[index] - timestamps[index - 1])
        if not math.isfinite(dt) or dt <= 0.0:
            raise JointConstraintFitError("timestamps must increase strictly within each episode")
        velocities.append((positions[index] - positions[index - 1]) / dt)
        velocity_times.append(float((timestamps[index] + timestamps[index - 1]) * 0.5))
        velocity_episodes.append(episodes[index])
    if len(velocities) < 2:
        raise JointConstraintFitError("trace has too few within-episode velocity samples")

    velocity_array = np.asarray(velocities, dtype=np.float64)
    accelerations: list[np.ndarray] = []
    for index in range(1, len(velocity_array)):
        if velocity_episodes[index] != velocity_episodes[index - 1]:
            continue
        dt = velocity_times[index] - velocity_times[index - 1]
        if not math.isfinite(dt) or dt <= 0.0:
            raise JointConstraintFitError("velocity sample times must increase within each episode")
        accelerations.append((velocity_array[index] - velocity_array[index - 1]) / dt)
    if not accelerations:
        raise JointConstraintFitError("trace has too few within-episode acceleration samples")
    acceleration_array = np.asarray(accelerations, dtype=np.float64)
    if not np.isfinite(velocity_array).all() or not np.isfinite(acceleration_array).all():
        raise JointConstraintFitError("derived dynamics contain non-finite values")
    return velocity_array, acceleration_array


def build_joint_constraint_artifact(
    input_path: str | Path,
    *,
    joint_names: Sequence[str] = DEFAULT_SO101_JOINT_NAMES,
    quantile: float = 0.95,
    safety_factor: float = 1.0,
    method: str = "offline within-episode finite-difference absolute quantile",
) -> dict[str, Any]:
    if not math.isfinite(float(quantile)) or not 0.0 < float(quantile) <= 1.0:
        raise JointConstraintFitError("quantile must be in (0, 1]")
    if not math.isfinite(float(safety_factor)) or float(safety_factor) <= 0.0:
        raise JointConstraintFitError("safety_factor must be finite and positive")
    if not isinstance(method, str) or not method.strip():
        raise JointConstraintFitError("method must be non-empty")
    source_path = Path(input_path).expanduser().resolve()
    names = tuple(joint_names)
    timestamps, positions, episodes = load_position_trace(source_path, joint_names=names)
    velocities, accelerations = derive_dynamics(timestamps, positions, episodes)
    velocity_limits = np.quantile(np.abs(velocities), quantile, axis=0, method="higher") * float(
        safety_factor
    )
    acceleration_limits = np.quantile(np.abs(accelerations), quantile, axis=0, method="higher") * float(
        safety_factor
    )
    if np.any(velocity_limits <= 0.0) or np.any(acceleration_limits <= 0.0):
        raise JointConstraintFitError("every joint requires positive measured velocity and acceleration")
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "coordinate_space": COORDINATE_SPACE,
        "joint_names": list(names),
        "max_velocity": velocity_limits.tolist(),
        "max_acceleration": acceleration_limits.tolist(),
        "method": method.strip(),
        "quantile": float(quantile),
        "safety_factor": float(safety_factor),
        "source": {
            "path": str(source_path),
            "format": "jsonl",
            "sha256": _sha256(source_path),
            "sample_count": int(len(timestamps)),
            "velocity_sample_count": int(len(velocities)),
            "acceleration_sample_count": int(len(accelerations)),
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build", help="Build constraints from a measured position trace")
    build.add_argument("--input", required=True)
    build.add_argument("--output", required=True)
    build.add_argument("--joint-names", nargs="+", default=list(DEFAULT_SO101_JOINT_NAMES))
    build.add_argument("--quantile", type=float, default=0.95)
    build.add_argument("--safety-factor", type=float, default=1.0)
    build.add_argument("--method", default="offline within-episode finite-difference absolute quantile")
    build.add_argument("--overwrite", action="store_true")
    validate = subparsers.add_parser("validate", help="Validate a pinned constraint artifact")
    validate.add_argument("--artifact", required=True)
    validate.add_argument("--sha256", required=True)
    validate.add_argument("--joint-names", nargs="+")
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "build":
        document = build_joint_constraint_artifact(
            args.input,
            joint_names=args.joint_names,
            quantile=args.quantile,
            safety_factor=args.safety_factor,
            method=args.method,
        )
        sha256 = write_artifact(document, args.output, overwrite=args.overwrite)
        load_joint_constraint_artifact(
            args.output, expected_sha256=sha256, joint_names=document["joint_names"]
        )
        print(f"wrote joint constraint artifact: {Path(args.output).expanduser().resolve()}")
        print(f"sha256: {sha256}")
        return
    artifact = load_joint_constraint_artifact(
        args.artifact, expected_sha256=args.sha256, joint_names=args.joint_names
    )
    print(json.dumps(artifact.audit_snapshot(), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
