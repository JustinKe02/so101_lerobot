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

"""Fit an SO-101 pure-delay, first-order actuator model from an offline trace.

This module deliberately has no robot, motor, camera, or rollout imports. It
only reads an existing JSONL/CSV file and writes a calibration artifact.

Example:

```shell
uv run python -m lerobot.scripts.lerobot_calibrate_so101_actuator \
  --input actuator_trace.jsonl \
  --output so101_actuator_calibration.json \
  --max-delay-s 0.25 \
  --max-tau-s 1.0
```

Each input row must contain a strictly increasing monotonic timestamp and six
command/observed values. JSONL accepts arrays in ``command`` and ``observed``.
CSV accepts those arrays as JSON cells or flattened columns such as
``command_0,...,command_5`` and ``observed_0,...,observed_5``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from lerobot.utils.import_utils import require_package

SCHEMA_VERSION = 1
ARTIFACT_TYPE = "lerobot.so101_actuator_calibration"
MONOTONIC_CLOCK_DOMAIN = "monotonic"
DEFAULT_SO101_JOINT_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)


class CalibrationError(ValueError):
    """Raised when a trace cannot produce a trustworthy calibration."""


@dataclass(frozen=True)
class CalibrationSamples:
    """Validated samples loaded from one immutable source file."""

    timestamps: NDArray[np.float64]
    commands: NDArray[np.float64]
    observed: NDArray[np.float64]
    source_path: Path
    source_format: str
    source_sha256: str

    @property
    def sample_count(self) -> int:
        return int(len(self.timestamps))


@dataclass(frozen=True)
class FitConfig:
    """Bounds and data-quality requirements for actuator identification."""

    max_delay_s: float = 0.30
    min_tau_s: float = 0.01
    max_tau_s: float = 1.00
    min_fit_samples: int = 30
    min_command_span: float = 1e-3
    coarse_delay_steps: int = 13
    coarse_tau_steps: int = 9
    local_start_count: int = 4
    max_nfev: int = 250

    def __post_init__(self) -> None:
        for name in ("max_delay_s", "min_tau_s", "max_tau_s", "min_command_span"):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise CalibrationError(f"{name} must be finite")
        if self.max_delay_s < 0.0:
            raise CalibrationError("max_delay_s must be non-negative")
        if self.min_tau_s <= 0.0:
            raise CalibrationError("min_tau_s must be positive")
        if self.max_tau_s <= self.min_tau_s:
            raise CalibrationError("max_tau_s must be greater than min_tau_s")
        if self.min_command_span <= 0.0:
            raise CalibrationError("min_command_span must be positive")
        for name, minimum in (
            ("min_fit_samples", 3),
            ("coarse_delay_steps", 2),
            ("coarse_tau_steps", 2),
            ("local_start_count", 1),
            ("max_nfev", 1),
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise CalibrationError(f"{name} must be an integer >= {minimum}")


@dataclass(frozen=True)
class JointFit:
    """Identified parameters and one-step prediction diagnostics."""

    joint_name: str
    command_delay_s: float
    tau_s: float
    residual_rmse: float
    residual_mae: float
    residual_rss: float
    r_squared: float | None
    sample_count: int
    command_span: float
    optimizer_nfev: int
    optimizer_status: int
    at_delay_bound: bool
    at_tau_bound: bool


def _source_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _line_context(path: Path, line_number: int) -> str:
    return f"{path}:{line_number}"


def _parse_finite_float(value: Any, *, name: str, context: str) -> float:
    if isinstance(value, bool):
        raise CalibrationError(f"{context}: {name} must be a finite number, not boolean")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise CalibrationError(f"{context}: {name} must be a finite number") from exc
    if not math.isfinite(result):
        raise CalibrationError(f"{context}: {name} must be finite")
    return result


def _parse_vector(value: Any, *, name: str, size: int, context: str) -> list[float]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise CalibrationError(
                f"{context}: {name} must be a JSON array with {size} numeric values"
            ) from exc
    if isinstance(value, (str, bytes, Mapping)) or not isinstance(value, Sequence):
        raise CalibrationError(f"{context}: {name} must be an array with {size} numeric values")
    if len(value) != size:
        raise CalibrationError(f"{context}: {name} must contain exactly {size} values, got {len(value)}")
    return [
        _parse_finite_float(item, name=f"{name}[{index}]", context=context)
        for index, item in enumerate(value)
    ]


def _resolve_timestamp(record: Mapping[str, Any], timestamp_key: str, *, context: str) -> float:
    if timestamp_key == "auto":
        available = [key for key in ("monotonic_timestamp", "timestamp") if key in record]
        if not available:
            raise CalibrationError(
                f"{context}: missing timestamp; expected 'monotonic_timestamp' or 'timestamp'"
            )
        key = available[0]
    else:
        key = timestamp_key
        if key not in record:
            raise CalibrationError(f"{context}: missing timestamp field {key!r}")
    return _parse_finite_float(record[key], name=key, context=context)


def _validate_clock_domain(record: Mapping[str, Any], *, context: str) -> None:
    value = record.get("clock_domain")
    if value is None or (isinstance(value, str) and not value.strip()):
        return
    if not isinstance(value, str) or value.strip().lower() != MONOTONIC_CLOCK_DOMAIN:
        raise CalibrationError(
            f"{context}: clock_domain must be {MONOTONIC_CLOCK_DOMAIN!r}; wall/unix clocks are rejected"
        )


def _flattened_csv_vector(
    record: Mapping[str, Any],
    *,
    name: str,
    size: int,
    context: str,
) -> list[float]:
    if name in record and record[name] not in (None, ""):
        return _parse_vector(record[name], name=name, size=size, context=context)

    naming_patterns = (
        tuple(f"{name}_{index}" for index in range(size)),
        tuple(f"{name}[{index}]" for index in range(size)),
        tuple(f"{name}.{index}" for index in range(size)),
    )
    for columns in naming_patterns:
        if all(column in record and record[column] not in (None, "") for column in columns):
            return [_parse_finite_float(record[column], name=column, context=context) for column in columns]
    expected = ",".join(naming_patterns[0])
    raise CalibrationError(f"{context}: missing {name!r} JSON array or complete flattened columns {expected}")


def _load_jsonl_rows(
    path: Path,
    *,
    timestamp_key: str,
    command_key: str,
    observed_key: str,
    action_dim: int,
) -> tuple[list[float], list[list[float]], list[list[float]]]:
    timestamps: list[float] = []
    commands: list[list[float]] = []
    observed: list[list[float]] = []
    with path.open(encoding="utf-8") as source:
        for line_number, raw_line in enumerate(source, start=1):
            if not raw_line.strip():
                continue
            context = _line_context(path, line_number)
            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise CalibrationError(f"{context}: invalid JSON") from exc
            if not isinstance(record, Mapping):
                raise CalibrationError(f"{context}: each JSONL row must be an object")
            _validate_clock_domain(record, context=context)
            timestamps.append(_resolve_timestamp(record, timestamp_key, context=context))
            if command_key not in record:
                raise CalibrationError(f"{context}: missing command field {command_key!r}")
            if observed_key not in record:
                raise CalibrationError(f"{context}: missing observed field {observed_key!r}")
            commands.append(
                _parse_vector(record[command_key], name=command_key, size=action_dim, context=context)
            )
            observed.append(
                _parse_vector(record[observed_key], name=observed_key, size=action_dim, context=context)
            )
    return timestamps, commands, observed


def _load_csv_rows(
    path: Path,
    *,
    timestamp_key: str,
    command_key: str,
    observed_key: str,
    action_dim: int,
) -> tuple[list[float], list[list[float]], list[list[float]]]:
    timestamps: list[float] = []
    commands: list[list[float]] = []
    observed: list[list[float]] = []
    with path.open(newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames is None:
            raise CalibrationError(f"{path}: CSV file has no header")
        for line_number, record in enumerate(reader, start=2):
            context = _line_context(path, line_number)
            _validate_clock_domain(record, context=context)
            timestamps.append(_resolve_timestamp(record, timestamp_key, context=context))
            commands.append(
                _flattened_csv_vector(
                    record,
                    name=command_key,
                    size=action_dim,
                    context=context,
                )
            )
            observed.append(
                _flattened_csv_vector(
                    record,
                    name=observed_key,
                    size=action_dim,
                    context=context,
                )
            )
    return timestamps, commands, observed


def load_calibration_samples(
    path: str | Path,
    *,
    action_dim: int = 6,
    timestamp_key: str = "auto",
    command_key: str = "command",
    observed_key: str = "observed",
) -> CalibrationSamples:
    """Load and strictly validate a monotonic JSONL/CSV calibration trace."""

    source_path = Path(path).expanduser()
    if not source_path.is_file():
        raise CalibrationError(f"input trace does not exist or is not a file: {source_path}")
    if isinstance(action_dim, bool) or not isinstance(action_dim, int) or action_dim < 1:
        raise CalibrationError("action_dim must be a positive integer")
    for name, value in (
        ("timestamp_key", timestamp_key),
        ("command_key", command_key),
        ("observed_key", observed_key),
    ):
        if not isinstance(value, str) or not value.strip():
            raise CalibrationError(f"{name} must be a non-empty string")

    suffix = source_path.suffix.lower()
    loader: Callable[..., tuple[list[float], list[list[float]], list[list[float]]]]
    if suffix in {".jsonl", ".ndjson"}:
        source_format = "jsonl"
        loader = _load_jsonl_rows
    elif suffix == ".csv":
        source_format = "csv"
        loader = _load_csv_rows
    else:
        raise CalibrationError("input trace must use a .jsonl, .ndjson, or .csv extension")

    timestamps, commands, observed = loader(
        source_path,
        timestamp_key=timestamp_key,
        command_key=command_key,
        observed_key=observed_key,
        action_dim=action_dim,
    )
    if len(timestamps) < 3:
        raise CalibrationError(f"{source_path}: at least 3 samples are required")

    timestamp_array = np.asarray(timestamps, dtype=np.float64)
    command_array = np.asarray(commands, dtype=np.float64)
    observed_array = np.asarray(observed, dtype=np.float64)
    non_increasing = np.flatnonzero(np.diff(timestamp_array) <= 0.0)
    if len(non_increasing):
        row = int(non_increasing[0] + 2)
        raise CalibrationError(
            f"{source_path}: timestamps must be strictly increasing in the monotonic clock domain; "
            f"sample {row} is not later than sample {row - 1}"
        )

    return CalibrationSamples(
        timestamps=timestamp_array,
        commands=command_array,
        observed=observed_array,
        source_path=source_path.resolve(),
        source_format=source_format,
        source_sha256=_source_sha256(source_path),
    )


def _load_least_squares() -> Callable[..., Any]:
    try:
        require_package("scipy", extra="scipy-dep")
        from scipy.optimize import least_squares
    except ImportError as exc:
        raise RuntimeError(
            "SO-101 actuator calibration requires SciPy. Install it with "
            "`uv sync --extra scipy-dep` or `pip install 'lerobot[scipy-dep]'`."
        ) from exc
    return least_squares


def _advance_first_order(position: float, command: float, duration_s: float, tau_s: float) -> float:
    if duration_s <= 0.0:
        return position
    decay = math.exp(-duration_s / tau_s)
    return command + (position - command) * decay


def _predict_transition(
    timestamps: NDArray[np.float64],
    command: NDArray[np.float64],
    position: float,
    start_s: float,
    end_s: float,
    delay_s: float,
    tau_s: float,
) -> float:
    """Integrate a delayed zero-order-held command over one observation interval."""

    active_index = int(np.searchsorted(timestamps, start_s - delay_s, side="right") - 1)
    if active_index < 0:
        raise CalibrationError("transition predates the available command history")

    cursor = start_s
    next_index = active_index + 1
    while next_index < len(timestamps):
        change_s = float(timestamps[next_index] + delay_s)
        if change_s >= end_s:
            break
        if change_s > cursor:
            position = _advance_first_order(
                position,
                float(command[active_index]),
                change_s - cursor,
                tau_s,
            )
            cursor = change_s
        active_index = next_index
        next_index += 1
    return _advance_first_order(
        position,
        float(command[active_index]),
        end_s - cursor,
        tau_s,
    )


def _transition_indices(timestamps: NDArray[np.float64], config: FitConfig) -> NDArray[np.int64]:
    # A fixed index set keeps the objective dimension constant across candidate delays.
    earliest_start = float(timestamps[0] + config.max_delay_s)
    starts = timestamps[:-1]
    indices = np.flatnonzero(starts >= earliest_start - 1e-12).astype(np.int64) + 1
    if len(indices) < config.min_fit_samples:
        duration = float(timestamps[-1] - timestamps[0])
        raise CalibrationError(
            "trace is too short after reserving command-delay history: "
            f"{len(indices)} fit samples available, {config.min_fit_samples} required "
            f"(duration={duration:.6g}s, max_delay_s={config.max_delay_s:.6g})"
        )
    return indices


def _joint_residuals(
    parameters: Sequence[float],
    *,
    timestamps: NDArray[np.float64],
    command: NDArray[np.float64],
    observed: NDArray[np.float64],
    indices: NDArray[np.int64],
    fixed_delay_s: float | None,
) -> NDArray[np.float64]:
    if fixed_delay_s is None:
        delay_s = float(parameters[0])
        tau_s = math.exp(float(parameters[1]))
    else:
        delay_s = fixed_delay_s
        tau_s = math.exp(float(parameters[0]))

    residuals = np.empty(len(indices), dtype=np.float64)
    for output_index, sample_index in enumerate(indices):
        prediction = _predict_transition(
            timestamps,
            command,
            float(observed[sample_index - 1]),
            float(timestamps[sample_index - 1]),
            float(timestamps[sample_index]),
            delay_s,
            tau_s,
        )
        residuals[output_index] = prediction - float(observed[sample_index])
    return residuals


def _fit_joint(
    joint_name: str,
    timestamps: NDArray[np.float64],
    command: NDArray[np.float64],
    observed: NDArray[np.float64],
    indices: NDArray[np.int64],
    config: FitConfig,
    least_squares: Callable[..., Any],
) -> JointFit:
    command_span = float(np.ptp(command[indices - 1]))
    if command_span < config.min_command_span:
        raise CalibrationError(
            f"joint {joint_name!r} has insufficient command excitation: "
            f"span={command_span:.6g}, required>={config.min_command_span:.6g}"
        )

    fixed_delay_s = 0.0 if config.max_delay_s == 0.0 else None
    delay_grid = (
        np.asarray([0.0], dtype=np.float64)
        if fixed_delay_s is not None
        else np.linspace(0.0, config.max_delay_s, config.coarse_delay_steps)
    )
    tau_grid = np.geomspace(config.min_tau_s, config.max_tau_s, config.coarse_tau_steps)
    coarse: list[tuple[float, float, float]] = []
    for delay_s in delay_grid:
        for tau_s in tau_grid:
            packed = (
                [math.log(float(tau_s))]
                if fixed_delay_s is not None
                else [float(delay_s), math.log(float(tau_s))]
            )
            residual = _joint_residuals(
                packed,
                timestamps=timestamps,
                command=command,
                observed=observed,
                indices=indices,
                fixed_delay_s=fixed_delay_s,
            )
            coarse.append((float(np.dot(residual, residual)), float(delay_s), float(tau_s)))
    coarse.sort(key=lambda item: item[0])

    starts: list[tuple[float, float]] = []
    for _, delay_s, tau_s in coarse:
        if any(
            abs(delay_s - previous_delay) < 1e-12 and abs(tau_s - previous_tau) < 1e-12
            for previous_delay, previous_tau in starts
        ):
            continue
        starts.append((delay_s, tau_s))
        if len(starts) == config.local_start_count:
            break

    solutions: list[tuple[float, Any, float, float]] = []
    log_tau_bounds = (math.log(config.min_tau_s), math.log(config.max_tau_s))
    for delay_start, tau_start in starts:
        if fixed_delay_s is not None:
            x0 = np.asarray([math.log(tau_start)], dtype=np.float64)
            bounds = ([log_tau_bounds[0]], [log_tau_bounds[1]])
            x_scale: str | list[float] = [max(log_tau_bounds[1] - log_tau_bounds[0], 1.0)]
        else:
            x0 = np.asarray([delay_start, math.log(tau_start)], dtype=np.float64)
            bounds = ([0.0, log_tau_bounds[0]], [config.max_delay_s, log_tau_bounds[1]])
            x_scale = [max(config.max_delay_s, 1e-3), max(log_tau_bounds[1] - log_tau_bounds[0], 1.0)]
        solution = least_squares(
            _joint_residuals,
            x0,
            bounds=bounds,
            x_scale=x_scale,
            max_nfev=config.max_nfev,
            kwargs={
                "timestamps": timestamps,
                "command": command,
                "observed": observed,
                "indices": indices,
                "fixed_delay_s": fixed_delay_s,
            },
        )
        delay_s = fixed_delay_s if fixed_delay_s is not None else float(solution.x[0])
        log_tau = float(solution.x[0] if fixed_delay_s is not None else solution.x[1])
        tau_s = math.exp(log_tau)
        residual = _joint_residuals(
            solution.x,
            timestamps=timestamps,
            command=command,
            observed=observed,
            indices=indices,
            fixed_delay_s=fixed_delay_s,
        )
        solutions.append((float(np.dot(residual, residual)), solution, float(delay_s), tau_s))

    if not solutions:
        raise CalibrationError(f"joint {joint_name!r}: optimizer produced no solution")
    rss, solution, delay_s, tau_s = min(solutions, key=lambda item: item[0])
    if not np.isfinite([rss, delay_s, tau_s]).all():
        raise CalibrationError(f"joint {joint_name!r}: optimizer produced non-finite parameters")

    residual = _joint_residuals(
        solution.x,
        timestamps=timestamps,
        command=command,
        observed=observed,
        indices=indices,
        fixed_delay_s=fixed_delay_s,
    )
    targets = observed[indices]
    centered_rss = float(np.dot(targets - np.mean(targets), targets - np.mean(targets)))
    r_squared = 1.0 - rss / centered_rss if centered_rss > np.finfo(np.float64).eps else None
    delay_tolerance = max(config.max_delay_s * 1e-4, 1e-7)
    tau_tolerance = max((config.max_tau_s - config.min_tau_s) * 1e-4, 1e-7)

    return JointFit(
        joint_name=joint_name,
        command_delay_s=delay_s,
        tau_s=tau_s,
        residual_rmse=float(np.sqrt(np.mean(np.square(residual)))),
        residual_mae=float(np.mean(np.abs(residual))),
        residual_rss=rss,
        r_squared=r_squared,
        sample_count=int(len(residual)),
        command_span=command_span,
        optimizer_nfev=int(solution.nfev),
        optimizer_status=int(solution.status),
        at_delay_bound=(
            fixed_delay_s is not None
            or delay_s <= delay_tolerance
            or delay_s >= config.max_delay_s - delay_tolerance
        ),
        at_tau_bound=(tau_s <= config.min_tau_s + tau_tolerance or tau_s >= config.max_tau_s - tau_tolerance),
    )


def fit_calibration(
    samples: CalibrationSamples,
    *,
    joint_names: Sequence[str] = DEFAULT_SO101_JOINT_NAMES,
    config: FitConfig | None = None,
) -> tuple[JointFit, ...]:
    """Fit independent delayed first-order dynamics for every SO-101 joint."""

    config = config or FitConfig()
    names = tuple(joint_names)
    action_dim = samples.commands.shape[1]
    if len(names) != action_dim:
        raise CalibrationError(f"expected {action_dim} joint names, got {len(names)}")
    if any(not isinstance(name, str) or not name.strip() for name in names):
        raise CalibrationError("joint names must be non-empty strings")
    if len(set(names)) != len(names):
        raise CalibrationError("joint names must be unique")
    if samples.commands.shape != samples.observed.shape:
        raise CalibrationError("commands and observed arrays must have identical shapes")
    if samples.commands.shape[0] != len(samples.timestamps):
        raise CalibrationError("sample timestamps and vectors have inconsistent lengths")

    indices = _transition_indices(samples.timestamps, config)
    least_squares = _load_least_squares()
    return tuple(
        _fit_joint(
            joint_name,
            samples.timestamps,
            samples.commands[:, joint_index],
            samples.observed[:, joint_index],
            indices,
            config,
            least_squares,
        )
        for joint_index, joint_name in enumerate(names)
    )


def build_artifact(
    samples: CalibrationSamples,
    fits: Sequence[JointFit],
    *,
    config: FitConfig,
) -> dict[str, Any]:
    """Build a JSON-serializable, versioned calibration artifact."""

    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "clock_domain": MONOTONIC_CLOCK_DOMAIN,
        "joint_names": [fit.joint_name for fit in fits],
        "command_delay_s": [fit.command_delay_s for fit in fits],
        "tau_s": [fit.tau_s for fit in fits],
        "model": {
            "type": "pure_delay_first_order",
            "equation": "dx/dt = (command(t - delay) - x) / tau",
            "command_interpolation": "zero_order_hold",
            "residual": "one_step_prediction",
        },
        "fit_diagnostics": [
            {
                "joint_name": fit.joint_name,
                "residual_rmse": fit.residual_rmse,
                "residual_mae": fit.residual_mae,
                "residual_rss": fit.residual_rss,
                "r_squared": fit.r_squared,
                "sample_count": fit.sample_count,
                "command_span": fit.command_span,
                "optimizer_nfev": fit.optimizer_nfev,
                "optimizer_status": fit.optimizer_status,
                "at_delay_bound": fit.at_delay_bound,
                "at_tau_bound": fit.at_tau_bound,
            }
            for fit in fits
        ],
        "fit_config": {
            "max_delay_s": config.max_delay_s,
            "min_tau_s": config.min_tau_s,
            "max_tau_s": config.max_tau_s,
            "min_fit_samples": config.min_fit_samples,
            "min_command_span": config.min_command_span,
        },
        "source": {
            "path": str(samples.source_path),
            "format": samples.source_format,
            "sha256": samples.source_sha256,
            "sample_count": samples.sample_count,
        },
    }


def write_artifact(artifact: Mapping[str, Any], output_path: str | Path, *, overwrite: bool = False) -> Path:
    """Atomically write an artifact, refusing accidental replacement by default."""

    destination = Path(output_path).expanduser().resolve()
    if destination.exists() and not overwrite:
        raise CalibrationError(f"output already exists: {destination}; pass --overwrite to replace it")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as output:
            temporary_path = Path(output.name)
            json.dump(artifact, output, indent=2, allow_nan=False)
            output.write("\n")
        os.replace(temporary_path, destination)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return destination


def calibrate_file(
    input_path: str | Path,
    *,
    output_path: str | Path | None = None,
    joint_names: Sequence[str] = DEFAULT_SO101_JOINT_NAMES,
    config: FitConfig | None = None,
    timestamp_key: str = "auto",
    command_key: str = "command",
    observed_key: str = "observed",
    overwrite: bool = False,
) -> dict[str, Any]:
    """Load, fit, and optionally write one offline actuator calibration."""

    names = tuple(joint_names)
    if len(names) != 6:
        raise CalibrationError(f"SO-101 calibration requires exactly 6 joint names, got {len(names)}")
    samples = load_calibration_samples(
        input_path,
        action_dim=len(names),
        timestamp_key=timestamp_key,
        command_key=command_key,
        observed_key=observed_key,
    )
    fit_config = config or FitConfig()
    fits = fit_calibration(samples, joint_names=names, config=fit_config)
    artifact = build_artifact(samples, fits, config=fit_config)
    if output_path is not None:
        destination = Path(output_path).expanduser().resolve()
        if destination == samples.source_path:
            raise CalibrationError("output path must differ from the input trace path")
        write_artifact(artifact, destination, overwrite=overwrite)
    return artifact


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit SO-101 command delay and first-order tau from an offline monotonic trace."
    )
    parser.add_argument("--input", type=Path, required=True, help="Input .jsonl/.ndjson/.csv trace")
    parser.add_argument("--output", type=Path, required=True, help="Output calibration .json artifact")
    parser.add_argument(
        "--joint-names",
        nargs=6,
        default=list(DEFAULT_SO101_JOINT_NAMES),
        metavar=("J1", "J2", "J3", "J4", "J5", "J6"),
        help="Six joint names in command/observed array order",
    )
    parser.add_argument("--timestamp-key", default="auto")
    parser.add_argument("--command-key", default="command")
    parser.add_argument("--observed-key", default="observed")
    parser.add_argument("--max-delay-s", type=float, default=FitConfig.max_delay_s)
    parser.add_argument("--min-tau-s", type=float, default=FitConfig.min_tau_s)
    parser.add_argument("--max-tau-s", type=float, default=FitConfig.max_tau_s)
    parser.add_argument("--min-fit-samples", type=int, default=FitConfig.min_fit_samples)
    parser.add_argument("--min-command-span", type=float, default=FitConfig.min_command_span)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        config = FitConfig(
            max_delay_s=args.max_delay_s,
            min_tau_s=args.min_tau_s,
            max_tau_s=args.max_tau_s,
            min_fit_samples=args.min_fit_samples,
            min_command_span=args.min_command_span,
        )
        artifact = calibrate_file(
            args.input,
            output_path=args.output,
            joint_names=args.joint_names,
            config=config,
            timestamp_key=args.timestamp_key,
            command_key=args.command_key,
            observed_key=args.observed_key,
            overwrite=args.overwrite,
        )
    except (CalibrationError, OSError, RuntimeError) as exc:
        raise SystemExit(f"SO-101 actuator calibration failed: {exc}") from exc

    print(f"Wrote calibration artifact: {args.output.expanduser().resolve()}")
    for joint_name, delay_s, tau_s in zip(
        artifact["joint_names"], artifact["command_delay_s"], artifact["tau_s"], strict=True
    ):
        print(f"  {joint_name}: delay={delay_s:.6f}s tau={tau_s:.6f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
