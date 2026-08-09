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

"""Offline speed-factor learning primitives for Realtime-VLA style rollouts.

The adapter predicts one dimensionless speed factor ``beta`` per action-path
segment. ``beta=1`` keeps the nominal control period, values above one request
a shorter period, and values below one request a longer period.  This module
does not send robot commands and intentionally does not ship a trained model.

Path features are geometric and independent of the execution rate::

    delta[i] = action[i + 1] - action[i]
    curvature[0] = 0
    curvature[i] = delta[i] - delta[i - 1]

Optional state and phase embeddings are sampled at the start of each segment.
The MLP output is bounded by construction, and segment reference periods are
computed as ``base_dt_s / beta``.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

THROTTLE_SCHEMA_ID = "lerobot.speed_throttle.v1"
ROBOT_ACTION_COORDINATE_SPACE = "robot_action_units"
POLICY_ACTION_COORDINATE_SPACE = "policy_action_units"
FEATURE_COORDINATE_SPACES = frozenset((ROBOT_ACTION_COORDINATE_SPACE, POLICY_ACTION_COORDINATE_SPACE))
SPEED_ADAPTER_FORMAT = "lerobot.speed_adapter"
SPEED_ADAPTER_FORMAT_VERSION = 1
SPEED_ADAPTER_WEIGHTS_NAME = "model.safetensors"
SPEED_ADAPTER_METADATA_NAME = "metadata.json"

_THROTTLE_REQUIRED_FIELDS = {
    "schema",
    "episode_id",
    "step_index",
    "feature_coordinate_space",
    "delta",
    "curvature",
    "beta_target",
}
_THROTTLE_OPTIONAL_FIELDS = {
    "timestamp_s",
    "state_embedding",
    "phase_embedding",
    "failure_event",
    "include_in_training",
    "metadata",
}

THROTTLE_JSONL_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": THROTTLE_SCHEMA_ID,
    "title": "LeRobot per-segment speed throttle sample",
    "description": (
        "One JSON object per action-path segment. beta_target is a dimensionless speed factor; "
        "failure_event marks a failed step used to construct exclusion windows, not a low-speed label."
    ),
    "type": "object",
    "additionalProperties": False,
    "required": sorted(_THROTTLE_REQUIRED_FIELDS),
    "properties": {
        "schema": {"const": THROTTLE_SCHEMA_ID},
        "episode_id": {"type": "string", "minLength": 1},
        "step_index": {"type": "integer", "minimum": 0},
        "feature_coordinate_space": {"enum": sorted(FEATURE_COORDINATE_SPACES)},
        "timestamp_s": {"type": "number"},
        "delta": {"type": "array", "minItems": 1, "items": {"type": "number"}},
        "curvature": {"type": "array", "minItems": 1, "items": {"type": "number"}},
        "state_embedding": {"type": "array", "minItems": 1, "items": {"type": "number"}},
        "phase_embedding": {"type": "array", "minItems": 1, "items": {"type": "number"}},
        "beta_target": {"type": "number", "exclusiveMinimum": 0},
        "failure_event": {"type": "boolean", "default": False},
        "include_in_training": {"type": "boolean", "default": True},
        "metadata": {"type": "object"},
    },
}


def throttle_jsonl_schema() -> dict[str, Any]:
    """Return a caller-owned copy of the formal JSONL record schema."""

    return copy.deepcopy(THROTTLE_JSONL_SCHEMA)


def _require_non_negative_integer(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _finite_float(value: Any, *, name: str, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if positive and result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _finite_vector(value: Any, *, name: str) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be an array of numbers")
    if not value:
        raise ValueError(f"{name} must not be empty")
    return tuple(_finite_float(item, name=f"{name}[{index}]") for index, item in enumerate(value))


def _optional_finite_vector(value: Any, *, name: str) -> tuple[float, ...] | None:
    if value is None:
        return None
    return _finite_vector(value, name=name)


@dataclass(frozen=True)
class ThrottleRecord:
    """Validated representation of one ``THROTTLE_SCHEMA_ID`` JSONL row."""

    episode_id: str
    step_index: int
    feature_coordinate_space: str
    delta: tuple[float, ...]
    curvature: tuple[float, ...]
    beta_target: float
    timestamp_s: float | None = None
    state_embedding: tuple[float, ...] | None = None
    phase_embedding: tuple[float, ...] | None = None
    failure_event: bool = False
    include_in_training: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> ThrottleRecord:
        if not isinstance(raw, Mapping):
            raise TypeError("throttle record must be a JSON object")
        missing = sorted(_THROTTLE_REQUIRED_FIELDS - raw.keys())
        if missing:
            raise ValueError(f"throttle record is missing required fields: {', '.join(missing)}")
        unknown = sorted(raw.keys() - _THROTTLE_REQUIRED_FIELDS - _THROTTLE_OPTIONAL_FIELDS)
        if unknown:
            raise ValueError(f"throttle record contains unknown fields: {', '.join(unknown)}")
        if raw["schema"] != THROTTLE_SCHEMA_ID:
            raise ValueError(
                f"unsupported throttle schema {raw['schema']!r}; expected {THROTTLE_SCHEMA_ID!r}"
            )
        episode_id = raw["episode_id"]
        if not isinstance(episode_id, str) or not episode_id.strip():
            raise ValueError("episode_id must be a non-empty string")
        step_index = _require_non_negative_integer(raw["step_index"], name="step_index")
        feature_coordinate_space = raw["feature_coordinate_space"]
        if feature_coordinate_space not in FEATURE_COORDINATE_SPACES:
            raise ValueError(
                "feature_coordinate_space must be one of "
                f"{sorted(FEATURE_COORDINATE_SPACES)}, got {feature_coordinate_space!r}"
            )
        delta = _finite_vector(raw["delta"], name="delta")
        curvature = _finite_vector(raw["curvature"], name="curvature")
        if len(delta) != len(curvature):
            raise ValueError("delta and curvature must have the same length")
        timestamp_raw = raw.get("timestamp_s")
        timestamp_s = None if timestamp_raw is None else _finite_float(timestamp_raw, name="timestamp_s")
        for field_name in ("failure_event", "include_in_training"):
            if field_name in raw and not isinstance(raw[field_name], bool):
                raise TypeError(f"{field_name} must be a boolean")
        metadata = raw.get("metadata", {})
        if not isinstance(metadata, Mapping):
            raise TypeError("metadata must be a JSON object")
        # Verify early that checkpoint provenance can serialize this data.
        try:
            json.dumps(metadata, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("metadata must contain only finite JSON values") from exc
        return cls(
            episode_id=episode_id,
            step_index=step_index,
            feature_coordinate_space=feature_coordinate_space,
            delta=delta,
            curvature=curvature,
            beta_target=_finite_float(raw["beta_target"], name="beta_target", positive=True),
            timestamp_s=timestamp_s,
            state_embedding=_optional_finite_vector(raw.get("state_embedding"), name="state_embedding"),
            phase_embedding=_optional_finite_vector(raw.get("phase_embedding"), name="phase_embedding"),
            failure_event=raw.get("failure_event", False),
            include_in_training=raw.get("include_in_training", True),
            metadata=dict(metadata),
        )

    def to_mapping(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema": THROTTLE_SCHEMA_ID,
            "episode_id": self.episode_id,
            "step_index": self.step_index,
            "feature_coordinate_space": self.feature_coordinate_space,
            "delta": list(self.delta),
            "curvature": list(self.curvature),
            "beta_target": self.beta_target,
            "failure_event": self.failure_event,
            "include_in_training": self.include_in_training,
        }
        if self.timestamp_s is not None:
            result["timestamp_s"] = self.timestamp_s
        if self.state_embedding is not None:
            result["state_embedding"] = list(self.state_embedding)
        if self.phase_embedding is not None:
            result["phase_embedding"] = list(self.phase_embedding)
        if self.metadata:
            result["metadata"] = dict(self.metadata)
        return result


def load_throttle_jsonl(path: str | Path) -> list[ThrottleRecord]:
    """Load and strictly validate per-segment throttle annotations.

    Blank lines are ignored. Duplicate ``(episode_id, step_index)`` keys are
    rejected because failure-window masking and grouped validation splitting
    both require an unambiguous episode timeline.
    """

    path = Path(path)
    records: list[ThrottleRecord] = []
    seen_keys: set[tuple[str, int]] = set()
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
                record = ThrottleRecord.from_mapping(raw)
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                raise ValueError(f"invalid throttle record at {path}:{line_number}: {exc}") from exc
            key = (record.episode_id, record.step_index)
            if key in seen_keys:
                raise ValueError(
                    f"duplicate throttle record at {path}:{line_number}: "
                    f"episode_id={record.episode_id!r}, step_index={record.step_index}"
                )
            seen_keys.add(key)
            records.append(record)
    return records


def failure_window_mask(
    records: Sequence[ThrottleRecord],
    *,
    preceding_steps: int = 0,
    following_steps: int = 0,
) -> torch.Tensor:
    """Return a mask selecting manually valid samples outside failure windows.

    For a failure at step ``f``, every record in the same episode whose index
    lies in ``[f - preceding_steps, f + following_steps]`` is excluded. The
    mask also excludes rows with ``include_in_training=false``. Failure rows
    are therefore excluded even when both window sizes are zero.
    """

    preceding_steps = _require_non_negative_integer(preceding_steps, name="preceding_steps")
    following_steps = _require_non_negative_integer(following_steps, name="following_steps")
    failures: dict[str, list[int]] = {}
    for record in records:
        if record.failure_event:
            failures.setdefault(record.episode_id, []).append(record.step_index)

    keep = []
    for record in records:
        inside_failure_window = any(
            failure_step - preceding_steps <= record.step_index <= failure_step + following_steps
            for failure_step in failures.get(record.episode_id, ())
        )
        keep.append(record.include_in_training and not inside_failure_window)
    return torch.tensor(keep, dtype=torch.bool)


@dataclass(frozen=True)
class SpeedFeatureLayout:
    action_dim: int
    state_dim: int
    phase_dim: int
    feature_coordinate_space: str
    feature_names: tuple[str, ...]

    @property
    def feature_dim(self) -> int:
        return 2 * self.action_dim + self.state_dim + self.phase_dim


@dataclass(frozen=True)
class ThrottleTensorData:
    features: torch.Tensor
    beta_targets: torch.Tensor
    layout: SpeedFeatureLayout


def _record_layout(records: Sequence[ThrottleRecord]) -> SpeedFeatureLayout:
    if not records:
        raise ValueError("at least one throttle record is required")
    first = records[0]
    action_dim = len(first.delta)
    state_dim = 0 if first.state_embedding is None else len(first.state_embedding)
    phase_dim = 0 if first.phase_embedding is None else len(first.phase_embedding)
    for index, record in enumerate(records[1:], start=1):
        dimensions = (
            len(record.delta),
            0 if record.state_embedding is None else len(record.state_embedding),
            0 if record.phase_embedding is None else len(record.phase_embedding),
        )
        if dimensions != (action_dim, state_dim, phase_dim):
            raise ValueError(
                "inconsistent feature dimensions at record "
                f"{index}: expected {(action_dim, state_dim, phase_dim)}, got {dimensions}"
            )
        if record.feature_coordinate_space != first.feature_coordinate_space:
            raise ValueError(
                "inconsistent feature_coordinate_space at record "
                f"{index}: expected {first.feature_coordinate_space!r}, "
                f"got {record.feature_coordinate_space!r}"
            )
    names = (
        *(f"delta.{index}" for index in range(action_dim)),
        *(f"curvature.{index}" for index in range(action_dim)),
        *(f"state_embedding.{index}" for index in range(state_dim)),
        *(f"phase_embedding.{index}" for index in range(phase_dim)),
    )
    return SpeedFeatureLayout(
        action_dim,
        state_dim,
        phase_dim,
        first.feature_coordinate_space,
        tuple(names),
    )


def throttle_records_to_tensors(records: Sequence[ThrottleRecord]) -> ThrottleTensorData:
    """Stack validated JSONL records in the adapter's canonical feature order."""

    layout = _record_layout(records)
    rows: list[tuple[float, ...]] = []
    for record in records:
        rows.append(
            (
                *record.delta,
                *record.curvature,
                *(record.state_embedding or ()),
                *(record.phase_embedding or ()),
            )
        )
    return ThrottleTensorData(
        features=torch.tensor(rows, dtype=torch.float32),
        beta_targets=torch.tensor([record.beta_target for record in records], dtype=torch.float32),
        layout=layout,
    )


def _path_tensor(value: Any, *, name: str, device: torch.device | None = None) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        tensor = value
        if device is not None and tensor.device != device:
            tensor = tensor.to(device=device)
        if not tensor.is_floating_point():
            tensor = tensor.to(dtype=torch.float32)
    else:
        tensor = torch.as_tensor(value, dtype=torch.float32, device=device)
    if tensor.ndim != 2:
        raise ValueError(f"{name} must have shape [steps, features]")
    if not torch.isfinite(tensor).all().item():
        raise ValueError(f"{name} must contain only finite values")
    return tensor


def _segment_aligned_embedding(
    value: Any,
    *,
    name: str,
    waypoint_count: int,
    segment_count: int,
    device: torch.device,
) -> torch.Tensor:
    tensor = _path_tensor(value, name=name, device=device)
    if tensor.shape[0] == waypoint_count:
        return tensor[:-1]
    if tensor.shape[0] == segment_count:
        return tensor
    raise ValueError(
        f"{name} must have {waypoint_count} waypoint rows or {segment_count} segment rows; "
        f"got {tensor.shape[0]}"
    )


def extract_action_path_features(
    actions: Any,
    *,
    state_embeddings: Any | None = None,
    phase_embeddings: Any | None = None,
) -> torch.Tensor:
    """Build one feature row per segment of a waypoint path.

    Optional embeddings may have one row per waypoint or one row per segment.
    Waypoint-aligned embeddings use the segment's starting waypoint (``[:-1]``).
    The returned tensor always has shape ``[num_actions - 1, feature_dim]``.
    """

    action_tensor = _path_tensor(actions, name="actions")
    if action_tensor.shape[0] < 2:
        raise ValueError("actions must contain at least two waypoints")
    deltas = action_tensor[1:] - action_tensor[:-1]
    curvature = torch.zeros_like(deltas)
    if len(deltas) > 1:
        curvature[1:] = deltas[1:] - deltas[:-1]
    features = [deltas, curvature]
    for value, name in (
        (state_embeddings, "state_embeddings"),
        (phase_embeddings, "phase_embeddings"),
    ):
        if value is not None:
            features.append(
                _segment_aligned_embedding(
                    value,
                    name=name,
                    waypoint_count=len(action_tensor),
                    segment_count=len(deltas),
                    device=action_tensor.device,
                ).to(dtype=action_tensor.dtype)
            )
    return torch.cat(features, dim=-1)


@dataclass(frozen=True)
class SpeedAdapterConfig:
    """Architecture and output contract for the lightweight speed MLP."""

    action_dim: int
    state_dim: int = 0
    phase_dim: int = 0
    hidden_dims: tuple[int, ...] = (64, 32)
    beta_min: float = 0.25
    beta_max: float = 2.0
    normalization_epsilon: float = 1e-6
    feature_coordinate_space: str = ROBOT_ACTION_COORDINATE_SPACE

    def __post_init__(self) -> None:
        for name in ("action_dim", "state_dim", "phase_dim"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if (name == "action_dim" and value < 1) or (name != "action_dim" and value < 0):
                raise ValueError(f"{name} has an invalid dimension: {value}")
        if not self.hidden_dims:
            raise ValueError("hidden_dims must contain at least one layer")
        if any(
            isinstance(width, bool) or not isinstance(width, int) or width < 1 for width in self.hidden_dims
        ):
            raise ValueError("hidden_dims must contain positive integers")
        beta_min = _finite_float(self.beta_min, name="beta_min", positive=True)
        beta_max = _finite_float(self.beta_max, name="beta_max", positive=True)
        if beta_min >= beta_max:
            raise ValueError("beta_min must be less than beta_max")
        _finite_float(
            self.normalization_epsilon,
            name="normalization_epsilon",
            positive=True,
        )
        if self.feature_coordinate_space not in FEATURE_COORDINATE_SPACES:
            raise ValueError(f"feature_coordinate_space must be one of {sorted(FEATURE_COORDINATE_SPACES)}")

    @property
    def feature_dim(self) -> int:
        return 2 * self.action_dim + self.state_dim + self.phase_dim


@dataclass(frozen=True)
class SpeedAdapterRuntimeConfig:
    """Rollout-time loading contract for an optional trained speed adapter."""

    enabled: bool = False
    checkpoint: str | None = None
    device: str = "cpu"
    verify_checksum: bool = True
    require_trained_checkpoint: bool = True
    feature_coordinate_space: str = ROBOT_ACTION_COORDINATE_SPACE

    def __post_init__(self) -> None:
        if self.checkpoint is not None and (
            not isinstance(self.checkpoint, str) or not self.checkpoint.strip()
        ):
            raise ValueError("speed_adapter.checkpoint must be a non-empty path when set")
        if self.enabled and self.checkpoint is None:
            raise ValueError("speed_adapter.checkpoint is required when speed_adapter.enabled=true")
        if not isinstance(self.device, str) or not self.device.strip():
            raise ValueError("speed_adapter.device must be a non-empty torch device")
        for name in ("verify_checksum", "require_trained_checkpoint"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"speed_adapter.{name} must be a boolean")
        if self.feature_coordinate_space not in FEATURE_COORDINATE_SPACES:
            raise ValueError(
                f"speed_adapter.feature_coordinate_space must be one of {sorted(FEATURE_COORDINATE_SPACES)}"
            )


class SpeedAdapter(nn.Module):
    """Small MLP with a sigmoid-mapped, bounded per-segment speed factor."""

    def __init__(
        self,
        config: SpeedAdapterConfig,
        *,
        feature_mean: torch.Tensor | Sequence[float] | None = None,
        feature_std: torch.Tensor | Sequence[float] | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        dimensions = (config.feature_dim, *config.hidden_dims, 1)
        layers: list[nn.Module] = []
        for input_dim, output_dim in zip(dimensions[:-2], dimensions[1:-1], strict=True):
            layers.extend((nn.Linear(input_dim, output_dim), nn.SiLU()))
        layers.append(nn.Linear(dimensions[-2], dimensions[-1]))
        self.network = nn.Sequential(*layers)
        mean = self._normalization_tensor(feature_mean, default=0.0, name="feature_mean")
        std = self._normalization_tensor(feature_std, default=1.0, name="feature_std")
        if torch.any(std <= 0).item():
            raise ValueError("feature_std must be strictly positive")
        self.register_buffer("feature_mean", mean)
        self.register_buffer("feature_std", std)

    def _normalization_tensor(
        self,
        value: torch.Tensor | Sequence[float] | None,
        *,
        default: float,
        name: str,
    ) -> torch.Tensor:
        if value is None:
            result = torch.full((self.config.feature_dim,), default, dtype=torch.float32)
        else:
            result = torch.as_tensor(value, dtype=torch.float32).detach().clone()
        if result.shape != (self.config.feature_dim,):
            raise ValueError(f"{name} must have shape [{self.config.feature_dim}]")
        if not torch.isfinite(result).all().item():
            raise ValueError(f"{name} must contain only finite values")
        return result

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if not isinstance(features, torch.Tensor):
            raise TypeError("features must be a torch.Tensor")
        if features.ndim < 1 or features.shape[-1] != self.config.feature_dim:
            raise ValueError(
                f"features must end with dimension {self.config.feature_dim}; got {tuple(features.shape)}"
            )
        if not torch.isfinite(features).all().item():
            raise ValueError("features must contain only finite values")
        normalized = (features - self.feature_mean) / self.feature_std.clamp_min(
            self.config.normalization_epsilon
        )
        logits = self.network(normalized).squeeze(-1)
        return self.config.beta_min + (self.config.beta_max - self.config.beta_min) * torch.sigmoid(logits)

    @torch.inference_mode()
    def predict_path(
        self,
        actions: Any,
        *,
        state_embeddings: Any | None = None,
        phase_embeddings: Any | None = None,
    ) -> torch.Tensor:
        """Extract path features and predict one beta for every segment."""

        features = extract_action_path_features(
            actions,
            state_embeddings=state_embeddings,
            phase_embeddings=phase_embeddings,
        ).to(device=self.feature_mean.device, dtype=self.feature_mean.dtype)
        return self(features)


def _validate_dt_bounds(
    base_dt_s: float,
    min_dt_s: float | None,
    max_dt_s: float | None,
) -> tuple[float, float | None, float | None]:
    base_dt_s = _finite_float(base_dt_s, name="base_dt_s", positive=True)
    if min_dt_s is not None:
        min_dt_s = _finite_float(min_dt_s, name="min_dt_s", positive=True)
    if max_dt_s is not None:
        max_dt_s = _finite_float(max_dt_s, name="max_dt_s", positive=True)
    if min_dt_s is not None and max_dt_s is not None and min_dt_s > max_dt_s:
        raise ValueError("min_dt_s must be less than or equal to max_dt_s")
    return base_dt_s, min_dt_s, max_dt_s


def beta_to_segment_dt_ref(
    beta: torch.Tensor | np.ndarray | Sequence[float] | float,
    *,
    base_dt_s: float,
    min_dt_s: float | None = None,
    max_dt_s: float | None = None,
) -> torch.Tensor | np.ndarray | float:
    """Convert dimensionless speed factors to per-segment reference periods.

    Tensor inputs return tensors and NumPy/sequence inputs return NumPy arrays.
    A scalar input returns a float. Optional period bounds provide a second,
    explicit scheduling guard independent of the adapter's beta bounds.
    """

    base_dt_s, min_dt_s, max_dt_s = _validate_dt_bounds(base_dt_s, min_dt_s, max_dt_s)
    if isinstance(beta, torch.Tensor):
        if beta.dtype == torch.bool:
            raise TypeError("beta must be floating point or integral speed factors")
        values = beta if beta.is_floating_point() else beta.to(dtype=torch.float32)
        if not torch.isfinite(values).all().item() or torch.any(values <= 0).item():
            raise ValueError("beta must contain only finite, positive values")
        durations = base_dt_s / values
        if min_dt_s is not None or max_dt_s is not None:
            durations = durations.clamp(
                min=min_dt_s if min_dt_s is not None else None,
                max=max_dt_s if max_dt_s is not None else None,
            )
        return durations

    array = np.asarray(beta)
    scalar_input = array.ndim == 0
    if array.dtype == np.bool_:
        raise TypeError("beta must be floating point or integral speed factors")
    values = np.asarray(array, dtype=np.float64)
    if not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError("beta must contain only finite, positive values")
    durations = base_dt_s / values
    if min_dt_s is not None or max_dt_s is not None:
        durations = np.clip(
            durations,
            min_dt_s if min_dt_s is not None else -np.inf,
            max_dt_s if max_dt_s is not None else np.inf,
        )
    if scalar_input:
        return float(durations)
    return durations


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_speed_adapter_checkpoint(
    output_dir: str | Path,
    model: SpeedAdapter,
    *,
    feature_names: Sequence[str],
    training_metadata: Mapping[str, Any] | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Save portable safetensors weights and auditable JSON metadata."""

    from safetensors.torch import save_file

    output_dir = Path(output_dir)
    weights_path = output_dir / SPEED_ADAPTER_WEIGHTS_NAME
    metadata_path = output_dir / SPEED_ADAPTER_METADATA_NAME
    if not overwrite and (weights_path.exists() or metadata_path.exists()):
        raise FileExistsError(f"speed adapter checkpoint already exists in {output_dir}")
    names = tuple(str(name) for name in feature_names)
    if len(names) != model.config.feature_dim or any(not name for name in names):
        raise ValueError(f"feature_names must contain {model.config.feature_dim} non-empty entries")
    training = dict(training_metadata or {})
    try:
        json.dumps(training, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("training_metadata must contain only finite JSON values") from exc
    output_dir.mkdir(parents=True, exist_ok=True)
    state_dict = {name: tensor.detach().cpu().contiguous() for name, tensor in model.state_dict().items()}
    save_file(state_dict, str(weights_path))
    num_training_samples = training.get("num_training_samples", 0)
    trained_from_data = (
        isinstance(num_training_samples, int)
        and not isinstance(num_training_samples, bool)
        and num_training_samples > 0
    )
    metadata: dict[str, Any] = {
        "format": SPEED_ADAPTER_FORMAT,
        "format_version": SPEED_ADAPTER_FORMAT_VERSION,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "architecture": asdict(model.config),
        "feature_contract": {
            "feature_names": list(names),
            "feature_coordinate_space": model.config.feature_coordinate_space,
            "path_definition": {
                "delta": "action[i+1] - action[i]",
                "curvature": "delta[i] - delta[i-1], with curvature[0] = 0",
                "embedding_alignment": "segment start; waypoint arrays use rows[:-1]",
            },
            "normalization_mean": model.feature_mean.detach().cpu().tolist(),
            "normalization_std": model.feature_std.detach().cpu().tolist(),
        },
        "output_contract": {
            "name": "beta",
            "units": "dimensionless speed factor",
            "beta_min": model.config.beta_min,
            "beta_max": model.config.beta_max,
            "segment_dt_ref": "base_dt_s / beta",
        },
        "provenance": {
            "trained_from_collected_throttle_data": trained_from_data,
        },
        "training": training,
        "weights": {
            "filename": SPEED_ADAPTER_WEIGHTS_NAME,
            "sha256": file_sha256(weights_path),
        },
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return metadata


def load_speed_adapter_checkpoint(
    checkpoint_dir: str | Path,
    *,
    device: str | torch.device = "cpu",
    verify_checksum: bool = True,
) -> tuple[SpeedAdapter, dict[str, Any]]:
    """Load a speed adapter and validate its format, dimensions, and checksum."""

    from safetensors.torch import load_file

    checkpoint_dir = Path(checkpoint_dir)
    weights_path = checkpoint_dir / SPEED_ADAPTER_WEIGHTS_NAME
    metadata_path = checkpoint_dir / SPEED_ADAPTER_METADATA_NAME
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("format") != SPEED_ADAPTER_FORMAT:
        raise ValueError(f"unsupported speed adapter format: {metadata.get('format')!r}")
    if metadata.get("format_version") != SPEED_ADAPTER_FORMAT_VERSION:
        raise ValueError(f"unsupported speed adapter format_version: {metadata.get('format_version')!r}")
    weights_metadata = metadata.get("weights")
    if not isinstance(weights_metadata, Mapping):
        raise ValueError("checkpoint metadata is missing weights information")
    if weights_metadata.get("filename") != SPEED_ADAPTER_WEIGHTS_NAME:
        raise ValueError("checkpoint metadata points to an unexpected weights filename")
    if verify_checksum and file_sha256(weights_path) != weights_metadata.get("sha256"):
        raise ValueError("speed adapter weights checksum does not match metadata")

    architecture = metadata.get("architecture")
    if not isinstance(architecture, Mapping):
        raise ValueError("checkpoint metadata is missing architecture")
    config_values = dict(architecture)
    config_values["hidden_dims"] = tuple(config_values.get("hidden_dims", ()))
    config = SpeedAdapterConfig(**config_values)
    feature_contract = metadata.get("feature_contract")
    if not isinstance(feature_contract, Mapping):
        raise ValueError("checkpoint metadata is missing feature_contract")
    feature_names = feature_contract.get("feature_names")
    if not isinstance(feature_names, list) or len(feature_names) != config.feature_dim:
        raise ValueError("checkpoint feature_names do not match the model feature dimension")

    state_dict = load_file(str(weights_path), device="cpu")
    model = SpeedAdapter(config)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model, metadata


def load_runtime_speed_adapter(
    config: SpeedAdapterRuntimeConfig,
    *,
    expected_action_dim: int | None = None,
    require_path_only_features: bool = True,
) -> tuple[SpeedAdapter, dict[str, Any]] | None:
    """Load and validate an enabled adapter before rollout hardware is opened.

    The current RTC integration supplies geometric action-path features only.
    A checkpoint trained with state or phase embeddings is rejected at startup
    unless its caller explicitly provides those embeddings and sets
    ``require_path_only_features=False``.
    """

    if not config.enabled:
        return None
    if config.checkpoint is None:  # guarded by the dataclass, kept for typed callers
        raise RuntimeError("enabled speed adapter has no checkpoint")
    try:
        model, metadata = load_speed_adapter_checkpoint(
            config.checkpoint,
            device=config.device,
            verify_checksum=config.verify_checksum,
        )
    except Exception as exc:
        raise RuntimeError(
            f"failed to load enabled speed adapter checkpoint {config.checkpoint!r}: {exc}"
        ) from exc

    if expected_action_dim is not None:
        if (
            isinstance(expected_action_dim, bool)
            or not isinstance(expected_action_dim, int)
            or expected_action_dim < 1
        ):
            raise ValueError("expected_action_dim must be a positive integer")
        if model.config.action_dim != expected_action_dim:
            raise RuntimeError(
                "speed adapter action dimension does not match the policy: "
                f"checkpoint={model.config.action_dim}, policy={expected_action_dim}"
            )
    if require_path_only_features and (model.config.state_dim or model.config.phase_dim):
        raise RuntimeError(
            "the current RTC speed-adapter integration supplies delta/curvature only, but the "
            f"checkpoint requires state_dim={model.config.state_dim}, phase_dim={model.config.phase_dim}"
        )
    feature_contract = metadata.get("feature_contract")
    checkpoint_coordinate_space = (
        feature_contract.get("feature_coordinate_space") if isinstance(feature_contract, Mapping) else None
    )
    if (
        checkpoint_coordinate_space != model.config.feature_coordinate_space
        or checkpoint_coordinate_space != config.feature_coordinate_space
    ):
        raise RuntimeError(
            "speed adapter feature coordinate space mismatch: "
            f"metadata={checkpoint_coordinate_space!r}, "
            f"architecture={model.config.feature_coordinate_space!r}, "
            f"runtime={config.feature_coordinate_space!r}"
        )
    provenance = metadata.get("provenance")
    trained_from_data = (
        isinstance(provenance, Mapping) and provenance.get("trained_from_collected_throttle_data") is True
    )
    if config.require_trained_checkpoint and not trained_from_data:
        raise RuntimeError("speed adapter checkpoint is not marked as trained from collected throttle data")
    return model, metadata
