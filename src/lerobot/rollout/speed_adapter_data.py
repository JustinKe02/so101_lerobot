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

"""Strict offline preparation of speed-adapter throttle training records."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .speed_adapter import (
    ROBOT_ACTION_COORDINATE_SPACE,
    THROTTLE_SCHEMA_ID,
    ThrottleRecord,
    extract_action_path_features,
    file_sha256,
)

TRACE_SCHEMA_VERSION = 2
TRACE_ACTION_FIELD = "raw_robot_chunk"
ANNOTATION_SCHEMA_ID = "lerobot.speed_throttle_annotation.v1"
# Template rows use the final schema ID but deliberately contain null labels,
# so filling labels is sufficient and an untouched template is rejected.
ANNOTATION_TEMPLATE_SCHEMA_ID = ANNOTATION_SCHEMA_ID
PREPARATION_FORMAT = "lerobot.speed_throttle_preparation"
PREPARATION_FORMAT_VERSION = 1

_ANNOTATION_REQUIRED = {
    "schema",
    "session_id",
    "episode_id",
    "chunk_sequence",
    "segment_count",
    "action_dim",
    "feature_coordinate_space",
    "beta",
    "failure",
    "include",
}
_ANNOTATION_OPTIONAL = {"notes"}

ANNOTATION_JSONL_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": ANNOTATION_SCHEMA_ID,
    "title": "Human speed-throttle annotation for one RealtimeTraceWriter v2 chunk",
    "description": (
        "Every trace inference chunk requires exactly one annotation. Scalar beta/include values "
        "apply to every segment. Scalar failure=true marks only the final segment; arrays map "
        "one-to-one to segments. No label has an implicit default."
    ),
    "type": "object",
    "additionalProperties": False,
    "required": sorted(_ANNOTATION_REQUIRED),
    "properties": {
        "schema": {"const": ANNOTATION_SCHEMA_ID},
        "session_id": {"type": "string", "minLength": 1},
        "episode_id": {"type": "string", "minLength": 1},
        "chunk_sequence": {"type": "integer", "minimum": 0},
        "segment_count": {"type": "integer", "minimum": 1},
        "action_dim": {"type": "integer", "minimum": 1},
        "feature_coordinate_space": {"const": ROBOT_ACTION_COORDINATE_SPACE},
        "beta": {
            "oneOf": [
                {"type": "number", "exclusiveMinimum": 0},
                {
                    "type": "array",
                    "minItems": 1,
                    "items": {"type": "number", "exclusiveMinimum": 0},
                },
            ]
        },
        "failure": {
            "oneOf": [
                {"type": "boolean"},
                {"type": "array", "minItems": 1, "items": {"type": "boolean"}},
            ]
        },
        "include": {
            "oneOf": [
                {"type": "boolean"},
                {"type": "array", "minItems": 1, "items": {"type": "boolean"}},
            ]
        },
        "notes": {"type": "string"},
    },
}


def annotation_jsonl_schema() -> dict[str, Any]:
    return json.loads(json.dumps(ANNOTATION_JSONL_SCHEMA))


def _read_jsonl(path: Path, *, label: str) -> list[tuple[int, Mapping[str, Any]]]:
    rows: list[tuple[int, Mapping[str, Any]]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid {label} JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(row, Mapping):
                raise ValueError(f"invalid {label} row at {path}:{line_number}: expected an object")
            rows.append((line_number, row))
    return rows


def _nonempty_string(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _nonnegative_int(value: Any, *, name: str, positive: bool = False) -> int:
    minimum = 1 if positive else 0
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{name} must be a {qualifier} integer")
    return value


def _full_action_chunk(value: Any, *, location: str) -> np.ndarray:
    if isinstance(value, Mapping) and value.get("storage") == "summary":
        raise ValueError(
            f"{location} contains a summarized action chunk; RealtimeTraceWriter schema v2 "
            "with complete inline chunks is required"
        )
    if not isinstance(value, list) or len(value) < 2 or any(not isinstance(row, list) for row in value):
        raise ValueError(f"{location} must be a complete two-dimensional action chunk")
    widths = {len(row) for row in value}
    if len(widths) != 1 or not widths or next(iter(widths)) < 1:
        raise ValueError(f"{location} action rows must have one non-zero, consistent width")
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) for row in value for item in row):
        raise ValueError(f"{location} action chunk must contain only numbers, not booleans")
    try:
        actions = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{location} action chunk must contain only numbers") from exc
    if actions.ndim != 2 or not np.isfinite(actions).all():
        raise ValueError(f"{location} action chunk must be finite and two-dimensional")
    return actions


@dataclass(frozen=True)
class TraceActionChunk:
    session_id: str
    chunk_sequence: int
    actions: np.ndarray

    @property
    def segment_count(self) -> int:
        return len(self.actions) - 1

    @property
    def action_dim(self) -> int:
        return self.actions.shape[1]

    @property
    def key(self) -> tuple[str, int]:
        return self.session_id, self.chunk_sequence


def load_realtime_trace_v2(path: str | Path) -> list[TraceActionChunk]:
    """Load all full robot-space inference chunks from trace schema v2."""

    path = Path(path)
    sessions: dict[str, int] = {}
    seen_events: set[tuple[str, int]] = set()
    last_sequence: dict[str, int] = {}
    chunks: list[TraceActionChunk] = []
    for line_number, row in _read_jsonl(path, label="trace"):
        location = f"{path}:{line_number}"
        session_id = _nonempty_string(row.get("session_id"), name=f"{location} session_id")
        sequence = _nonnegative_int(row.get("sequence"), name=f"{location} sequence")
        event_key = (session_id, sequence)
        if event_key in seen_events:
            raise ValueError(f"duplicate trace event key at {location}: {event_key}")
        seen_events.add(event_key)
        event = row.get("event")
        if event == "session_start":
            if sequence != 0:
                raise ValueError(f"session_start at {location} must have sequence=0")
            schema_version = row.get("schema_version")
            if isinstance(schema_version, bool) or schema_version != TRACE_SCHEMA_VERSION:
                raise ValueError(
                    f"unsupported trace schema at {location}: expected schema_version="
                    f"{TRACE_SCHEMA_VERSION}, got {schema_version!r}"
                )
            if session_id in sessions:
                raise ValueError(f"duplicate session_start for session {session_id!r}")
            sessions[session_id] = line_number
        elif session_id not in sessions:
            raise ValueError(
                f"trace event at {location} appears before its schema-v{TRACE_SCHEMA_VERSION} session_start"
            )
        prior_sequence = last_sequence.get(session_id)
        if prior_sequence is not None and sequence <= prior_sequence:
            raise ValueError(
                f"trace sequence is not strictly increasing at {location}: "
                f"previous={prior_sequence}, current={sequence}"
            )
        last_sequence[session_id] = sequence
        if event == "inference_chunk":
            actions = _full_action_chunk(
                row.get(TRACE_ACTION_FIELD),
                location=f"{location} {TRACE_ACTION_FIELD}",
            )
            trace_space = row.get("planner_feature_coordinate_space")
            if trace_space not in (None, ROBOT_ACTION_COORDINATE_SPACE):
                raise ValueError(
                    f"trace coordinate space mismatch at {location}: "
                    f"expected {ROBOT_ACTION_COORDINATE_SPACE!r}, got {trace_space!r}"
                )
            chunks.append(TraceActionChunk(session_id, sequence, actions))

    if not chunks:
        raise ValueError(f"trace contains no inference_chunk events: {path}")
    for chunk in chunks:
        if chunk.session_id not in sessions:
            raise ValueError(f"trace chunk {chunk.key} has no schema v{TRACE_SCHEMA_VERSION} session_start")
    return chunks


def _number_vector(value: Any, *, name: str) -> float | tuple[float, ...]:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a positive number or array")
    if isinstance(value, (int, float)):
        result = float(value)
        if not math.isfinite(result) or result <= 0:
            raise ValueError(f"{name} must be finite and positive")
        return result
    if not isinstance(value, list) or not value:
        raise TypeError(f"{name} must be a positive number or non-empty array")
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value):
        raise TypeError(f"{name} must contain numbers, not booleans")
    result = tuple(float(item) for item in value)
    if any(not math.isfinite(item) or item <= 0 for item in result):
        raise ValueError(f"{name} must contain finite positive numbers")
    return result


def _bool_vector(value: Any, *, name: str) -> bool | tuple[bool, ...]:
    if isinstance(value, bool):
        return value
    if not isinstance(value, list) or not value or any(not isinstance(item, bool) for item in value):
        raise TypeError(f"{name} must be a boolean or non-empty boolean array")
    return tuple(value)


@dataclass(frozen=True)
class ThrottleAnnotation:
    session_id: str
    episode_id: str
    chunk_sequence: int
    segment_count: int
    action_dim: int
    feature_coordinate_space: str
    beta: float | tuple[float, ...]
    failure: bool | tuple[bool, ...]
    include: bool | tuple[bool, ...]
    notes: str | None = None

    @property
    def key(self) -> tuple[str, int]:
        return self.session_id, self.chunk_sequence


def load_throttle_annotations(path: str | Path) -> list[ThrottleAnnotation]:
    path = Path(path)
    annotations: list[ThrottleAnnotation] = []
    seen: set[tuple[str, int]] = set()
    for line_number, row in _read_jsonl(path, label="annotation"):
        location = f"{path}:{line_number}"
        missing = sorted(_ANNOTATION_REQUIRED - row.keys())
        if missing:
            raise ValueError(f"annotation at {location} is missing labels/fields: {', '.join(missing)}")
        unknown = sorted(row.keys() - _ANNOTATION_REQUIRED - _ANNOTATION_OPTIONAL)
        if unknown:
            raise ValueError(f"annotation at {location} has unknown fields: {', '.join(unknown)}")
        if row["schema"] != ANNOTATION_SCHEMA_ID:
            raise ValueError(
                f"unsupported annotation schema at {location}: {row['schema']!r}; "
                f"expected {ANNOTATION_SCHEMA_ID!r}"
            )
        feature_space = row["feature_coordinate_space"]
        if feature_space != ROBOT_ACTION_COORDINATE_SPACE:
            raise ValueError(
                f"annotation coordinate space mismatch at {location}: "
                f"expected {ROBOT_ACTION_COORDINATE_SPACE!r}, got {feature_space!r}"
            )
        notes = row.get("notes")
        if notes is not None and not isinstance(notes, str):
            raise TypeError(f"annotation notes at {location} must be a string")
        annotation = ThrottleAnnotation(
            session_id=_nonempty_string(row["session_id"], name=f"{location} session_id"),
            episode_id=_nonempty_string(row["episode_id"], name=f"{location} episode_id"),
            chunk_sequence=_nonnegative_int(row["chunk_sequence"], name=f"{location} chunk_sequence"),
            segment_count=_nonnegative_int(
                row["segment_count"], name=f"{location} segment_count", positive=True
            ),
            action_dim=_nonnegative_int(row["action_dim"], name=f"{location} action_dim", positive=True),
            feature_coordinate_space=feature_space,
            beta=_number_vector(row["beta"], name=f"{location} beta"),
            failure=_bool_vector(row["failure"], name=f"{location} failure"),
            include=_bool_vector(row["include"], name=f"{location} include"),
            notes=notes,
        )
        if annotation.key in seen:
            raise ValueError(f"duplicate annotation key at {location}: {annotation.key}")
        seen.add(annotation.key)
        annotations.append(annotation)
    if not annotations:
        raise ValueError(f"annotation file contains no rows: {path}")
    return annotations


def _expand_beta(value: float | tuple[float, ...], count: int, *, key: tuple[str, int]) -> tuple[float, ...]:
    if isinstance(value, float):
        return (value,) * count
    if len(value) != count:
        raise ValueError(f"annotation {key} beta length must be {count}, got {len(value)}")
    return value


def _expand_include(value: bool | tuple[bool, ...], count: int, *, key: tuple[str, int]) -> tuple[bool, ...]:
    if isinstance(value, bool):
        return (value,) * count
    if len(value) != count:
        raise ValueError(f"annotation {key} include length must be {count}, got {len(value)}")
    return value


def _expand_failure(value: bool | tuple[bool, ...], count: int, *, key: tuple[str, int]) -> tuple[bool, ...]:
    if isinstance(value, bool):
        return ((False,) * (count - 1) + (value,)) if value else (False,) * count
    if len(value) != count:
        raise ValueError(f"annotation {key} failure length must be {count}, got {len(value)}")
    return value


def prepare_throttle_records(
    chunks: Sequence[TraceActionChunk],
    annotations: Sequence[ThrottleAnnotation],
    *,
    trace_sha256: str,
    annotation_sha256: str,
) -> list[ThrottleRecord]:
    """Join trace chunks and human labels, then emit strict per-segment rows."""

    chunk_map = {chunk.key: chunk for chunk in chunks}
    annotation_map = {annotation.key: annotation for annotation in annotations}
    missing = sorted(chunk_map.keys() - annotation_map.keys())
    extra = sorted(annotation_map.keys() - chunk_map.keys())
    if missing:
        raise ValueError(f"missing annotations for trace chunks: {missing}")
    if extra:
        raise ValueError(f"annotations reference unknown trace chunks: {extra}")
    action_dims = {chunk.action_dim for chunk in chunks}
    if len(action_dims) != 1:
        raise ValueError(f"trace chunks have inconsistent action dimensions: {sorted(action_dims)}")

    episode_sessions: dict[str, str] = {}
    step_cursors: dict[str, int] = {}
    records: list[ThrottleRecord] = []
    for chunk in chunks:
        annotation = annotation_map[chunk.key]
        if annotation.segment_count != chunk.segment_count:
            raise ValueError(
                f"annotation {chunk.key} segment_count={annotation.segment_count} does not match "
                f"trace segment_count={chunk.segment_count}"
            )
        if annotation.action_dim != chunk.action_dim:
            raise ValueError(
                f"annotation {chunk.key} action_dim={annotation.action_dim} does not match "
                f"trace action_dim={chunk.action_dim}"
            )
        prior_session = episode_sessions.setdefault(annotation.episode_id, annotation.session_id)
        if prior_session != annotation.session_id:
            raise ValueError(
                f"episode_id {annotation.episode_id!r} is reused across sessions; episode IDs "
                "must be globally unambiguous in one throttle dataset"
            )
        betas = _expand_beta(annotation.beta, chunk.segment_count, key=chunk.key)
        failures = _expand_failure(annotation.failure, chunk.segment_count, key=chunk.key)
        includes = _expand_include(annotation.include, chunk.segment_count, key=chunk.key)
        features = extract_action_path_features(chunk.actions).cpu().numpy()
        action_dim = chunk.action_dim
        start_step = step_cursors.get(annotation.episode_id, 0)
        for segment_index in range(chunk.segment_count):
            records.append(
                ThrottleRecord(
                    episode_id=annotation.episode_id,
                    step_index=start_step + segment_index,
                    feature_coordinate_space=annotation.feature_coordinate_space,
                    delta=tuple(float(item) for item in features[segment_index, :action_dim]),
                    curvature=tuple(float(item) for item in features[segment_index, action_dim:]),
                    beta_target=betas[segment_index],
                    failure_event=failures[segment_index],
                    include_in_training=includes[segment_index],
                    metadata={
                        "source": {
                            "trace_schema_version": TRACE_SCHEMA_VERSION,
                            "trace_action_field": TRACE_ACTION_FIELD,
                            "trace_sha256": trace_sha256,
                            "annotation_schema": ANNOTATION_SCHEMA_ID,
                            "annotation_sha256": annotation_sha256,
                            "session_id": chunk.session_id,
                            "chunk_sequence": chunk.chunk_sequence,
                            "segment_index": segment_index,
                        },
                        "annotation_notes": annotation.notes,
                    },
                )
            )
        step_cursors[annotation.episode_id] = start_step + chunk.segment_count
    return records


def annotation_template_rows(chunks: Sequence[TraceActionChunk]) -> list[dict[str, Any]]:
    """Return deliberately incomplete human-editable rows with no generated labels."""

    return [
        {
            "schema": ANNOTATION_TEMPLATE_SCHEMA_ID,
            "session_id": chunk.session_id,
            "episode_id": None,
            "chunk_sequence": chunk.chunk_sequence,
            "segment_count": chunk.segment_count,
            "action_dim": chunk.action_dim,
            "feature_coordinate_space": ROBOT_ACTION_COORDINATE_SPACE,
            "beta": None,
            "failure": None,
            "include": None,
        }
        for chunk in chunks
    ]


def write_jsonl(path: str | Path, rows: Sequence[Mapping[str, Any]], *, overwrite: bool = False) -> None:
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")


def convert_trace_annotations(
    trace_path: str | Path,
    annotation_path: str | Path,
    output_path: str | Path,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    trace_path = Path(trace_path)
    annotation_path = Path(annotation_path)
    output_path = Path(output_path)
    provenance_path = output_path.with_suffix(output_path.suffix + ".provenance.json")
    if provenance_path.exists() and not overwrite:
        raise FileExistsError(f"provenance output already exists: {provenance_path}")
    trace_hash = file_sha256(trace_path)
    annotation_hash = file_sha256(annotation_path)
    chunks = load_realtime_trace_v2(trace_path)
    annotations = load_throttle_annotations(annotation_path)
    records = prepare_throttle_records(
        chunks,
        annotations,
        trace_sha256=trace_hash,
        annotation_sha256=annotation_hash,
    )
    write_jsonl(output_path, [record.to_mapping() for record in records], overwrite=overwrite)
    provenance = {
        "format": PREPARATION_FORMAT,
        "format_version": PREPARATION_FORMAT_VERSION,
        "output_schema": THROTTLE_SCHEMA_ID,
        "feature_coordinate_space": ROBOT_ACTION_COORDINATE_SPACE,
        "source_trace": {"path": str(trace_path), "sha256": trace_hash},
        "source_annotations": {"path": str(annotation_path), "sha256": annotation_hash},
        "output": {"path": str(output_path), "sha256": file_sha256(output_path)},
        "num_chunks": len(chunks),
        "num_segments": len(records),
    }
    provenance_path.write_text(
        json.dumps(provenance, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return provenance
