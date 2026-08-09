from __future__ import annotations

import json

import pytest

from lerobot.rollout.speed_adapter import (
    ROBOT_ACTION_COORDINATE_SPACE,
    file_sha256,
    load_throttle_jsonl,
)
from lerobot.rollout.speed_adapter_data import (
    ANNOTATION_SCHEMA_ID,
    ANNOTATION_TEMPLATE_SCHEMA_ID,
    convert_trace_annotations,
)
from lerobot.rollout.trajectory import RealtimeTraceWriter
from lerobot.scripts.lerobot_prepare_speed_throttle import main


def _write_trace(path, chunks: list[list[list[float]]], *, schema_version: int = 2) -> list[int]:
    if schema_version == 2:
        writer = RealtimeTraceWriter(path, session_id="session-a")
        sequences = []
        for chunk in chunks:
            sequences.append(writer._sequence)
            writer.write("inference_chunk", raw_robot_chunk=chunk)
        writer.close()
        return sequences
    rows = [
        {
            "event": "session_start",
            "session_id": "session-a",
            "sequence": 0,
            "schema_version": schema_version,
        },
        {
            "event": "inference_chunk",
            "session_id": "session-a",
            "sequence": 1,
            "raw_robot_chunk": chunks[0],
        },
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return [1]


def _annotation(
    sequence: int,
    *,
    beta=1.5,
    failure=False,
    include=True,
    episode_id: str = "episode-a",
    segment_count: int = 2,
    coordinate_space: str = ROBOT_ACTION_COORDINATE_SPACE,
) -> dict:
    return {
        "schema": ANNOTATION_SCHEMA_ID,
        "session_id": "session-a",
        "episode_id": episode_id,
        "chunk_sequence": sequence,
        "segment_count": segment_count,
        "action_dim": 2,
        "feature_coordinate_space": coordinate_space,
        "beta": beta,
        "failure": failure,
        "include": include,
    }


def _write_annotations(path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_convert_trace_and_scalar_annotations_to_strict_segment_records(tmp_path) -> None:
    trace_path = tmp_path / "trace.jsonl"
    annotation_path = tmp_path / "annotations.jsonl"
    output_path = tmp_path / "throttle.jsonl"
    sequences = _write_trace(trace_path, [[[0.0, 0.0], [1.0, 0.0], [1.0, 2.0]]])
    _write_annotations(
        annotation_path,
        [_annotation(sequences[0], beta=1.5, failure=True, include=True)],
    )

    assert (
        main(
            [
                "--trace",
                str(trace_path),
                "--annotations",
                str(annotation_path),
                "--output",
                str(output_path),
            ]
        )
        == 0
    )
    records = load_throttle_jsonl(output_path)

    assert [record.beta_target for record in records] == [1.5, 1.5]
    assert [record.failure_event for record in records] == [False, True]
    assert [record.delta for record in records] == [(1.0, 0.0), (0.0, 2.0)]
    assert [record.curvature for record in records] == [(0.0, 0.0), (-1.0, 2.0)]
    assert records[0].metadata["source"]["chunk_sequence"] == sequences[0]
    assert records[0].metadata["source"]["trace_sha256"] == file_sha256(trace_path)
    assert records[0].metadata["source"]["annotation_sha256"] == file_sha256(annotation_path)
    sidecar = json.loads((tmp_path / "throttle.jsonl.provenance.json").read_text(encoding="utf-8"))
    assert sidecar["output"]["sha256"] == file_sha256(output_path)


def test_per_segment_labels_and_episode_step_indices_span_chunks(tmp_path) -> None:
    trace_path = tmp_path / "trace.jsonl"
    annotation_path = tmp_path / "annotations.jsonl"
    output_path = tmp_path / "throttle.jsonl"
    chunk = [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]
    sequences = _write_trace(trace_path, [chunk, chunk])
    _write_annotations(
        annotation_path,
        [
            _annotation(sequences[0], beta=[0.8, 1.2], failure=[False, False], include=[True, False]),
            _annotation(sequences[1], beta=[1.1, 1.3], failure=[True, False], include=True),
        ],
    )

    convert_trace_annotations(trace_path, annotation_path, output_path)
    records = load_throttle_jsonl(output_path)

    assert [record.step_index for record in records] == [0, 1, 2, 3]
    assert [record.beta_target for record in records] == pytest.approx([0.8, 1.2, 1.1, 1.3])
    assert [record.failure_event for record in records] == [False, False, True, False]
    assert [record.include_in_training for record in records] == [True, False, True, True]


def test_template_export_contains_no_synthetic_labels(tmp_path) -> None:
    trace_path = tmp_path / "trace.jsonl"
    template_path = tmp_path / "template.jsonl"
    _write_trace(trace_path, [[[0.0], [1.0], [2.0]]])

    assert main(["--trace", str(trace_path), "--export-annotation-template", str(template_path)]) == 0
    template = json.loads(template_path.read_text(encoding="utf-8"))

    assert template["schema"] == ANNOTATION_TEMPLATE_SCHEMA_ID
    assert template["episode_id"] is None
    assert template["beta"] is None
    assert template["failure"] is None
    assert template["include"] is None


def test_print_annotation_schema_does_not_require_files(capsys) -> None:
    assert main(["--print-annotation-schema"]) == 0
    schema = json.loads(capsys.readouterr().out)
    assert schema["$id"] == ANNOTATION_SCHEMA_ID
    assert "beta" in schema["required"]


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda row: row.pop("beta"), "missing labels/fields: beta"),
        (lambda row: row.update(feature_coordinate_space="policy_action_units"), "coordinate space mismatch"),
        (lambda row: row.update(segment_count=3), "does not match trace segment_count"),
        (lambda row: row.update(beta=[1.0]), "beta length must be 2"),
    ],
)
def test_rejects_missing_or_incompatible_annotations(tmp_path, mutation, message) -> None:
    trace_path = tmp_path / "trace.jsonl"
    annotation_path = tmp_path / "annotations.jsonl"
    output_path = tmp_path / "throttle.jsonl"
    sequence = _write_trace(trace_path, [[[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]])[0]
    row = _annotation(sequence)
    mutation(row)
    _write_annotations(annotation_path, [row])

    with pytest.raises((TypeError, ValueError), match=message):
        convert_trace_annotations(trace_path, annotation_path, output_path)

    assert not output_path.exists()


def test_rejects_duplicate_and_missing_chunk_annotations(tmp_path) -> None:
    trace_path = tmp_path / "trace.jsonl"
    annotation_path = tmp_path / "annotations.jsonl"
    output_path = tmp_path / "throttle.jsonl"
    sequences = _write_trace(
        trace_path,
        [
            [[0.0], [1.0]],
            [[1.0], [2.0]],
        ],
    )
    duplicate = _annotation(sequences[0], segment_count=1)
    duplicate["action_dim"] = 1
    _write_annotations(annotation_path, [duplicate, duplicate])
    with pytest.raises(ValueError, match="duplicate annotation key"):
        convert_trace_annotations(trace_path, annotation_path, output_path)

    _write_annotations(annotation_path, [duplicate])
    with pytest.raises(ValueError, match="missing annotations for trace chunks"):
        convert_trace_annotations(trace_path, annotation_path, output_path)


def test_rejects_old_or_summarized_trace_chunks(tmp_path) -> None:
    old_trace = tmp_path / "old.jsonl"
    annotations = tmp_path / "annotations.jsonl"
    output = tmp_path / "output.jsonl"
    _write_trace(old_trace, [[[0.0], [1.0]]], schema_version=1)
    row = _annotation(1, segment_count=1)
    row["action_dim"] = 1
    _write_annotations(annotations, [row])

    with pytest.raises(ValueError, match="unsupported trace schema"):
        convert_trace_annotations(old_trace, annotations, output)

    summarized_trace = tmp_path / "summarized.jsonl"
    rows = [
        {
            "event": "session_start",
            "session_id": "session-a",
            "sequence": 0,
            "schema_version": 2,
        },
        {
            "event": "inference_chunk",
            "session_id": "session-a",
            "sequence": 1,
            "raw_robot_chunk": {"shape": [50, 6], "storage": "summary"},
        },
    ]
    summarized_trace.write_text("\n".join(json.dumps(item) for item in rows) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="summarized action chunk"):
        convert_trace_annotations(summarized_trace, annotations, output)
