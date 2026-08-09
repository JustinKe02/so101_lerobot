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

from __future__ import annotations

import json
from pathlib import Path
from threading import Event
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from lerobot.policies.pi05.realtime_vla_v2_triton import PI05RealtimeVLATritonOutput
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.rollout import context as rollout_context
from lerobot.rollout.configs import PI05ActionBackend
from lerobot.rollout.inference.factory import RTCInferenceConfig
from lerobot.rollout.pi05_parity import (
    PI05ParityReportArtifactConfig,
    PI05ParityReportError,
    build_pi05_parity_report,
    file_sha256,
    load_and_validate_pi05_parity_report,
    write_pi05_parity_report_atomic,
)
from lerobot.rollout.trajectory import RealtimeTraceWriter
from lerobot.scripts.lerobot_pi05_realtime_vla_v2_parity import (
    ParityRequest,
    main as parity_main,
    request_fingerprint,
    save_request,
    save_result,
)


def _write_config(path: Path, training_max_delay: int) -> None:
    path.write_text(
        json.dumps({"rtc_training_max_delay": training_max_delay}),
        encoding="utf-8",
    )


def _write_serialized_result(
    path: Path,
    *,
    implementation: str,
    prefix: int,
    fingerprint: str,
    actions: np.ndarray,
    seed: int = 1000,
    noise: np.ndarray | None = None,
) -> None:
    metadata = {
        "schema": 1,
        "implementation": implementation,
        "request_fingerprint": fingerprint,
        "prefill_length": prefix,
        "seed": seed,
    }
    if noise is None:
        noise = np.random.default_rng(seed).standard_normal((50, 32)).astype(np.float32)
    np.savez_compressed(
        path,
        metadata=np.asarray(json.dumps(metadata)),
        actions=actions,
        noise=noise,
    )


def _make_request(prefix: int) -> ParityRequest:
    images = {
        "top": np.zeros((4, 5, 3), dtype=np.float32),
        "wrist": np.ones((3, 4, 5), dtype=np.float32),
    }
    state = np.linspace(-0.5, 0.5, 6, dtype=np.float32)
    prefill = np.full((prefix, 6), 0.1 * prefix, dtype=np.float32)
    fingerprint = request_fingerprint(
        images=images,
        normalized_state=state,
        normalized_prefill_actions=prefill,
        prompt="pick",
        camera_keys=("top", "wrist"),
        image_value_range="zero_one",
    )
    return ParityRequest(
        images=images,
        normalized_state=state,
        normalized_prefill_actions=prefill,
        prompt="pick",
        camera_keys=("top", "wrist"),
        image_value_range="zero_one",
        fingerprint=fingerprint,
    )


def _rewrite_npz(path: Path, mutator) -> None:
    with np.load(path, allow_pickle=False) as archive:
        payload = {key: np.array(archive[key]) for key in archive.files}
    metadata = json.loads(payload["metadata"].item())
    mutator(payload, metadata)
    payload["metadata"] = np.asarray(json.dumps(metadata))
    np.savez_compressed(path, **payload)


def _repin_report(report_path: Path, document: dict) -> PI05ParityReportArtifactConfig:
    report_path.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return PI05ParityReportArtifactConfig(
        enabled=True,
        path=str(report_path),
        sha256=file_sha256(report_path),
    )


def _refresh_prefix_hashes(document: dict, prefix: int) -> None:
    result = document["prefix_results"][prefix]
    for field in ("request", "pytorch_output", "triton_output"):
        result[field]["sha256"] = file_sha256(result[field]["path"])


def _build_artifact_bundle(
    tmp_path: Path,
    *,
    training_max_delay: int = 2,
    triton_action_delta: float = 0.0,
):
    checkpoint = tmp_path / "pretrained_model"
    checkpoint.mkdir(parents=True)
    _write_config(checkpoint / "config.json", training_max_delay)
    (checkpoint / "model.safetensors").write_bytes(b"source checkpoint weights")
    triton_weights = tmp_path / "pi05.pkl"
    triton_weights.write_bytes(b"triton export weights")

    prefix_results = []
    for prefix in range(training_max_delay + 1):
        request_path = tmp_path / f"request-{prefix}.npz"
        pytorch_output = tmp_path / f"pytorch-{prefix}.npz"
        triton_output = tmp_path / f"triton-{prefix}.npz"
        request = _make_request(prefix)
        save_request(request_path, request)
        fingerprint = request.fingerprint
        reference_actions = np.full((50, 6), prefix, dtype=np.float32)
        _write_serialized_result(
            pytorch_output,
            implementation="lerobot-pytorch",
            prefix=prefix,
            fingerprint=fingerprint,
            actions=reference_actions,
        )
        _write_serialized_result(
            triton_output,
            implementation="realtime-vla-v2-triton",
            prefix=prefix,
            fingerprint=fingerprint,
            actions=reference_actions + triton_action_delta,
        )
        error = np.abs((reference_actions + triton_action_delta) - reference_actions)
        prefix_results.append(
            {
                "prefix_length": prefix,
                "passed": True,
                "max_abs_error": float(error.max(initial=0.0)),
                "mean_abs_error": float(error.mean()),
                "shape": (50, 6),
                "seed": 1000,
                "request_fingerprint": fingerprint,
                "request_path": str(request_path),
                "pytorch_output_path": str(pytorch_output),
                "triton_output_path": str(triton_output),
            }
        )
    report = build_pi05_parity_report(
        checkpoint_path=checkpoint,
        triton_weights_path=triton_weights,
        triton_weights_sha256=file_sha256(triton_weights),
        triton_model_config_path=checkpoint / "config.json",
        training_max_delay=training_max_delay,
        atol=0.08,
        rtol=0.02,
        prefix_results=prefix_results,
    )
    report_path = tmp_path / "aggregate-report.json"
    report_sha256 = write_pi05_parity_report_atomic(report_path, report)
    config = PI05ParityReportArtifactConfig(
        enabled=True,
        path=str(report_path),
        sha256=report_sha256,
    )
    return checkpoint, triton_weights, prefix_results, report, report_path, config


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"enabled": True}, "path is required"),
        ({"enabled": True, "path": "report.json"}, "sha256 is required"),
        (
            {"enabled": True, "path": "report.json", "sha256": "bad"},
            "64 hexadecimal",
        ),
    ],
)
def test_parity_report_config_requires_checksum_pin(kwargs: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        PI05ParityReportArtifactConfig(**kwargs)


def test_strict_loader_validates_complete_prefix_and_artifact_chain(tmp_path: Path) -> None:
    checkpoint, weights, _, _, _, config = _build_artifact_bundle(tmp_path)

    validated = load_and_validate_pi05_parity_report(
        config,
        checkpoint_path=checkpoint,
        triton_weights_path=weights,
        triton_weights_sha256=file_sha256(weights),
        triton_model_config_path=checkpoint / "config.json",
        training_max_delay=2,
    )

    assert validated.prefixes == (0, 1, 2)
    assert validated.training_max_delay == 2
    assert validated.atol == pytest.approx(0.08)
    assert validated.rtol == pytest.approx(0.02)
    assert validated.audit_snapshot()["passed"] is True
    trace_snapshot: dict = {}
    rollout_context._inject_pi05_parity_trace_provenance(trace_snapshot, validated)
    assert trace_snapshot["resolved_pi05_triton_parity_report"] == validated.audit_snapshot()
    trace_path = tmp_path / "trace.jsonl"
    trace = RealtimeTraceWriter(trace_path, config_snapshot=trace_snapshot)
    trace.close()
    session_start = json.loads(trace_path.read_text(encoding="utf-8").splitlines()[0])
    assert (
        session_start["config_snapshot"]["resolved_pi05_triton_parity_report"] == validated.audit_snapshot()
    )


def test_aggregate_builder_rejects_incomplete_prefix_coverage(tmp_path: Path) -> None:
    checkpoint, weights, prefix_results, _, _, _ = _build_artifact_bundle(tmp_path)

    with pytest.raises(PI05ParityReportError, match=r"missing=\[1\]"):
        build_pi05_parity_report(
            checkpoint_path=checkpoint,
            triton_weights_path=weights,
            triton_weights_sha256=file_sha256(weights),
            triton_model_config_path=checkpoint / "config.json",
            training_max_delay=2,
            atol=0.08,
            rtol=0.02,
            prefix_results=[prefix_results[0], prefix_results[2]],
        )


def test_strict_loader_rejects_report_or_evidence_tampering(tmp_path: Path) -> None:
    checkpoint, weights, prefix_results, _, report_path, config = _build_artifact_bundle(tmp_path)
    report_path.write_text("{}", encoding="utf-8")
    with pytest.raises(PI05ParityReportError, match="report SHA-256 mismatch"):
        load_and_validate_pi05_parity_report(
            config,
            checkpoint_path=checkpoint,
            triton_weights_path=weights,
            triton_weights_sha256=file_sha256(weights),
            triton_model_config_path=checkpoint / "config.json",
            training_max_delay=2,
        )

    _, _, _, _, _, fresh_config = _build_artifact_bundle(tmp_path / "fresh")
    evidence_path = Path(prefix_results[0]["pytorch_output_path"])
    # The original bundle is already invalidated above; use the fresh report's
    # first referenced result to prove referenced evidence is also checked.
    fresh_document = json.loads(Path(fresh_config.path).read_text(encoding="utf-8"))
    fresh_evidence = Path(fresh_document["prefix_results"][0]["pytorch_output"]["path"])
    fresh_evidence.write_bytes(b"tampered")
    fresh_checkpoint = Path(fresh_document["source_checkpoint"]["path"])
    fresh_weights = Path(fresh_document["triton_export"]["weights"]["path"])
    with pytest.raises(PI05ParityReportError, match="pytorch_output SHA-256 mismatch"):
        load_and_validate_pi05_parity_report(
            fresh_config,
            checkpoint_path=fresh_checkpoint,
            triton_weights_path=fresh_weights,
            triton_weights_sha256=fresh_document["triton_export"]["weights"]["sha256"],
            triton_model_config_path=fresh_checkpoint / "config.json",
            training_max_delay=2,
        )
    assert evidence_path.is_file()


def test_strict_loader_recomputes_and_rejects_forged_passed_result(tmp_path: Path) -> None:
    checkpoint, weights, _, _, _, config = _build_artifact_bundle(
        tmp_path,
        triton_action_delta=1.0,
    )

    with pytest.raises(PI05ParityReportError, match="passed disagrees with recomputed NPZ evidence"):
        load_and_validate_pi05_parity_report(
            config,
            checkpoint_path=checkpoint,
            triton_weights_path=weights,
            triton_weights_sha256=file_sha256(weights),
            triton_model_config_path=checkpoint / "config.json",
            training_max_delay=2,
        )


def test_strict_loader_rejects_forged_request_fingerprint_even_when_repinning(
    tmp_path: Path,
) -> None:
    checkpoint, weights, _, _, report_path, _ = _build_artifact_bundle(tmp_path)
    document = json.loads(report_path.read_text(encoding="utf-8"))
    prefix_result = document["prefix_results"][0]
    forged_fingerprint = "f" * 64

    def forge_request(_payload: dict, metadata: dict) -> None:
        metadata["fingerprint"] = forged_fingerprint

    def forge_result(_payload: dict, metadata: dict) -> None:
        metadata["request_fingerprint"] = forged_fingerprint

    _rewrite_npz(Path(prefix_result["request"]["path"]), forge_request)
    _rewrite_npz(Path(prefix_result["pytorch_output"]["path"]), forge_result)
    _rewrite_npz(Path(prefix_result["triton_output"]["path"]), forge_result)
    prefix_result["request_fingerprint"] = forged_fingerprint
    _refresh_prefix_hashes(document, 0)
    config = _repin_report(report_path, document)

    with pytest.raises(PI05ParityReportError, match="request.*fingerprint mismatch"):
        load_and_validate_pi05_parity_report(
            config,
            checkpoint_path=checkpoint,
            triton_weights_path=weights,
            triton_weights_sha256=file_sha256(weights),
            triton_model_config_path=checkpoint / "config.json",
            training_max_delay=2,
        )


@pytest.mark.parametrize("tamper", ["seed", "noise"])
def test_strict_loader_rejects_noise_not_generated_by_reported_seed(
    tmp_path: Path,
    tamper: str,
) -> None:
    checkpoint, weights, _, _, report_path, _ = _build_artifact_bundle(tmp_path)
    document = json.loads(report_path.read_text(encoding="utf-8"))
    prefix_result = document["prefix_results"][0]

    def mutate_result(payload: dict, metadata: dict) -> None:
        if tamper == "seed":
            metadata["seed"] += 1
        else:
            payload["noise"][0, 0] += np.float32(1.0)

    for field in ("pytorch_output", "triton_output"):
        _rewrite_npz(Path(prefix_result[field]["path"]), mutate_result)
    if tamper == "seed":
        prefix_result["seed"] += 1
    _refresh_prefix_hashes(document, 0)
    config = _repin_report(report_path, document)

    with pytest.raises(PI05ParityReportError, match="noise does not match fixed diffusion noise"):
        load_and_validate_pi05_parity_report(
            config,
            checkpoint_path=checkpoint,
            triton_weights_path=weights,
            triton_weights_sha256=file_sha256(weights),
            triton_model_config_path=checkpoint / "config.json",
            training_max_delay=2,
        )


def test_strict_loader_rejects_request_content_tampering_even_when_repinning(
    tmp_path: Path,
) -> None:
    checkpoint, weights, _, _, report_path, _ = _build_artifact_bundle(tmp_path)
    document = json.loads(report_path.read_text(encoding="utf-8"))
    prefix_result = document["prefix_results"][1]

    def tamper_state(payload: dict, _metadata: dict) -> None:
        payload["normalized_state"][0] += np.float32(0.25)

    _rewrite_npz(Path(prefix_result["request"]["path"]), tamper_state)
    _refresh_prefix_hashes(document, 1)
    config = _repin_report(report_path, document)

    with pytest.raises(PI05ParityReportError, match="request.*fingerprint mismatch"):
        load_and_validate_pi05_parity_report(
            config,
            checkpoint_path=checkpoint,
            triton_weights_path=weights,
            triton_weights_sha256=file_sha256(weights),
            triton_model_config_path=checkpoint / "config.json",
            training_max_delay=2,
        )


def test_aggregate_cli_atomically_writes_cpu_report(tmp_path: Path, capsys) -> None:
    checkpoint = tmp_path / "pretrained_model"
    checkpoint.mkdir()
    _write_config(checkpoint / "config.json", 2)
    (checkpoint / "model.safetensors").write_bytes(b"checkpoint")
    weights = tmp_path / "weights.pkl"
    weights.write_bytes(b"weights")
    pairs: list[tuple[int, Path, Path, Path]] = []
    for prefix in range(3):
        request = _make_request(prefix)
        request_path = tmp_path / f"request-{prefix}.npz"
        save_request(request_path, request)
        noise = np.random.default_rng(7).standard_normal((50, 32)).astype(np.float32)
        output = PI05RealtimeVLATritonOutput(
            actions=np.full((50, 6), prefix, dtype=np.float32),
            prefill_length=prefix,
        )
        pytorch_path = tmp_path / f"pytorch-{prefix}.npz"
        triton_path = tmp_path / f"triton-{prefix}.npz"
        save_result(
            pytorch_path,
            output,
            noise,
            implementation="lerobot-pytorch",
            request=request,
            seed=7,
        )
        save_result(
            triton_path,
            output,
            noise,
            implementation="realtime-vla-v2-triton",
            request=request,
            seed=7,
        )
        pairs.append((prefix, request_path, pytorch_path, triton_path))

    output_path = tmp_path / "reports" / "aggregate.json"
    argv = [
        "aggregate",
        "--checkpoint",
        str(checkpoint),
        "--weights",
        str(weights),
        "--weights-sha256",
        file_sha256(weights),
        "--model-config",
        str(checkpoint / "config.json"),
        "--training-max-delay",
        "2",
        "--output",
        str(output_path),
    ]
    for prefix, request_path, pytorch_path, triton_path in pairs:
        argv.extend(["--pair", str(prefix), str(request_path), str(pytorch_path), str(triton_path)])

    assert parity_main(argv) == 0
    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["passed"] is True
    assert report["schema_version"] == 2
    assert [result["prefix_length"] for result in report["prefix_results"]] == [0, 1, 2]
    assert all("request" in result for result in report["prefix_results"])
    assert not list(output_path.parent.glob(f".{output_path.name}.*.tmp"))
    assert file_sha256(output_path) in capsys.readouterr().out


def test_bad_parity_report_fails_before_cuda_backend_or_hardware(tmp_path: Path) -> None:
    checkpoint = tmp_path / "pretrained_model"
    checkpoint.mkdir()
    _write_config(checkpoint / "config.json", 2)
    weights = tmp_path / "weights.pkl"
    weights.write_bytes(b"weights")
    report_path = tmp_path / "report.json"
    report_path.write_text("{}", encoding="utf-8")
    cfg = SimpleNamespace(
        inference=RTCInferenceConfig(
            rtc=RTCConfig(enabled=True, execution_horizon=10),
            mode="trained_prefix",
            timing_mode="actual_consumed",
            guidance_delay_mode="fixed",
        ),
        pi05_action_backend=PI05ActionBackend.TRITON,
        pi05_triton_parity_report=PI05ParityReportArtifactConfig(
            enabled=True,
            path=str(report_path),
            sha256="0" * 64,
        ),
        pi05_triton_export_weights=str(weights),
        pi05_triton_weights_sha256=file_sha256(weights),
        policy=SimpleNamespace(
            pretrained_path=str(checkpoint),
            rtc_training_max_delay=2,
        ),
        resolved_pi05_triton_model_config=lambda: str(checkpoint / "config.json"),
    )

    with (
        patch.object(rollout_context, "get_policy_class") as get_policy_class,
        patch.object(rollout_context, "make_robot_from_config") as make_robot,
        pytest.raises(PI05ParityReportError, match="report SHA-256 mismatch"),
    ):
        rollout_context._build_rollout_context(
            cfg,
            Event(),
            hardware_state=rollout_context._HardwareBuildState(),
        )

    get_policy_class.assert_not_called()
    make_robot.assert_not_called()
