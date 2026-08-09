import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest

from lerobot.rollout.actuator_calibration import load_actuator_calibration_artifact
from lerobot.scripts import lerobot_calibrate_so101_actuator as calibration
from lerobot.scripts.lerobot_calibrate_so101_actuator import (
    CalibrationError,
    FitConfig,
    calibrate_file,
    load_calibration_samples,
)


def _synthetic_trace(sample_count: int = 420, dt_s: float = 0.02):
    rng = np.random.default_rng(1741)
    timestamps = np.round(np.arange(sample_count, dtype=np.float64) * dt_s, 9)
    delays = np.asarray([0.00, 0.02, 0.04, 0.06, 0.08, 0.10])
    taus = np.asarray([0.05, 0.07, 0.10, 0.14, 0.18, 0.24])
    delay_steps = np.rint(delays / dt_s).astype(np.int64)

    commands = np.zeros((sample_count, 6), dtype=np.float64)
    block_size = 8
    for block_start in range(0, sample_count, block_size):
        commands[block_start : block_start + block_size] = rng.uniform(-60.0, 60.0, size=6)

    observed = np.zeros_like(commands)
    observed[0] = commands[0]
    for sample_index in range(1, sample_count):
        for joint_index in range(6):
            command_index = max(0, sample_index - 1 - delay_steps[joint_index])
            active_command = commands[command_index, joint_index]
            decay = math.exp(-dt_s / taus[joint_index])
            observed[sample_index, joint_index] = (
                active_command + (observed[sample_index - 1, joint_index] - active_command) * decay
            )
    observed += rng.normal(0.0, 2e-4, size=observed.shape)
    return timestamps, commands, observed, delays, taus


def _write_jsonl(
    path: Path,
    timestamps: np.ndarray,
    commands: np.ndarray,
    observed: np.ndarray,
) -> None:
    with path.open("w", encoding="utf-8") as output:
        for timestamp, command, position in zip(timestamps, commands, observed, strict=True):
            output.write(
                json.dumps(
                    {
                        "monotonic_timestamp": float(timestamp),
                        "clock_domain": "monotonic",
                        "command": command.tolist(),
                        "observed": position.tolist(),
                    }
                )
                + "\n"
            )


def test_recovers_per_joint_delay_and_tau_and_writes_auditable_artifact(tmp_path: Path) -> None:
    timestamps, commands, observed, expected_delays, expected_taus = _synthetic_trace()
    trace_path = tmp_path / "trace.jsonl"
    artifact_path = tmp_path / "calibration.json"
    _write_jsonl(trace_path, timestamps, commands, observed)

    artifact = calibrate_file(
        trace_path,
        output_path=artifact_path,
        config=FitConfig(
            max_delay_s=0.14,
            min_tau_s=0.02,
            max_tau_s=0.40,
            min_fit_samples=100,
        ),
    )

    np.testing.assert_allclose(artifact["command_delay_s"], expected_delays, atol=0.004)
    np.testing.assert_allclose(artifact["tau_s"], expected_taus, rtol=0.025, atol=0.002)
    assert artifact["schema_version"] == 1
    assert artifact["clock_domain"] == "monotonic"
    assert artifact["joint_names"] == list(calibration.DEFAULT_SO101_JOINT_NAMES)
    assert artifact["source"]["sha256"] == hashlib.sha256(trace_path.read_bytes()).hexdigest()
    assert artifact["source"]["sample_count"] == len(timestamps)
    assert all(item["sample_count"] >= 100 for item in artifact["fit_diagnostics"])
    assert json.loads(artifact_path.read_text(encoding="utf-8")) == artifact

    artifact_sha256 = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    loaded = load_actuator_calibration_artifact(
        artifact_path,
        expected_sha256=artifact_sha256,
        action_dim=6,
        joint_names=calibration.DEFAULT_SO101_JOINT_NAMES,
    )
    np.testing.assert_allclose(loaded.command_delay_s, expected_delays, atol=0.004)
    np.testing.assert_allclose(loaded.tau_s, expected_taus, rtol=0.025, atol=0.002)


def test_loads_csv_with_flattened_vectors(tmp_path: Path) -> None:
    path = tmp_path / "trace.csv"
    fieldnames = [
        "timestamp",
        "clock_domain",
        *(f"command_{index}" for index in range(6)),
        *(f"observed_{index}" for index in range(6)),
    ]
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        for index in range(4):
            row = {"timestamp": index * 0.02, "clock_domain": "monotonic"}
            row.update({f"command_{joint}": index + joint for joint in range(6)})
            row.update({f"observed_{joint}": index - joint for joint in range(6)})
            writer.writerow(row)

    samples = load_calibration_samples(path)

    assert samples.source_format == "csv"
    assert samples.commands.shape == (4, 6)
    np.testing.assert_array_equal(samples.commands[2], np.arange(6) + 2)
    np.testing.assert_array_equal(samples.observed[2], 2 - np.arange(6))


@pytest.mark.parametrize(
    "rows",
    [
        [
            {"timestamp": 1.0, "clock_domain": "monotonic"},
            {"timestamp": 1.0, "clock_domain": "monotonic"},
            {"timestamp": 1.1, "clock_domain": "monotonic"},
        ],
        [
            {"timestamp": 1.0, "clock_domain": "unix"},
            {"timestamp": 1.1, "clock_domain": "unix"},
            {"timestamp": 1.2, "clock_domain": "unix"},
        ],
    ],
)
def test_rejects_bad_clock(rows: list[dict], tmp_path: Path) -> None:
    path = tmp_path / "bad_clock.jsonl"
    with path.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps({**row, "command": [0] * 6, "observed": [0] * 6}) + "\n")

    with pytest.raises(CalibrationError, match="clock_domain|strictly increasing"):
        load_calibration_samples(path)


def test_missing_scipy_has_actionable_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing_package(*_args, **_kwargs):
        raise ImportError("missing")

    monkeypatch.setattr(calibration, "require_package", missing_package)

    with pytest.raises(RuntimeError, match=r"uv sync --extra scipy-dep"):
        calibration._load_least_squares()
