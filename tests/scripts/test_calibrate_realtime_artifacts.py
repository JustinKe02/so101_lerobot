from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from lerobot.rollout.joint_constraints import load_joint_constraint_artifact
from lerobot.rollout.sensor_timing_calibration import load_sensor_timing_calibration_artifact
from lerobot.scripts.lerobot_calibrate_joint_constraints import (
    DEFAULT_SO101_JOINT_NAMES,
    build_joint_constraint_artifact,
    write_artifact as write_joint_artifact,
)
from lerobot.scripts.lerobot_calibrate_sensor_timing import (
    build_sensor_timing_artifact,
    write_artifact,
)


def test_sensor_cli_builder_uses_higher_quantile_and_writes_loadable_artifact(tmp_path: Path) -> None:
    source = tmp_path / "sensor.jsonl"
    rows = [
        {
            "clock_domain": "monotonic",
            "camera_capture_delay_s": {"top": top, "wrist": wrist},
            "state_observation_delay_s": state,
            "camera_skew_s": skew,
        }
        for top, wrist, state, skew in (
            (0.010, 0.020, 0.005, 0.002),
            (0.012, 0.025, 0.007, 0.004),
            (0.018, 0.030, 0.009, 0.006),
        )
    ]
    source.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    document = build_sensor_timing_artifact(
        source,
        method="paired monotonic LED and bus reference",
        quantile=0.95,
        camera_keys=("top", "wrist"),
    )
    output = tmp_path / "sensor.json"
    sha256 = write_artifact(document, output)
    artifact = load_sensor_timing_calibration_artifact(
        output, expected_sha256=sha256, camera_keys=("top", "wrist")
    )

    assert artifact.camera_delay_by_key == {"top": 0.018, "wrist": 0.03}
    assert artifact.source_sample_count == 3


def test_joint_cli_builder_skips_episode_boundaries_and_writes_loadable_artifact(tmp_path: Path) -> None:
    source = tmp_path / "joints.jsonl"
    rows = []
    dt = 0.1
    for episode_index in range(2):
        for step in range(5):
            position = [
                float((joint_index + 1) * step * step + episode_index * 1000) for joint_index in range(6)
            ]
            rows.append(
                {
                    "clock_domain": "monotonic",
                    "monotonic_timestamp": episode_index * 10.0 + step * dt,
                    "episode_index": episode_index,
                    "observed": position,
                }
            )
    source.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    document = build_joint_constraint_artifact(source, quantile=1.0, safety_factor=1.2)
    output = tmp_path / "joint.json"
    sha256 = write_joint_artifact(document, output)
    artifact = load_joint_constraint_artifact(
        output, expected_sha256=sha256, joint_names=DEFAULT_SO101_JOINT_NAMES
    )

    assert artifact.source_sample_count == 10
    assert artifact.velocity_sample_count == 8
    assert artifact.acceleration_sample_count == 6
    np.testing.assert_allclose(artifact.max_velocity, np.asarray([70, 140, 210, 280, 350, 420]) * 1.2)
