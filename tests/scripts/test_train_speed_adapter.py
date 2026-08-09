from __future__ import annotations

import json

from lerobot.rollout.speed_adapter import (
    ROBOT_ACTION_COORDINATE_SPACE,
    SPEED_ADAPTER_METADATA_NAME,
    SPEED_ADAPTER_WEIGHTS_NAME,
    THROTTLE_SCHEMA_ID,
    load_speed_adapter_checkpoint,
)
from lerobot.scripts.lerobot_train_speed_adapter import main


def _row(episode: str, step: int, *, failure: bool = False) -> dict:
    return {
        "schema": THROTTLE_SCHEMA_ID,
        "episode_id": episode,
        "step_index": step,
        "feature_coordinate_space": ROBOT_ACTION_COORDINATE_SPACE,
        "delta": [0.1 * step, 0.2],
        "curvature": [0.01 * step, -0.02],
        "state_embedding": [float(step) / 10],
        "phase_embedding": [float(step % 2)],
        "beta_target": 0.8 + 0.05 * step,
        "failure_event": failure,
    }


def test_training_cli_writes_loadable_checkpoint_with_real_data_provenance(tmp_path) -> None:
    data_path = tmp_path / "throttle.jsonl"
    rows = [
        *[_row("episode-a", step, failure=step == 3) for step in range(5)],
        *[_row("episode-b", step) for step in range(5)],
    ]
    data_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    output_dir = tmp_path / "checkpoint"

    status = main(
        [
            "--data",
            str(data_path),
            "--output-dir",
            str(output_dir),
            "--beta-min",
            "0.5",
            "--beta-max",
            "1.5",
            "--epochs",
            "3",
            "--batch-size",
            "4",
            "--validation-fraction",
            "0.5",
            "--failure-preceding-steps",
            "1",
            "--device",
            "cpu",
            "--log-every",
            "3",
        ]
    )

    assert status == 0
    assert (output_dir / SPEED_ADAPTER_WEIGHTS_NAME).is_file()
    assert (output_dir / SPEED_ADAPTER_METADATA_NAME).is_file()
    model, metadata = load_speed_adapter_checkpoint(output_dir)
    assert model.config.action_dim == 2
    assert model.config.state_dim == 1
    assert model.config.phase_dim == 1
    assert metadata["provenance"]["trained_from_collected_throttle_data"] is True
    assert metadata["training"]["num_input_samples"] == 10
    assert metadata["training"]["num_total_exclusions"] == 2
    assert metadata["training"]["split"]["strategy"] == "episode_grouped"
