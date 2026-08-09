from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from lerobot.rollout.speed_adapter import (
    ROBOT_ACTION_COORDINATE_SPACE,
    SPEED_ADAPTER_METADATA_NAME,
    SPEED_ADAPTER_WEIGHTS_NAME,
    THROTTLE_SCHEMA_ID,
    SpeedAdapter,
    SpeedAdapterConfig,
    SpeedAdapterRuntimeConfig,
    ThrottleRecord,
    beta_to_segment_dt_ref,
    extract_action_path_features,
    failure_window_mask,
    load_runtime_speed_adapter,
    load_speed_adapter_checkpoint,
    load_throttle_jsonl,
    save_speed_adapter_checkpoint,
    throttle_records_to_tensors,
)


def _record(
    episode_id: str,
    step_index: int,
    *,
    failure_event: bool = False,
    include_in_training: bool = True,
) -> ThrottleRecord:
    return ThrottleRecord(
        episode_id=episode_id,
        step_index=step_index,
        feature_coordinate_space=ROBOT_ACTION_COORDINATE_SPACE,
        delta=(float(step_index), 1.0),
        curvature=(0.0, -0.5),
        state_embedding=(0.25,),
        phase_embedding=(1.0, 0.0),
        beta_target=1.0,
        failure_event=failure_event,
        include_in_training=include_in_training,
    )


def test_extract_action_path_features_uses_vector_delta_curvature_and_segment_start() -> None:
    actions = torch.tensor([[0.0, 0.0], [1.0, 0.0], [1.0, 2.0]])
    states = torch.tensor([[10.0], [11.0], [12.0]])
    phases = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

    features = extract_action_path_features(
        actions,
        state_embeddings=states,
        phase_embeddings=phases,
    )

    torch.testing.assert_close(
        features,
        torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0, 10.0, 1.0, 0.0],
                [0.0, 2.0, -1.0, 2.0, 11.0, 0.0, 1.0],
            ]
        ),
    )


def test_speed_adapter_output_is_bounded_and_predict_path_is_per_segment() -> None:
    config = SpeedAdapterConfig(action_dim=2, hidden_dims=(4,), beta_min=0.5, beta_max=1.5)
    model = SpeedAdapter(config)
    for parameter in model.parameters():
        torch.nn.init.zeros_(parameter)

    beta = model.predict_path([[0.0, 0.0], [1.0, 0.0], [3.0, 1.0]])

    assert beta.shape == (2,)
    torch.testing.assert_close(beta, torch.ones(2))
    random_beta = model(torch.randn(20, config.feature_dim))
    assert torch.all(random_beta >= config.beta_min)
    assert torch.all(random_beta <= config.beta_max)


def test_beta_to_segment_dt_ref_preserves_container_contract_and_applies_period_bounds() -> None:
    tensor_result = beta_to_segment_dt_ref(
        torch.tensor([0.5, 1.0, 2.0]),
        base_dt_s=0.05,
        min_dt_s=0.03,
        max_dt_s=0.08,
    )
    torch.testing.assert_close(tensor_result, torch.tensor([0.08, 0.05, 0.03]))

    array_result = beta_to_segment_dt_ref(np.array([0.5, 2.0]), base_dt_s=0.1)
    np.testing.assert_allclose(array_result, [0.2, 0.05])
    assert beta_to_segment_dt_ref(2.0, base_dt_s=0.1) == pytest.approx(0.05)

    with pytest.raises(ValueError, match="positive"):
        beta_to_segment_dt_ref([1.0, 0.0], base_dt_s=0.05)


def test_failure_window_mask_is_episode_local_and_combines_manual_exclusion() -> None:
    records = [_record("a", step, failure_event=step == 2) for step in range(5)]
    records.extend(
        [
            _record("b", 1),
            _record("b", 2, include_in_training=False),
            _record("b", 3),
        ]
    )

    mask = failure_window_mask(records, preceding_steps=1, following_steps=1)

    assert mask.tolist() == [True, False, False, False, True, True, False, True]


def test_jsonl_loader_and_tensor_layout_are_strict_and_auditable(tmp_path) -> None:
    path = tmp_path / "throttle.jsonl"
    rows = [_record("episode-1", 0).to_mapping(), _record("episode-1", 1).to_mapping()]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    records = load_throttle_jsonl(path)
    data = throttle_records_to_tensors(records)

    assert len(records) == 2
    assert data.features.shape == (2, 7)
    assert data.layout.feature_names == (
        "delta.0",
        "delta.1",
        "curvature.0",
        "curvature.1",
        "state_embedding.0",
        "phase_embedding.0",
        "phase_embedding.1",
    )

    rows.append({**rows[-1], "unexpected": 1})
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown fields"):
        load_throttle_jsonl(path)


def test_checkpoint_round_trip_preserves_predictions_and_provenance(tmp_path) -> None:
    config = SpeedAdapterConfig(
        action_dim=2,
        state_dim=1,
        phase_dim=2,
        hidden_dims=(8, 4),
        beta_min=0.4,
        beta_max=1.6,
    )
    mean = torch.arange(config.feature_dim, dtype=torch.float32)
    model = SpeedAdapter(config, feature_mean=mean, feature_std=torch.full_like(mean, 2.0))
    features = torch.randn(5, config.feature_dim)
    expected = model(features)
    feature_names = tuple(f"feature.{index}" for index in range(config.feature_dim))

    metadata = save_speed_adapter_checkpoint(
        tmp_path,
        model,
        feature_names=feature_names,
        training_metadata={
            "num_training_samples": 12,
            "source_jsonl_sha256": "abc123",
        },
    )
    loaded, loaded_metadata = load_speed_adapter_checkpoint(tmp_path)

    assert (tmp_path / SPEED_ADAPTER_WEIGHTS_NAME).is_file()
    assert (tmp_path / SPEED_ADAPTER_METADATA_NAME).is_file()
    assert metadata["provenance"]["trained_from_collected_throttle_data"] is True
    assert loaded_metadata["feature_contract"]["feature_names"] == list(feature_names)
    torch.testing.assert_close(loaded(features), expected)


def test_jsonl_schema_id_is_required() -> None:
    raw = _record("episode", 0).to_mapping()
    raw["schema"] = "some.other.schema"

    with pytest.raises(ValueError, match=THROTTLE_SCHEMA_ID):
        ThrottleRecord.from_mapping(raw)


def test_runtime_loader_requires_explicit_valid_trained_checkpoint(tmp_path) -> None:
    with pytest.raises(ValueError, match="checkpoint is required"):
        SpeedAdapterRuntimeConfig(enabled=True)

    missing = SpeedAdapterRuntimeConfig(enabled=True, checkpoint=str(tmp_path / "missing"))
    with pytest.raises(RuntimeError, match="failed to load enabled"):
        load_runtime_speed_adapter(missing, expected_action_dim=2)

    checkpoint = tmp_path / "trained"
    model = SpeedAdapter(SpeedAdapterConfig(action_dim=2, hidden_dims=(4,)))
    save_speed_adapter_checkpoint(
        checkpoint,
        model,
        feature_names=("delta.0", "delta.1", "curvature.0", "curvature.1"),
        training_metadata={"num_training_samples": 4},
    )
    loaded = load_runtime_speed_adapter(
        SpeedAdapterRuntimeConfig(enabled=True, checkpoint=str(checkpoint)),
        expected_action_dim=2,
    )
    assert loaded is not None
    assert loaded[0].config.action_dim == 2


def test_runtime_loader_rejects_untrained_or_embedding_dependent_checkpoint(tmp_path) -> None:
    feature_names = ("delta.0", "delta.1", "curvature.0", "curvature.1")
    untrained_dir = tmp_path / "untrained"
    save_speed_adapter_checkpoint(
        untrained_dir,
        SpeedAdapter(SpeedAdapterConfig(action_dim=2, hidden_dims=(4,))),
        feature_names=feature_names,
    )
    with pytest.raises(RuntimeError, match="not marked as trained"):
        load_runtime_speed_adapter(SpeedAdapterRuntimeConfig(enabled=True, checkpoint=str(untrained_dir)))

    embedding_dir = tmp_path / "embedding"
    save_speed_adapter_checkpoint(
        embedding_dir,
        SpeedAdapter(SpeedAdapterConfig(action_dim=2, state_dim=1, hidden_dims=(4,))),
        feature_names=(*feature_names, "state_embedding.0"),
        training_metadata={"num_training_samples": 4},
    )
    with pytest.raises(RuntimeError, match="supplies delta/curvature only"):
        load_runtime_speed_adapter(SpeedAdapterRuntimeConfig(enabled=True, checkpoint=str(embedding_dir)))


def test_runtime_loader_rejects_feature_coordinate_space_mismatch(tmp_path) -> None:
    checkpoint = tmp_path / "robot-space"
    save_speed_adapter_checkpoint(
        checkpoint,
        SpeedAdapter(SpeedAdapterConfig(action_dim=2, hidden_dims=(4,))),
        feature_names=("delta.0", "delta.1", "curvature.0", "curvature.1"),
        training_metadata={"num_training_samples": 4},
    )

    with pytest.raises(RuntimeError, match="coordinate space mismatch"):
        load_runtime_speed_adapter(
            SpeedAdapterRuntimeConfig(
                enabled=True,
                checkpoint=str(checkpoint),
                feature_coordinate_space="policy_action_units",
            )
        )
