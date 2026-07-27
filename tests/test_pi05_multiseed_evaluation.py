#!/usr/bin/env python

import json
from argparse import Namespace
from copy import deepcopy
from pathlib import Path

import pytest
import torch

from examples.inference.evaluate_pi05_multiseed import (
    DEFAULT_SEEDS,
    DEPLOYMENT_GATE_PROFILE,
    METRIC_SEMANTICS,
    REPORT_SCHEMA_VERSION,
    DatasetIndex,
    GateThresholds,
    MetricsAccumulator,
    ModelSpec,
    SelectedFrame,
    build_parser,
    compute_prediction_record,
    detect_gripper_events,
    evaluate_formal_selection_coverage,
    evaluate_formal_stage_safety_gate,
    evaluate_gate,
    evaluate_model,
    evaluate_model_coverage,
    evaluate_relative_baseline_gate,
    evaluate_robustness_gate,
    extract_execution_state_chunk,
    extract_ground_truth_chunk,
    fingerprint_checkpoint_assets,
    fingerprint_dataset_assets,
    gate_exit_code,
    limit_frames_evenly,
    limit_frames_preserving_phase_anchors,
    load_baseline_model_report,
    make_shared_noise,
    parse_model_spec,
    resolve_run_shape,
    resolve_state_indices,
    select_episode_frames,
    validate_args,
)


def test_parse_model_spec_supports_label_and_bare_path():
    labeled = parse_model_spec("full=/tmp/checkpoint")
    bare = parse_model_spec("/tmp/adapter")

    assert labeled.label == "full"
    assert labeled.checkpoint == Path("/tmp/checkpoint")
    assert bare.label == "adapter"
    assert bare.checkpoint == Path("/tmp/adapter")


def test_quick_mode_resolves_five_frames_and_three_shared_seeds():
    args = build_parser().parse_args(["--model", "fake=/tmp/model", "--quick"])

    seeds, frame_limit = resolve_run_shape(args)

    assert seeds == [0, 42, 1000]
    assert frame_limit == 5


def test_detect_gripper_events_clusters_contiguous_motion_at_peak():
    actions = torch.zeros(10, 2)
    actions[:, 1] = torch.tensor([0, 0, 0, 2, 5, 5, 5, 1, 0, 0], dtype=torch.float32)

    events = detect_gripper_events(actions, gripper_index=1, threshold=1.0)

    assert events == [4, 7]


def test_episode_selection_merges_duplicate_phase_and_event_frames():
    row_indices = list(range(5, 16))
    frame_indices = torch.arange(20)
    absolute_indices = torch.arange(100, 120)
    actions = torch.zeros(11, 2)
    actions[:, 1] = torch.tensor([0, 0, 0, 2, 5, 5, 5, 5, 5, 5, 5], dtype=torch.float32)

    selected = select_episode_frames(
        episode_index=3,
        row_indices=row_indices,
        frame_indices=frame_indices,
        absolute_indices=absolute_indices,
        episode_actions=actions,
        gripper_index=1,
        gripper_event_threshold=1.0,
        gripper_neighbor_radius=1,
    )

    assert len({frame.row_index for frame in selected}) == len(selected)
    event_frame = next(frame for frame in selected if frame.local_index == 4)
    assert {"gripper_event", "gripper_at"}.issubset(event_frame.stages)
    assert event_frame.gripper_contexts == {(4, 0)}
    assert selected[0].local_index == 0
    assert selected[-1].local_index == 10


def test_ground_truth_chunk_never_pads_or_crosses_episode_tail():
    episode_actions = torch.arange(12, dtype=torch.float32).reshape(6, 2)

    chunk = extract_ground_truth_chunk(episode_actions, local_index=4, chunk_size=50)

    assert chunk.shape == (2, 2)
    torch.testing.assert_close(chunk, episode_actions[4:])


def test_execution_state_chunk_stays_in_episode_and_maps_joint_names():
    episode_states = torch.tensor(
        [
            [10.0, 100.0, 1.0],
            [20.0, 200.0, 2.0],
            [30.0, 300.0, 3.0],
        ]
    )

    chunk = extract_execution_state_chunk(
        episode_states,
        local_index=1,
        action_names=["arm.pos", "gripper.pos"],
        state_names=["gripper.pos", "unused.pos", "arm.pos"],
        chunk_size=50,
    )

    torch.testing.assert_close(chunk, torch.tensor([[2.0, 20.0], [3.0, 30.0]]))
    with pytest.raises(ValueError, match="missing action joints"):
        resolve_state_indices(["arm.pos", "missing.pos"], ["arm.pos"])


def test_even_frame_limit_is_deterministic_and_keeps_endpoints():
    actions = torch.zeros(20, 1)
    selected = select_episode_frames(
        episode_index=0,
        row_indices=list(range(20)),
        frame_indices=torch.arange(20),
        absolute_indices=torch.arange(20),
        episode_actions=actions,
        gripper_index=0,
        gripper_event_threshold=1.0,
        gripper_neighbor_radius=0,
    )

    limited = limit_frames_evenly(selected, 5)

    assert len(limited) == 5
    assert limited[0] is selected[0]
    assert limited[-1] is selected[-1]


def test_full_limit_preserves_phase_anchors_before_event_only_frames():
    actions = torch.zeros(20, 1)
    actions[8:12, 0] = torch.tensor([2.0, 4.0, 6.0, 8.0])
    selected = select_episode_frames(
        episode_index=0,
        row_indices=list(range(20)),
        frame_indices=torch.arange(20),
        absolute_indices=torch.arange(20),
        episode_actions=actions,
        gripper_index=0,
        gripper_event_threshold=1.0,
        gripper_neighbor_radius=2,
    )
    phase_names = {"start", "p10", "p25", "p50", "p75", "p90", "end"}
    anchors = [frame for frame in selected if frame.stages & phase_names]

    limited = limit_frames_preserving_phase_anchors(selected, len(anchors) + 1)

    assert {frame.row_index for frame in anchors}.issubset({frame.row_index for frame in limited})
    assert len(limited) == len(anchors) + 1


def test_shared_noise_is_reproducible_and_seed_specific():
    first = make_shared_noise([0, 42], chunk_size=50, max_action_dim=32)
    second = make_shared_noise([0, 42], chunk_size=50, max_action_dim=32)

    assert first[0].shape == (1, 50, 32)
    assert torch.equal(first[0], second[0])
    assert torch.equal(first[42], second[42])
    assert not torch.equal(first[0], first[42])


def test_prediction_metrics_cover_action13_clamp_jump_and_gripper_timing():
    ground_truth = torch.zeros(14, 2)
    ground_truth[3:, 1] = 2.0
    prediction = ground_truth.clone()
    prediction[:, 0] = 6.0
    prediction[3, 1] = 0.0
    prediction[4:, 1] = 2.0

    record = compute_prediction_record(
        prediction,
        ground_truth,
        torch.zeros_like(ground_truth),
        gripper_index=1,
        max_relative_target=5.0,
        gripper_event_threshold=1.0,
    )

    assert record.action13_errors is not None
    assert record.first_action_clamp.tolist() == [True, False]
    assert record.gt_gripper_event == 3
    assert record.predicted_gripper_event == 4
    assert record.gt_gripper_direction == 1
    assert record.predicted_gripper_direction == 1
    assert bool(record.action13_state_jump_exceeds[0]) is True
    assert record.jumps.shape == (13, 2)


def test_action13_execution_uses_future_state_while_query_state_is_diagnostic():
    ground_truth = torch.arange(14, dtype=torch.float32).reshape(14, 1)
    execution_states = ground_truth.clone()

    record = compute_prediction_record(
        ground_truth,
        ground_truth,
        execution_states,
        gripper_index=0,
        max_relative_target=5.0,
        gripper_event_threshold=1.0,
    )

    assert float(record.action13_state_jumps[0]) == 13.0
    assert bool(record.action13_state_jump_exceeds[0]) is True
    assert float(record.action13_execution_state_jumps[0]) == 0.0
    assert bool(record.action13_execution_state_jump_exceeds[0]) is False
    assert bool(record.action13_predicted_only_clamp[0]) is False


def test_gt_clamp_is_frame_level_diagnostic_not_predicted_only_failure():
    ground_truth = torch.full((14, 1), 6.0)
    execution_states = torch.zeros_like(ground_truth)
    record = compute_prediction_record(
        ground_truth,
        ground_truth,
        execution_states,
        gripper_index=0,
        max_relative_target=5.0,
        gripper_event_threshold=1.0,
    )
    accumulator = MetricsAccumulator(["arm.pos"], max_relative_target=5.0)
    accumulator.add_dataset_safety(ground_truth, execution_states)
    for _ in range(3):
        accumulator.add_prediction(record)
    accumulator.add_seed_std(torch.zeros_like(ground_truth))

    summary = accumulator.summarize()
    gate = evaluate_gate(summary, GateThresholds(), allow_unavailable=True)

    assert summary["dataset_safety"]["frame_count"] == 1
    assert summary["prediction_count"] == 3
    assert summary["dataset_safety"]["all_joints"]["gt_action0_clamp_rate"] == 1.0
    assert summary["all_joints"]["first_action_clamp_rate"] == 1.0
    assert summary["all_joints"]["action0_predicted_only_clamp_rate"] == 0.0
    assert summary["all_joints"]["action13_predicted_only_clamp_rate"] == 0.0
    assert gate["checks"]["first_action_clamp_rate"]["diagnostic_only"] is True
    assert gate["passed"] is True


def test_accumulator_emits_per_joint_metrics_and_gate_passed():
    ground_truth = torch.zeros(14, 2)
    ground_truth[3:, 1] = 2.0
    prediction = ground_truth.clone()
    record = compute_prediction_record(
        prediction,
        ground_truth,
        torch.zeros_like(ground_truth),
        gripper_index=1,
        max_relative_target=5.0,
        gripper_event_threshold=1.0,
    )
    accumulator = MetricsAccumulator(["arm.pos", "gripper.pos"], max_relative_target=5.0)
    accumulator.add_prediction(record)
    no_event = compute_prediction_record(
        torch.zeros_like(ground_truth),
        torch.zeros_like(ground_truth),
        torch.zeros_like(ground_truth),
        gripper_index=1,
        max_relative_target=5.0,
        gripper_event_threshold=1.0,
    )
    accumulator.add_prediction(no_event)
    accumulator.add_seed_std(torch.zeros_like(ground_truth))

    summary = accumulator.summarize()
    gate = evaluate_gate(summary, GateThresholds())

    assert summary["per_joint"]["arm.pos"]["mae"]["mean"] == 0.0
    assert summary["all_joints"]["action13_abs_error"]["count"] == 4
    assert summary["gripper_event_timing"]["timing_abs_error_steps"]["mean"] == 0.0
    assert gate["passed"] is True


def test_gate_fails_when_mae_exceeds_threshold():
    ground_truth = torch.zeros(14, 2)
    ground_truth[3:, 1] = 2.0
    prediction = ground_truth + 1.0
    record = compute_prediction_record(
        prediction,
        ground_truth,
        torch.zeros_like(ground_truth),
        gripper_index=1,
        max_relative_target=5.0,
        gripper_event_threshold=1.0,
    )
    accumulator = MetricsAccumulator(["arm.pos", "gripper.pos"], max_relative_target=5.0)
    accumulator.add_prediction(record)
    accumulator.add_seed_std(torch.zeros_like(ground_truth))

    gate = evaluate_gate(accumulator.summarize(), GateThresholds(max_mae=0.5))

    assert gate["passed"] is False
    assert gate["checks"]["mae"]["passed"] is False


def test_predicted_only_clamp_is_a_hard_gate():
    ground_truth = torch.zeros(14, 1)
    prediction = torch.full_like(ground_truth, 6.0)
    record = compute_prediction_record(
        prediction,
        ground_truth,
        torch.zeros_like(ground_truth),
        gripper_index=0,
        max_relative_target=5.0,
        gripper_event_threshold=1.0,
    )
    accumulator = MetricsAccumulator(["arm.pos"], max_relative_target=5.0)
    accumulator.add_dataset_safety(ground_truth, torch.zeros_like(ground_truth))
    accumulator.add_prediction(record)
    accumulator.add_seed_std(torch.zeros_like(ground_truth))

    gate = evaluate_gate(
        accumulator.summarize(),
        GateThresholds(
            max_mae=10.0,
            max_mae_p95=10.0,
            max_mae_max=10.0,
            max_action0_error=10.0,
            max_action13_error=10.0,
            max_gripper_false_positive_rate=1.0,
        ),
        allow_unavailable=True,
    )

    assert gate["checks"]["action0_predicted_only_clamp_rate"]["passed"] is False
    assert gate["checks"]["action13_predicted_only_clamp_rate"]["passed"] is False
    assert gate["passed"] is False


def test_action13_coverage_counts_frames_and_predictions_and_fails_formal_below_90_percent():
    accumulator = MetricsAccumulator(["arm.pos"], max_relative_target=5.0)
    for steps in (14, 13):
        ground_truth = torch.zeros(steps, 1)
        accumulator.add_dataset_safety(ground_truth, ground_truth)
        record = compute_prediction_record(
            ground_truth,
            ground_truth,
            ground_truth,
            gripper_index=0,
            max_relative_target=5.0,
            gripper_event_threshold=1.0,
        )
        accumulator.add_prediction(record)
        accumulator.add_prediction(record)
    summary = accumulator.summarize()

    formal = evaluate_model_coverage(
        summary=summary,
        expected_predictions=4,
        finite_predictions=4,
        nonfinite_action_values=0,
        seed_count=2,
        require_semantic_coverage=True,
    )
    quick = evaluate_model_coverage(
        summary=summary,
        expected_predictions=4,
        finite_predictions=4,
        nonfinite_action_values=0,
        seed_count=2,
        require_semantic_coverage=False,
    )

    check = formal["checks"]["action13_coverage"]
    assert check["eligible_frame_count"] == 1
    assert check["ineligible_tail_frame_count"] == 1
    assert check["eligible_prediction_count"] == 2
    assert check["expected_eligible_prediction_count"] == 2
    assert check["frame_coverage_rate"] == 0.5
    assert formal["passed"] is False
    assert quick["passed"] is True


def test_quick_robustness_skips_stage_extrema_but_formal_enforces_them():
    robustness = {
        "worst_seed_mae": {"seed": 0, "mae": 0.0},
        "worst_frame_mae": {"mae": 0.0},
        "worst_stage_joint": {
            "mae_p95": {"value": 99.0},
            "mae_max": {"value": 99.0},
            "seed_std_p95": {"value": 99.0},
            "seed_std_max": {"value": 99.0},
        },
    }

    quick = evaluate_robustness_gate(robustness, GateThresholds(), include_stage_extrema=False)
    formal = evaluate_robustness_gate(robustness, GateThresholds(), include_stage_extrema=True)

    assert quick["passed"] is True
    assert quick["checks"]["worst_stage_joint_seed_std_p95"]["required"] is False
    assert "skipped" in quick["checks"]["worst_stage_joint_seed_std_p95"]
    assert formal["passed"] is False


def test_formal_stage_safety_gate_catches_a_localized_regression():
    checks = {
        name: {"value": 0.0, "evaluated": True, "passed": True}
        for name in (
            "action0_predicted_only_clamp_rate",
            "action13_predicted_only_clamp_rate",
            "chunk_jump_exceedance_rate",
            "gripper_timing_abs_error_steps",
            "gripper_event_miss_rate",
            "gripper_false_positive_rate",
            "gripper_direction_consistency_rate",
        )
    }
    reports = {
        stage: {
            "metrics": {"prediction_count": 10},
            "gate": {"checks": deepcopy(checks)},
        }
        for stage in ("start", "p10", "p25", "p50", "p75", "p90", "end", "gripper_event")
    }
    reports["gripper_event"]["gate"]["checks"]["action0_predicted_only_clamp_rate"]["passed"] = False

    result = evaluate_formal_stage_safety_gate(reports, minimum_predictions_per_stage=10)

    assert result["passed"] is False
    assert result["stages"]["gripper_event"]["passed"] is False
    assert result["stages"]["start"]["passed"] is True


def test_quick_skips_small_sample_predicted_only_rates_but_keeps_residual_gates():
    ground_truth = torch.zeros(14, 1)
    record = compute_prediction_record(
        ground_truth,
        ground_truth,
        ground_truth,
        gripper_index=0,
        max_relative_target=5.0,
        gripper_event_threshold=1.0,
    )
    accumulator = MetricsAccumulator(["arm.pos"], max_relative_target=5.0)
    accumulator.add_dataset_safety(ground_truth, ground_truth)
    accumulator.add_prediction(record)
    accumulator.add_seed_std(torch.zeros_like(ground_truth))
    summary = accumulator.summarize()
    summary["all_joints"]["action0_predicted_only_clamp_rate"] = 1.0
    summary["all_joints"]["action13_predicted_only_clamp_rate"] = 1.0

    quick = evaluate_gate(
        summary,
        GateThresholds(),
        allow_unavailable=True,
        include_predicted_only_rates=False,
    )
    formal = evaluate_gate(summary, GateThresholds(), allow_unavailable=True)

    assert quick["passed"] is True
    assert quick["checks"]["action0_predicted_only_clamp_rate"]["required"] is False
    assert formal["passed"] is False


def test_custom_seed_list_must_be_unique():
    args = Namespace(seeds=[1, 1], quick=False, max_frames=None)

    with pytest.raises(ValueError, match="unique"):
        resolve_run_shape(args)


def test_quick_mode_rejects_custom_seeds():
    args = Namespace(seeds=[1, 2, 3], quick=True, max_frames=400)

    with pytest.raises(ValueError, match="quick fixes"):
        resolve_run_shape(args)


@pytest.mark.parametrize("option", ["nan", "inf", "-inf"])
@pytest.mark.parametrize(
    "argument",
    ["--max-relative-target", "--gripper-event-threshold"],
)
def test_nonfinite_safety_thresholds_are_rejected(argument, option):
    args = build_parser().parse_args(["--model", "fake=/tmp/model", f"{argument}={option}"])

    with pytest.raises(ValueError, match="finite and positive"):
        validate_args(args)


def test_quick_rejects_formal_baseline_and_same_output_path(tmp_path):
    baseline = tmp_path / "baseline.json"
    quick_args = build_parser().parse_args(
        ["--model", "fake=/tmp/model", "--quick", "--baseline-json", str(baseline)]
    )
    same_path_args = build_parser().parse_args(
        [
            "--model",
            "fake=/tmp/model",
            "--baseline-json",
            str(baseline),
            "--output-json",
            str(baseline),
        ]
    )

    with pytest.raises(ValueError, match="only valid for formal"):
        validate_args(quick_args)
    with pytest.raises(ValueError, match="must be different"):
        validate_args(same_path_args)


def test_formal_selection_coverage_requires_40_episodes_all_anchors_and_400_frames():
    phase_names = ["start", "p10", "p25", "p50", "p75", "p90", "end"]
    frames = []
    row_index = 0
    for episode_index in range(40):
        for stage in phase_names:
            frames.append(
                SelectedFrame(
                    row_index=row_index,
                    absolute_index=row_index,
                    episode_index=episode_index,
                    frame_index=row_index,
                    local_index=row_index,
                    stages={stage},
                )
            )
            row_index += 1
        for _ in range(3):
            frames.append(
                SelectedFrame(
                    row_index=row_index,
                    absolute_index=row_index,
                    episode_index=episode_index,
                    frame_index=row_index,
                    local_index=row_index,
                    stages={"gripper_event"},
                )
            )
            row_index += 1

    episode_lengths = {
        episode_index: max(frame.local_index for frame in frames if frame.episode_index == episode_index) + 14
        for episode_index in range(40)
    }
    coverage = evaluate_formal_selection_coverage(frames, list(DEFAULT_SEEDS), episode_lengths)
    missing_anchor = evaluate_formal_selection_coverage(
        [frame for frame in frames if not (frame.episode_index == 0 and "p90" in frame.stages)],
        list(DEFAULT_SEEDS),
        episode_lengths,
    )
    short_episode = evaluate_formal_selection_coverage(
        frames,
        list(DEFAULT_SEEDS),
        {**episode_lengths, 0: 5},
    )

    assert len(frames) == 400
    assert coverage["passed"] is True
    assert missing_anchor["passed"] is False
    assert missing_anchor["checks"]["seven_phase_anchors_per_episode"]["passed"] is False
    assert short_episode["passed"] is False
    assert short_episode["checks"]["six_action13_eligible_anchors_per_episode"]["passed"] is False


def test_relative_baseline_gate_limits_rate_regression_to_five_percentage_points():
    ground_truth = torch.zeros(14, 2)
    ground_truth[3:, 1] = 2.0
    event_record = compute_prediction_record(
        ground_truth,
        ground_truth,
        torch.zeros_like(ground_truth),
        gripper_index=1,
        max_relative_target=5.0,
        gripper_event_threshold=1.0,
    )
    no_event_record = compute_prediction_record(
        torch.zeros_like(ground_truth),
        torch.zeros_like(ground_truth),
        torch.zeros_like(ground_truth),
        gripper_index=1,
        max_relative_target=5.0,
        gripper_event_threshold=1.0,
    )
    accumulator = MetricsAccumulator(["arm.pos", "gripper.pos"], max_relative_target=5.0)
    accumulator.add_prediction(event_record)
    accumulator.add_prediction(no_event_record)
    accumulator.add_seed_std(torch.zeros_like(ground_truth))
    baseline_summary = accumulator.summarize()
    baseline_report = {"stages": {"overall": {"metrics": baseline_summary}}}

    passing = evaluate_relative_baseline_gate(
        baseline_summary,
        baseline_report,
        GateThresholds(),
        rate_margin=0.05,
        error_ratio=2.0,
    )
    regressed_summary = deepcopy(baseline_summary)
    regressed_summary["all_joints"]["action0_predicted_only_clamp_rate"] = 0.06
    failing = evaluate_relative_baseline_gate(
        regressed_summary,
        baseline_report,
        GateThresholds(max_action0_predicted_only_clamp_rate=1.0),
        rate_margin=0.05,
        error_ratio=2.0,
    )

    assert passing["passed"] is True
    assert failing["passed"] is False
    assert failing["checks"]["action0_predicted_only_clamp_rate"]["effective_threshold"] == 0.05


def test_gate_exit_code_is_nonzero_on_smoke_or_formal_failure():
    assert gate_exit_code(run_mode="quick_smoke", smoke_passed=True, formal_passed=None) == 0
    assert gate_exit_code(run_mode="quick_smoke", smoke_passed=False, formal_passed=None) == 1
    assert gate_exit_code(run_mode="formal", smoke_passed=True, formal_passed=True) == 0
    assert gate_exit_code(run_mode="formal", smoke_passed=True, formal_passed=False) == 1


def test_model_coverage_fails_on_nonfinite_actions():
    ground_truth = torch.zeros(14, 2)
    ground_truth[3:, 1] = 2.0
    event_record = compute_prediction_record(
        ground_truth,
        ground_truth,
        torch.zeros_like(ground_truth),
        gripper_index=1,
        max_relative_target=5.0,
        gripper_event_threshold=1.0,
    )
    no_event_record = compute_prediction_record(
        torch.zeros_like(ground_truth),
        torch.zeros_like(ground_truth),
        torch.zeros_like(ground_truth),
        gripper_index=1,
        max_relative_target=5.0,
        gripper_event_threshold=1.0,
    )
    accumulator = MetricsAccumulator(["arm.pos", "gripper.pos"], max_relative_target=5.0)
    accumulator.add_prediction(event_record)
    accumulator.add_prediction(no_event_record)
    accumulator.add_seed_std(torch.zeros_like(ground_truth))

    coverage = evaluate_model_coverage(
        summary=accumulator.summarize(),
        expected_predictions=3,
        finite_predictions=2,
        nonfinite_action_values=1,
    )

    assert coverage["passed"] is False
    assert coverage["checks"]["all_predictions_finite"]["passed"] is False
    assert coverage["checks"]["prediction_count"]["passed"] is False


def test_baseline_report_requires_matching_formal_manifest(tmp_path):
    frame = SelectedFrame(
        row_index=0,
        absolute_index=10,
        episode_index=2,
        frame_index=3,
        local_index=3,
        stages={"p10"},
    )
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    (dataset_root / "data.bin").write_bytes(b"dataset")
    dataset_fingerprint = fingerprint_dataset_assets(dataset_root)
    checkpoint_root = tmp_path / "checkpoint"
    checkpoint_root.mkdir()
    (checkpoint_root / "config.json").write_text("{}", encoding="utf-8")
    (checkpoint_root / "model.safetensors").write_bytes(b"weights")
    report_path = tmp_path / "baseline.json"
    selection_contract = {
        "quick": False,
        "frame_limit": 400,
        "selected_frame_count": 1,
        "phase_fractions": {
            "start": 0.0,
            "p10": 0.1,
            "p25": 0.25,
            "p50": 0.5,
            "p75": 0.75,
            "p90": 0.9,
            "end": 1.0,
        },
        "gripper_joint": "gripper.pos",
        "gripper_event_threshold": 1.0,
        "gripper_neighbor_radius": 2,
        "max_relative_target": 5.0,
    }
    model_report = {
        "smoke_passed": True,
        "formal_passed": True,
        "passed": True,
        "hardware_action_sent": False,
        "checkpoint_asset_fingerprint": fingerprint_checkpoint_assets(checkpoint_root),
        "stages": {"overall": {"metrics": {}}},
    }
    baseline = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "report_status": "complete",
        "run_mode": "formal",
        "metric_semantics": METRIC_SEMANTICS,
        "gate_profile": {"name": DEPLOYMENT_GATE_PROFILE, "formal_eligible": True},
        "thresholds": vars(GateThresholds()),
        "formal_passed": True,
        "smoke_passed": True,
        "passed": True,
        "hardware_action_sent": False,
        "formal_selection_coverage": {"passed": True},
        "dataset": {
            "root": str(dataset_root),
            "repo_id": "fake/repo",
            "asset_fingerprint": dataset_fingerprint,
            "action_names": ["arm.pos", "gripper.pos"],
            "state_names": ["arm.pos", "gripper.pos"],
        },
        "noise": {"seeds": list(DEFAULT_SEEDS)},
        "selection": {**selection_contract, "frames": [frame.to_json()]},
        "expected_models": ["full"],
        "completed_models": ["full"],
        "models": {"full": model_report},
    }
    report_path.write_text(json.dumps(baseline), encoding="utf-8")

    loaded = load_baseline_model_report(
        report_path,
        label="full",
        dataset_root=dataset_root,
        repo_id="fake/repo",
        dataset_asset_fingerprint=dataset_fingerprint,
        action_names=["arm.pos", "gripper.pos"],
        state_names=["arm.pos", "gripper.pos"],
        seeds=list(DEFAULT_SEEDS),
        frames=[frame],
        selection_contract=selection_contract,
        thresholds=GateThresholds(),
    )

    assert loaded is model_report or loaded == model_report

    baseline["selection"]["max_relative_target"] = 10.0
    report_path.write_text(json.dumps(baseline), encoding="utf-8")
    with pytest.raises(ValueError, match="max_relative_target"):
        load_baseline_model_report(
            report_path,
            label="full",
            dataset_root=dataset_root,
            repo_id="fake/repo",
            dataset_asset_fingerprint=dataset_fingerprint,
            action_names=["arm.pos", "gripper.pos"],
            state_names=["arm.pos", "gripper.pos"],
            seeds=list(DEFAULT_SEEDS),
            frames=[frame],
            selection_contract=selection_contract,
            thresholds=GateThresholds(),
        )


def test_evaluate_model_cpu_smoke_uses_explicit_noise_and_ignores_tail_padding(monkeypatch):
    import examples.inference.evaluate_pi05_multiseed as evaluation

    actions = torch.zeros(20, 2)
    actions[3:, 1] = 2.0
    states = actions.clone()
    dataset_index = DatasetIndex(
        actions=actions,
        states=states,
        episode_indices=torch.zeros(20, dtype=torch.int64),
        frame_indices=torch.arange(20),
        absolute_indices=torch.arange(20),
        rows_by_episode={0: list(range(20))},
        action_names=["arm.pos", "gripper.pos"],
        state_names=["arm.pos", "gripper.pos"],
    )

    class FakeDataset:
        def __getitem__(self, index):
            return {
                "observation.state": dataset_index.states[index],
                "action": dataset_index.actions[index],
            }

    class IdentityPipeline:
        def reset(self):
            return None

        def __call__(self, value):
            return value

    class FakePolicy:
        def __init__(self):
            self.noises = []

        def reset(self):
            return None

        def predict_action_chunk(self, batch, *, noise):
            assert batch["observation.state"].shape == (1, 2)
            self.noises.append(noise.clone())
            prediction = torch.zeros(1, 50, 2)
            if float(batch["observation.state"][0, 1]) == 0.0:
                prediction[:, 3:, 1] = 2.0
            else:
                prediction[:, :, 1] = 2.0
            return prediction

    fake_policy = FakePolicy()
    monkeypatch.setattr(
        evaluation,
        "load_policy_and_processors",
        lambda spec, config, device: (fake_policy, IdentityPipeline(), IdentityPipeline()),
    )
    noise_bank = make_shared_noise([0, 42, 1000], chunk_size=50, max_action_dim=32)

    report = evaluate_model(
        spec=ModelSpec("fake", Path("/tmp/fake")),
        config=object(),
        descriptor={"label": "fake", "checkpoint": "/tmp/fake"},
        dataset=FakeDataset(),
        dataset_index=dataset_index,
        frames=[
            SelectedFrame(
                row_index=0,
                absolute_index=0,
                episode_index=0,
                frame_index=0,
                local_index=0,
                stages={"start"},
            ),
            SelectedFrame(
                row_index=10,
                absolute_index=10,
                episode_index=0,
                frame_index=10,
                local_index=10,
                stages={"p50"},
            ),
        ],
        noise_bank=noise_bank,
        seeds=[0, 42, 1000],
        device="cpu",
        chunk_size=50,
        gripper_index=1,
        max_relative_target=5.0,
        gripper_event_threshold=1.0,
        thresholds=GateThresholds(),
        run_mode="quick_smoke",
    )

    assert len(fake_policy.noises) == 6
    assert torch.equal(fake_policy.noises[0], noise_bank[0])
    assert report["frames"][0]["valid_gt_steps"] == 20
    assert report["frames"][0]["future_state_valid_steps"] == 20
    assert report["frames"][0]["tail_padding_ignored_steps"] == 30
    assert report["frames"][0]["dataset_safety"]["action13"] is not None
    assert report["frames"][1]["dataset_safety"]["action13"] is None
    assert report["stages"]["overall"]["metrics"]["all_joints"]["mae"]["mean"] == 0.0
    assert report["smoke_passed"] is True
    assert report["formal_passed"] is None
    assert report["passed"] is None
    assert report["hardware_action_sent"] is False
