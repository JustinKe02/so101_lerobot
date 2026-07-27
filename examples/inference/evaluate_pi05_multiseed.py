#!/usr/bin/env python

"""Offline multi-seed stability gate for local PI0.5 checkpoints.

The script reads only local checkpoints and a local LeRobot dataset. It never
constructs a robot, connects to hardware, or sends an action.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

# Keep all Hugging Face-backed loading local. These variables must be set
# before importing LeRobot policy or dataset modules.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


DEFAULT_ROOT = Path("/data/cqy_workspace/tk/lerobot_src")
DEFAULT_DATASET_ROOT = DEFAULT_ROOT / "data/so101_test_data"
DEFAULT_REPO_ID = "admin123/so101_test_data"
DEFAULT_SEEDS = (0, 1, 2, 3, 4, 5, 6, 7, 42, 1000)
QUICK_SEEDS = (0, 42, 1000)
REPORT_SCHEMA_VERSION = 2
DEPLOYMENT_GATE_PROFILE = "deployment-v2"
METRIC_SEMANTICS = {
    "comparison_space": "postprocessed_robot_units",
    "gt_alignment": "same_episode_future_action",
    "execution_state_alignment": "same_episode_future_observation_state",
    "action_offsets": [0, 13],
    "legacy_action13_query_state_fields": [
        "action13_state_jump",
        "action13_state_jump_exceedance_rate",
    ],
    "legacy_fields_are_diagnostic_only": True,
}
FORMAL_SAFETY_STAGES = (
    "start",
    "p10",
    "p25",
    "p50",
    "p75",
    "p90",
    "end",
    "gripper_event",
)
FORMAL_STAGE_SAFETY_CHECKS = (
    "action0_predicted_only_clamp_rate",
    "action13_predicted_only_clamp_rate",
    "chunk_jump_exceedance_rate",
    "gripper_timing_abs_error_steps",
    "gripper_event_miss_rate",
    "gripper_false_positive_rate",
    "gripper_direction_consistency_rate",
)
PHASE_FRACTIONS = (
    ("start", 0.0),
    ("p10", 0.10),
    ("p25", 0.25),
    ("p50", 0.50),
    ("p75", 0.75),
    ("p90", 0.90),
    ("end", 1.0),
)


@dataclass(frozen=True)
class ModelSpec:
    label: str
    checkpoint: Path


@dataclass
class SelectedFrame:
    row_index: int
    absolute_index: int
    episode_index: int
    frame_index: int
    local_index: int
    stages: set[str] = field(default_factory=set)
    gripper_contexts: set[tuple[int, int]] = field(default_factory=set)

    def to_json(self) -> dict[str, Any]:
        return {
            "row_index": self.row_index,
            "absolute_index": self.absolute_index,
            "episode_index": self.episode_index,
            "frame_index": self.frame_index,
            "local_index": self.local_index,
            "stages": sorted(self.stages),
            "gripper_contexts": [
                {"event_local_index": event_index, "offset": offset}
                for event_index, offset in sorted(self.gripper_contexts)
            ],
        }


@dataclass(frozen=True)
class GateThresholds:
    max_mae: float = 2.0
    max_mae_p95: float = 5.0
    max_mae_max: float = 20.0
    max_seed_std: float = 1.0
    max_seed_std_p95: float = 2.0
    max_seed_std_max: float = 10.0
    max_worst_frame_mae: float = 5.0
    max_worst_seed_mae: float = 3.0
    max_action0_error: float = 2.0
    max_action0_error_p95: float = 3.0
    max_action0_error_max: float = 5.0
    max_action13_error: float = 3.0
    max_action13_error_p95: float = 3.0
    max_action13_error_max: float = 5.0
    max_action0_predicted_only_clamp_rate: float = 0.05
    max_action13_predicted_only_clamp_rate: float = 0.05
    max_chunk_jump_rate: float = 0.05
    min_action13_coverage_rate: float = 0.90
    max_gripper_timing_error_steps: float = 5.0
    max_gripper_event_miss_rate: float = 0.10
    max_gripper_false_positive_rate: float = 0.10
    min_gripper_direction_consistency: float = 0.90


@dataclass
class DatasetIndex:
    actions: torch.Tensor
    states: torch.Tensor
    episode_indices: torch.Tensor
    frame_indices: torch.Tensor
    absolute_indices: torch.Tensor
    rows_by_episode: dict[int, list[int]]
    action_names: list[str]
    state_names: list[str]


@dataclass
class PredictionRecord:
    errors: torch.Tensor
    action0_errors: torch.Tensor
    action13_errors: torch.Tensor | None
    action0_state_jumps: torch.Tensor
    action13_state_jumps: torch.Tensor | None
    action13_execution_state_jumps: torch.Tensor | None
    first_action_clamp: torch.Tensor
    action0_predicted_only_clamp: torch.Tensor
    action13_state_jump_exceeds: torch.Tensor | None
    action13_execution_state_jump_exceeds: torch.Tensor | None
    action13_predicted_only_clamp: torch.Tensor | None
    action0_clamp_residuals: torch.Tensor
    action0_excess_clamp_residuals: torch.Tensor
    action13_execution_clamp_residuals: torch.Tensor | None
    action13_excess_clamp_residuals: torch.Tensor | None
    jumps: torch.Tensor
    gt_gripper_event: int | None
    predicted_gripper_event: int | None
    gt_gripper_direction: int | None
    predicted_gripper_direction: int | None


def parse_model_spec(value: str) -> ModelSpec:
    """Parse ``LABEL=PATH`` or a bare checkpoint path."""
    if "=" in value:
        label, raw_path = value.split("=", 1)
        if not label.strip():
            raise argparse.ArgumentTypeError("model label must not be empty")
    else:
        raw_path = value
        label = Path(raw_path).name
    if not raw_path.strip():
        raise argparse.ArgumentTypeError("model checkpoint path must not be empty")
    return ModelSpec(label=label.strip(), checkpoint=Path(raw_path).expanduser())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        dest="models",
        action="append",
        type=parse_model_spec,
        required=True,
        metavar="LABEL=PATH",
        help="Repeat for each full checkpoint or PEFT adapter to compare.",
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--episodes", type=int, nargs="+")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--video-backend", default="pyav")
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--quick", action="store_true", help="Evaluate five frames with seeds 0, 42, 1000.")
    parser.add_argument(
        "--max-frames",
        type=int,
        default=400,
        help="Cap the full candidate pool; use 0 for all selected frames.",
    )
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument("--max-relative-target", type=float, default=5.0)
    parser.add_argument("--gripper-joint", default="gripper.pos")
    parser.add_argument("--gripper-event-threshold", type=float, default=1.0)
    parser.add_argument("--gripper-neighbor-radius", type=int, default=2)
    parser.add_argument("--max-mae", type=float, default=GateThresholds.max_mae)
    parser.add_argument("--max-mae-p95", type=float, default=GateThresholds.max_mae_p95)
    parser.add_argument("--max-mae-max", type=float, default=GateThresholds.max_mae_max)
    parser.add_argument("--max-seed-std", type=float, default=GateThresholds.max_seed_std)
    parser.add_argument("--max-seed-std-p95", type=float, default=GateThresholds.max_seed_std_p95)
    parser.add_argument("--max-seed-std-max", type=float, default=GateThresholds.max_seed_std_max)
    parser.add_argument(
        "--max-worst-frame-mae",
        type=float,
        default=GateThresholds.max_worst_frame_mae,
    )
    parser.add_argument(
        "--max-worst-seed-mae",
        type=float,
        default=GateThresholds.max_worst_seed_mae,
    )
    parser.add_argument("--max-action0-error", type=float, default=GateThresholds.max_action0_error)
    parser.add_argument("--max-action0-error-p95", type=float, default=GateThresholds.max_action0_error_p95)
    parser.add_argument("--max-action0-error-max", type=float, default=GateThresholds.max_action0_error_max)
    parser.add_argument("--max-action13-error", type=float, default=GateThresholds.max_action13_error)
    parser.add_argument("--max-action13-error-p95", type=float, default=GateThresholds.max_action13_error_p95)
    parser.add_argument("--max-action13-error-max", type=float, default=GateThresholds.max_action13_error_max)
    parser.add_argument(
        "--max-action0-predicted-only-clamp-rate",
        "--max-first-action-clamp-rate",
        dest="max_action0_predicted_only_clamp_rate",
        type=float,
        default=GateThresholds.max_action0_predicted_only_clamp_rate,
        help="Maximum action0 predicted-only clamp rate; raw clamp rate remains diagnostic.",
    )
    parser.add_argument("--max-chunk-jump-rate", type=float, default=GateThresholds.max_chunk_jump_rate)
    parser.add_argument(
        "--max-action13-predicted-only-clamp-rate",
        "--max-action13-state-jump-rate",
        dest="max_action13_predicted_only_clamp_rate",
        type=float,
        default=GateThresholds.max_action13_predicted_only_clamp_rate,
        help="Maximum action13 predicted-only execution clamp rate.",
    )
    parser.add_argument(
        "--min-action13-coverage-rate",
        type=float,
        default=GateThresholds.min_action13_coverage_rate,
    )
    parser.add_argument(
        "--max-gripper-timing-error-steps",
        type=float,
        default=GateThresholds.max_gripper_timing_error_steps,
    )
    parser.add_argument(
        "--max-gripper-event-miss-rate",
        type=float,
        default=GateThresholds.max_gripper_event_miss_rate,
    )
    parser.add_argument(
        "--max-gripper-false-positive-rate",
        type=float,
        default=GateThresholds.max_gripper_false_positive_rate,
    )
    parser.add_argument(
        "--min-gripper-direction-consistency",
        type=float,
        default=GateThresholds.min_gripper_direction_consistency,
    )
    parser.add_argument("--baseline-json", type=Path)
    parser.add_argument("--baseline-label", default="full")
    parser.add_argument("--max-baseline-rate-regression", type=float, default=0.05)
    parser.add_argument("--max-baseline-error-ratio", type=float, default=2.0)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=DEFAULT_ROOT / "outputs/eval/pi05_multiseed.json",
    )
    return parser


def resolve_run_shape(args: argparse.Namespace) -> tuple[list[int], int | None]:
    if args.quick and args.seeds is not None:
        raise ValueError("--quick fixes the seed set to 0, 42, 1000; omit --seeds")
    seeds = list(QUICK_SEEDS) if args.quick else list(args.seeds or DEFAULT_SEEDS)
    if len(seeds) == 0 or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be non-empty and unique")
    frame_limit = 5 if args.quick else (None if args.max_frames in (None, 0) else args.max_frames)
    if frame_limit is not None and frame_limit < 1:
        raise ValueError("max-frames must be non-negative")
    return seeds, frame_limit


def validate_args(args: argparse.Namespace) -> None:
    labels = [spec.label for spec in args.models]
    if len(set(labels)) != len(labels):
        raise ValueError("model labels must be unique")
    if args.chunk_size != 50:
        raise ValueError("PI0.5 stability gate requires --chunk-size=50")
    if not math.isfinite(args.max_relative_target) or args.max_relative_target <= 0.0:
        raise ValueError("max-relative-target must be finite and positive")
    if not math.isfinite(args.gripper_event_threshold) or args.gripper_event_threshold <= 0.0:
        raise ValueError("gripper-event-threshold must be finite and positive")
    if args.gripper_neighbor_radius < 0:
        raise ValueError("gripper-neighbor-radius must be non-negative")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    for value in (
        args.max_mae,
        args.max_mae_p95,
        args.max_mae_max,
        args.max_seed_std,
        args.max_seed_std_p95,
        args.max_seed_std_max,
        args.max_worst_frame_mae,
        args.max_worst_seed_mae,
        args.max_action0_error,
        args.max_action0_error_p95,
        args.max_action0_error_max,
        args.max_action13_error,
        args.max_action13_error_p95,
        args.max_action13_error_max,
        args.max_gripper_timing_error_steps,
    ):
        if value < 0.0 or not math.isfinite(value):
            raise ValueError("error thresholds must be finite and non-negative")
    for value in (
        args.max_action0_predicted_only_clamp_rate,
        args.max_action13_predicted_only_clamp_rate,
        args.max_chunk_jump_rate,
        args.min_action13_coverage_rate,
        args.max_gripper_event_miss_rate,
        args.max_gripper_false_positive_rate,
        args.min_gripper_direction_consistency,
        args.max_baseline_rate_regression,
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError("rate thresholds must be in [0, 1]")
    if args.max_baseline_error_ratio < 1.0 or not math.isfinite(args.max_baseline_error_ratio):
        raise ValueError("max-baseline-error-ratio must be finite and at least 1")
    if args.quick and args.baseline_json is not None:
        raise ValueError("--baseline-json is only valid for formal evaluation")
    if (
        args.baseline_json is not None
        and args.baseline_json.expanduser().resolve() == args.output_json.expanduser().resolve()
    ):
        raise ValueError("baseline-json and output-json must be different files")


def detect_gripper_events(actions: torch.Tensor, gripper_index: int, threshold: float) -> list[int]:
    """Return one peak transition index for each contiguous gripper movement."""
    if actions.ndim != 2 or not 0 <= gripper_index < actions.shape[1]:
        raise ValueError("actions must be [T, A] and contain gripper_index")
    if len(actions) < 2:
        return []
    differences = actions[1:, gripper_index].sub(actions[:-1, gripper_index]).abs()
    active = torch.nonzero(differences >= threshold, as_tuple=False).flatten().tolist()
    if not active:
        return []

    clusters: list[list[int]] = [[active[0]]]
    for index in active[1:]:
        if index == clusters[-1][-1] + 1:
            clusters[-1].append(index)
        else:
            clusters.append([index])

    events = []
    for cluster in clusters:
        peak = max(cluster, key=lambda index: (float(differences[index]), -index))
        events.append(peak + 1)
    return events


def select_episode_frames(
    *,
    episode_index: int,
    row_indices: list[int],
    frame_indices: torch.Tensor,
    absolute_indices: torch.Tensor,
    episode_actions: torch.Tensor,
    gripper_index: int,
    gripper_event_threshold: float,
    gripper_neighbor_radius: int,
) -> list[SelectedFrame]:
    """Select phase anchors and gripper-event neighbors for one episode."""
    if len(row_indices) == 0 or len(row_indices) != len(episode_actions):
        raise ValueError("episode row/action lengths must be equal and non-zero")
    selected: dict[int, SelectedFrame] = {}

    def add(local_index: int, stage: str, context: tuple[int, int] | None = None) -> None:
        local_index = max(0, min(len(row_indices) - 1, local_index))
        row_index = row_indices[local_index]
        item = selected.get(row_index)
        if item is None:
            item = SelectedFrame(
                row_index=row_index,
                absolute_index=int(absolute_indices[row_index]),
                episode_index=episode_index,
                frame_index=int(frame_indices[row_index]),
                local_index=local_index,
            )
            selected[row_index] = item
        item.stages.add(stage)
        if context is not None:
            item.gripper_contexts.add(context)

    for stage, fraction in PHASE_FRACTIONS:
        add(round((len(row_indices) - 1) * fraction), stage)

    for event_index in detect_gripper_events(
        episode_actions,
        gripper_index=gripper_index,
        threshold=gripper_event_threshold,
    ):
        for offset in range(-gripper_neighbor_radius, gripper_neighbor_radius + 1):
            local_index = event_index + offset
            if not 0 <= local_index < len(row_indices):
                continue
            add(local_index, "gripper_event", (event_index, offset))
            if offset < 0:
                add(local_index, "gripper_pre")
            elif offset == 0:
                add(local_index, "gripper_at")
            else:
                add(local_index, "gripper_post")

    return sorted(selected.values(), key=lambda item: item.row_index)


def limit_frames_evenly(frames: list[SelectedFrame], limit: int | None) -> list[SelectedFrame]:
    """Apply a deterministic evenly-spaced cap without duplicating endpoints."""
    if limit is None or len(frames) <= limit:
        return frames
    if limit == 1:
        return [frames[0]]
    positions = [round(index * (len(frames) - 1) / (limit - 1)) for index in range(limit)]
    return [frames[position] for position in positions]


def limit_frames_preserving_phase_anchors(
    frames: list[SelectedFrame], limit: int | None
) -> list[SelectedFrame]:
    """Preserve all episode phase anchors before sampling extra event frames."""
    if limit is None or len(frames) <= limit:
        return frames
    phase_names = {stage for stage, _ in PHASE_FRACTIONS}
    anchors = [frame for frame in frames if frame.stages & phase_names]
    if len(anchors) > limit:
        # Quick smoke mode intentionally cannot cover every episode/stage.
        return limit_frames_evenly(frames, limit)
    anchor_rows = {frame.row_index for frame in anchors}
    event_only = [frame for frame in frames if frame.row_index not in anchor_rows]
    extras = limit_frames_evenly(event_only, limit - len(anchors))
    return sorted([*anchors, *extras], key=lambda frame: frame.row_index)


def extract_ground_truth_chunk(
    episode_actions: torch.Tensor, local_index: int, chunk_size: int = 50
) -> torch.Tensor:
    """Return only real in-episode actions; never cross or pad the episode tail."""
    if episode_actions.ndim != 2:
        raise ValueError("episode_actions must have shape [T, A]")
    if not 0 <= local_index < len(episode_actions):
        raise IndexError("local_index is outside the episode")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    return episode_actions[local_index : min(len(episode_actions), local_index + chunk_size)].clone()


def resolve_state_indices(action_names: list[str], state_names: list[str]) -> list[int]:
    """Resolve observation-state columns into action-joint order, failing closed."""
    if len(set(action_names)) != len(action_names) or len(set(state_names)) != len(state_names):
        raise ValueError("action/state feature names must be unique")
    state_index = {name: index for index, name in enumerate(state_names)}
    missing = [name for name in action_names if name not in state_index]
    if missing:
        raise ValueError(f"observation.state is missing action joints: {missing}")
    return [state_index[name] for name in action_names]


def extract_execution_state_chunk(
    episode_states: torch.Tensor,
    local_index: int,
    *,
    action_names: list[str],
    state_names: list[str],
    chunk_size: int = 50,
) -> torch.Tensor:
    """Return same-episode future states reordered into action-joint order."""
    state_chunk = extract_ground_truth_chunk(episode_states, local_index, chunk_size)
    state_indices = resolve_state_indices(action_names, state_names)
    return state_chunk[:, state_indices]


def make_shared_noise(seeds: list[int], chunk_size: int, max_action_dim: int) -> dict[int, torch.Tensor]:
    """Pre-generate one CPU Gaussian noise tensor per seed for all models/frames."""
    if chunk_size < 1 or max_action_dim < 1:
        raise ValueError("noise dimensions must be positive")
    result = {}
    for seed in seeds:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        result[seed] = torch.randn(
            1,
            chunk_size,
            max_action_dim,
            generator=generator,
            dtype=torch.float32,
        )
    return result


def first_gripper_event_details(
    sequence: torch.Tensor, initial_value: float, threshold: float
) -> tuple[int | None, int | None]:
    """Return the first gripper event step and its signed direction."""
    if sequence.ndim != 1:
        raise ValueError("gripper sequence must be one-dimensional")
    previous = torch.tensor(initial_value, dtype=sequence.dtype, device=sequence.device)
    for index, value in enumerate(sequence):
        delta = float(value - previous)
        if abs(delta) >= threshold:
            return index, 1 if delta > 0.0 else -1
        previous = value
    return None, None


def first_gripper_event(sequence: torch.Tensor, initial_value: float, threshold: float) -> int | None:
    """Find the first command step whose gripper delta reaches the event threshold."""
    return first_gripper_event_details(sequence, initial_value, threshold)[0]


def compute_prediction_record(
    prediction: torch.Tensor,
    ground_truth: torch.Tensor,
    execution_states: torch.Tensor,
    *,
    gripper_index: int,
    max_relative_target: float,
    gripper_event_threshold: float,
) -> PredictionRecord:
    """Compute tensor-valued metrics for one frame and one noise seed."""
    prediction = prediction.detach().float().cpu()
    ground_truth = ground_truth.detach().float().cpu()
    execution_states = execution_states.detach().float().cpu()
    if prediction.ndim != 2 or ground_truth.ndim != 2 or execution_states.ndim != 2:
        raise ValueError("prediction, ground_truth, and execution_states must have shape [T, A]")
    if not (
        prediction.shape[1] == ground_truth.shape[1] == execution_states.shape[1]
        and len(execution_states) == len(ground_truth)
    ):
        raise ValueError("prediction, ground truth, and execution-state horizons/dimensions must align")
    if len(ground_truth) == 0 or len(prediction) < len(ground_truth):
        raise ValueError("prediction must cover the non-empty ground-truth horizon")
    if not torch.isfinite(prediction).all():
        raise RuntimeError("model produced non-finite actions")
    if not torch.isfinite(ground_truth).all() or not torch.isfinite(execution_states).all():
        raise RuntimeError("dataset produced non-finite actions or states")

    valid_prediction = prediction[: len(ground_truth)]
    errors = valid_prediction.sub(ground_truth).abs()
    action13_errors = errors[13] if len(errors) > 13 else None
    current_state = execution_states[0]
    action0_state_jumps = prediction[0].sub(current_state).abs()
    gt_action0_state_jumps = ground_truth[0].sub(current_state).abs()
    action13_state_jumps = prediction[13].sub(current_state).abs() if len(ground_truth) > 13 else None
    first_action_clamp = action0_state_jumps > max_relative_target
    gt_first_action_clamp = gt_action0_state_jumps > max_relative_target
    action0_predicted_only_clamp = first_action_clamp & ~gt_first_action_clamp
    action13_state_jump_exceeds = (
        action13_state_jumps > max_relative_target if action13_state_jumps is not None else None
    )
    action13_execution_state_jumps = (
        prediction[13].sub(execution_states[13]).abs() if len(ground_truth) > 13 else None
    )
    gt_action13_execution_state_jumps = (
        ground_truth[13].sub(execution_states[13]).abs() if len(ground_truth) > 13 else None
    )
    action13_execution_state_jump_exceeds = (
        action13_execution_state_jumps > max_relative_target
        if action13_execution_state_jumps is not None
        else None
    )
    gt_action13_execution_clamp = (
        gt_action13_execution_state_jumps > max_relative_target
        if gt_action13_execution_state_jumps is not None
        else None
    )
    action13_predicted_only_clamp = (
        action13_execution_state_jump_exceeds & ~gt_action13_execution_clamp
        if action13_execution_state_jump_exceeds is not None and gt_action13_execution_clamp is not None
        else None
    )
    action0_clamp_residuals = action0_state_jumps.sub(max_relative_target).clamp_min(0.0)
    action0_excess_clamp_residuals = action0_state_jumps.sub(
        torch.maximum(gt_action0_state_jumps, torch.full_like(gt_action0_state_jumps, max_relative_target))
    ).clamp_min(0.0)
    action13_execution_clamp_residuals = (
        action13_execution_state_jumps.sub(max_relative_target).clamp_min(0.0)
        if action13_execution_state_jumps is not None
        else None
    )
    action13_excess_clamp_residuals = (
        action13_execution_state_jumps.sub(
            torch.maximum(
                gt_action13_execution_state_jumps,
                torch.full_like(gt_action13_execution_state_jumps, max_relative_target),
            )
        ).clamp_min(0.0)
        if action13_execution_state_jumps is not None and gt_action13_execution_state_jumps is not None
        else None
    )
    jumps = (
        valid_prediction[1:].sub(valid_prediction[:-1]).abs()
        if len(valid_prediction) > 1
        else torch.empty((0, prediction.shape[1]), dtype=torch.float32)
    )
    gt_event, gt_direction = first_gripper_event_details(
        ground_truth[:, gripper_index],
        float(current_state[gripper_index]),
        gripper_event_threshold,
    )
    predicted_event, predicted_direction = first_gripper_event_details(
        valid_prediction[:, gripper_index],
        float(current_state[gripper_index]),
        gripper_event_threshold,
    )
    return PredictionRecord(
        errors=errors,
        action0_errors=errors[0],
        action13_errors=action13_errors,
        action0_state_jumps=action0_state_jumps,
        action13_state_jumps=action13_state_jumps,
        action13_execution_state_jumps=action13_execution_state_jumps,
        first_action_clamp=first_action_clamp,
        action0_predicted_only_clamp=action0_predicted_only_clamp,
        action13_state_jump_exceeds=action13_state_jump_exceeds,
        action13_execution_state_jump_exceeds=action13_execution_state_jump_exceeds,
        action13_predicted_only_clamp=action13_predicted_only_clamp,
        action0_clamp_residuals=action0_clamp_residuals,
        action0_excess_clamp_residuals=action0_excess_clamp_residuals,
        action13_execution_clamp_residuals=action13_execution_clamp_residuals,
        action13_excess_clamp_residuals=action13_excess_clamp_residuals,
        jumps=jumps,
        gt_gripper_event=gt_event,
        predicted_gripper_event=predicted_event,
        gt_gripper_direction=gt_direction,
        predicted_gripper_direction=predicted_direction,
    )


def dataset_safety_frame_to_json(
    ground_truth: torch.Tensor,
    execution_states: torch.Tensor,
    joint_names: list[str],
    max_relative_target: float,
) -> dict[str, Any]:
    """Serialize seed-invariant, same-episode safety context for one selected frame."""
    if ground_truth.ndim != 2 or execution_states.shape != ground_truth.shape:
        raise ValueError("ground truth and execution states must align as [T, A]")
    if len(ground_truth) == 0 or ground_truth.shape[1] != len(joint_names):
        raise ValueError("dataset safety requires a non-empty action-aligned frame")

    def offset_metrics(offset: int) -> dict[str, Any] | None:
        if len(ground_truth) <= offset:
            return None
        gap = ground_truth[offset].sub(execution_states[offset]).abs()
        clamp = gap > max_relative_target
        residual = gap.sub(max_relative_target).clamp_min(0.0)
        return {
            "offset": offset,
            "gap_mean": float(gap.mean()),
            "gap_max": float(gap.max()),
            "gap_by_joint": {joint_name: float(gap[index]) for index, joint_name in enumerate(joint_names)},
            "clamp": bool(clamp.any()),
            "clamp_by_joint": {
                joint_name: bool(clamp[index]) for index, joint_name in enumerate(joint_names)
            },
            "clamp_residual_max": float(residual.max()),
            "clamp_residual_by_joint": {
                joint_name: float(residual[index]) for index, joint_name in enumerate(joint_names)
            },
        }

    return {
        "future_state_valid_steps": len(execution_states),
        "action0": offset_metrics(0),
        "action13": offset_metrics(13),
    }


def scalar_stats(values: list[float]) -> dict[str, int | float | None]:
    if not values:
        return {"count": 0, "mean": None, "std": None, "p95": None, "max": None}
    tensor = torch.tensor(values, dtype=torch.float64)
    return {
        "count": len(values),
        "mean": float(tensor.mean()),
        "std": float(tensor.std(unbiased=False)),
        "p95": float(torch.quantile(tensor, 0.95)),
        "max": float(tensor.max()),
    }


def rate(values: list[bool]) -> float | None:
    return statistics.fmean(values) if values else None


class MetricsAccumulator:
    """Aggregate per-joint and all-joint stability metrics for one stage."""

    _TENSOR_METRICS = (
        "mae",
        "seed_std",
        "action0_abs_error",
        "action13_abs_error",
        "action0_state_jump",
        "action13_state_jump",
        "action13_execution_state_jump",
        "action0_clamp_residual",
        "action0_excess_clamp_residual",
        "action13_execution_clamp_residual",
        "action13_excess_clamp_residual",
        "chunk_jump_abs",
    )
    _DATASET_SAFETY_METRICS = (
        "gt_action0_state_jump",
        "gt_action0_clamp_residual",
        "gt_action13_execution_state_jump",
        "gt_action13_execution_clamp_residual",
    )

    def __init__(self, joint_names: list[str], max_relative_target: float) -> None:
        self.joint_names = joint_names
        self.max_relative_target = max_relative_target
        self.values = {metric: {joint: [] for joint in joint_names} for metric in self._TENSOR_METRICS}
        self.dataset_safety_values = {metric: [] for metric in self._DATASET_SAFETY_METRICS}
        self.clamp_by_joint = {joint: [] for joint in joint_names}
        self.predicted_only_clamp_by_joint = {joint: [] for joint in joint_names}
        self.action13_jump_by_joint = {joint: [] for joint in joint_names}
        self.action13_execution_jump_by_joint = {joint: [] for joint in joint_names}
        self.action13_predicted_only_clamp_by_joint = {joint: [] for joint in joint_names}
        self.jump_exceeds_by_joint = {joint: [] for joint in joint_names}
        self.any_first_action_clamp: list[bool] = []
        self.any_action0_predicted_only_clamp: list[bool] = []
        self.any_action13_state_jump: list[bool] = []
        self.any_action13_execution_state_jump: list[bool] = []
        self.any_action13_predicted_only_clamp: list[bool] = []
        self.any_gt_action0_clamp: list[bool] = []
        self.any_gt_action13_execution_clamp: list[bool] = []
        self.any_step_jump: list[bool] = []
        self.chunk_has_jump: list[bool] = []
        self.gripper_gt_events = 0
        self.gripper_no_gt_events = 0
        self.gripper_matched_events = 0
        self.gripper_missing_events = 0
        self.gripper_false_positive_events = 0
        self.gripper_direction_matches = 0
        self.gripper_timing_errors: list[float] = []
        self.prediction_count = 0
        self.frame_count = 0
        self.action13_eligible_prediction_count = 0
        self.action13_ineligible_tail_prediction_count = 0
        self.dataset_safety_frame_count = 0
        self.action13_eligible_frame_count = 0
        self.action13_ineligible_tail_frame_count = 0

    def _append_tensor(self, metric: str, tensor: torch.Tensor | None) -> None:
        if tensor is None or tensor.numel() == 0:
            return
        tensor = tensor.detach().float().cpu().reshape(-1, len(self.joint_names))
        for joint_index, joint_name in enumerate(self.joint_names):
            self.values[metric][joint_name].extend(float(value) for value in tensor[:, joint_index])

    def _append_dataset_safety_tensor(self, metric: str, tensor: torch.Tensor | None) -> None:
        if tensor is None or tensor.numel() == 0:
            return
        self.dataset_safety_values[metric].extend(
            float(value) for value in tensor.detach().float().cpu().flatten()
        )

    def add_dataset_safety(
        self,
        ground_truth: torch.Tensor,
        execution_states: torch.Tensor,
    ) -> None:
        """Add seed-invariant demonstration safety metrics once per selected frame."""
        ground_truth = ground_truth.detach().float().cpu()
        execution_states = execution_states.detach().float().cpu()
        if ground_truth.ndim != 2 or execution_states.shape != ground_truth.shape:
            raise ValueError("ground truth and execution states must align as [T, A]")
        if len(ground_truth) == 0 or ground_truth.shape[1] != len(self.joint_names):
            raise ValueError("dataset safety requires a non-empty action-aligned frame")

        self.dataset_safety_frame_count += 1
        action0_jump = ground_truth[0].sub(execution_states[0]).abs()
        action0_clamp = action0_jump > self.max_relative_target
        self._append_dataset_safety_tensor("gt_action0_state_jump", action0_jump)
        self._append_dataset_safety_tensor(
            "gt_action0_clamp_residual",
            action0_jump.sub(self.max_relative_target).clamp_min(0.0),
        )
        self.any_gt_action0_clamp.append(bool(action0_clamp.any()))

        if len(ground_truth) <= 13:
            self.action13_ineligible_tail_frame_count += 1
            return

        self.action13_eligible_frame_count += 1
        action13_jump = ground_truth[13].sub(execution_states[13]).abs()
        action13_clamp = action13_jump > self.max_relative_target
        self._append_dataset_safety_tensor("gt_action13_execution_state_jump", action13_jump)
        self._append_dataset_safety_tensor(
            "gt_action13_execution_clamp_residual",
            action13_jump.sub(self.max_relative_target).clamp_min(0.0),
        )
        self.any_gt_action13_execution_clamp.append(bool(action13_clamp.any()))

    def add_prediction(self, record: PredictionRecord) -> None:
        self.prediction_count += 1
        self._append_tensor("mae", record.errors)
        self._append_tensor("action0_abs_error", record.action0_errors)
        self._append_tensor("action13_abs_error", record.action13_errors)
        self._append_tensor("action0_state_jump", record.action0_state_jumps)
        self._append_tensor("action13_state_jump", record.action13_state_jumps)
        self._append_tensor("action13_execution_state_jump", record.action13_execution_state_jumps)
        self._append_tensor("action0_clamp_residual", record.action0_clamp_residuals)
        self._append_tensor("action0_excess_clamp_residual", record.action0_excess_clamp_residuals)
        self._append_tensor("action13_execution_clamp_residual", record.action13_execution_clamp_residuals)
        self._append_tensor("action13_excess_clamp_residual", record.action13_excess_clamp_residuals)
        self._append_tensor("chunk_jump_abs", record.jumps)

        if record.action13_errors is None:
            self.action13_ineligible_tail_prediction_count += 1
        else:
            self.action13_eligible_prediction_count += 1

        for joint_index, joint_name in enumerate(self.joint_names):
            self.clamp_by_joint[joint_name].append(bool(record.first_action_clamp[joint_index]))
            self.predicted_only_clamp_by_joint[joint_name].append(
                bool(record.action0_predicted_only_clamp[joint_index])
            )
            if record.action13_state_jump_exceeds is not None:
                self.action13_jump_by_joint[joint_name].append(
                    bool(record.action13_state_jump_exceeds[joint_index])
                )
            if record.action13_execution_state_jump_exceeds is not None:
                self.action13_execution_jump_by_joint[joint_name].append(
                    bool(record.action13_execution_state_jump_exceeds[joint_index])
                )
            if record.action13_predicted_only_clamp is not None:
                self.action13_predicted_only_clamp_by_joint[joint_name].append(
                    bool(record.action13_predicted_only_clamp[joint_index])
                )
            if record.jumps.numel() > 0:
                self.jump_exceeds_by_joint[joint_name].extend(
                    bool(value) for value in (record.jumps[:, joint_index] > self.max_relative_target)
                )
        self.any_first_action_clamp.append(bool(record.first_action_clamp.any()))
        self.any_action0_predicted_only_clamp.append(bool(record.action0_predicted_only_clamp.any()))
        if record.action13_state_jump_exceeds is not None:
            self.any_action13_state_jump.append(bool(record.action13_state_jump_exceeds.any()))
        if record.action13_execution_state_jump_exceeds is not None:
            self.any_action13_execution_state_jump.append(
                bool(record.action13_execution_state_jump_exceeds.any())
            )
        if record.action13_predicted_only_clamp is not None:
            self.any_action13_predicted_only_clamp.append(bool(record.action13_predicted_only_clamp.any()))
        if record.jumps.numel() > 0:
            jump_mask = record.jumps > self.max_relative_target
            self.any_step_jump.extend(bool(value) for value in jump_mask.any(dim=1))
            self.chunk_has_jump.append(bool(jump_mask.any()))

        if record.gt_gripper_event is not None:
            self.gripper_gt_events += 1
            if record.predicted_gripper_event is None:
                self.gripper_missing_events += 1
            else:
                self.gripper_matched_events += 1
                self.gripper_timing_errors.append(
                    abs(record.predicted_gripper_event - record.gt_gripper_event)
                )
                if record.predicted_gripper_direction == record.gt_gripper_direction:
                    self.gripper_direction_matches += 1
        else:
            self.gripper_no_gt_events += 1
            if record.predicted_gripper_event is not None:
                self.gripper_false_positive_events += 1

    def add_seed_std(self, seed_std: torch.Tensor) -> None:
        self.frame_count += 1
        self._append_tensor("seed_std", seed_std)

    def summarize(self) -> dict[str, Any]:
        per_joint = {}
        for joint_name in self.joint_names:
            joint_summary = {
                metric: scalar_stats(self.values[metric][joint_name]) for metric in self._TENSOR_METRICS
            }
            joint_summary["first_action_clamp_rate"] = rate(self.clamp_by_joint[joint_name])
            joint_summary["action0_predicted_only_clamp_rate"] = rate(
                self.predicted_only_clamp_by_joint[joint_name]
            )
            joint_summary["action13_state_jump_exceedance_rate"] = rate(
                self.action13_jump_by_joint[joint_name]
            )
            joint_summary["action13_execution_state_jump_exceedance_rate"] = rate(
                self.action13_execution_jump_by_joint[joint_name]
            )
            joint_summary["action13_predicted_only_clamp_rate"] = rate(
                self.action13_predicted_only_clamp_by_joint[joint_name]
            )
            joint_summary["chunk_jump_exceedance_rate"] = rate(self.jump_exceeds_by_joint[joint_name])
            per_joint[joint_name] = joint_summary

        all_joints = {}
        for metric in self._TENSOR_METRICS:
            combined = []
            for joint_name in self.joint_names:
                combined.extend(self.values[metric][joint_name])
            all_joints[metric] = scalar_stats(combined)
        all_joints.update(
            {
                "first_action_clamp_rate": rate(self.any_first_action_clamp),
                "action0_predicted_only_clamp_rate": rate(self.any_action0_predicted_only_clamp),
                "action13_state_jump_exceedance_rate": rate(self.any_action13_state_jump),
                "action13_execution_state_jump_exceedance_rate": rate(self.any_action13_execution_state_jump),
                "action13_predicted_only_clamp_rate": rate(self.any_action13_predicted_only_clamp),
                "chunk_jump_exceedance_rate": rate(self.any_step_jump),
                "chunk_has_jump_rate": rate(self.chunk_has_jump),
            }
        )

        gripper_miss_rate = (
            self.gripper_missing_events / self.gripper_gt_events if self.gripper_gt_events else None
        )
        gripper_false_positive_rate = (
            self.gripper_false_positive_events / self.gripper_no_gt_events
            if self.gripper_no_gt_events
            else None
        )
        gripper_direction_consistency = (
            self.gripper_direction_matches / self.gripper_matched_events
            if self.gripper_matched_events
            else None
        )
        gripper = {
            "ground_truth_event_count": self.gripper_gt_events,
            "no_ground_truth_event_count": self.gripper_no_gt_events,
            "matched_event_count": self.gripper_matched_events,
            "missing_event_count": self.gripper_missing_events,
            "false_positive_event_count": self.gripper_false_positive_events,
            "direction_match_count": self.gripper_direction_matches,
            "miss_rate": gripper_miss_rate,
            "false_positive_rate": gripper_false_positive_rate,
            "direction_consistency_rate": gripper_direction_consistency,
            "timing_abs_error_steps": scalar_stats(self.gripper_timing_errors),
        }
        dataset_safety_all_joints = {
            metric: scalar_stats(values) for metric, values in self.dataset_safety_values.items()
        }
        dataset_safety_all_joints.update(
            {
                "gt_action0_clamp_rate": rate(self.any_gt_action0_clamp),
                "gt_action13_execution_clamp_rate": rate(self.any_gt_action13_execution_clamp),
            }
        )
        action13_frame_coverage_rate = (
            self.action13_eligible_frame_count / self.dataset_safety_frame_count
            if self.dataset_safety_frame_count
            else None
        )
        action13_prediction_coverage_rate = (
            self.action13_eligible_prediction_count / self.prediction_count if self.prediction_count else None
        )
        return {
            "frame_count": self.frame_count,
            "prediction_count": self.prediction_count,
            "all_joints": all_joints,
            "per_joint": per_joint,
            "action13_coverage": {
                "eligible_frame_count": self.action13_eligible_frame_count,
                "ineligible_tail_frame_count": self.action13_ineligible_tail_frame_count,
                "total_frame_count": self.dataset_safety_frame_count,
                "frame_coverage_rate": action13_frame_coverage_rate,
                "eligible_prediction_count": self.action13_eligible_prediction_count,
                "ineligible_tail_prediction_count": self.action13_ineligible_tail_prediction_count,
                "total_prediction_count": self.prediction_count,
                "prediction_coverage_rate": action13_prediction_coverage_rate,
            },
            "dataset_safety": {
                "aggregation_unit": "selected_frame",
                "frame_count": self.dataset_safety_frame_count,
                "all_joints": dataset_safety_all_joints,
            },
            "metric_roles": {
                "first_action_clamp_rate": "diagnostic_raw_query_state",
                "action13_state_jump_exceedance_rate": "diagnostic_legacy_query_state",
                "action13_execution_state_jump_exceedance_rate": "diagnostic_raw_execution_state",
                "action0_predicted_only_clamp_rate": "hard_gate",
                "action13_predicted_only_clamp_rate": "hard_gate",
            },
            "gripper_event_timing": gripper,
        }


def _threshold_check(value: float | None, threshold: float) -> dict[str, Any]:
    return {
        "value": value,
        "threshold": threshold,
        "operator": "<=",
        "evaluated": value is not None,
        "passed": value is not None and value <= threshold,
    }


def _minimum_check(value: float | None, threshold: float) -> dict[str, Any]:
    return {
        "value": value,
        "threshold": threshold,
        "operator": ">=",
        "evaluated": value is not None,
        "passed": value is not None and value >= threshold,
    }


def evaluate_gate(
    summary: dict[str, Any],
    thresholds: GateThresholds,
    *,
    allow_unavailable: bool = False,
    include_predicted_only_rates: bool = True,
) -> dict[str, Any]:
    all_joints = summary["all_joints"]
    gripper = summary["gripper_event_timing"]
    required_checks = {
        "mae": _threshold_check(all_joints["mae"]["mean"], thresholds.max_mae),
        "mae_p95": _threshold_check(all_joints["mae"]["p95"], thresholds.max_mae_p95),
        "mae_max": _threshold_check(all_joints["mae"]["max"], thresholds.max_mae_max),
        "seed_std": _threshold_check(all_joints["seed_std"]["mean"], thresholds.max_seed_std),
        "seed_std_p95": _threshold_check(all_joints["seed_std"]["p95"], thresholds.max_seed_std_p95),
        "seed_std_max": _threshold_check(all_joints["seed_std"]["max"], thresholds.max_seed_std_max),
        "action0_abs_error": _threshold_check(
            all_joints["action0_abs_error"]["mean"], thresholds.max_action0_error
        ),
        "action0_abs_error_p95": _threshold_check(
            all_joints["action0_abs_error"]["p95"], thresholds.max_action0_error_p95
        ),
        "action0_abs_error_max": _threshold_check(
            all_joints["action0_abs_error"]["max"], thresholds.max_action0_error_max
        ),
        "action13_abs_error": _threshold_check(
            all_joints["action13_abs_error"]["mean"], thresholds.max_action13_error
        ),
        "action13_abs_error_p95": _threshold_check(
            all_joints["action13_abs_error"]["p95"], thresholds.max_action13_error_p95
        ),
        "action13_abs_error_max": _threshold_check(
            all_joints["action13_abs_error"]["max"], thresholds.max_action13_error_max
        ),
        "action0_predicted_only_clamp_rate": _threshold_check(
            all_joints["action0_predicted_only_clamp_rate"],
            thresholds.max_action0_predicted_only_clamp_rate,
        ),
        "action13_predicted_only_clamp_rate": _threshold_check(
            all_joints["action13_predicted_only_clamp_rate"],
            thresholds.max_action13_predicted_only_clamp_rate,
        ),
        "action0_excess_clamp_residual_p95": _threshold_check(
            all_joints["action0_excess_clamp_residual"]["p95"],
            thresholds.max_action0_error_p95,
        ),
        "action0_excess_clamp_residual_max": _threshold_check(
            all_joints["action0_excess_clamp_residual"]["max"],
            thresholds.max_action0_error_max,
        ),
        "action13_excess_clamp_residual_p95": _threshold_check(
            all_joints["action13_excess_clamp_residual"]["p95"],
            thresholds.max_action13_error_p95,
        ),
        "action13_excess_clamp_residual_max": _threshold_check(
            all_joints["action13_excess_clamp_residual"]["max"],
            thresholds.max_action13_error_max,
        ),
        "chunk_jump_exceedance_rate": _threshold_check(
            all_joints["chunk_jump_exceedance_rate"], thresholds.max_chunk_jump_rate
        ),
        "gripper_timing_abs_error_steps": _threshold_check(
            gripper["timing_abs_error_steps"]["mean"],
            thresholds.max_gripper_timing_error_steps,
        ),
        "gripper_event_miss_rate": _threshold_check(
            gripper["miss_rate"], thresholds.max_gripper_event_miss_rate
        ),
        "gripper_false_positive_rate": _threshold_check(
            gripper["false_positive_rate"], thresholds.max_gripper_false_positive_rate
        ),
        "gripper_direction_consistency_rate": _minimum_check(
            gripper["direction_consistency_rate"], thresholds.min_gripper_direction_consistency
        ),
    }
    diagnostic_checks = {
        "first_action_clamp_rate": _threshold_check(
            all_joints["first_action_clamp_rate"],
            thresholds.max_action0_predicted_only_clamp_rate,
        ),
        "action13_state_jump_exceedance_rate": _threshold_check(
            all_joints["action13_state_jump_exceedance_rate"],
            thresholds.max_action13_predicted_only_clamp_rate,
        ),
        "action13_execution_state_jump_exceedance_rate": _threshold_check(
            all_joints["action13_execution_state_jump_exceedance_rate"],
            thresholds.max_action13_predicted_only_clamp_rate,
        ),
    }
    for name, check in diagnostic_checks.items():
        check["diagnostic_only"] = True
        check["required"] = False
        check["semantics"] = (
            "legacy_action13_vs_query_state"
            if name == "action13_state_jump_exceedance_rate"
            else "raw_clamp_rate"
        )
    if allow_unavailable:
        for check in required_checks.values():
            if not check["evaluated"]:
                check["passed"] = True
                check["skipped"] = "not required in quick_smoke"
    if not include_predicted_only_rates:
        for name in (
            "action0_predicted_only_clamp_rate",
            "action13_predicted_only_clamp_rate",
        ):
            check = required_checks[name]
            check["observed_passed"] = check["passed"]
            check["passed"] = True
            check["required"] = False
            check["skipped"] = "quick_smoke sample count is too small for a 5% rate gate"
    for check in required_checks.values():
        check.setdefault("required", True)
    checks = {**required_checks, **diagnostic_checks}
    return {
        "passed": all(check["passed"] for check in required_checks.values()),
        "checks": checks,
    }


def build_robustness_summary(
    *,
    seed_maes: dict[int, list[float]],
    frame_maes: list[dict[str, Any]],
    stage_reports: dict[str, Any],
) -> dict[str, Any]:
    per_seed = {str(seed): scalar_stats(values) for seed, values in sorted(seed_maes.items())}
    valid_seed_means = [
        (int(seed), metrics["mean"]) for seed, metrics in per_seed.items() if metrics["mean"] is not None
    ]
    worst_seed = (
        {
            "seed": max(valid_seed_means, key=lambda item: item[1])[0],
            "mae": max(value for _, value in valid_seed_means),
        }
        if valid_seed_means
        else None
    )
    worst_frame = max(frame_maes, key=lambda item: item["mae"]) if frame_maes else None

    extrema: dict[str, dict[str, Any] | None] = {
        "mae_p95": None,
        "mae_max": None,
        "seed_std_p95": None,
        "seed_std_max": None,
    }
    for stage_name, stage_report in stage_reports.items():
        for joint_name, joint_metrics in stage_report["metrics"]["per_joint"].items():
            for metric_name in ("mae", "seed_std"):
                for statistic in ("p95", "max"):
                    value = joint_metrics[metric_name][statistic]
                    key = f"{metric_name}_{statistic}"
                    if value is not None and (extrema[key] is None or value > extrema[key]["value"]):
                        extrema[key] = {
                            "value": value,
                            "stage": stage_name,
                            "joint": joint_name,
                        }
    return {
        "per_seed_mae": per_seed,
        "worst_seed_mae": worst_seed,
        "worst_frame_mae": worst_frame,
        "worst_stage_joint": extrema,
    }


def evaluate_robustness_gate(
    robustness: dict[str, Any],
    thresholds: GateThresholds,
    *,
    include_stage_extrema: bool = True,
) -> dict[str, Any]:
    worst_seed = robustness["worst_seed_mae"]
    worst_frame = robustness["worst_frame_mae"]
    extrema = robustness["worst_stage_joint"]
    checks = {
        "worst_seed_mae": _threshold_check(
            worst_seed["mae"] if worst_seed is not None else None,
            thresholds.max_worst_seed_mae,
        ),
        "worst_frame_mae": _threshold_check(
            worst_frame["mae"] if worst_frame is not None else None,
            thresholds.max_worst_frame_mae,
        ),
        "worst_stage_joint_mae_p95": _threshold_check(
            extrema["mae_p95"]["value"] if extrema["mae_p95"] is not None else None,
            thresholds.max_mae_p95,
        ),
        "worst_stage_joint_mae_max": _threshold_check(
            extrema["mae_max"]["value"] if extrema["mae_max"] is not None else None,
            thresholds.max_mae_max,
        ),
        "worst_stage_joint_seed_std_p95": _threshold_check(
            extrema["seed_std_p95"]["value"] if extrema["seed_std_p95"] is not None else None,
            thresholds.max_seed_std_p95,
        ),
        "worst_stage_joint_seed_std_max": _threshold_check(
            extrema["seed_std_max"]["value"] if extrema["seed_std_max"] is not None else None,
            thresholds.max_seed_std_max,
        ),
    }
    stage_extrema_checks = {
        "worst_stage_joint_mae_p95",
        "worst_stage_joint_mae_max",
        "worst_stage_joint_seed_std_p95",
        "worst_stage_joint_seed_std_max",
    }
    for name, check in checks.items():
        check["required"] = include_stage_extrema or name not in stage_extrema_checks
        if not check["required"]:
            check["observed_passed"] = check["passed"]
            check["passed"] = True
            check["skipped"] = "quick_smoke stage extrema are diagnostic only"
    return {
        "passed": all(check["passed"] for check in checks.values()),
        "checks": checks,
    }


def evaluate_formal_stage_safety_gate(
    stage_reports: dict[str, Any],
    *,
    minimum_predictions_per_stage: int,
) -> dict[str, Any]:
    """Prevent overall aggregation from hiding a safety regression in a task stage."""
    if minimum_predictions_per_stage < 1:
        raise ValueError("minimum_predictions_per_stage must be positive")
    missing_stages = [stage for stage in FORMAL_SAFETY_STAGES if stage not in stage_reports]
    reports = {}
    for stage in FORMAL_SAFETY_STAGES:
        if stage not in stage_reports:
            continue
        stage_report = stage_reports[stage]
        prediction_count = stage_report["metrics"]["prediction_count"]
        coverage_check = {
            "value": prediction_count,
            "minimum": minimum_predictions_per_stage,
            "operator": ">=",
            "evaluated": True,
            "passed": prediction_count >= minimum_predictions_per_stage,
        }
        checks = {"prediction_count": coverage_check}
        for name in FORMAL_STAGE_SAFETY_CHECKS:
            check = dict(stage_report["gate"]["checks"][name])
            if not check["evaluated"]:
                check["passed"] = True
                check["required"] = False
                check["skipped"] = "metric has no eligible samples in this stage"
            else:
                check["required"] = True
            checks[name] = check
        reports[stage] = {
            "passed": all(check["passed"] for check in checks.values()),
            "checks": checks,
        }
    return {
        "passed": not missing_stages and all(report["passed"] for report in reports.values()),
        "required_stages": list(FORMAL_SAFETY_STAGES),
        "missing_stages": missing_stages,
        "minimum_predictions_per_stage": minimum_predictions_per_stage,
        "stages": reports,
    }


def evaluate_model_coverage(
    *,
    summary: dict[str, Any],
    expected_predictions: int,
    finite_predictions: int,
    nonfinite_action_values: int,
    seed_count: int = 1,
    min_action13_coverage_rate: float = GateThresholds.min_action13_coverage_rate,
    require_semantic_coverage: bool = True,
) -> dict[str, Any]:
    action13_count = summary["all_joints"]["action13_abs_error"]["count"]
    action13_coverage = summary["action13_coverage"]
    eligible_frames = action13_coverage["eligible_frame_count"]
    eligible_predictions = action13_coverage["eligible_prediction_count"]
    expected_eligible_predictions = eligible_frames * seed_count
    action13_coverage_rate = action13_coverage["frame_coverage_rate"]
    action13_counts_match = eligible_predictions == expected_eligible_predictions
    gripper_event_count = summary["gripper_event_timing"]["ground_truth_event_count"]
    no_gripper_event_count = summary["gripper_event_timing"]["no_ground_truth_event_count"]
    checks = {
        "all_predictions_finite": {
            "nonfinite_action_values": nonfinite_action_values,
            "passed": nonfinite_action_values == 0,
        },
        "prediction_count": {
            "value": finite_predictions,
            "expected": expected_predictions,
            "passed": finite_predictions == expected_predictions,
        },
        "action13_coverage": {
            "value": action13_count,
            "eligible_frame_count": eligible_frames,
            "ineligible_tail_frame_count": action13_coverage["ineligible_tail_frame_count"],
            "total_frame_count": action13_coverage["total_frame_count"],
            "frame_coverage_rate": action13_coverage_rate,
            "eligible_prediction_count": eligible_predictions,
            "expected_eligible_prediction_count": expected_eligible_predictions,
            "ineligible_tail_prediction_count": action13_coverage["ineligible_tail_prediction_count"],
            "minimum_rate": min_action13_coverage_rate,
            "required": require_semantic_coverage,
            "counts_match": action13_counts_match,
            "passed": action13_counts_match
            and (
                not require_semantic_coverage
                or (
                    action13_coverage_rate is not None
                    and action13_coverage_rate >= min_action13_coverage_rate
                )
            ),
        },
        "gripper_event_coverage": {
            "value": gripper_event_count,
            "minimum": 1,
            "required": require_semantic_coverage,
            "passed": gripper_event_count > 0 or not require_semantic_coverage,
        },
        "gripper_no_event_coverage": {
            "value": no_gripper_event_count,
            "minimum": 1,
            "required": require_semantic_coverage,
            "passed": no_gripper_event_count > 0 or not require_semantic_coverage,
        },
    }
    return {
        "passed": all(check["passed"] for check in checks.values()),
        "checks": checks,
    }


def _relative_upper_check(
    candidate: float | None,
    baseline: float | None,
    *,
    absolute_threshold: float,
    rate_margin: float | None = None,
    error_ratio: float | None = None,
) -> dict[str, Any]:
    if candidate is None or baseline is None:
        return {
            "candidate": candidate,
            "baseline": baseline,
            "evaluated": False,
            "passed": False,
        }
    relative_threshold = (
        baseline + rate_margin
        if rate_margin is not None
        else max(baseline * float(error_ratio), baseline + 1e-6)
    )
    effective_threshold = min(absolute_threshold, relative_threshold)
    return {
        "candidate": candidate,
        "baseline": baseline,
        "absolute_threshold": absolute_threshold,
        "relative_threshold": relative_threshold,
        "effective_threshold": effective_threshold,
        "operator": "<=",
        "evaluated": True,
        "passed": candidate <= effective_threshold,
    }


def evaluate_relative_baseline_gate(
    candidate_summary: dict[str, Any],
    baseline_model_report: dict[str, Any],
    thresholds: GateThresholds,
    *,
    rate_margin: float,
    error_ratio: float,
) -> dict[str, Any]:
    baseline_summary = baseline_model_report["stages"]["overall"]["metrics"]
    candidate_all = candidate_summary["all_joints"]
    baseline_all = baseline_summary["all_joints"]
    candidate_gripper = candidate_summary["gripper_event_timing"]
    baseline_gripper = baseline_summary["gripper_event_timing"]
    required_checks = {
        "mae": _relative_upper_check(
            candidate_all["mae"]["mean"],
            baseline_all["mae"]["mean"],
            absolute_threshold=thresholds.max_mae,
            error_ratio=error_ratio,
        ),
        "seed_std": _relative_upper_check(
            candidate_all["seed_std"]["mean"],
            baseline_all["seed_std"]["mean"],
            absolute_threshold=thresholds.max_seed_std,
            error_ratio=error_ratio,
        ),
        "action0_predicted_only_clamp_rate": _relative_upper_check(
            candidate_all["action0_predicted_only_clamp_rate"],
            baseline_all["action0_predicted_only_clamp_rate"],
            absolute_threshold=thresholds.max_action0_predicted_only_clamp_rate,
            rate_margin=rate_margin,
        ),
        "action13_predicted_only_clamp_rate": _relative_upper_check(
            candidate_all["action13_predicted_only_clamp_rate"],
            baseline_all["action13_predicted_only_clamp_rate"],
            absolute_threshold=thresholds.max_action13_predicted_only_clamp_rate,
            rate_margin=rate_margin,
        ),
        "chunk_jump_exceedance_rate": _relative_upper_check(
            candidate_all["chunk_jump_exceedance_rate"],
            baseline_all["chunk_jump_exceedance_rate"],
            absolute_threshold=thresholds.max_chunk_jump_rate,
            rate_margin=rate_margin,
        ),
        "gripper_event_miss_rate": _relative_upper_check(
            candidate_gripper["miss_rate"],
            baseline_gripper["miss_rate"],
            absolute_threshold=thresholds.max_gripper_event_miss_rate,
            rate_margin=rate_margin,
        ),
        "gripper_false_positive_rate": _relative_upper_check(
            candidate_gripper["false_positive_rate"],
            baseline_gripper["false_positive_rate"],
            absolute_threshold=thresholds.max_gripper_false_positive_rate,
            rate_margin=rate_margin,
        ),
    }
    diagnostic_checks = {
        "first_action_clamp_rate": _relative_upper_check(
            candidate_all["first_action_clamp_rate"],
            baseline_all["first_action_clamp_rate"],
            absolute_threshold=thresholds.max_action0_predicted_only_clamp_rate,
            rate_margin=rate_margin,
        ),
        "action13_state_jump_exceedance_rate": _relative_upper_check(
            candidate_all["action13_state_jump_exceedance_rate"],
            baseline_all["action13_state_jump_exceedance_rate"],
            absolute_threshold=thresholds.max_action13_predicted_only_clamp_rate,
            rate_margin=rate_margin,
        ),
        "action13_execution_state_jump_exceedance_rate": _relative_upper_check(
            candidate_all["action13_execution_state_jump_exceedance_rate"],
            baseline_all["action13_execution_state_jump_exceedance_rate"],
            absolute_threshold=thresholds.max_action13_predicted_only_clamp_rate,
            rate_margin=rate_margin,
        ),
    }
    for check in required_checks.values():
        check["required"] = True
    for name, check in diagnostic_checks.items():
        check["required"] = False
        check["diagnostic_only"] = True
        check["semantics"] = (
            "legacy_action13_vs_query_state"
            if name == "action13_state_jump_exceedance_rate"
            else "raw_clamp_rate"
        )
    checks = {**required_checks, **diagnostic_checks}
    return {
        "passed": all(check["passed"] for check in required_checks.values()),
        "checks": checks,
    }


def load_baseline_model_report(
    path: Path,
    *,
    label: str,
    dataset_root: Path,
    repo_id: str,
    dataset_asset_fingerprint: dict[str, Any],
    action_names: list[str],
    state_names: list[str],
    seeds: list[int],
    frames: list[SelectedFrame],
    selection_contract: dict[str, Any],
    thresholds: GateThresholds,
) -> dict[str, Any]:
    report = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if report.get("schema_version") != REPORT_SCHEMA_VERSION:
        raise ValueError(
            f"baseline report schema must be {REPORT_SCHEMA_VERSION} for predicted-only clamp semantics"
        )
    if report.get("report_status") != "complete":
        raise ValueError("baseline report must be complete")
    if report.get("run_mode") != "formal":
        raise ValueError("baseline report must come from formal evaluation")
    if report.get("hardware_action_sent") is not False:
        raise ValueError("baseline report must declare hardware_action_sent=false")
    if report.get("formal_passed") is not True:
        raise ValueError("baseline report must have formal_passed=true")
    if report.get("smoke_passed") is not True or report.get("passed") is not True:
        raise ValueError("baseline report top-level gates are inconsistent")
    if report.get("formal_selection_coverage", {}).get("passed") is not True:
        raise ValueError("baseline report formal selection coverage did not pass")
    if report.get("metric_semantics") != METRIC_SEMANTICS:
        raise ValueError("baseline metric semantics differ from the current evaluator")
    if (
        report.get("gate_profile", {}).get("name") != DEPLOYMENT_GATE_PROFILE
        or report.get("gate_profile", {}).get("formal_eligible") is not True
    ):
        raise ValueError("baseline report must use the deployment-v2 gate profile")
    if report.get("thresholds") != asdict(thresholds) or thresholds != GateThresholds():
        raise ValueError("baseline thresholds differ from the immutable deployment profile")
    expected_models = report.get("expected_models")
    completed_models = report.get("completed_models")
    model_labels = list(report.get("models", {}))
    if expected_models != completed_models or set(model_labels) != set(expected_models or []):
        raise ValueError("baseline model set is incomplete")
    if Path(report["dataset"]["root"]).resolve() != dataset_root.resolve():
        raise ValueError("baseline dataset root differs from the current run")
    if report["dataset"].get("repo_id") != repo_id:
        raise ValueError("baseline dataset repo_id differs from the current run")
    if report["dataset"].get("asset_fingerprint") != dataset_asset_fingerprint:
        raise ValueError("baseline dataset asset fingerprint differs from the current run")
    if (
        report["dataset"].get("action_names") != action_names
        or report["dataset"].get("state_names") != state_names
    ):
        raise ValueError("baseline action/state feature contract differs from the current run")
    if report["noise"]["seeds"] != seeds:
        raise ValueError("baseline seed set/order differs from the current run")
    for key, expected in selection_contract.items():
        if report["selection"].get(key) != expected:
            raise ValueError(f"baseline selection parameter {key!r} differs from the current run")
    if report["selection"].get("frames") != [frame.to_json() for frame in frames]:
        raise ValueError("baseline frame manifest differs from the current run")
    try:
        model_report = report["models"][label]
    except KeyError as exc:
        raise ValueError(f"baseline report does not contain model label {label!r}") from exc
    if (
        model_report.get("smoke_passed") is not True
        or model_report.get("formal_passed") is not True
        or model_report.get("passed") is not True
        or model_report.get("hardware_action_sent") is not False
    ):
        raise ValueError(f"baseline model {label!r} did not pass its formal gate")
    verify_checkpoint_asset_fingerprint(model_report)
    return model_report


def _has_model_weights(path: Path) -> bool:
    return (path / "model.safetensors").is_file() or (path / "model.safetensors.index.json").is_file()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint_dataset_assets(root: Path) -> dict[str, Any]:
    """Fingerprint every local dataset file, including videos and numeric metadata."""
    root = root.resolve()
    files = sorted((path for path in root.rglob("*") if path.is_file()), key=lambda path: str(path))
    if not files:
        raise FileNotFoundError(f"No dataset assets found in {root}")
    assets = []
    aggregate = hashlib.sha256()
    for path in files:
        relative_path = str(path.relative_to(root))
        digest = _sha256_file(path)
        assets.append(
            {
                "relative_path": relative_path,
                "size": path.stat().st_size,
                "sha256": digest,
            }
        )
        aggregate.update(relative_path.encode())
        aggregate.update(bytes.fromhex(digest))
    return {"sha256": aggregate.hexdigest(), "assets": assets}


def verify_dataset_asset_fingerprint(root: Path, fingerprint: dict[str, Any]) -> None:
    current = fingerprint_dataset_assets(root)
    if current != fingerprint:
        raise ValueError("baseline dataset assets differ from the current local dataset")


def fingerprint_checkpoint_assets(
    checkpoint: Path,
    base_model: Path | None = None,
) -> dict[str, Any]:
    """Bind a report to the exact model and processor files loaded locally."""
    roots = [("checkpoint", checkpoint.resolve())]
    if base_model is not None:
        roots.append(("base_model", base_model.resolve()))
    assets = []
    for role, root in roots:
        files = [
            path
            for path in root.iterdir()
            if path.is_file()
            and (
                path.suffix == ".safetensors"
                or path.name == "adapter_model.bin"
                or path.name.endswith(".safetensors.index.json")
                or path.name
                in {
                    "config.json",
                    "adapter_config.json",
                    "policy_preprocessor.json",
                    "policy_postprocessor.json",
                }
            )
        ]
        for path in sorted(files, key=lambda item: item.name):
            assets.append(
                {
                    "role": role,
                    "path": str(path.resolve()),
                    "size": path.stat().st_size,
                    "sha256": _sha256_file(path),
                }
            )
    if not assets:
        raise FileNotFoundError(f"No checkpoint assets found in {checkpoint}")
    aggregate = hashlib.sha256()
    for asset in assets:
        aggregate.update(asset["role"].encode())
        aggregate.update(Path(asset["path"]).name.encode())
        aggregate.update(bytes.fromhex(asset["sha256"]))
    return {"sha256": aggregate.hexdigest(), "assets": assets}


def verify_checkpoint_asset_fingerprint(model_report: dict[str, Any]) -> None:
    fingerprint = model_report.get("checkpoint_asset_fingerprint")
    if not isinstance(fingerprint, dict) or not fingerprint.get("assets"):
        raise ValueError("baseline model is missing its checkpoint asset fingerprint")
    aggregate = hashlib.sha256()
    for asset in fingerprint["assets"]:
        path = Path(asset["path"])
        if not path.is_file() or path.stat().st_size != asset["size"]:
            raise ValueError(f"baseline checkpoint asset is missing or changed: {path}")
        digest = _sha256_file(path)
        if digest != asset["sha256"]:
            raise ValueError(f"baseline checkpoint asset hash changed: {path}")
        aggregate.update(asset["role"].encode())
        aggregate.update(path.name.encode())
        aggregate.update(bytes.fromhex(digest))
    if aggregate.hexdigest() != fingerprint.get("sha256"):
        raise ValueError("baseline checkpoint aggregate fingerprint is inconsistent")


def inspect_checkpoint(spec: ModelSpec, device: str) -> tuple[Any, dict[str, Any]]:
    """Validate a local full/PEFT checkpoint and return its PI0.5 config."""
    from lerobot.configs import PreTrainedConfig
    from lerobot.policies import PI05Config

    checkpoint = spec.checkpoint.resolve()
    required = (
        checkpoint / "config.json",
        checkpoint / "policy_preprocessor.json",
        checkpoint / "policy_postprocessor.json",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing checkpoint assets:\n" + "\n".join(missing))

    is_adapter = (checkpoint / "adapter_config.json").is_file()
    base_model = None
    if is_adapter:
        from peft import PeftConfig

        if not (
            (checkpoint / "adapter_model.safetensors").is_file()
            or (checkpoint / "adapter_model.bin").is_file()
        ):
            raise FileNotFoundError(f"Missing PEFT adapter weights in {checkpoint}")
        peft_config = PeftConfig.from_pretrained(str(checkpoint), local_files_only=True)
        if not peft_config.base_model_name_or_path:
            raise ValueError(f"PEFT adapter {checkpoint} does not identify a base model")
        base_model = Path(peft_config.base_model_name_or_path).expanduser().resolve()
        if not (base_model / "config.json").is_file() or not _has_model_weights(base_model):
            raise FileNotFoundError(f"PEFT base model is not complete and local: {base_model}")
    elif not _has_model_weights(checkpoint):
        raise FileNotFoundError(f"Missing full model weights in {checkpoint}")

    config = PreTrainedConfig.from_pretrained(checkpoint, local_files_only=True)
    if not isinstance(config, PI05Config) or config.type != "pi05":
        raise ValueError(f"{checkpoint} is policy type {config.type!r}, expected 'pi05'")
    config.device = device
    config.gradient_checkpointing = False
    config.compile_model = False
    config.rtc_config = None
    descriptor = {
        "label": spec.label,
        "checkpoint": str(checkpoint),
        "checkpoint_kind": "peft_adapter" if is_adapter else "full",
        "base_model": str(base_model) if base_model is not None else None,
        "processor_source": str(checkpoint),
        "policy_type": config.type,
        "chunk_size": int(config.chunk_size),
        "max_action_dim": int(config.max_action_dim),
        "action_dim": int(config.output_features["action"].shape[0]),
        "num_inference_steps": int(config.num_inference_steps),
        "use_relative_actions": bool(config.use_relative_actions),
        "checkpoint_asset_fingerprint": fingerprint_checkpoint_assets(checkpoint, base_model),
    }
    return config, descriptor


def load_policy_and_processors(spec: ModelSpec, config: Any, device: str):
    """Load either a full PI0.5 checkpoint or a local PEFT base+adapter."""
    from lerobot.policies import get_policy_class, make_pre_post_processors

    checkpoint = spec.checkpoint.resolve()
    policy_class = get_policy_class(config.type)
    if (checkpoint / "adapter_config.json").is_file():
        from peft import PeftConfig, PeftModel

        peft_config = PeftConfig.from_pretrained(str(checkpoint), local_files_only=True)
        policy = policy_class.from_pretrained(
            pretrained_name_or_path=peft_config.base_model_name_or_path,
            config=config,
            local_files_only=True,
            strict=True,
        )
        policy = PeftModel.from_pretrained(
            policy,
            str(checkpoint),
            config=peft_config,
            is_trainable=False,
            local_files_only=True,
        )
    else:
        policy = policy_class.from_pretrained(
            checkpoint,
            config=config,
            local_files_only=True,
            strict=True,
        )
    policy = policy.to(device).eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=config,
        pretrained_path=str(checkpoint),
        preprocessor_overrides={"device_processor": {"device": device}},
    )
    return policy, preprocessor, postprocessor


def _stack_dataset_column(column: Any, *, dtype: torch.dtype | None = None) -> torch.Tensor:
    values = [torch.as_tensor(value) for value in column]
    if not values:
        raise ValueError("dataset column is empty")
    result = torch.stack(values)
    return result.to(dtype=dtype) if dtype is not None else result


def build_dataset_index(dataset: Any) -> DatasetIndex:
    """Materialize only small numeric columns needed for selection and GT."""
    hf_dataset = dataset.hf_dataset
    actions = _stack_dataset_column(hf_dataset["action"], dtype=torch.float32)
    states = _stack_dataset_column(hf_dataset["observation.state"], dtype=torch.float32)
    episode_indices = _stack_dataset_column(hf_dataset["episode_index"], dtype=torch.int64).flatten()
    frame_indices = _stack_dataset_column(hf_dataset["frame_index"], dtype=torch.int64).flatten()
    absolute_indices = _stack_dataset_column(hf_dataset["index"], dtype=torch.int64).flatten()
    if not (
        len(actions) == len(states) == len(episode_indices) == len(frame_indices) == len(absolute_indices)
    ):
        raise ValueError("dataset numeric columns have inconsistent lengths")

    rows_by_episode: dict[int, list[int]] = defaultdict(list)
    for row_index, episode_index in enumerate(episode_indices.tolist()):
        rows_by_episode[int(episode_index)].append(row_index)
    for rows in rows_by_episode.values():
        rows.sort(key=lambda row_index: int(frame_indices[row_index]))

    action_names = list(dataset.meta.features["action"].get("names") or [])
    state_names = list(dataset.meta.features["observation.state"].get("names") or [])
    if len(action_names) != actions.shape[1] or len(state_names) != states.shape[1]:
        raise ValueError("dataset action/state feature names are missing or dimensionally inconsistent")
    resolve_state_indices(action_names, state_names)
    return DatasetIndex(
        actions=actions,
        states=states,
        episode_indices=episode_indices,
        frame_indices=frame_indices,
        absolute_indices=absolute_indices,
        rows_by_episode=dict(rows_by_episode),
        action_names=action_names,
        state_names=state_names,
    )


def build_frame_selection(
    dataset_index: DatasetIndex,
    *,
    episodes: list[int] | None,
    gripper_index: int,
    gripper_event_threshold: float,
    gripper_neighbor_radius: int,
    frame_limit: int | None,
) -> list[SelectedFrame]:
    available = sorted(dataset_index.rows_by_episode)
    selected_episodes = available if episodes is None else episodes
    if len(set(selected_episodes)) != len(selected_episodes):
        raise ValueError("episodes must be unique")
    missing = sorted(set(selected_episodes) - set(available))
    if missing:
        raise IndexError(f"dataset does not contain episodes: {missing}")

    frames = []
    for episode_index in selected_episodes:
        rows = dataset_index.rows_by_episode[episode_index]
        frames.extend(
            select_episode_frames(
                episode_index=episode_index,
                row_indices=rows,
                frame_indices=dataset_index.frame_indices,
                absolute_indices=dataset_index.absolute_indices,
                episode_actions=dataset_index.actions[rows],
                gripper_index=gripper_index,
                gripper_event_threshold=gripper_event_threshold,
                gripper_neighbor_radius=gripper_neighbor_radius,
            )
        )
    return limit_frames_preserving_phase_anchors(
        sorted(frames, key=lambda item: item.row_index),
        frame_limit,
    )


def evaluate_formal_selection_coverage(
    frames: list[SelectedFrame],
    seeds: list[int],
    episode_lengths: dict[int, int],
) -> dict[str, Any]:
    """Validate the immutable coverage contract for a formal model gate."""
    phase_names = {stage for stage, _ in PHASE_FRACTIONS}
    episode_indices = sorted({frame.episode_index for frame in frames})
    anchors_by_episode = {
        episode_index: {
            stage
            for frame in frames
            if frame.episode_index == episode_index
            for stage in frame.stages & phase_names
        }
        for episode_index in episode_indices
    }
    incomplete_episodes = {
        str(episode_index): sorted(phase_names - stages)
        for episode_index, stages in anchors_by_episode.items()
        if stages != phase_names
    }
    future_state_stages = phase_names - {"end"}
    missing_episode_lengths = sorted(set(episode_indices) - set(episode_lengths))
    incomplete_future_state_anchors = {}
    for episode_index in episode_indices:
        episode_length = episode_lengths.get(episode_index)
        if episode_length is None:
            continue
        eligible_stages = {
            stage
            for frame in frames
            if frame.episode_index == episode_index and episode_length - frame.local_index >= 14
            for stage in frame.stages & future_state_stages
        }
        missing_stages = future_state_stages - eligible_stages
        if missing_stages:
            incomplete_future_state_anchors[str(episode_index)] = sorted(missing_stages)
    checks = {
        "episode_count": {
            "value": len(episode_indices),
            "expected": 40,
            "passed": len(episode_indices) == 40,
        },
        "seven_phase_anchors_per_episode": {
            "incomplete_episodes": incomplete_episodes,
            "passed": len(episode_indices) == 40 and not incomplete_episodes,
        },
        "six_action13_eligible_anchors_per_episode": {
            "required_stages": sorted(future_state_stages),
            "missing_episode_lengths": missing_episode_lengths,
            "incomplete_episodes": incomplete_future_state_anchors,
            "passed": len(episode_indices) == 40
            and not missing_episode_lengths
            and not incomplete_future_state_anchors,
        },
        "frame_count": {
            "value": len(frames),
            "minimum": 400,
            "passed": len(frames) >= 400,
        },
        "default_seed_set": {
            "value": seeds,
            "expected": list(DEFAULT_SEEDS),
            "passed": len(seeds) == len(DEFAULT_SEEDS)
            and len(set(seeds)) == len(seeds)
            and set(seeds) == set(DEFAULT_SEEDS),
        },
    }
    return {
        "passed": all(check["passed"] for check in checks.values()),
        "checks": checks,
    }


def prediction_record_to_json(record: PredictionRecord, joint_names: list[str]) -> dict[str, Any]:
    action13 = record.action13_errors
    return {
        "mae": float(record.errors.mean()),
        "mae_by_joint": {
            joint_name: float(record.errors[:, index].mean()) for index, joint_name in enumerate(joint_names)
        },
        "action0_abs_error": float(record.action0_errors.mean()),
        "action0_abs_error_by_joint": {
            joint_name: float(record.action0_errors[index]) for index, joint_name in enumerate(joint_names)
        },
        "action13_abs_error": float(action13.mean()) if action13 is not None else None,
        "action13_abs_error_by_joint": (
            {joint_name: float(action13[index]) for index, joint_name in enumerate(joint_names)}
            if action13 is not None
            else None
        ),
        "action0_state_jump": float(record.action0_state_jumps.mean()),
        "action0_state_jump_p95": float(torch.quantile(record.action0_state_jumps, 0.95)),
        "action0_state_jump_max": float(record.action0_state_jumps.max()),
        "action0_state_jump_by_joint": {
            joint_name: float(record.action0_state_jumps[index])
            for index, joint_name in enumerate(joint_names)
        },
        "action13_state_jump": (
            float(record.action13_state_jumps.mean()) if record.action13_state_jumps is not None else None
        ),
        "action13_state_jump_p95": (
            float(torch.quantile(record.action13_state_jumps, 0.95))
            if record.action13_state_jumps is not None
            else None
        ),
        "action13_state_jump_max": (
            float(record.action13_state_jumps.max()) if record.action13_state_jumps is not None else None
        ),
        "action13_state_jump_by_joint": (
            {
                joint_name: float(record.action13_state_jumps[index])
                for index, joint_name in enumerate(joint_names)
            }
            if record.action13_state_jumps is not None
            else None
        ),
        "action13_state_jump_diagnostic_only": True,
        "action13_state_jump_semantics": "legacy_action13_vs_query_state",
        "action13_execution_state_jump": (
            float(record.action13_execution_state_jumps.mean())
            if record.action13_execution_state_jumps is not None
            else None
        ),
        "action13_execution_state_jump_max": (
            float(record.action13_execution_state_jumps.max())
            if record.action13_execution_state_jumps is not None
            else None
        ),
        "action13_execution_state_jump_by_joint": (
            {
                joint_name: float(record.action13_execution_state_jumps[index])
                for index, joint_name in enumerate(joint_names)
            }
            if record.action13_execution_state_jumps is not None
            else None
        ),
        "first_action_clamp_predicted": bool(record.first_action_clamp.any()),
        "first_action_clamp_diagnostic_only": True,
        "first_action_clamp_by_joint": {
            joint_name: bool(record.first_action_clamp[index]) for index, joint_name in enumerate(joint_names)
        },
        "action0_predicted_only_clamp": bool(record.action0_predicted_only_clamp.any()),
        "action0_predicted_only_clamp_by_joint": {
            joint_name: bool(record.action0_predicted_only_clamp[index])
            for index, joint_name in enumerate(joint_names)
        },
        "action13_execution_clamp_predicted": (
            bool(record.action13_execution_state_jump_exceeds.any())
            if record.action13_execution_state_jump_exceeds is not None
            else None
        ),
        "action13_predicted_only_clamp": (
            bool(record.action13_predicted_only_clamp.any())
            if record.action13_predicted_only_clamp is not None
            else None
        ),
        "action0_clamp_residual_max": float(record.action0_clamp_residuals.max()),
        "action0_excess_clamp_residual_max": float(record.action0_excess_clamp_residuals.max()),
        "action13_execution_clamp_residual_max": (
            float(record.action13_execution_clamp_residuals.max())
            if record.action13_execution_clamp_residuals is not None
            else None
        ),
        "action13_excess_clamp_residual_max": (
            float(record.action13_excess_clamp_residuals.max())
            if record.action13_excess_clamp_residuals is not None
            else None
        ),
        "chunk_jump_max": float(record.jumps.max()) if record.jumps.numel() else None,
        "gt_gripper_event_step": record.gt_gripper_event,
        "predicted_gripper_event_step": record.predicted_gripper_event,
        "gt_gripper_direction": record.gt_gripper_direction,
        "predicted_gripper_direction": record.predicted_gripper_direction,
        "gripper_direction_matches": (
            record.gt_gripper_direction == record.predicted_gripper_direction
            if record.gt_gripper_direction is not None and record.predicted_gripper_direction is not None
            else None
        ),
        "gripper_timing_abs_error_steps": (
            abs(record.predicted_gripper_event - record.gt_gripper_event)
            if record.gt_gripper_event is not None and record.predicted_gripper_event is not None
            else None
        ),
    }


def evaluate_model(
    *,
    spec: ModelSpec,
    config: Any,
    descriptor: dict[str, Any],
    dataset: Any,
    dataset_index: DatasetIndex,
    frames: list[SelectedFrame],
    noise_bank: dict[int, torch.Tensor],
    seeds: list[int],
    device: str,
    chunk_size: int,
    gripper_index: int,
    max_relative_target: float,
    gripper_event_threshold: float,
    thresholds: GateThresholds,
    run_mode: str = "formal",
    formal_selection_coverage: dict[str, Any] | None = None,
    formal_gate_eligible: bool = True,
    baseline_model_report: dict[str, Any] | None = None,
    baseline_rate_margin: float = 0.05,
    baseline_error_ratio: float = 2.0,
) -> dict[str, Any]:
    started_at = datetime.now(UTC)
    policy, preprocessor, postprocessor = load_policy_and_processors(spec, config, device)
    accumulators: dict[str, MetricsAccumulator] = {
        "overall": MetricsAccumulator(dataset_index.action_names, max_relative_target)
    }
    frame_reports = []
    seed_maes: dict[int, list[float]] = {seed: [] for seed in seeds}
    frame_maes: list[dict[str, Any]] = []
    finite_predictions = 0
    nonfinite_action_values = 0

    try:
        progress_interval = max(1, len(frames) // 10)
        for frame_number, frame in enumerate(frames, start=1):
            rows = dataset_index.rows_by_episode[frame.episode_index]
            if not 0 <= frame.local_index < len(rows) or rows[frame.local_index] != frame.row_index:
                raise ValueError("selected frame does not match its same-episode row index")
            episode_actions = dataset_index.actions[rows]
            episode_states = dataset_index.states[rows]
            ground_truth = extract_ground_truth_chunk(
                episode_actions,
                frame.local_index,
                chunk_size=chunk_size,
            )
            execution_states = extract_execution_state_chunk(
                episode_states,
                frame.local_index,
                action_names=dataset_index.action_names,
                state_names=dataset_index.state_names,
                chunk_size=chunk_size,
            )
            if len(execution_states) != len(ground_truth):
                raise ValueError("same-episode action and execution-state horizons differ")
            dataset_safety = dataset_safety_frame_to_json(
                ground_truth,
                execution_states,
                dataset_index.action_names,
                max_relative_target,
            )

            preprocessor.reset()
            postprocessor.reset()
            policy.reset()
            raw_sample = torch.utils.data.default_collate([dataset[frame.row_index]])
            processed_sample = preprocessor(raw_sample)

            predictions = []
            records = []
            per_seed = {}
            with torch.inference_mode():
                for seed in seeds:
                    noise = noise_bank[seed].to(device=device).clone()
                    normalized_actions = policy.predict_action_chunk(
                        processed_sample,
                        noise=noise,
                    )
                    normalized_nonfinite_count = int((~torch.isfinite(normalized_actions)).sum())
                    if normalized_nonfinite_count:
                        nonfinite_action_values += normalized_nonfinite_count
                        per_seed[str(seed)] = {
                            "nonfinite_action_values": normalized_nonfinite_count,
                            "nonfinite_space": "normalized",
                            "metrics_available": False,
                        }
                        continue
                    robot_actions = postprocessor(normalized_actions).squeeze(0).detach().float().cpu()
                    if robot_actions.shape != (chunk_size, len(dataset_index.action_names)):
                        raise ValueError(
                            f"{spec.label} returned robot action shape {tuple(robot_actions.shape)}, "
                            f"expected {(chunk_size, len(dataset_index.action_names))}"
                        )
                    nonfinite_count = int((~torch.isfinite(robot_actions)).sum())
                    if nonfinite_count:
                        nonfinite_action_values += nonfinite_count
                        per_seed[str(seed)] = {
                            "nonfinite_action_values": nonfinite_count,
                            "nonfinite_space": "robot_units",
                            "metrics_available": False,
                        }
                        continue
                    record = compute_prediction_record(
                        robot_actions,
                        ground_truth,
                        execution_states,
                        gripper_index=gripper_index,
                        max_relative_target=max_relative_target,
                        gripper_event_threshold=gripper_event_threshold,
                    )
                    predictions.append(robot_actions)
                    records.append(record)
                    finite_predictions += 1
                    seed_maes[seed].append(float(record.errors.mean()))
                    per_seed[str(seed)] = prediction_record_to_json(record, dataset_index.action_names)

            valid_steps = len(ground_truth)
            seed_std = (
                torch.stack(predictions).std(dim=0, unbiased=False)[:valid_steps]
                if len(predictions) == len(seeds)
                else None
            )
            stage_names = {"overall", *frame.stages}
            for stage_name in stage_names:
                accumulator = accumulators.setdefault(
                    stage_name,
                    MetricsAccumulator(dataset_index.action_names, max_relative_target),
                )
                accumulator.add_dataset_safety(ground_truth, execution_states)
                for record in records:
                    accumulator.add_prediction(record)
                if seed_std is not None:
                    accumulator.add_seed_std(seed_std)

            if records:
                frame_maes.append(
                    {
                        "episode_index": frame.episode_index,
                        "frame_index": frame.frame_index,
                        "absolute_index": frame.absolute_index,
                        "mae": statistics.fmean(float(record.errors.mean()) for record in records),
                    }
                )

            frame_reports.append(
                {
                    **frame.to_json(),
                    "valid_gt_steps": valid_steps,
                    "future_state_valid_steps": len(execution_states),
                    "tail_padding_ignored_steps": chunk_size - valid_steps,
                    "action13_eligible": valid_steps >= 14,
                    "action13_target_frame_index": (
                        int(dataset_index.frame_indices[rows[frame.local_index + 13]])
                        if valid_steps >= 14
                        else None
                    ),
                    "action13_ineligible_reason": None if valid_steps >= 14 else "episode_tail",
                    "dataset_safety": dataset_safety,
                    "seed_std_mean": float(seed_std.mean()) if seed_std is not None else None,
                    "seed_std_by_joint": (
                        {
                            joint_name: float(seed_std[:, index].mean())
                            for index, joint_name in enumerate(dataset_index.action_names)
                        }
                        if seed_std is not None
                        else None
                    ),
                    "seeds": per_seed,
                }
            )
            if frame_number % progress_interval == 0 or frame_number == len(frames):
                elapsed_seconds = (datetime.now(UTC) - started_at).total_seconds()
                print(
                    f"{spec.label}: progress={frame_number}/{len(frames)} "
                    f"finite_predictions={finite_predictions} elapsed={elapsed_seconds:.1f}s",
                    flush=True,
                )
    finally:
        del policy
        if device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()

    stage_reports = {}
    for stage_name, accumulator in sorted(accumulators.items()):
        metrics = accumulator.summarize()
        stage_reports[stage_name] = {
            "metrics": metrics,
            "gate": evaluate_gate(
                metrics,
                thresholds,
                allow_unavailable=run_mode == "quick_smoke",
                include_predicted_only_rates=run_mode == "formal",
            ),
        }
    robustness = build_robustness_summary(
        seed_maes=seed_maes,
        frame_maes=frame_maes,
        stage_reports=stage_reports,
    )
    robustness_gate = evaluate_robustness_gate(
        robustness,
        thresholds,
        include_stage_extrema=run_mode == "formal",
    )
    formal_stage_safety_gate = (
        evaluate_formal_stage_safety_gate(
            stage_reports,
            minimum_predictions_per_stage=len(seeds),
        )
        if run_mode == "formal"
        else {
            "passed": True,
            "evaluated": False,
            "reason": "quick_smoke does not have formal stage coverage",
        }
    )
    model_coverage = evaluate_model_coverage(
        summary=stage_reports["overall"]["metrics"],
        expected_predictions=len(frames) * len(seeds),
        finite_predictions=finite_predictions,
        nonfinite_action_values=nonfinite_action_values,
        seed_count=len(seeds),
        min_action13_coverage_rate=thresholds.min_action13_coverage_rate,
        require_semantic_coverage=run_mode == "formal",
    )
    relative_gate = (
        evaluate_relative_baseline_gate(
            stage_reports["overall"]["metrics"],
            baseline_model_report,
            thresholds,
            rate_margin=baseline_rate_margin,
            error_ratio=baseline_error_ratio,
        )
        if baseline_model_report is not None
        else None
    )
    absolute_passed = (
        stage_reports["overall"]["gate"]["passed"]
        and robustness_gate["passed"]
        and formal_stage_safety_gate["passed"]
    )
    smoke_passed = absolute_passed and model_coverage["passed"]
    formal_passed = (
        None
        if run_mode == "quick_smoke"
        else bool(
            smoke_passed
            and formal_gate_eligible
            and formal_selection_coverage is not None
            and formal_selection_coverage["passed"]
            and (relative_gate is None or relative_gate["passed"])
        )
    )
    completed_at = datetime.now(UTC)
    return {
        **descriptor,
        "seeds": seeds,
        "frames_evaluated": len(frames),
        "stages": stage_reports,
        "robustness": robustness,
        "robustness_gate": robustness_gate,
        "formal_stage_safety_gate": formal_stage_safety_gate,
        "model_coverage": model_coverage,
        "relative_baseline_gate": relative_gate,
        "frames": frame_reports,
        "metric_semantics": METRIC_SEMANTICS,
        "formal_gate_eligible": formal_gate_eligible,
        "smoke_passed": smoke_passed,
        "formal_passed": formal_passed,
        "passed": formal_passed,
        "nonfinite_action_values": nonfinite_action_values,
        "duration_seconds": (completed_at - started_at).total_seconds(),
        "hardware_action_sent": False,
    }


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(report, indent=2, ensure_ascii=True, allow_nan=False)
    temporary_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary_path.write_text(serialized, encoding="utf-8")
    os.replace(temporary_path, path)


def gate_exit_code(*, run_mode: str, smoke_passed: bool, formal_passed: bool | None) -> int:
    """Return zero only when the active smoke/formal gate passed."""
    succeeded = smoke_passed if run_mode == "quick_smoke" else formal_passed is True
    return 0 if succeeded else 1


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(args)
    seeds, frame_limit = resolve_run_shape(args)
    dataset_root = args.dataset_root.expanduser().resolve()
    if not (dataset_root / "meta/info.json").is_file():
        raise FileNotFoundError(f"Local LeRobot dataset is missing meta/info.json: {dataset_root}")
    dataset_asset_fingerprint = fingerprint_dataset_assets(dataset_root)

    from lerobot.datasets import LeRobotDataset

    dataset = LeRobotDataset(
        repo_id=args.repo_id,
        root=dataset_root,
        download_videos=False,
        video_backend=args.video_backend,
    )
    dataset_index = build_dataset_index(dataset)
    try:
        gripper_index = dataset_index.action_names.index(args.gripper_joint)
    except ValueError as exc:
        raise ValueError(
            f"gripper joint {args.gripper_joint!r} is not in dataset actions: {dataset_index.action_names}"
        ) from exc

    frames = build_frame_selection(
        dataset_index,
        episodes=args.episodes,
        gripper_index=gripper_index,
        gripper_event_threshold=args.gripper_event_threshold,
        gripper_neighbor_radius=args.gripper_neighbor_radius,
        frame_limit=frame_limit,
    )
    if not frames:
        raise RuntimeError("frame selection produced no samples")
    run_mode = "quick_smoke" if args.quick else "formal"
    formal_selection_coverage = (
        {"passed": False, "evaluated": False, "reason": "quick_smoke is never a formal gate"}
        if args.quick
        else evaluate_formal_selection_coverage(
            frames,
            seeds,
            {episode: len(rows) for episode, rows in dataset_index.rows_by_episode.items()},
        )
    )
    output_path = args.output_json.expanduser().resolve()
    if not args.quick and not formal_selection_coverage["passed"]:
        coverage_failure_report = {
            "schema_version": REPORT_SCHEMA_VERSION,
            "report_status": "coverage_failure",
            "created_at": datetime.now(UTC).isoformat(),
            "run_mode": run_mode,
            "metric_semantics": METRIC_SEMANTICS,
            "dataset": {
                "repo_id": args.repo_id,
                "root": str(dataset_root),
                "episodes": args.episodes or sorted(dataset_index.rows_by_episode),
                "action_names": dataset_index.action_names,
                "state_names": dataset_index.state_names,
                "asset_fingerprint": dataset_asset_fingerprint,
            },
            "selection": {
                "selected_frame_count": len(frames),
                "gripper_joint": args.gripper_joint,
                "gripper_event_threshold": args.gripper_event_threshold,
                "gripper_neighbor_radius": args.gripper_neighbor_radius,
                "max_relative_target": args.max_relative_target,
                "frames": [frame.to_json() for frame in frames],
            },
            "noise": {"seeds": seeds},
            "formal_selection_coverage": formal_selection_coverage,
            "smoke_passed": False,
            "formal_passed": False,
            "passed": False,
            "hardware_action_sent": False,
        }
        write_report(output_path, coverage_failure_report)
        print(
            json.dumps(
                {
                    "formal_passed": False,
                    "reason": "formal selection coverage failed",
                    "coverage": formal_selection_coverage,
                    "output_json": str(output_path),
                    "hardware_action_sent": False,
                },
                indent=2,
            )
        )
        return 2

    inspected_models = []
    expected_shape = None
    for spec in args.models:
        resolved_spec = ModelSpec(spec.label, spec.checkpoint.resolve())
        config, descriptor = inspect_checkpoint(resolved_spec, args.device)
        shape = (int(config.chunk_size), int(config.max_action_dim))
        if expected_shape is None:
            expected_shape = shape
        elif shape != expected_shape:
            raise ValueError(
                f"all models must share chunk/max-action dimensions; got {shape} and {expected_shape}"
            )
        if int(config.chunk_size) != args.chunk_size:
            raise ValueError(
                f"{resolved_spec.label} chunk_size={config.chunk_size}, expected {args.chunk_size}"
            )
        if int(config.output_features["action"].shape[0]) != len(dataset_index.action_names):
            raise ValueError(f"{resolved_spec.label} action dimension does not match the dataset")
        config_names = list(config.action_feature_names or [])
        if config_names and config_names != dataset_index.action_names:
            raise ValueError(
                f"{resolved_spec.label} action feature order differs from dataset: "
                f"{config_names} != {dataset_index.action_names}"
            )
        inspected_models.append((resolved_spec, config, descriptor))

    noise_bank = make_shared_noise(
        seeds,
        chunk_size=expected_shape[0],
        max_action_dim=expected_shape[1],
    )
    thresholds = GateThresholds(
        max_mae=args.max_mae,
        max_mae_p95=args.max_mae_p95,
        max_mae_max=args.max_mae_max,
        max_seed_std=args.max_seed_std,
        max_seed_std_p95=args.max_seed_std_p95,
        max_seed_std_max=args.max_seed_std_max,
        max_worst_frame_mae=args.max_worst_frame_mae,
        max_worst_seed_mae=args.max_worst_seed_mae,
        max_action0_error=args.max_action0_error,
        max_action0_error_p95=args.max_action0_error_p95,
        max_action0_error_max=args.max_action0_error_max,
        max_action13_error=args.max_action13_error,
        max_action13_error_p95=args.max_action13_error_p95,
        max_action13_error_max=args.max_action13_error_max,
        max_action0_predicted_only_clamp_rate=args.max_action0_predicted_only_clamp_rate,
        max_action13_predicted_only_clamp_rate=args.max_action13_predicted_only_clamp_rate,
        max_chunk_jump_rate=args.max_chunk_jump_rate,
        min_action13_coverage_rate=args.min_action13_coverage_rate,
        max_gripper_timing_error_steps=args.max_gripper_timing_error_steps,
        max_gripper_event_miss_rate=args.max_gripper_event_miss_rate,
        max_gripper_false_positive_rate=args.max_gripper_false_positive_rate,
        min_gripper_direction_consistency=args.min_gripper_direction_consistency,
    )
    formal_gate_eligible = (
        thresholds == GateThresholds()
        and args.max_relative_target == 5.0
        and args.gripper_event_threshold == 1.0
        and args.gripper_neighbor_radius == 2
        and args.max_baseline_rate_regression == 0.05
        and args.max_baseline_error_ratio == 2.0
    )
    gate_profile = {
        "name": DEPLOYMENT_GATE_PROFILE if formal_gate_eligible else "experimental",
        "formal_eligible": formal_gate_eligible,
        "default_thresholds": asdict(GateThresholds()),
    }
    baseline_model_report = (
        load_baseline_model_report(
            args.baseline_json,
            label=args.baseline_label,
            dataset_root=dataset_root,
            repo_id=args.repo_id,
            dataset_asset_fingerprint=dataset_asset_fingerprint,
            action_names=dataset_index.action_names,
            state_names=dataset_index.state_names,
            seeds=seeds,
            frames=frames,
            selection_contract={
                "quick": False,
                "frame_limit": frame_limit,
                "selected_frame_count": len(frames),
                "phase_fractions": dict(PHASE_FRACTIONS),
                "gripper_joint": args.gripper_joint,
                "gripper_event_threshold": args.gripper_event_threshold,
                "gripper_neighbor_radius": args.gripper_neighbor_radius,
                "max_relative_target": args.max_relative_target,
            },
            thresholds=thresholds,
        )
        if args.baseline_json is not None
        else None
    )
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "report_status": "running",
        "created_at": datetime.now(UTC).isoformat(),
        "run_mode": run_mode,
        "dataset": {
            "repo_id": args.repo_id,
            "root": str(dataset_root),
            "fps": dataset.fps,
            "episodes": args.episodes or sorted(dataset_index.rows_by_episode),
            "action_names": dataset_index.action_names,
            "state_names": dataset_index.state_names,
            "asset_fingerprint": dataset_asset_fingerprint,
        },
        "metric_semantics": METRIC_SEMANTICS,
        "selection": {
            "quick": args.quick,
            "frame_limit": frame_limit,
            "selected_frame_count": len(frames),
            "phase_fractions": dict(PHASE_FRACTIONS),
            "gripper_joint": args.gripper_joint,
            "gripper_event_threshold": args.gripper_event_threshold,
            "gripper_neighbor_radius": args.gripper_neighbor_radius,
            "max_relative_target": args.max_relative_target,
            "frames": [frame.to_json() for frame in frames],
        },
        "formal_selection_coverage": formal_selection_coverage,
        "noise": {
            "seeds": seeds,
            "shape": [1, expected_shape[0], expected_shape[1]],
            "pre_generated_on_cpu": True,
            "shared_across_models_and_frames": True,
            "passed_explicitly_to_predict_action_chunk": True,
        },
        "thresholds": asdict(thresholds),
        "gate_profile": gate_profile,
        "relative_baseline": {
            "enabled": baseline_model_report is not None,
            "report": str(args.baseline_json.expanduser().resolve())
            if args.baseline_json is not None
            else None,
            "model_label": args.baseline_label if baseline_model_report is not None else None,
            "max_rate_regression": args.max_baseline_rate_regression,
            "max_error_ratio": args.max_baseline_error_ratio,
        },
        "models": {},
        "expected_models": [spec.label for spec, _, _ in inspected_models],
        "completed_models": [],
        "smoke_passed": False,
        "formal_passed": None if args.quick else False,
        "passed": None if args.quick else False,
        "hardware_action_sent": False,
    }
    write_report(output_path, report)

    try:
        for spec, config, descriptor in inspected_models:
            print(f"Evaluating {spec.label}: {spec.checkpoint}", flush=True)
            model_report = evaluate_model(
                spec=spec,
                config=config,
                descriptor=descriptor,
                dataset=dataset,
                dataset_index=dataset_index,
                frames=frames,
                noise_bank=noise_bank,
                seeds=seeds,
                device=args.device,
                chunk_size=args.chunk_size,
                gripper_index=gripper_index,
                max_relative_target=args.max_relative_target,
                gripper_event_threshold=args.gripper_event_threshold,
                thresholds=thresholds,
                run_mode=run_mode,
                formal_selection_coverage=formal_selection_coverage,
                formal_gate_eligible=formal_gate_eligible,
                baseline_model_report=baseline_model_report,
                baseline_rate_margin=args.max_baseline_rate_regression,
                baseline_error_ratio=args.max_baseline_error_ratio,
            )
            report["models"][spec.label] = model_report
            report["completed_models"].append(spec.label)
            write_report(output_path, report)
            print(
                f"{spec.label}: smoke_passed={model_report['smoke_passed']} "
                f"formal_passed={model_report['formal_passed']} report={output_path}",
                flush=True,
            )
    except BaseException as exc:
        report["report_status"] = "failed"
        report["failure"] = {"type": type(exc).__name__, "message": str(exc)}
        report["smoke_passed"] = False
        report["formal_passed"] = None if args.quick else False
        report["passed"] = None if args.quick else False
        write_report(output_path, report)
        raise

    report["report_status"] = "complete"
    report["smoke_passed"] = all(
        report["models"][label]["smoke_passed"] for label in report["expected_models"]
    )
    report["formal_passed"] = (
        None
        if args.quick
        else all(report["models"][label]["formal_passed"] for label in report["expected_models"])
    )
    report["passed"] = report["formal_passed"]
    write_report(output_path, report)

    print(
        json.dumps(
            {
                "run_mode": run_mode,
                "smoke_passed": report["smoke_passed"],
                "formal_passed": report["formal_passed"],
                "passed": report["formal_passed"],
                "models": {
                    label: {
                        "smoke_passed": model_report["smoke_passed"],
                        "formal_passed": model_report["formal_passed"],
                    }
                    for label, model_report in report["models"].items()
                },
                "output_json": str(output_path),
                "hardware_action_sent": False,
            },
            indent=2,
        )
    )
    return gate_exit_code(
        run_mode=run_mode,
        smoke_passed=report["smoke_passed"],
        formal_passed=report["formal_passed"],
    )


if __name__ == "__main__":
    raise SystemExit(main())
