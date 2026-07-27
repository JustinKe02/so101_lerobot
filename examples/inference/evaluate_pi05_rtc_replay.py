#!/usr/bin/env python

"""Offline deployment-window replay for PI0.5 RTC.

This evaluator uses the production ``ActionQueue`` merge semantics with
injected control-step delays. It only reads a local dataset and checkpoint;
it never constructs or connects a robot.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.inference.evaluate_pi05_multiseed import (  # noqa: E402
    DEFAULT_DATASET_ROOT,
    DEFAULT_REPO_ID,
    DatasetIndex,
    ModelSpec,
    build_dataset_index,
    fingerprint_checkpoint_assets,
    fingerprint_dataset_assets,
    first_gripper_event,
    inspect_checkpoint,
    load_policy_and_processors,
    scalar_stats,
    write_report,
)
from lerobot.configs import RTCAttentionSchedule  # noqa: E402
from lerobot.policies.rtc import ActionQueue, reanchor_relative_rtc_prefix  # noqa: E402
from lerobot.policies.rtc.configuration_rtc import RTCConfig  # noqa: E402
from lerobot.processor import NormalizerProcessorStep, RelativeActionsProcessorStep  # noqa: E402
from lerobot.rollout.action_filter import (  # noqa: E402
    DEFAULT_MAX_ACCELERATION,
    DEFAULT_MAX_VELOCITY,
    SecondOrderActionFilter,
    build_limit_tensor,
)

DEFAULT_ROOT = Path("/data/cqy_workspace/tk/lerobot_src")
DEFAULT_MODEL = (
    DEFAULT_ROOT / "outputs/train/pi05_so101_local_10epochs_bs32/checkpoints/005613/pretrained_model"
)
DEFAULT_OUTPUT = DEFAULT_ROOT / "outputs/eval/pi05_full_rtc_replay_v1.json"
DEFAULT_ABSOLUTE_INDICES = (45, 15715, 4984, 3255, 15760, 13066)
DEFAULT_SEEDS = (0, 1, 3, 4, 6, 42, 1000)
DEFAULT_QUEUE_THRESHOLDS = (20, 30, 35, 40, 45)
DEFAULT_ACTUAL_CONSUMED_STEPS = (4, 5)
REPORT_SCHEMA_VERSION = 1


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class ReplayThresholds:
    max_mae_mean: float = 2.0
    max_mae_p95: float = 5.0
    max_error: float = 20.0
    max_predicted_only_clamp_rate: float = 0.05
    max_excess_clamp_residual: float = 5.0
    max_splice_jump: float = 5.0
    max_command_delta: float = 5.0
    max_repeat_error: float = 1e-6
    max_latency_p95_steps: float = 5.0
    max_latency_steps: float = 9.0


@dataclass
class QueueTransitionStart:
    queue: ActionQueue
    inference_start: Any
    initial_merge: Any
    actions_before_inference: torch.Tensor
    replan_interval_steps: int
    execution_horizon: int


@dataclass
class QueueTransitionTrace:
    initial_merge: Any
    transition_merge: Any
    actions_before_inference: torch.Tensor
    old_actions_during_inference: torch.Tensor
    executed_current_actions: torch.Tensor
    executed_model_indices: list[int]
    queue_size_after_execution: int
    underflow_count: int


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--video-backend", default="pyav")
    parser.add_argument("--absolute-indices", type=int, nargs="+", default=DEFAULT_ABSOLUTE_INDICES)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument(
        "--queue-thresholds",
        type=int,
        nargs="+",
        default=DEFAULT_QUEUE_THRESHOLDS,
    )
    parser.add_argument("--guidance-delay", type=int, default=5)
    parser.add_argument(
        "--actual-consumed-steps",
        "--delays",
        dest="actual_consumed_steps",
        type=int,
        nargs="+",
        default=DEFAULT_ACTUAL_CONSUMED_STEPS,
    )
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument("--execution-horizon", type=int, default=10)
    parser.add_argument("--max-guidance-weight", type=float, default=10.0)
    parser.add_argument(
        "--disable-guidance",
        action="store_true",
        help="Generate the replacement chunk without an RTC prefix for prefix-health diagnostics.",
    )
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--max-relative-target", type=float, default=5.0)
    parser.add_argument(
        "--action-filter",
        action="store_true",
        help=(
            "Model the deployment second-order output filter (strategy doc §6, doc default caps): "
            "all replay metrics then evaluate the filtered command stream the robot would receive."
        ),
    )
    parser.add_argument("--gripper-joint", default="gripper.pos")
    parser.add_argument("--gripper-event-threshold", type=float, default=1.0)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.chunk_size <= 0:
        raise ValueError("chunk-size must be positive")
    if args.execution_horizon <= 0:
        raise ValueError("execution-horizon must be positive")
    if not math.isfinite(args.max_guidance_weight) or args.max_guidance_weight <= 0:
        raise ValueError("max-guidance-weight must be finite and positive")
    if not math.isfinite(args.fps) or args.fps <= 0:
        raise ValueError("fps must be finite and positive")
    if not math.isfinite(args.max_relative_target) or args.max_relative_target <= 0:
        raise ValueError("max-relative-target must be finite and positive")
    if not args.absolute_indices or len(set(args.absolute_indices)) != len(args.absolute_indices):
        raise ValueError("absolute-indices must be non-empty and unique")
    if not args.seeds or len(set(args.seeds)) != len(args.seeds):
        raise ValueError("seeds must be non-empty and unique")
    if not args.queue_thresholds or len(set(args.queue_thresholds)) != len(args.queue_thresholds):
        raise ValueError("queue-thresholds must be non-empty and unique")
    invalid_queues = [value for value in args.queue_thresholds if not 0 <= value < args.chunk_size]
    if invalid_queues:
        raise ValueError(f"queue-thresholds must be in [0, chunk-size): {invalid_queues}")
    if not 0 <= args.guidance_delay < args.execution_horizon:
        raise ValueError("guidance-delay must be in [0, execution-horizon)")
    actual_steps = args.actual_consumed_steps
    if not actual_steps or len(set(actual_steps)) != len(actual_steps):
        raise ValueError("actual-consumed-steps must be non-empty and unique")
    invalid_actual_steps = [value for value in actual_steps if not 0 <= value < args.execution_horizon]
    if invalid_actual_steps:
        raise ValueError(f"actual-consumed-steps must be in [0, execution-horizon): {invalid_actual_steps}")


def _pop_actions(queue: ActionQueue, count: int, action_dim: int) -> tuple[torch.Tensor, int]:
    actions: list[torch.Tensor] = []
    underflow_count = 0
    for _ in range(count):
        action = queue.get()
        if action is None:
            underflow_count += 1
        else:
            actions.append(action)
    if not actions:
        return torch.empty((0, action_dim), dtype=torch.float32), underflow_count
    return torch.stack(actions), underflow_count


def begin_queue_transition(
    previous_original: torch.Tensor,
    previous_processed: torch.Tensor,
    *,
    queue_threshold: int,
    execution_horizon: int,
) -> QueueTransitionStart:
    """Merge an initial chunk and advance to the next inference start."""
    if previous_original.ndim != 2 or previous_processed.ndim != 2:
        raise ValueError("action chunks must be 2D [T, A]")
    if previous_original.shape != previous_processed.shape:
        raise ValueError("original and processed chunks must have the same shape")
    chunk_size = len(previous_original)
    if not 0 <= queue_threshold < chunk_size:
        raise ValueError("queue_threshold must be in [0, chunk_size)")

    queue = ActionQueue(RTCConfig(enabled=True, execution_horizon=execution_horizon))
    initial_start = queue.snapshot()
    initial_merge = queue.merge_actual_consumed(
        previous_original,
        previous_processed,
        initial_start,
    )
    replan_interval = chunk_size - queue_threshold
    actions_before, underflow = _pop_actions(
        queue,
        replan_interval,
        previous_processed.shape[1],
    )
    if underflow:
        raise RuntimeError("initial action chunk underflowed before the next inference")
    return QueueTransitionStart(
        queue=queue,
        inference_start=queue.snapshot(),
        initial_merge=initial_merge,
        actions_before_inference=actions_before,
        replan_interval_steps=replan_interval,
        execution_horizon=execution_horizon,
    )


def finish_queue_transition(
    start: QueueTransitionStart,
    current_original: torch.Tensor,
    current_processed: torch.Tensor,
    *,
    consumed_during_inference: int,
) -> QueueTransitionTrace:
    """Consume the injected delay, merge, and collect the next fixed-delay window."""
    action_dim = current_processed.shape[1]
    old_actions, old_underflow = _pop_actions(
        start.queue,
        consumed_during_inference,
        action_dim,
    )
    merge = start.queue.merge_actual_consumed(
        current_original,
        current_processed,
        start.inference_start,
        max_actual_consumed_steps=start.execution_horizon,
    )
    current_actions, current_underflow = _pop_actions(
        start.queue,
        start.replan_interval_steps,
        action_dim,
    )
    return QueueTransitionTrace(
        initial_merge=start.initial_merge,
        transition_merge=merge,
        actions_before_inference=start.actions_before_inference,
        old_actions_during_inference=old_actions,
        executed_current_actions=current_actions,
        executed_model_indices=list(range(merge.merge_skip, merge.merge_skip + len(current_actions))),
        queue_size_after_execution=start.queue.qsize(),
        underflow_count=old_underflow + current_underflow,
    )


def replay_initial_chunk(
    original_actions: torch.Tensor,
    processed_actions: torch.Tensor,
    *,
    queue_threshold: int,
    next_inference_consumed_steps: int,
    execution_horizon: int,
) -> QueueTransitionTrace:
    """Collect actions dispatched before the second inference result is ready."""
    start = begin_queue_transition(
        original_actions,
        processed_actions,
        queue_threshold=queue_threshold,
        execution_horizon=execution_horizon,
    )
    action_dim = processed_actions.shape[1]
    during_next_inference, underflow = _pop_actions(
        start.queue,
        next_inference_consumed_steps,
        action_dim,
    )
    executed = torch.cat((start.actions_before_inference, during_next_inference), dim=0)
    return QueueTransitionTrace(
        initial_merge=start.initial_merge,
        transition_merge=start.initial_merge,
        actions_before_inference=torch.empty((0, action_dim), dtype=processed_actions.dtype),
        old_actions_during_inference=torch.empty(
            (0, action_dim),
            dtype=processed_actions.dtype,
        ),
        executed_current_actions=executed,
        executed_model_indices=list(range(len(executed))),
        queue_size_after_execution=start.queue.qsize(),
        underflow_count=underflow,
    )


def _noise(seed: int, absolute_index: int, role: int, chunk_size: int, max_action_dim: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    mixed_seed = (seed * 1_000_003 + absolute_index * 97 + role * 53) % (2**63 - 1)
    generator.manual_seed(mixed_seed)
    return torch.randn((1, chunk_size, max_action_dim), generator=generator, dtype=torch.float32)


def _fit_prefix(prefix: torch.Tensor, target_steps: int) -> torch.Tensor:
    if prefix.ndim != 2:
        raise ValueError("RTC prefix must be 2D [T, A]")
    if len(prefix) >= target_steps:
        return prefix[:target_steps]
    result = torch.zeros(
        (target_steps, prefix.shape[1]),
        dtype=prefix.dtype,
        device=prefix.device,
    )
    result[: len(prefix)] = prefix
    return result


def _row_for_absolute_index(dataset_index: DatasetIndex, absolute_index: int) -> int:
    matches = torch.nonzero(dataset_index.absolute_indices == absolute_index, as_tuple=False).flatten()
    if len(matches) != 1:
        raise ValueError(f"absolute index {absolute_index} resolved to {len(matches)} rows")
    return int(matches.item())


def _processor_steps(preprocessor: Any, action_names: list[str]):
    relative_step = next(
        (
            step
            for step in preprocessor.steps
            if isinstance(step, RelativeActionsProcessorStep) and step.enabled
        ),
        None,
    )
    normalizer_step = next(
        (step for step in preprocessor.steps if isinstance(step, NormalizerProcessorStep)),
        None,
    )
    if relative_step is not None and relative_step.action_names is None:
        relative_step.action_names = list(action_names)
    return relative_step, normalizer_step


def _processed_sample(dataset: Any, row_index: int, preprocessor: Any) -> dict[str, torch.Tensor]:
    raw_sample = torch.utils.data.default_collate([dataset[row_index]])
    return preprocessor(raw_sample)


def _joint_dynamics(actions: torch.Tensor, fps: float) -> dict[str, float | None]:
    velocity = torch.diff(actions, dim=0) * fps
    acceleration = torch.diff(velocity, dim=0) * fps
    jerk = torch.diff(acceleration, dim=0) * fps
    return {
        "velocity_abs_max": float(velocity.abs().max()) if velocity.numel() else None,
        "acceleration_abs_max": float(acceleration.abs().max()) if acceleration.numel() else None,
        "jerk_abs_max": float(jerk.abs().max()) if jerk.numel() else None,
    }


def _evaluate_gate(records: list[dict[str, Any]], thresholds: ReplayThresholds) -> dict[str, Any]:
    errors = [value for record in records for row in record["abs_error"] for value in row]
    record_maes = [record["mae"] for record in records]
    predicted_only = [value for record in records for row in record["predicted_only_clamp"] for value in row]
    excess = [value for record in records for row in record["excess_clamp_residual"] for value in row]
    splice = [value for record in records for value in record["splice_jump_by_joint"]]
    command_delta = [record["command_delta_max"] for record in records]
    latency_steps = [record["inference_latency_steps"] for record in records]
    repeat = [record["repeat_max_abs"] for record in records]
    invariant_failures = sum(len(record["invariant_failures"]) for record in records)
    nonfinite = sum(record["nonfinite_action_values"] for record in records)
    predicted_only_rate = sum(bool(value) for value in predicted_only) / len(predicted_only)
    checks = {
        "coverage": {"passed": bool(records), "value": len(records), "minimum": 1},
        "invariants": {"passed": invariant_failures == 0, "value": invariant_failures, "maximum": 0},
        "nonfinite": {"passed": nonfinite == 0, "value": nonfinite, "maximum": 0},
        "mae_mean": {
            "passed": sum(record_maes) / len(record_maes) <= thresholds.max_mae_mean,
            "value": sum(record_maes) / len(record_maes),
            "maximum": thresholds.max_mae_mean,
            "diagnostic_only": True,
        },
        "error_p95": {
            "passed": scalar_stats(errors)["p95"] <= thresholds.max_mae_p95,
            "value": scalar_stats(errors)["p95"],
            "maximum": thresholds.max_mae_p95,
            "diagnostic_only": True,
        },
        "error_max": {
            "passed": max(errors) <= thresholds.max_error,
            "value": max(errors),
            "maximum": thresholds.max_error,
            "diagnostic_only": True,
        },
        "predicted_only_clamp_rate": {
            "passed": predicted_only_rate <= thresholds.max_predicted_only_clamp_rate,
            "value": predicted_only_rate,
            "maximum": thresholds.max_predicted_only_clamp_rate,
            "diagnostic_only": True,
        },
        "excess_clamp_residual_max": {
            "passed": max(excess) <= thresholds.max_excess_clamp_residual,
            "value": max(excess),
            "maximum": thresholds.max_excess_clamp_residual,
            "diagnostic_only": True,
        },
        "splice_jump_max": {
            "passed": max(splice) <= thresholds.max_splice_jump,
            "value": max(splice),
            "maximum": thresholds.max_splice_jump,
        },
        "command_delta_max": {
            "passed": max(command_delta) <= thresholds.max_command_delta,
            "value": max(command_delta),
            "maximum": thresholds.max_command_delta,
        },
        "latency_p95_steps": {
            "passed": scalar_stats(latency_steps)["p95"] <= thresholds.max_latency_p95_steps,
            "value": scalar_stats(latency_steps)["p95"],
            "maximum": thresholds.max_latency_p95_steps,
        },
        "latency_max_steps": {
            "passed": max(latency_steps) <= thresholds.max_latency_steps,
            "value": max(latency_steps),
            "maximum": thresholds.max_latency_steps,
        },
        "repeat_max_abs": {
            "passed": max(repeat) <= thresholds.max_repeat_error,
            "value": max(repeat),
            "maximum": thresholds.max_repeat_error,
        },
    }
    safety_passed = all(
        check["passed"] for check in checks.values() if not check.get("diagnostic_only", False)
    )
    semantic_diagnostics_passed = all(
        check["passed"] for check in checks.values() if check.get("diagnostic_only", False)
    )
    return {
        "passed": safety_passed,
        "safety_passed": safety_passed,
        "semantic_diagnostics_passed": semantic_diagnostics_passed,
        "checks": checks,
    }


def _record_transition(
    *,
    trace: QueueTransitionTrace,
    dataset_index: DatasetIndex,
    rows: list[int],
    local_index: int,
    current_state_indices: list[int],
    seed: int,
    absolute_index: int,
    queue_threshold: int,
    transition_kind: str,
    guidance_enabled: bool,
    guidance_delay: int,
    actual_consumed_steps: int,
    inference_latency_ms: float,
    repeat_max_abs: float,
    max_relative_target: float,
    fps: float,
    gripper_index: int,
    gripper_event_threshold: float,
    filter_limits: tuple[torch.Tensor, torch.Tensor] | None,
) -> dict[str, Any]:
    model_indices = trace.executed_model_indices
    target_local_indices = [local_index + index for index in model_indices]
    ground_truth = dataset_index.actions[[rows[index] for index in target_local_indices]]
    execution_states = dataset_index.states[[rows[index] for index in target_local_indices]][
        :, current_state_indices
    ]
    prediction_raw = trace.executed_current_actions.float().cpu()
    action_filter_info: dict[str, Any] = {"enabled": False}
    if filter_limits is not None and len(prediction_raw):
        # Seed the command integrator the way the deployment filter would be
        # seeded entering this window: at the measured pose for the very first
        # chunk, or at the last command dispatched before the splice.  Velocity
        # starts at zero (per-window independence approximation).
        if transition_kind != "initial" and len(trace.old_actions_during_inference):
            seed_position = trace.old_actions_during_inference[-1].cpu()
            seed_kind = "last_old_action"
        else:
            seed_position = execution_states[0]
            seed_kind = "execution_state"
        output_filter = SecondOrderActionFilter(filter_limits[0], filter_limits[1], 1.0 / fps)
        filtered = output_filter.filter_sequence(
            prediction_raw.to(torch.float64),
            seed_position.to(torch.float64),
        ).float()
        action_filter_info = {
            "enabled": True,
            "seed_kind": seed_kind,
            "steps": output_filter.step_count,
            "interventions": output_filter.intervention_count,
            "max_abs_intervention": output_filter.max_abs_intervention,
            "max_abs_delta_vs_raw": float((filtered - prediction_raw).abs().max()),
        }
        prediction = filtered
    else:
        prediction = prediction_raw
    errors = (prediction - ground_truth).abs()
    predicted_residual = (prediction - execution_states).abs().sub(max_relative_target).clamp_min(0)
    gt_residual = (ground_truth - execution_states).abs().sub(max_relative_target).clamp_min(0)
    predicted_only = (predicted_residual > 0) & (gt_residual == 0)
    excess_residual = (predicted_residual - gt_residual).clamp_min(0)
    if transition_kind == "initial":
        splice_jump = (prediction[0] - execution_states[0]).abs()
    elif len(trace.old_actions_during_inference) and len(prediction):
        splice_jump = (prediction[0] - trace.old_actions_during_inference[-1].cpu()).abs()
    else:
        splice_jump = torch.zeros(prediction.shape[1])
    within_chunk_delta = torch.diff(prediction, dim=0).abs()
    command_delta_max = max(
        float(splice_jump.max()),
        float(within_chunk_delta.max()) if within_chunk_delta.numel() else 0.0,
    )
    initial_gripper = float(execution_states[0, gripper_index])
    predicted_event = first_gripper_event(
        prediction[:, gripper_index],
        initial_gripper,
        gripper_event_threshold,
    )
    gt_event = first_gripper_event(
        ground_truth[:, gripper_index],
        initial_gripper,
        gripper_event_threshold,
    )
    predicted_direction = (
        int(torch.sign(prediction[predicted_event, gripper_index] - initial_gripper).item())
        if predicted_event is not None
        else None
    )
    gt_direction = (
        int(torch.sign(ground_truth[gt_event, gripper_index] - initial_gripper).item())
        if gt_event is not None
        else None
    )
    invariant_failures = []
    if not trace.initial_merge.merged or trace.initial_merge.stale:
        invariant_failures.append("initial_merge_rejected")
    if transition_kind != "initial" and (not trace.transition_merge.merged or trace.transition_merge.stale):
        invariant_failures.append("transition_merge_rejected")
    if transition_kind != "initial" and trace.transition_merge.actual_consumed_steps != actual_consumed_steps:
        invariant_failures.append("actual_consumed_mismatch")
    if transition_kind != "initial" and trace.transition_merge.merge_skip != actual_consumed_steps:
        invariant_failures.append("merge_skip_mismatch")
    if transition_kind != "initial" and trace.transition_merge.skip_was_clamped:
        invariant_failures.append("merge_skip_clamped")
    if trace.underflow_count:
        invariant_failures.append("queue_underflow")
    nonfinite = int((~torch.isfinite(prediction)).sum())
    return {
        "absolute_index": absolute_index,
        "episode_index": int(dataset_index.episode_indices[rows[local_index]]),
        "frame_index": int(dataset_index.frame_indices[rows[local_index]]),
        "local_index": local_index,
        "seed": seed,
        "queue_threshold": queue_threshold,
        "transition_kind": transition_kind,
        "guidance_enabled": guidance_enabled,
        "replan_interval_steps": trace.actions_before_inference.shape[0],
        "guidance_delay_steps": guidance_delay,
        "injected_actual_consumed_steps": actual_consumed_steps,
        "actual_consumed_steps": trace.transition_merge.actual_consumed_steps,
        "merge_skip": trace.transition_merge.merge_skip,
        "executed_model_indices": model_indices,
        "queue_size_after_execution": trace.queue_size_after_execution,
        "prediction": prediction.tolist(),
        "prediction_unfiltered": prediction_raw.tolist() if action_filter_info["enabled"] else None,
        "action_filter": action_filter_info,
        "ground_truth": ground_truth.tolist(),
        "execution_state": execution_states.tolist(),
        "abs_error": errors.tolist(),
        "mae": float(errors.mean()),
        "max_abs_error": float(errors.max()),
        "predicted_only_clamp": predicted_only.tolist(),
        "excess_clamp_residual": excess_residual.tolist(),
        "excess_clamp_residual_max": float(excess_residual.max()),
        "splice_jump_by_joint": splice_jump.tolist(),
        "splice_jump_max": float(splice_jump.max()),
        "command_delta_max": command_delta_max,
        "inference_latency_ms": inference_latency_ms,
        "inference_latency_steps": inference_latency_ms * fps / 1000.0,
        "repeat_max_abs": repeat_max_abs,
        "gripper": {
            "predicted_event_step": predicted_event,
            "gt_event_step": gt_event,
            "timing_abs_error_steps": (
                abs(predicted_event - gt_event)
                if predicted_event is not None and gt_event is not None
                else None
            ),
            "predicted_direction": predicted_direction,
            "gt_direction": gt_direction,
            "direction_matches": (
                predicted_direction == gt_direction
                if predicted_direction is not None and gt_direction is not None
                else None
            ),
        },
        "dynamics": _joint_dynamics(prediction, fps),
        "nonfinite_action_values": nonfinite,
        "invariant_failures": invariant_failures,
    }


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    from lerobot.datasets import LeRobotDataset

    dataset_root = args.dataset_root.expanduser().resolve()
    model_path = args.model.expanduser().resolve()
    dataset = LeRobotDataset(
        repo_id=args.repo_id,
        root=dataset_root,
        download_videos=False,
        video_backend=args.video_backend,
    )
    dataset_index = build_dataset_index(dataset)
    gripper_index = dataset_index.action_names.index(args.gripper_joint)
    state_indices = [dataset_index.state_names.index(name) for name in dataset_index.action_names]
    filter_limits = None
    if args.action_filter:
        filter_limits = (
            build_limit_tensor(dataset_index.action_names, DEFAULT_MAX_VELOCITY),
            build_limit_tensor(dataset_index.action_names, DEFAULT_MAX_ACCELERATION),
        )
    spec = ModelSpec("full", model_path)
    config, descriptor = inspect_checkpoint(spec, args.device)
    if int(config.chunk_size) != args.chunk_size:
        raise ValueError(f"checkpoint chunk_size={config.chunk_size}, expected {args.chunk_size}")
    rtc_config = RTCConfig(
        enabled=True,
        execution_horizon=args.execution_horizon,
        max_guidance_weight=args.max_guidance_weight,
        prefix_attention_schedule=RTCAttentionSchedule.EXP,
    )
    config.rtc_config = rtc_config
    policy, preprocessor, postprocessor = load_policy_and_processors(spec, config, args.device)
    policy.config.rtc_config = rtc_config
    policy.init_rtc_processor()
    relative_step, normalizer_step = _processor_steps(preprocessor, dataset_index.action_names)

    records: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    try:
        total = (
            len(args.absolute_indices)
            * len(args.seeds)
            * len(args.queue_thresholds)
            * len(args.actual_consumed_steps)
        )
        completed = 0
        for absolute_index in args.absolute_indices:
            row_index = _row_for_absolute_index(dataset_index, absolute_index)
            episode_index = int(dataset_index.episode_indices[row_index])
            rows = dataset_index.rows_by_episode[episode_index]
            local_index = rows.index(row_index)
            for queue_threshold in args.queue_thresholds:
                interval = args.chunk_size - queue_threshold
                previous_local = local_index - interval
                max_delay = max(args.actual_consumed_steps)
                has_future_context = local_index + max_delay + interval <= len(rows)
                if previous_local < 0 and local_index == 0 and has_future_context:
                    for seed in args.seeds:
                        preprocessor.reset()
                        postprocessor.reset()
                        policy.reset()
                        initial_noise = _noise(
                            seed,
                            absolute_index,
                            2,
                            args.chunk_size,
                            int(config.max_action_dim),
                        ).to(args.device)
                        initial_raw_sample = torch.utils.data.default_collate([dataset[row_index]])
                        if args.device.startswith("cuda"):
                            torch.cuda.synchronize(args.device)
                        inference_started = time.perf_counter()
                        initial_sample = preprocessor(initial_raw_sample)
                        initial_original = policy.predict_action_chunk(
                            initial_sample,
                            noise=initial_noise.clone(),
                        )
                        initial_processed = postprocessor(initial_original).squeeze(0).detach().float().cpu()
                        if args.device.startswith("cuda"):
                            torch.cuda.synchronize(args.device)
                        inference_latency_ms = (time.perf_counter() - inference_started) * 1000.0
                        repeated_initial = policy.predict_action_chunk(
                            initial_sample,
                            noise=initial_noise.clone(),
                        )
                        repeat_max_abs = float(
                            (initial_original.float() - repeated_initial.float()).abs().max()
                        )
                        initial_original_cpu = initial_original.squeeze(0).detach().float().cpu()
                        for actual_consumed_steps in args.actual_consumed_steps:
                            trace = replay_initial_chunk(
                                initial_original_cpu,
                                initial_processed,
                                queue_threshold=queue_threshold,
                                next_inference_consumed_steps=actual_consumed_steps,
                                execution_horizon=args.execution_horizon,
                            )
                            records.append(
                                _record_transition(
                                    trace=trace,
                                    dataset_index=dataset_index,
                                    rows=rows,
                                    local_index=local_index,
                                    current_state_indices=state_indices,
                                    seed=seed,
                                    absolute_index=absolute_index,
                                    queue_threshold=queue_threshold,
                                    transition_kind="initial",
                                    guidance_enabled=False,
                                    guidance_delay=0,
                                    actual_consumed_steps=actual_consumed_steps,
                                    inference_latency_ms=inference_latency_ms,
                                    repeat_max_abs=repeat_max_abs,
                                    max_relative_target=args.max_relative_target,
                                    fps=args.fps,
                                    gripper_index=gripper_index,
                                    gripper_event_threshold=args.gripper_event_threshold,
                                    filter_limits=filter_limits,
                                )
                            )
                            completed += 1
                            if completed % max(1, total // 20) == 0 or completed == total:
                                print(f"RTC replay progress={completed}/{total}", flush=True)
                    continue
                if previous_local < 0 or not has_future_context:
                    skipped.append(
                        {
                            "absolute_index": absolute_index,
                            "queue_threshold": queue_threshold,
                            "reason": "insufficient_same_episode_context",
                        }
                    )
                    completed += len(args.seeds) * len(args.actual_consumed_steps)
                    continue
                previous_row = rows[previous_local]
                for seed in args.seeds:
                    preprocessor.reset()
                    postprocessor.reset()
                    policy.reset()
                    previous_sample = _processed_sample(dataset, previous_row, preprocessor)
                    previous_noise = _noise(
                        seed,
                        int(dataset_index.absolute_indices[previous_row]),
                        0,
                        args.chunk_size,
                        int(config.max_action_dim),
                    ).to(args.device)
                    previous_original = policy.predict_action_chunk(
                        previous_sample,
                        noise=previous_noise,
                    )
                    previous_processed = postprocessor(previous_original).squeeze(0).detach().float().cpu()
                    previous_original_cpu = previous_original.squeeze(0).detach().float().cpu()
                    prefix_start = begin_queue_transition(
                        previous_original_cpu,
                        previous_processed,
                        queue_threshold=queue_threshold,
                        execution_horizon=args.execution_horizon,
                    )

                    current_noise = _noise(
                        seed,
                        absolute_index,
                        1,
                        args.chunk_size,
                        int(config.max_action_dim),
                    ).to(args.device)
                    current_raw_sample = torch.utils.data.default_collate([dataset[row_index]])
                    if args.device.startswith("cuda"):
                        torch.cuda.synchronize(args.device)
                    inference_started = time.perf_counter()
                    current_sample = preprocessor(current_raw_sample)
                    prefix = prefix_start.inference_start.original_leftover
                    if relative_step is not None:
                        current_state = relative_step.get_cached_state()
                        if current_state is None or prefix_start.inference_start.processed_leftover is None:
                            raise RuntimeError(
                                "relative RTC replay is missing current state or absolute prefix"
                            )
                        prefix = reanchor_relative_rtc_prefix(
                            prev_actions_absolute=prefix_start.inference_start.processed_leftover,
                            current_state=current_state,
                            relative_step=relative_step,
                            normalizer_step=normalizer_step,
                            policy_device=args.device,
                        )
                    if prefix is None:
                        raise RuntimeError("RTC replay did not produce a previous action prefix")
                    prefix = _fit_prefix(prefix.to(args.device), args.execution_horizon)
                    prediction_kwargs = {"noise": current_noise.clone()}
                    if not args.disable_guidance:
                        prediction_kwargs.update(
                            {
                                "inference_delay": args.guidance_delay,
                                "prev_chunk_left_over": prefix,
                            }
                        )
                    current_original = policy.predict_action_chunk(
                        current_sample,
                        **prediction_kwargs,
                    )
                    current_processed = postprocessor(current_original).squeeze(0).detach().float().cpu()
                    if args.device.startswith("cuda"):
                        torch.cuda.synchronize(args.device)
                    inference_latency_ms = (time.perf_counter() - inference_started) * 1000.0
                    prediction_kwargs["noise"] = current_noise.clone()
                    repeated_original = policy.predict_action_chunk(
                        current_sample,
                        **prediction_kwargs,
                    )
                    repeat_max_abs = float((current_original.float() - repeated_original.float()).abs().max())
                    current_original_cpu = current_original.squeeze(0).detach().float().cpu()
                    for actual_consumed_steps in args.actual_consumed_steps:
                        start = begin_queue_transition(
                            previous_original_cpu,
                            previous_processed,
                            queue_threshold=queue_threshold,
                            execution_horizon=args.execution_horizon,
                        )
                        trace = finish_queue_transition(
                            start,
                            current_original_cpu,
                            current_processed,
                            consumed_during_inference=actual_consumed_steps,
                        )
                        records.append(
                            _record_transition(
                                trace=trace,
                                dataset_index=dataset_index,
                                rows=rows,
                                local_index=local_index,
                                current_state_indices=state_indices,
                                seed=seed,
                                absolute_index=absolute_index,
                                queue_threshold=queue_threshold,
                                transition_kind="steady",
                                guidance_enabled=not args.disable_guidance,
                                guidance_delay=args.guidance_delay,
                                actual_consumed_steps=actual_consumed_steps,
                                inference_latency_ms=inference_latency_ms,
                                repeat_max_abs=repeat_max_abs,
                                max_relative_target=args.max_relative_target,
                                fps=args.fps,
                                gripper_index=gripper_index,
                                gripper_event_threshold=args.gripper_event_threshold,
                                filter_limits=filter_limits,
                            )
                        )
                        completed += 1
                        if completed % max(1, total // 20) == 0 or completed == total:
                            print(f"RTC replay progress={completed}/{total}", flush=True)
    finally:
        del policy
        if args.device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()

    thresholds = ReplayThresholds()
    groups: dict[str, Any] = {}
    for queue_threshold in args.queue_thresholds:
        for actual_consumed_steps in args.actual_consumed_steps:
            group_records = [
                record
                for record in records
                if record["queue_threshold"] == queue_threshold
                and record["injected_actual_consumed_steps"] == actual_consumed_steps
            ]
            guidance_label = "unguided" if args.disable_guidance else f"g{args.guidance_delay}"
            key = f"q{queue_threshold}_{guidance_label}_a{actual_consumed_steps}"
            groups[key] = {
                "queue_threshold": queue_threshold,
                "steady_guidance_enabled": not args.disable_guidance,
                "guidance_delay_steps": args.guidance_delay,
                "actual_consumed_steps": actual_consumed_steps,
                "replan_interval_steps": args.chunk_size - queue_threshold,
                "expected_replan_hz": args.fps / (args.chunk_size - queue_threshold),
                "record_count": len(group_records),
                "gate": _evaluate_gate(group_records, thresholds)
                if group_records
                else {
                    "passed": False,
                    "checks": {"coverage": {"passed": False, "value": 0, "minimum": 1}},
                },
            }
    candidate_keys = [
        (
            f"q45_unguided_a{actual_consumed_steps}"
            if args.disable_guidance
            else f"q45_g{args.guidance_delay}_a{actual_consumed_steps}"
        )
        for actual_consumed_steps in args.actual_consumed_steps
        if 45 in args.queue_thresholds
    ]
    deployment_passed = bool(candidate_keys) and all(groups[key]["gate"]["passed"] for key in candidate_keys)
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "report_status": "complete",
        "created_at": datetime.now(UTC).isoformat(),
        "hardware_action_sent": False,
        "evaluator": {
            "path": str(Path(__file__).resolve()),
            "sha256": _file_sha256(Path(__file__).resolve()),
        },
        "checkpoint": descriptor,
        "checkpoint_asset_fingerprint": fingerprint_checkpoint_assets(model_path),
        "dataset": {
            "root": str(dataset_root),
            "repo_id": args.repo_id,
            "asset_fingerprint": fingerprint_dataset_assets(dataset_root),
            "action_names": dataset_index.action_names,
            "state_names": dataset_index.state_names,
        },
        "configuration": {
            "chunk_size": args.chunk_size,
            "execution_horizon": args.execution_horizon,
            "max_guidance_weight": args.max_guidance_weight,
            "guidance_enabled": not args.disable_guidance,
            "prefix_attention_schedule": "EXP",
            "fps": args.fps,
            "queue_thresholds": args.queue_thresholds,
            "guidance_delay_steps": args.guidance_delay,
            "actual_consumed_steps": args.actual_consumed_steps,
            "absolute_indices": args.absolute_indices,
            "seeds": args.seeds,
            "max_relative_target": args.max_relative_target,
            "action_filter": {
                "enabled": bool(args.action_filter),
                "max_velocity": DEFAULT_MAX_VELOCITY,
                "max_acceleration": DEFAULT_MAX_ACCELERATION,
            },
        },
        "thresholds": asdict(thresholds),
        "groups": groups,
        "records": records,
        "skipped": skipped,
        "deployment_candidate": ("q45_unguided" if args.disable_guidance else "q45_fixed_guidance_5"),
        "required_rollout_config": {
            "timing_mode": "actual_consumed",
            "guidance_delay_mode": "fixed",
            "fixed_guidance_delay_steps": args.guidance_delay,
            "queue_threshold": 45,
            "execution_horizon": args.execution_horizon,
            "max_guidance_weight": args.max_guidance_weight,
            "prefix_attention_schedule": "EXP",
            "enforce_guided_execution_window": True,
            "prefix_backend": "pytorch",
            "action_backend": "pytorch",
            "action_filter_enabled": bool(args.action_filter),
        },
        "deployment_passed": deployment_passed,
        "passed": deployment_passed,
    }


def main() -> int:
    args = build_parser().parse_args()
    validate_args(args)
    output_path = args.output_json.expanduser().resolve()
    running_report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "report_status": "running",
        "created_at": datetime.now(UTC).isoformat(),
        "hardware_action_sent": False,
        "passed": False,
    }
    write_report(output_path, running_report)
    try:
        report = evaluate(args)
    except BaseException as exc:
        running_report.update(
            {
                "report_status": "failed",
                "failure": {"type": type(exc).__name__, "message": str(exc)},
                "passed": False,
            }
        )
        write_report(output_path, running_report)
        raise
    write_report(output_path, report)
    print(
        json.dumps(
            {
                "report_status": report["report_status"],
                "deployment_candidate": report["deployment_candidate"],
                "deployment_passed": report["deployment_passed"],
                "output_json": str(output_path),
                "hardware_action_sent": False,
            },
            indent=2,
        )
    )
    return 0 if report["deployment_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
