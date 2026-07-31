#!/usr/bin/env python

"""Run the real PI0.5 VLASH engine against local dataset observations.

This smoke test loads a local checkpoint and exercises asynchronous chunk
switching at the requested control rate. It never constructs a robot or sends
an action to hardware.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.inference.evaluate_pi05_multiseed import (  # noqa: E402
    ModelSpec,
    inspect_checkpoint,
    load_policy_and_processors,
    write_report,
)
from lerobot.datasets import LeRobotDataset  # noqa: E402
from lerobot.rollout.inference.factory import VLASHInferenceConfig  # noqa: E402
from lerobot.rollout.inference.vlash import VLASHInferenceEngine  # noqa: E402

ROOT = REPO_ROOT


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "data/so101_test_data")
    parser.add_argument("--repo-id", default="admin123/so101_test_data")
    parser.add_argument("--video-backend", default="pyav")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--task", default="Put the block in the bin")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--control-steps", type=int, default=30)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--execution-horizon", type=int, default=10)
    parser.add_argument("--inference-overlap-steps", type=int, default=5)
    parser.add_argument("--max-future-state-delta", type=float, default=5.0)
    parser.add_argument("--deadline-miss-limit", type=int, default=1)
    parser.add_argument("--initial-timeout-s", type=float, default=120.0)
    parser.add_argument("--p95-limit-ms", type=float, default=150.0)
    parser.add_argument("--max-limit-ms", type=float, default=167.0)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=ROOT / "outputs/eval/pi05_vlash_replay.json",
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.episode < 0 or args.start_frame < 0:
        raise ValueError("episode and start-frame must be non-negative")
    if args.control_steps < args.execution_horizon * 2:
        raise ValueError("control-steps must cover at least two execution horizons")
    for name in ("fps", "initial_timeout_s", "p95_limit_ms", "max_limit_ms"):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name.replace('_', '-')} must be finite and positive")


def _to_numpy_image(value: Any) -> np.ndarray:
    image = torch.as_tensor(value).detach().cpu()
    if image.ndim != 3:
        raise ValueError(f"expected rank-3 image, got {tuple(image.shape)}")
    if image.shape[0] in (1, 3):
        image = image.permute(1, 2, 0)
    return image.contiguous().numpy()


def build_raw_observation(sample: dict[str, Any], features: dict[str, dict]) -> dict[str, Any]:
    state_feature = features["observation.state"]
    state = torch.as_tensor(sample["observation.state"]).detach().cpu().reshape(-1)
    state_names = state_feature["names"]
    if len(state) != len(state_names):
        raise ValueError(f"state dimension {len(state)} does not match names {len(state_names)}")

    observation = {name: float(state[index]) for index, name in enumerate(state_names)}
    for key, feature in features.items():
        if not key.startswith("observation.images."):
            continue
        image = _to_numpy_image(sample[key])
        expected_shape = tuple(feature["shape"])
        if image.shape != expected_shape:
            raise ValueError(f"{key} has shape {image.shape}, expected {expected_shape}")
        observation[key.removeprefix("observation.images.")] = image
    return observation


def build_recorded_action(sample: dict[str, Any], features: dict[str, dict]) -> dict[str, float]:
    action = torch.as_tensor(sample["action"]).detach().cpu().reshape(-1)
    action_names = features["action"]["names"]
    if len(action) != len(action_names):
        raise ValueError(f"action dimension {len(action)} does not match names {len(action_names)}")
    return {name: float(action[index]) for index, name in enumerate(action_names)}


def wait_for_initial_chunk(engine: VLASHInferenceEngine, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    while engine.stats_snapshot().active_actions == 0:
        if engine.failed:
            raise RuntimeError(f"VLASH initial inference failed: {engine.fatal_error}")
        if time.monotonic() >= deadline:
            raise TimeoutError("timed out waiting for VLASH initial action chunk")
        time.sleep(0.005)


def main() -> int:
    args = build_parser().parse_args()
    validate_args(args)
    checkpoint = args.model.expanduser().resolve()
    dataset_root = args.dataset_root.expanduser().resolve()
    output_path = args.output_json.expanduser().resolve()

    with (dataset_root / "meta/info.json").open(encoding="utf-8") as handle:
        dataset_info = json.load(handle)
    features = dataset_info["features"]
    dataset = LeRobotDataset(
        repo_id=args.repo_id,
        root=dataset_root,
        download_videos=False,
        video_backend=args.video_backend,
    )
    episode_indices = [int(value) for value in dataset.hf_dataset["episode_index"]]
    rows = [index for index, episode in enumerate(episode_indices) if episode == args.episode]
    selected_rows = rows[args.start_frame : args.start_frame + args.control_steps]
    if len(selected_rows) != args.control_steps:
        raise ValueError(
            f"episode {args.episode} has only {len(rows)} rows; cannot select "
            f"{args.control_steps} from local frame {args.start_frame}"
        )
    samples = [dataset[row] for row in selected_rows]
    observations = [build_raw_observation(sample, features) for sample in samples]
    recorded_actions = [build_recorded_action(sample, features) for sample in samples]

    spec = ModelSpec("vlash", checkpoint)
    config, descriptor = inspect_checkpoint(spec, args.device)
    vlash_config = VLASHInferenceConfig(
        execution_horizon=args.execution_horizon,
        inference_overlap_steps=args.inference_overlap_steps,
        max_future_state_delta=args.max_future_state_delta,
        deadline_miss_limit=args.deadline_miss_limit,
        timing_diagnostics=True,
    )
    vlash_config.validate_policy(config)
    policy, preprocessor, postprocessor = load_policy_and_processors(spec, config, args.device)

    engine = VLASHInferenceEngine(
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        robot_wrapper=SimpleNamespace(robot_type=dataset_info.get("robot_type", "so_follower")),
        hw_features=features,
        ordered_action_keys=list(features["observation.state"]["names"]),
        task=args.task,
        fps=args.fps,
        device=args.device,
        execution_horizon=args.execution_horizon,
        inference_overlap_steps=args.inference_overlap_steps,
        max_future_state_delta=args.max_future_state_delta,
        deadline_miss_limit=args.deadline_miss_limit,
        timing_diagnostics=True,
    )

    actions: list[torch.Tensor] = []
    control_overruns = 0
    failure: str | None = None
    started = time.perf_counter()
    engine.reset()
    engine.start()
    engine.resume()
    initial_latency_ms: float | None = None
    try:
        engine.notify_observation(observations[0])
        wait_for_initial_chunk(engine, args.initial_timeout_s)
        initial_latency_ms = engine.stats_snapshot().latency_max_ms
        engine.clear_latency_window()
        interval = 1.0 / args.fps
        for observation, recorded_action in zip(observations, recorded_actions, strict=True):
            tick_started = time.perf_counter()
            engine.notify_observation(observation)
            action = engine.get_action(None)
            if action is None:
                failure = str(engine.fatal_error or "VLASH returned no action")
                break
            if action.ndim != 1 or not torch.isfinite(action).all():
                failure = f"invalid action shape or values: shape={tuple(action.shape)}"
                break
            actions.append(action)
            engine.notify_action_result(recorded_action, recorded_action, observation)
            remaining = interval - (time.perf_counter() - tick_started)
            if remaining > 0:
                time.sleep(remaining)
            else:
                control_overruns += 1
    finally:
        engine.stop()

    stats = engine.stats_snapshot()
    p95_passed = stats.latency_p95_ms is not None and stats.latency_p95_ms < args.p95_limit_ms
    max_passed = stats.latency_max_ms is not None and stats.latency_max_ms < args.max_limit_ms
    passed = bool(
        failure is None
        and len(actions) == args.control_steps
        and stats.inference_count >= 3
        and stats.deadline_misses == 0
        and p95_passed
        and max_passed
    )
    report = {
        "checkpoint": descriptor,
        "dataset": {
            "repo_id": args.repo_id,
            "root": str(dataset_root),
            "episode": args.episode,
            "start_frame": args.start_frame,
            "control_steps": args.control_steps,
        },
        "vlash": {
            "execution_horizon": args.execution_horizon,
            "inference_overlap_steps": args.inference_overlap_steps,
            "max_future_state_delta": args.max_future_state_delta,
            "deadline_miss_limit": args.deadline_miss_limit,
            "fps": args.fps,
            "feedback_source": "recorded_dataset_action",
        },
        "stats": asdict(stats),
        "initial_latency_ms": initial_latency_ms,
        "actions_produced": len(actions),
        "action_shape": list(actions[0].shape) if actions else None,
        "all_actions_finite": bool(actions and all(torch.isfinite(action).all() for action in actions)),
        "control_overruns": control_overruns,
        "failure": failure,
        "checks": {
            "p95_latency_ms": {
                "value": stats.latency_p95_ms,
                "limit": args.p95_limit_ms,
                "passed": p95_passed,
            },
            "max_latency_ms": {
                "value": stats.latency_max_ms,
                "limit": args.max_limit_ms,
                "passed": max_passed,
            },
            "deadline_misses": {
                "value": stats.deadline_misses,
                "limit": 0,
                "passed": stats.deadline_misses == 0,
            },
        },
        "passed": passed,
        "duration_seconds": time.perf_counter() - started,
        "hardware_action_sent": False,
    }
    write_report(output_path, report)
    print(
        json.dumps(
            {
                "passed": passed,
                "stats": report["stats"],
                "failure": failure,
                "output_json": str(output_path),
                "hardware_action_sent": False,
            },
            indent=2,
        )
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
