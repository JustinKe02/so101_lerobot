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

"""Train the offline Realtime-VLA per-segment speed adapter from JSONL labels.

Example:

    uv run python -m lerobot.scripts.lerobot_train_speed_adapter \
        --data throttle_samples.jsonl \
        --output-dir outputs/speed_adapter \
        --beta-min 0.5 \
        --beta-max 1.5

This command only reads annotations and writes a checkpoint. It does not open
cameras, connect to a robot, or execute actions.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from lerobot.rollout.speed_adapter import (
    SpeedAdapter,
    SpeedAdapterConfig,
    ThrottleRecord,
    failure_window_mask,
    file_sha256,
    load_throttle_jsonl,
    save_speed_adapter_checkpoint,
    throttle_jsonl_schema,
    throttle_records_to_tensors,
)


@dataclass(frozen=True)
class TrainingResult:
    model: SpeedAdapter
    feature_names: tuple[str, ...]
    metadata: dict


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, help="Throttle JSONL conforming to lerobot.speed_throttle.v1")
    parser.add_argument("--output-dir", type=Path, help="Checkpoint destination directory")
    parser.add_argument(
        "--print-schema",
        action="store_true",
        help="Print the accepted JSON schema and exit without training",
    )
    parser.add_argument("--beta-min", type=float, help="Hard lower bound for predicted beta")
    parser.add_argument("--beta-max", type=float, help="Hard upper bound for predicted beta")
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[64, 32])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument(
        "--failure-preceding-steps",
        type=int,
        default=0,
        help="Exclude this many steps before every failure_event",
    )
    parser.add_argument(
        "--failure-following-steps",
        type=int,
        default=0,
        help="Exclude this many steps after every failure_event",
    )
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or a concrete torch device")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _positive_integer(value: int, *, name: str) -> int:
    if isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")
    return device


def _validate_training_arguments(args: argparse.Namespace) -> None:
    if args.data is None or args.output_dir is None:
        raise ValueError("--data and --output-dir are required for training")
    if args.beta_min is None or args.beta_max is None:
        raise ValueError("--beta-min and --beta-max are required; deployment bounds must be explicit")
    _positive_integer(args.epochs, name="epochs")
    _positive_integer(args.batch_size, name="batch_size")
    _positive_integer(args.log_every, name="log_every")
    if any(isinstance(width, bool) or width < 1 for width in args.hidden_dims):
        raise ValueError("hidden-dims must contain positive integers")
    for value, name in (
        (args.learning_rate, "learning-rate"),
        (args.weight_decay, "weight-decay"),
    ):
        if not math.isfinite(value) or (value <= 0 if name == "learning-rate" else value < 0):
            raise ValueError(f"{name} has an invalid value: {value}")
    if not math.isfinite(args.validation_fraction) or not 0 <= args.validation_fraction < 1:
        raise ValueError("validation-fraction must be in [0, 1)")
    for value, name in (
        (args.failure_preceding_steps, "failure-preceding-steps"),
        (args.failure_following_steps, "failure-following-steps"),
    ):
        if isinstance(value, bool) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")


def _grouped_split(
    records: Sequence[ThrottleRecord],
    selected_indices: torch.Tensor,
    *,
    validation_fraction: float,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    selected = [int(index) for index in selected_indices.tolist()]
    episodes = sorted({records[index].episode_id for index in selected})
    if validation_fraction == 0 or len(episodes) < 2:
        return (
            torch.tensor(selected, dtype=torch.long),
            torch.empty(0, dtype=torch.long),
            {
                "strategy": "all_train" if validation_fraction == 0 else "all_train_single_episode",
                "train_episodes": episodes,
                "validation_episodes": [],
            },
        )

    shuffled = episodes.copy()
    random.Random(seed).shuffle(shuffled)
    validation_episode_count = max(1, round(len(shuffled) * validation_fraction))
    validation_episode_count = min(validation_episode_count, len(shuffled) - 1)
    validation_episodes = set(shuffled[:validation_episode_count])
    train = [index for index in selected if records[index].episode_id not in validation_episodes]
    validation = [index for index in selected if records[index].episode_id in validation_episodes]
    return (
        torch.tensor(train, dtype=torch.long),
        torch.tensor(validation, dtype=torch.long),
        {
            "strategy": "episode_grouped",
            "train_episodes": sorted({records[index].episode_id for index in train}),
            "validation_episodes": sorted(validation_episodes),
        },
    )


def _relative_speed_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Smooth L1 on log speed treats equal relative beta errors equally."""

    return functional.smooth_l1_loss(torch.log(prediction), torch.log(target))


@torch.inference_mode()
def _evaluate(model: nn.Module, features: torch.Tensor, targets: torch.Tensor) -> float | None:
    if len(features) == 0:
        return None
    model.eval()
    return float(_relative_speed_loss(model(features), targets).item())


def train_speed_adapter(args: argparse.Namespace) -> TrainingResult:
    """Train from real JSONL rows supplied by the caller and return the best model."""

    _validate_training_arguments(args)
    records = load_throttle_jsonl(args.data)
    if not records:
        raise ValueError(f"no throttle records found in {args.data}")
    tensor_data = throttle_records_to_tensors(records)
    keep_mask = failure_window_mask(
        records,
        preceding_steps=args.failure_preceding_steps,
        following_steps=args.failure_following_steps,
    )
    selected_indices = keep_mask.nonzero(as_tuple=False).flatten()
    if len(selected_indices) < 2:
        raise ValueError(
            "fewer than two training-eligible throttle records remain after failure/manual masking"
        )
    config = SpeedAdapterConfig(
        action_dim=tensor_data.layout.action_dim,
        state_dim=tensor_data.layout.state_dim,
        phase_dim=tensor_data.layout.phase_dim,
        hidden_dims=tuple(args.hidden_dims),
        beta_min=args.beta_min,
        beta_max=args.beta_max,
        feature_coordinate_space=tensor_data.layout.feature_coordinate_space,
    )
    selected_targets = tensor_data.beta_targets[selected_indices]
    outside_bounds = (selected_targets < config.beta_min) | (selected_targets > config.beta_max)
    if outside_bounds.any().item():
        original_indices = selected_indices[outside_bounds].tolist()
        raise ValueError(
            "beta_target values must lie inside the requested output bounds; "
            f"offending record indices: {original_indices[:10]}"
        )

    train_indices, validation_indices, split_metadata = _grouped_split(
        records,
        selected_indices,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
    )
    train_features = tensor_data.features[train_indices]
    train_targets = tensor_data.beta_targets[train_indices]
    validation_features = tensor_data.features[validation_indices]
    validation_targets = tensor_data.beta_targets[validation_indices]
    feature_mean = train_features.mean(dim=0)
    feature_std = train_features.std(dim=0, unbiased=False)
    feature_std = torch.where(feature_std < config.normalization_epsilon, 1.0, feature_std)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = _resolve_device(args.device)
    model = SpeedAdapter(config, feature_mean=feature_mean, feature_std=feature_std).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        TensorDataset(train_features, train_targets),
        batch_size=min(args.batch_size, len(train_features)),
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    validation_features = validation_features.to(device)
    validation_targets = validation_targets.to(device)

    best_state: dict[str, torch.Tensor] | None = None
    best_loss = math.inf
    best_epoch = 0
    final_train_loss = math.inf
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_samples = 0
        for batch_features, batch_targets in loader:
            batch_features = batch_features.to(device)
            batch_targets = batch_targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(batch_features)
            loss = _relative_speed_loss(prediction, batch_targets)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(batch_features)
            total_samples += len(batch_features)
        final_train_loss = total_loss / total_samples
        validation_loss = _evaluate(model, validation_features, validation_targets)
        selection_loss = validation_loss if validation_loss is not None else final_train_loss
        if selection_loss < best_loss:
            best_loss = selection_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
        if epoch == 1 or epoch == args.epochs or epoch % args.log_every == 0:
            validation_text = "n/a" if validation_loss is None else f"{validation_loss:.6f}"
            print(
                f"epoch={epoch}/{args.epochs} train_log_beta_smooth_l1={final_train_loss:.6f} "
                f"validation={validation_text}"
            )

    if best_state is None:
        raise RuntimeError("training did not produce a checkpoint state")
    model.load_state_dict(best_state)
    model.eval()
    final_validation_loss = _evaluate(model, validation_features, validation_targets)
    excluded_manual = sum(not record.include_in_training for record in records)
    excluded_total = len(records) - int(keep_mask.sum().item())
    training_metadata = {
        "source_jsonl": str(args.data),
        "source_jsonl_sha256": file_sha256(args.data),
        "schema": "lerobot.speed_throttle.v1",
        "feature_coordinate_space": tensor_data.layout.feature_coordinate_space,
        "num_input_samples": len(records),
        "num_eligible_samples": int(keep_mask.sum().item()),
        "num_training_samples": len(train_indices),
        "num_validation_samples": len(validation_indices),
        "num_failure_events": sum(record.failure_event for record in records),
        "num_manual_exclusions": excluded_manual,
        "num_total_exclusions": excluded_total,
        "failure_window": {
            "preceding_steps": args.failure_preceding_steps,
            "following_steps": args.failure_following_steps,
        },
        "split": split_metadata,
        "optimization": {
            "epochs": args.epochs,
            "best_epoch": best_epoch,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "loss": "smooth_l1(log(predicted_beta), log(beta_target))",
            "best_selection_loss": best_loss,
            "final_train_loss": final_train_loss,
            "best_validation_loss": final_validation_loss,
            "seed": args.seed,
            "device": str(device),
        },
    }
    return TrainingResult(model, tensor_data.layout.feature_names, training_metadata)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.print_schema:
        print(json.dumps(throttle_jsonl_schema(), indent=2, sort_keys=True))
        return 0
    try:
        result = train_speed_adapter(args)
        metadata = save_speed_adapter_checkpoint(
            args.output_dir,
            result.model,
            feature_names=result.feature_names,
            training_metadata=result.metadata,
            overwrite=args.overwrite,
        )
    except (FileNotFoundError, FileExistsError, RuntimeError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    print(
        f"saved speed adapter to {args.output_dir} "
        f"(training_samples={metadata['training']['num_training_samples']}, "
        f"beta=[{metadata['output_contract']['beta_min']}, "
        f"{metadata['output_contract']['beta_max']}])"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
