#!/usr/bin/env python

"""Run one offline PI0.5 inference from a local LeRobot dataset sample.

This script never connects to a robot and never sends an action to hardware.
"""

import argparse
import json
import os
from pathlib import Path

DEFAULT_ROOT = Path("/data/cqy_workspace/tk/lerobot_src")
DEFAULT_MODEL = DEFAULT_ROOT / "outputs/train/pi05_so101_local_10epochs_bs32/checkpoints/005613/pretrained_model"
DEFAULT_DATASET = DEFAULT_ROOT / "data/so101_test_data"
DEFAULT_HF_HOME = Path("/data/cqy_workspace/tk/hf_cache/huggingface")

# Keep model and tokenizer loading local. These must be set before importing
# Hugging Face-backed LeRobot modules.
os.environ.setdefault("HF_HOME", str(DEFAULT_HF_HOME))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch  # noqa: E402

from lerobot.configs import PreTrainedConfig  # noqa: E402
from lerobot.datasets import LeRobotDataset  # noqa: E402
from lerobot.policies import get_policy_class, make_pre_post_processors  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--repo-id", default="admin123/so101_test_data")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def require_local_assets(model: Path, dataset_root: Path, device: str) -> None:
    required = (
        model / "config.json",
        model / "model.safetensors",
        model / "policy_preprocessor.json",
        model / "policy_postprocessor.json",
        dataset_root / "meta/info.json",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required local assets:\n" + "\n".join(missing))
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")


def main() -> None:
    args = parse_args()
    model_path = args.model.resolve()
    dataset_root = args.dataset_root.resolve()
    require_local_assets(model_path, dataset_root, args.device)

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    config = PreTrainedConfig.from_pretrained(model_path, local_files_only=True)
    config.device = args.device
    config.gradient_checkpointing = False
    config.compile_model = False

    policy_class = get_policy_class(config.type)
    policy = policy_class.from_pretrained(
        model_path,
        config=config,
        local_files_only=True,
        strict=True,
    )
    policy.eval()
    policy.reset()

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=config,
        pretrained_path=str(model_path),
        preprocessor_overrides={"device_processor": {"device": args.device}},
    )

    dataset = LeRobotDataset(
        repo_id=args.repo_id,
        root=dataset_root,
        video_backend="pyav",
    )
    if not 0 <= args.sample_index < len(dataset):
        raise IndexError(f"sample-index must be in [0, {len(dataset) - 1}]")

    sample = torch.utils.data.default_collate([dataset[args.sample_index]])
    processed_sample = preprocessor(sample)

    with torch.inference_mode():
        normalized_action = policy.select_action(processed_sample)
        action = postprocessor(normalized_action)

    action_cpu = action.detach().float().cpu()
    if not torch.isfinite(action_cpu).all():
        raise RuntimeError("Model produced a non-finite action")

    result = {
        "model": str(model_path),
        "policy_type": config.type,
        "dataset": str(dataset_root),
        "sample_index": args.sample_index,
        "task": sample.get("task"),
        "action_shape": list(action_cpu.shape),
        "action": action_cpu.squeeze(0).tolist(),
        "hardware_action_sent": False,
    }
    print(json.dumps(result, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
