#!/usr/bin/env python

"""Compare PI0.5 PyTorch and TensorRT prefix backends under RTC guidance.

This diagnostic only reads local model and dataset files. It never connects to
robot hardware and never sends an action.
"""

import argparse
import json
import math
import os
import statistics
import time
from pathlib import Path


DEFAULT_ROOT = Path("/data/cqy_workspace/tk/lerobot_src")
DEFAULT_MODEL = (
    DEFAULT_ROOT
    / "outputs/train/pi05_so101_expert_only_10epochs_bs32_seed1000/checkpoints/005613/pretrained_model"
)
DEFAULT_ENGINE = DEFAULT_MODEL / "pi05_tensorrt/prefix_cache_bf16.plan"
DEFAULT_DATASET = DEFAULT_ROOT / "data/so101_test_data"
DEFAULT_HF_HOME = Path("/data/cqy_workspace/tk/hf_cache/huggingface")

os.environ.setdefault("HF_HOME", str(DEFAULT_HF_HOME))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch  # noqa: E402

from lerobot.configs import PreTrainedConfig, RTCAttentionSchedule  # noqa: E402
from lerobot.datasets import LeRobotDataset  # noqa: E402
from lerobot.policies import get_policy_class, make_pre_post_processors  # noqa: E402
from lerobot.policies.pi05.tensorrt_prefix import PI05TensorRTPrefixCache  # noqa: E402
from lerobot.policies.rtc.configuration_rtc import RTCConfig  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--engine", type=Path, default=DEFAULT_ENGINE)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--repo-id", default="admin123/so101_test_data")
    parser.add_argument("--previous-index", type=int, default=0)
    parser.add_argument("--current-index", type=int, default=30)
    parser.add_argument("--leftover-start", type=int, default=30)
    parser.add_argument("--execution-horizon", type=int, default=10)
    parser.add_argument("--delays", type=int, nargs="+", default=[2, 3, 4, 5, 6])
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--max-guidance-weight", type=float, default=10.0)
    parser.add_argument("--timing-repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def require_assets(args: argparse.Namespace) -> None:
    required = (
        args.model / "config.json",
        args.model / "model.safetensors",
        args.model / "policy_preprocessor.json",
        args.model / "policy_postprocessor.json",
        args.engine,
        args.dataset_root / "meta/info.json",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required local assets:\n" + "\n".join(missing))
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if args.fps <= 0:
        raise ValueError("fps must be positive")
    if args.timing_repeats < 1:
        raise ValueError("timing-repeats must be at least 1")


def synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timed_call(function):
    synchronize()
    start = time.perf_counter()
    result = function()
    synchronize()
    return result, (time.perf_counter() - start) * 1000.0


def make_noise(config, device: str, seed: int) -> torch.Tensor:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return torch.randn(
        1,
        config.chunk_size,
        config.max_action_dim,
        dtype=torch.float32,
        device=device,
        generator=generator,
    )


def difference_metrics(reference: torch.Tensor, actual: torch.Tensor) -> dict:
    difference = (reference.float() - actual.float()).abs()
    return {
        "mean_abs": difference.mean().item(),
        "max_abs": difference.max().item(),
    }


def run_backend(
    policy,
    backend,
    sample,
    noise: torch.Tensor,
    prev_actions: torch.Tensor,
    delay: int,
    repeats: int,
    fps: float,
):
    policy.model.set_prefix_cache_backend(backend)
    actions = None
    timings = []
    for _ in range(repeats):
        actions, elapsed_ms = timed_call(
            lambda: policy.predict_action_chunk(
                sample,
                noise=noise.clone(),
                inference_delay=delay,
                prev_chunk_left_over=prev_actions.clone(),
            )
        )
        timings.append(elapsed_ms)
    return actions, {
        "median_ms": statistics.median(timings),
        "min_ms": min(timings),
        "max_ms": max(timings),
        "delay_steps_at_fps": math.ceil(statistics.median(timings) / 1000.0 * fps),
    }


def select_runtime_delay(delay_reports: list[dict], backend: str) -> int:
    timing_key = f"{backend}_timing"
    return min(
        delay_reports,
        key=lambda item: (
            abs(item["delay"] - item[timing_key]["delay_steps_at_fps"]),
            item["delay"],
        ),
    )["delay"]


def main() -> None:
    args = parse_args()
    args.model = args.model.resolve()
    args.engine = args.engine.resolve()
    args.dataset_root = args.dataset_root.resolve()
    require_assets(args)

    config = PreTrainedConfig.from_pretrained(args.model, local_files_only=True)
    config.device = args.device
    config.gradient_checkpointing = False
    config.compile_model = False
    config.rtc_config = RTCConfig(
        enabled=True,
        execution_horizon=args.execution_horizon,
        max_guidance_weight=args.max_guidance_weight,
        prefix_attention_schedule=RTCAttentionSchedule.EXP,
    )

    policy_class = get_policy_class(config.type)
    policy = policy_class.from_pretrained(
        args.model,
        config=config,
        local_files_only=True,
        strict=True,
    )
    policy.init_rtc_processor()
    policy.to(args.device).eval()

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=config,
        pretrained_path=str(args.model),
        preprocessor_overrides={"device_processor": {"device": args.device}},
    )
    dataset = LeRobotDataset(repo_id=args.repo_id, root=args.dataset_root, video_backend="pyav")
    for index in (args.previous_index, args.current_index):
        if not 0 <= index < len(dataset):
            raise IndexError(f"sample index {index} outside [0, {len(dataset) - 1}]")

    previous_raw = torch.utils.data.default_collate([dataset[args.previous_index]])
    current_raw = torch.utils.data.default_collate([dataset[args.current_index]])
    previous_sample = preprocessor(previous_raw)
    current_sample = preprocessor(current_raw)

    cache_dtype = policy.model.paligemma_with_expert.paligemma.model.language_model.layers[
        0
    ].self_attn.k_proj.weight.dtype
    tensorrt_backend = PI05TensorRTPrefixCache(
        args.engine,
        device=args.device,
        cache_dtype=cache_dtype,
    )

    initial_noise = make_noise(config, args.device, args.seed)
    guided_noise = make_noise(config, args.device, args.seed + 1)

    # Build one common previous chunk with PyTorch. Both backends then receive
    # exactly the same RTC prefix so prefix-cache differences are isolated.
    policy.model.set_prefix_cache_backend(None)
    with torch.no_grad():
        previous_actions = policy.predict_action_chunk(previous_sample, noise=initial_noise.clone())
    leftover_end = args.leftover_start + args.execution_horizon
    if leftover_end > previous_actions.shape[1]:
        raise ValueError(
            f"leftover range [{args.leftover_start}, {leftover_end}) exceeds chunk size "
            f"{previous_actions.shape[1]}"
        )
    prev_actions = previous_actions[0, args.leftover_start:leftover_end].clone()

    delay_reports = []
    outputs = {}
    for delay in args.delays:
        pytorch_actions, pytorch_timing = run_backend(
            policy,
            None,
            current_sample,
            guided_noise,
            prev_actions,
            delay,
            args.timing_repeats,
            args.fps,
        )
        tensorrt_actions, tensorrt_timing = run_backend(
            policy,
            tensorrt_backend,
            current_sample,
            guided_noise,
            prev_actions,
            delay,
            args.timing_repeats,
            args.fps,
        )
        outputs[("pytorch", delay)] = pytorch_actions
        outputs[("tensorrt", delay)] = tensorrt_actions

        pytorch_robot = postprocessor(pytorch_actions)
        tensorrt_robot = postprocessor(tensorrt_actions)
        delay_reports.append(
            {
                "delay": delay,
                "normalized_action": difference_metrics(pytorch_actions, tensorrt_actions),
                "robot_units": difference_metrics(pytorch_robot, tensorrt_robot),
                "pytorch_timing": pytorch_timing,
                "tensorrt_timing": tensorrt_timing,
            }
        )

    pytorch_runtime_delay = select_runtime_delay(delay_reports, "pytorch")
    tensorrt_runtime_delay = select_runtime_delay(delay_reports, "tensorrt")

    pytorch_runtime_actions = outputs[("pytorch", pytorch_runtime_delay)]
    tensorrt_runtime_actions = outputs[("tensorrt", tensorrt_runtime_delay)]
    pytorch_runtime_robot = postprocessor(pytorch_runtime_actions)
    tensorrt_runtime_robot = postprocessor(tensorrt_runtime_actions)
    pytorch_first_index = min(pytorch_runtime_delay, pytorch_runtime_actions.shape[1] - 1)
    tensorrt_first_index = min(tensorrt_runtime_delay, tensorrt_runtime_actions.shape[1] - 1)

    runtime_comparison = {
        "pytorch_guidance_delay": pytorch_runtime_delay,
        "tensorrt_guidance_delay": tensorrt_runtime_delay,
        "pytorch_first_executed_index": pytorch_first_index,
        "tensorrt_first_executed_index": tensorrt_first_index,
        "full_chunks_normalized": difference_metrics(pytorch_runtime_actions, tensorrt_runtime_actions),
        "full_chunks_robot_units": difference_metrics(pytorch_runtime_robot, tensorrt_runtime_robot),
        "first_executed_action_robot_units": difference_metrics(
            pytorch_runtime_robot[:, pytorch_first_index],
            tensorrt_runtime_robot[:, tensorrt_first_index],
        ),
    }

    report = {
        "model": str(args.model),
        "engine": str(args.engine),
        "dataset": str(args.dataset_root),
        "previous_index": args.previous_index,
        "current_index": args.current_index,
        "rtc": {
            "execution_horizon": args.execution_horizon,
            "max_guidance_weight": args.max_guidance_weight,
            "prefix_attention_schedule": "EXP",
            "fps": args.fps,
            "leftover_range": [args.leftover_start, leftover_end],
        },
        "delays": delay_reports,
        "runtime_comparison": runtime_comparison,
        "hardware_action_sent": False,
    }
    output_json = args.output_json or args.engine.with_suffix(f"{args.engine.suffix}.rtc_parity.json")
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"RTC parity report written to {output_json}")


if __name__ == "__main__":
    main()
