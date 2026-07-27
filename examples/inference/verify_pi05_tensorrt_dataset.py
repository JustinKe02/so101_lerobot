#!/usr/bin/env python

"""Verify PI0.5 TensorRT prefix parity on real local dataset frames.

This script never connects to a robot and never sends an action to hardware.
"""

import argparse
import json
import os
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

from lerobot.configs import PreTrainedConfig  # noqa: E402
from lerobot.datasets import LeRobotDataset  # noqa: E402
from lerobot.policies import get_policy_class, make_pre_post_processors  # noqa: E402
from lerobot.policies.pi05.tensorrt_prefix import (  # noqa: E402
    PI05PrefixCacheExport,
    PI05TensorRTPrefixCache,
    flatten_dynamic_cache,
    prefix_cache_output_names,
)
from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--engine", type=Path, default=DEFAULT_ENGINE)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--repo-id", default="admin123/so101_test_data")
    parser.add_argument("--sample-indices", type=int, nargs="+", default=[0, 4490, 8980, 13470, 17959])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timing-repeats", type=int, default=3)
    parser.add_argument("--kv-mean-threshold", type=float, default=1e-1)
    parser.add_argument("--action-mean-threshold", type=float, default=1e-2)
    parser.add_argument("--action-max-threshold", type=float, default=1e-1)
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def require_local_assets(model: Path, engine: Path, dataset_root: Path, device: str) -> None:
    required = (
        model / "config.json",
        model / "model.safetensors",
        model / "policy_preprocessor.json",
        model / "policy_postprocessor.json",
        engine,
        dataset_root / "meta/info.json",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required local assets:\n" + "\n".join(missing))
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")


def timed_cuda_call(function):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    result = function()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return result, (time.perf_counter() - start) * 1000


def benchmark_cuda_call(function, repeats: int) -> float:
    timings = [timed_cuda_call(function)[1] for _ in range(repeats)]
    return sum(timings) / len(timings)


def summarize_differences(reference, actual, names: list[str]) -> dict:
    differences = []
    worst_name = None
    worst_mean = 0.0
    worst_max = 0.0
    for name, reference_tensor, actual_tensor in zip(names, reference, actual, strict=True):
        difference = (reference_tensor.float() - actual_tensor.float()).abs()
        differences.append(difference.flatten())
        mean_abs = difference.mean().item()
        max_abs = difference.max().item()
        if max_abs > worst_max:
            worst_name = name
            worst_max = max_abs
        worst_mean = max(worst_mean, mean_abs)

    all_differences = torch.cat(differences)
    quantiles = torch.quantile(
        all_differences,
        torch.tensor([0.5, 0.99, 0.999, 0.9999], device=all_differences.device),
    ).cpu()
    return {
        "num_elements": all_differences.numel(),
        "worst_tensor": worst_name,
        "worst_mean_abs": worst_mean,
        "worst_max_abs": worst_max,
        "p50_abs": quantiles[0].item(),
        "p99_abs": quantiles[1].item(),
        "p999_abs": quantiles[2].item(),
        "p9999_abs": quantiles[3].item(),
        "count_gt_1": int((all_differences > 1.0).sum().item()),
        "count_gt_2": int((all_differences > 2.0).sum().item()),
        "count_gt_4": int((all_differences > 4.0).sum().item()),
    }


def main() -> None:
    args = parse_args()
    model_path = args.model.resolve()
    engine_path = args.engine.resolve()
    dataset_root = args.dataset_root.resolve()
    output_json = args.output_json or engine_path.with_suffix(f"{engine_path.suffix}.dataset_parity.json")
    require_local_assets(model_path, engine_path, dataset_root, args.device)

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

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=config,
        pretrained_path=str(model_path),
        preprocessor_overrides={"device_processor": {"device": args.device}},
    )
    dataset = LeRobotDataset(repo_id=args.repo_id, root=dataset_root, video_backend="pyav")
    invalid_indices = [index for index in args.sample_indices if not 0 <= index < len(dataset)]
    if invalid_indices:
        raise IndexError(f"sample indices outside [0, {len(dataset) - 1}]: {invalid_indices}")
    if args.timing_repeats < 1:
        raise ValueError("timing-repeats must be at least 1")

    model = policy.model
    num_cameras = len(config.image_features)
    wrapper = PI05PrefixCacheExport(model, num_cameras).to(args.device).eval()
    cache_dtype = model.paligemma_with_expert.paligemma.model.language_model.layers[
        0
    ].self_attn.k_proj.weight.dtype
    backend = PI05TensorRTPrefixCache(engine_path, device=args.device, cache_dtype=cache_dtype)
    output_names = prefix_cache_output_names(backend.num_layers)[1:]

    sample_reports = []
    for sample_index in args.sample_indices:
        sample = torch.utils.data.default_collate([dataset[sample_index]])
        processed_sample = preprocessor(sample)
        images, img_masks = policy._preprocess_images(processed_sample)
        tokens = processed_sample[OBS_LANGUAGE_TOKENS]
        token_masks = processed_sample[OBS_LANGUAGE_ATTENTION_MASK]

        with torch.inference_mode():
            reference_prefix, _ = timed_cuda_call(
                lambda images=images, img_masks=img_masks, tokens=tokens, token_masks=token_masks: wrapper(
                    *images, *img_masks, tokens, token_masks
                )
            )
            (actual_masks, actual_cache), _ = timed_cuda_call(
                lambda images=images, img_masks=img_masks, tokens=tokens, token_masks=token_masks: backend(
                    images, img_masks, tokens, token_masks
                )
            )
            actual_prefix = flatten_dynamic_cache(actual_cache)
            kv_metrics = summarize_differences(reference_prefix[1:], actual_prefix, output_names)
            pytorch_prefix_ms = benchmark_cuda_call(
                lambda images=images, img_masks=img_masks, tokens=tokens, token_masks=token_masks: wrapper(
                    *images, *img_masks, tokens, token_masks
                ),
                args.timing_repeats,
            )
            tensorrt_prefix_ms = benchmark_cuda_call(
                lambda images=images, img_masks=img_masks, tokens=tokens, token_masks=token_masks: backend(
                    images, img_masks, tokens, token_masks
                ),
                args.timing_repeats,
            )

            generator = torch.Generator(device=args.device)
            generator.manual_seed(args.seed + sample_index)
            noise = torch.randn(
                1,
                config.chunk_size,
                config.max_action_dim,
                dtype=torch.float32,
                device=args.device,
                generator=generator,
            )

            model.set_prefix_cache_backend(None)
            reference_actions, _ = timed_cuda_call(
                lambda processed_sample=processed_sample, noise=noise: policy.predict_action_chunk(
                    processed_sample, noise=noise.clone()
                )
            )
            model.set_prefix_cache_backend(backend)
            actual_actions, _ = timed_cuda_call(
                lambda processed_sample=processed_sample, noise=noise: policy.predict_action_chunk(
                    processed_sample, noise=noise.clone()
                )
            )
            model.set_prefix_cache_backend(None)

            action_difference = (reference_actions.float() - actual_actions.float()).abs()
            reference_robot_actions = postprocessor(reference_actions)
            actual_robot_actions = postprocessor(actual_actions)
            robot_action_difference = (reference_robot_actions.float() - actual_robot_actions.float()).abs()

            model.set_prefix_cache_backend(None)
            pytorch_action_ms = benchmark_cuda_call(
                lambda processed_sample=processed_sample, noise=noise: policy.predict_action_chunk(
                    processed_sample, noise=noise.clone()
                ),
                args.timing_repeats,
            )
            model.set_prefix_cache_backend(backend)
            tensorrt_action_ms = benchmark_cuda_call(
                lambda processed_sample=processed_sample, noise=noise: policy.predict_action_chunk(
                    processed_sample, noise=noise.clone()
                ),
                args.timing_repeats,
            )
            model.set_prefix_cache_backend(None)

        sample_reports.append(
            {
                "sample_index": sample_index,
                "episode_index": int(sample["episode_index"].item()),
                "frame_index": int(sample["frame_index"].item()),
                "prefix_masks_equal": bool(torch.equal(reference_prefix[0], actual_masks)),
                "kv": kv_metrics,
                "action": {
                    "mean_abs": action_difference.mean().item(),
                    "max_abs": action_difference.max().item(),
                    "robot_units_mean_abs": robot_action_difference.mean().item(),
                    "robot_units_max_abs": robot_action_difference.max().item(),
                },
                "latency_ms": {
                    "pytorch_prefix": pytorch_prefix_ms,
                    "tensorrt_prefix": tensorrt_prefix_ms,
                    "pytorch_full_action": pytorch_action_ms,
                    "tensorrt_full_action": tensorrt_action_ms,
                },
            }
        )

    aggregate = {
        "all_prefix_masks_equal": all(item["prefix_masks_equal"] for item in sample_reports),
        "worst_kv_mean_abs": max(item["kv"]["worst_mean_abs"] for item in sample_reports),
        "worst_kv_max_abs": max(item["kv"]["worst_max_abs"] for item in sample_reports),
        "worst_action_mean_abs": max(item["action"]["mean_abs"] for item in sample_reports),
        "worst_action_max_abs": max(item["action"]["max_abs"] for item in sample_reports),
        "mean_pytorch_prefix_ms": sum(item["latency_ms"]["pytorch_prefix"] for item in sample_reports)
        / len(sample_reports),
        "mean_tensorrt_prefix_ms": sum(item["latency_ms"]["tensorrt_prefix"] for item in sample_reports)
        / len(sample_reports),
        "mean_pytorch_full_action_ms": sum(
            item["latency_ms"]["pytorch_full_action"] for item in sample_reports
        )
        / len(sample_reports),
        "mean_tensorrt_full_action_ms": sum(
            item["latency_ms"]["tensorrt_full_action"] for item in sample_reports
        )
        / len(sample_reports),
    }
    aggregate["prefix_speedup"] = aggregate["mean_pytorch_prefix_ms"] / aggregate["mean_tensorrt_prefix_ms"]
    aggregate["full_action_speedup"] = (
        aggregate["mean_pytorch_full_action_ms"] / aggregate["mean_tensorrt_full_action_ms"]
    )
    aggregate["passed"] = (
        aggregate["all_prefix_masks_equal"]
        and aggregate["worst_kv_mean_abs"] <= args.kv_mean_threshold
        and aggregate["worst_action_mean_abs"] <= args.action_mean_threshold
        and aggregate["worst_action_max_abs"] <= args.action_max_threshold
    )

    report = {
        "model": str(model_path),
        "engine": str(engine_path),
        "dataset": str(dataset_root),
        "sample_indices": args.sample_indices,
        "timing_repeats": args.timing_repeats,
        "thresholds": {
            "kv_mean": args.kv_mean_threshold,
            "action_mean": args.action_mean_threshold,
            "action_max": args.action_max_threshold,
        },
        "aggregate": aggregate,
        "samples": sample_reports,
        "hardware_action_sent": False,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not aggregate["passed"]:
        raise RuntimeError(f"Dataset parity thresholds failed; report written to {output_json}")
    print(f"Dataset parity report written to {output_json}")


if __name__ == "__main__":
    main()
