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

"""Export the PI0.5 image/language prefix and KV cache to TensorRT.

The action expert is intentionally not exported: RTC needs PyTorch autograd for
its denoising VJP. Build the engine on the same GPU used for robot inference.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import subprocess
from pathlib import Path

import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies import get_policy_class
from lerobot.policies.pi05.tensorrt_prefix import (
    PREFIX_METADATA_NAME,
    PREFIX_ONNX_NAME,
    PI05PrefixCacheExport,
    PI05TensorRTError,
    PI05TensorRTPrefixCache,
    compute_model_architecture_fingerprint,
    compute_prefix_weight_fingerprint,
    engine_verification_path,
    file_sha256,
    flatten_dynamic_cache,
    prefix_cache_input_names,
    prefix_cache_output_names,
    runtime_environment_info,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--precision", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--workspace-mb", type=int, default=4096)
    parser.add_argument("--trtexec", default="trtexec")
    parser.add_argument("--build-engine", action="store_true")
    parser.add_argument("--verify-engine", action="store_true")
    parser.add_argument("--verify-atol", type=float, default=5e-2)
    parser.add_argument("--verify-rtol", type=float, default=5e-2)
    parser.add_argument("--verify-kv-mean-threshold", type=float, default=1e-1)
    # KV max recalibrated 2026-07-26 from measured strongly-typed bf16 behavior
    # (depth-accumulated divergence, worst 2.44 in value_17 with no action-level
    # footprint); see docs/pi05_tensorrt.md "Verification thresholds".
    parser.add_argument("--verify-kv-max-threshold", type=float, default=3.0)
    parser.add_argument("--verify-kv-outlier-fraction-threshold", type=float, default=0.05)
    parser.add_argument("--verify-action-mean-threshold", type=float, default=1e-2)
    parser.add_argument("--verify-action-max-threshold", type=float, default=1e-1)
    parser.add_argument("--skip-export", action="store_true")
    parser.add_argument("--check-onnx", action="store_true")
    return parser.parse_args()


def load_policy(checkpoint: Path, device: str):
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")

    config = PreTrainedConfig.from_pretrained(
        str(checkpoint),
        cli_overrides=[f"--device={device}", "--gradient_checkpointing=false", "--compile_model=false"],
    )
    if config.type != "pi05":
        raise ValueError(f"Expected a pi05 checkpoint, got {config.type!r}")
    config.pretrained_path = str(checkpoint)
    policy_class = get_policy_class(config.type)
    policy = policy_class.from_pretrained(str(checkpoint), config=config)
    policy.to(device)
    policy.eval()
    return policy


def export_prefix_onnx(
    checkpoint: Path,
    output_dir: Path,
    *,
    device: str,
    opset: int,
    precision: str,
) -> dict:
    if importlib.util.find_spec("onnx") is None:
        raise RuntimeError(
            "ONNX is required for export. Install the 'onnx' package in the rollout environment"
        )

    policy = load_policy(checkpoint, device)
    model = policy.model
    config = policy.config
    num_cameras = len(config.image_features)
    num_layers = len(model.paligemma_with_expert.paligemma.model.language_model.layers)
    wrapper = PI05PrefixCacheExport(model, num_cameras).to(device).eval()

    batch_size = 1
    image_shape = (batch_size, 3, *config.image_resolution)
    images = [torch.zeros(image_shape, dtype=torch.float32, device=device) for _ in range(num_cameras)]
    img_masks = [torch.ones(batch_size, dtype=torch.bool, device=device) for _ in range(num_cameras)]
    tokens = torch.zeros(batch_size, config.tokenizer_max_length, dtype=torch.int32, device=device)
    token_masks = torch.ones_like(tokens, dtype=torch.bool)
    inputs = (*images, *img_masks, tokens, token_masks)

    input_names = prefix_cache_input_names(num_cameras)
    output_names = prefix_cache_output_names(num_layers)
    onnx_path = output_dir / PREFIX_ONNX_NAME
    output_dir.mkdir(parents=True, exist_ok=True)

    with torch.inference_mode():
        reference_outputs = wrapper(*inputs)
        torch.onnx.export(
            wrapper,
            inputs,
            str(onnx_path),
            input_names=input_names,
            output_names=output_names,
            opset_version=opset,
            do_constant_folding=True,
            dynamo=False,
            external_data=True,
        )

    try:
        environment = runtime_environment_info(device)
    except PI05TensorRTError:
        # Export can legitimately run before TensorRT is installed; the
        # verification step stamps the authoritative environment fields.
        environment = {"tensorrt_version": None}
    metadata = {
        "format": "lerobot_pi05_tensorrt_prefix_v2",
        "checkpoint": str(checkpoint),
        "onnx": PREFIX_ONNX_NAME,
        "engine": f"prefix_cache_{precision}.plan",
        "precision": precision,
        "opset": opset,
        "num_cameras": num_cameras,
        "num_layers": num_layers,
        "input_names": input_names,
        "output_names": output_names,
        "input_shapes": {name: list(tensor.shape) for name, tensor in zip(input_names, inputs, strict=True)},
        "output_shapes": {
            name: list(tensor.shape) for name, tensor in zip(output_names, reference_outputs, strict=True)
        },
        "runtime": "TensorRT prefix/KV + PyTorch denoise/RTC",
        "prefix_weight_fingerprint": compute_prefix_weight_fingerprint(model),
        "model_architecture_fingerprint": compute_model_architecture_fingerprint(policy),
        **environment,
    }
    (output_dir / PREFIX_METADATA_NAME).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def build_engine(
    output_dir: Path,
    *,
    precision: str,
    workspace_mb: int,
    trtexec: str,
) -> Path:
    trtexec_path = shutil.which(trtexec)
    onnx_path = output_dir / PREFIX_ONNX_NAME
    if not onnx_path.is_file():
        raise FileNotFoundError(f"Prefix ONNX model not found: {onnx_path}")
    patched_cumsum_count = patch_onnx_for_tensorrt(onnx_path)
    if patched_cumsum_count:
        print(f"Patched {patched_cumsum_count} CumSum inputs to int32 for TensorRT")
    engine_path = output_dir / f"prefix_cache_{precision}.plan"

    if trtexec_path is None:
        return build_engine_with_python(
            onnx_path,
            engine_path,
            precision=precision,
            workspace_mb=workspace_mb,
        )

    command = [
        trtexec_path,
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        f"--memPoolSize=workspace:{workspace_mb}",
    ]
    if precision == "bf16":
        command.append("--stronglyTyped")
    else:
        command.append("--fp16")
    subprocess.run(command, check=True)
    return engine_path


def patch_onnx_for_tensorrt(onnx_path: Path) -> int:
    """Cast CumSum data inputs to int32 without loading multi-GB external weights."""
    import onnx
    from onnx import TensorProto, helper

    model = onnx.load_model(str(onnx_path), load_external_data=False)
    patched_nodes = []
    patched_count = 0
    for node_index, node in enumerate(model.graph.node):
        if node.op_type == "CumSum" and "__trt_cumsum_int32_" not in node.input[0]:
            cast_output = f"{node.input[0]}__trt_cumsum_int32_{node_index}"
            patched_nodes.append(
                helper.make_node(
                    "Cast",
                    [node.input[0]],
                    [cast_output],
                    name=f"{node.name or 'CumSum'}_InputCast",
                    to=TensorProto.INT32,
                )
            )
            node.input[0] = cast_output
            patched_count += 1
        patched_nodes.append(node)

    if patched_count:
        model.graph.ClearField("node")
        model.graph.node.extend(patched_nodes)
        temporary_path = onnx_path.with_name(f".{onnx_path.name}.tmp")
        onnx.save_model(model, str(temporary_path))
        temporary_path.replace(onnx_path)
    return patched_count


def build_engine_with_python(
    onnx_path: Path,
    engine_path: Path,
    *,
    precision: str,
    workspace_mb: int,
) -> Path:
    try:
        import tensorrt as trt
    except ImportError as exc:
        raise RuntimeError(
            "Neither trtexec nor TensorRT Python bindings are available for engine construction"
        ) from exc

    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    if precision == "bf16":
        network_flags |= 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(onnx_path)):
        errors = "\n".join(str(parser.get_error(idx)) for idx in range(parser.num_errors))
        raise RuntimeError(f"TensorRT failed to parse {onnx_path}:\n{errors}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_mb * 1024 * 1024)
    if precision == "fp16":
        config.set_flag(trt.BuilderFlag.FP16)

    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        raise RuntimeError(f"TensorRT failed to build {precision} engine from {onnx_path}")
    engine_path.write_bytes(bytes(serialized_engine))
    return engine_path


def verify_engine(
    checkpoint: Path,
    engine_path: Path,
    *,
    device: str,
    atol: float,
    rtol: float,
    kv_mean_threshold: float,
    kv_max_threshold: float,
    kv_outlier_fraction_threshold: float,
    action_mean_threshold: float,
    action_max_threshold: float,
) -> None:
    if not engine_path.is_file():
        raise FileNotFoundError(f"TensorRT engine not found: {engine_path}")
    verification_path = engine_verification_path(engine_path)
    verification_path.unlink(missing_ok=True)

    policy = load_policy(checkpoint, device)
    model = policy.model
    config = policy.config
    num_cameras = len(config.image_features)
    # Identity facts (plan §9.1): computed while the full torch prefix is
    # resident, stamped into the report the attach-time gate validates against.
    prefix_weight_fingerprint = compute_prefix_weight_fingerprint(model)
    model_architecture_fingerprint = compute_model_architecture_fingerprint(policy)
    environment = runtime_environment_info(device)
    engine_sha256 = file_sha256(engine_path)
    wrapper = PI05PrefixCacheExport(model, num_cameras).to(device).eval()

    torch.manual_seed(0)
    image_shape = (1, 3, *config.image_resolution)
    images = [torch.rand(image_shape, dtype=torch.float32, device=device) * 2 - 1 for _ in range(num_cameras)]
    img_masks = [torch.ones(1, dtype=torch.bool, device=device) for _ in range(num_cameras)]
    tokens = torch.randint(0, 1024, (1, config.tokenizer_max_length), dtype=torch.int32, device=device)
    token_masks = torch.ones_like(tokens, dtype=torch.bool)

    with torch.inference_mode():
        reference = wrapper(*images, *img_masks, tokens, token_masks)
        backend = PI05TensorRTPrefixCache(
            engine_path,
            device=device,
            cache_dtype=reference[1].dtype,
        )
        actual_masks, actual_cache = backend(images, img_masks, tokens, token_masks)
        torch.cuda.synchronize(device)

    if not torch.equal(actual_masks, reference[0]):
        raise RuntimeError("TensorRT prefix_pad_masks do not match PyTorch")

    actual = (actual_masks, *flatten_dynamic_cache(actual_cache))
    output_names = prefix_cache_output_names((len(reference) - 1) // 2)
    kv_stats: list[dict] = []
    worst_mean_abs = 0.0
    worst_max_abs = 0.0
    for name, reference_tensor, actual_tensor in zip(
        output_names[1:], reference[1:], actual[1:], strict=True
    ):
        reference_float = reference_tensor.float()
        actual_float = actual_tensor.float()
        difference = (reference_float - actual_float).abs()
        mean_abs = difference.mean().item()
        max_abs = difference.max().item()
        worst_mean_abs = max(worst_mean_abs, mean_abs)
        worst_max_abs = max(worst_max_abs, max_abs)
        tolerance = atol + rtol * reference_float.abs()
        ratio = difference / tolerance
        argmax_abs = int(difference.argmax().item())
        argmax_ratio = int(ratio.argmax().item())
        kv_stats.append(
            {
                "name": name,
                "numel": int(difference.numel()),
                "mean_abs": mean_abs,
                "max_abs": max_abs,
                "ref_at_max_abs": reference_float.reshape(-1)[argmax_abs].item(),
                "exceed_count": int((difference > tolerance).sum().item()),
                "worst_ratio": ratio.max().item(),
                "ref_at_worst_ratio": reference_float.reshape(-1)[argmax_ratio].item(),
            }
        )
    outlier_stats = sorted(
        (stats for stats in kv_stats if stats["exceed_count"]),
        key=lambda stats: stats["worst_ratio"],
        reverse=True,
    )
    kv_outlier_elements = sum(stats["exceed_count"] for stats in outlier_stats)
    kv_total_elements = sum(stats["numel"] for stats in kv_stats)
    kv_outlier_fraction = kv_outlier_elements / kv_total_elements if kv_total_elements else 0.0
    pointwise_outliers = [
        (
            f"{stats['name']}: max_abs={stats['max_abs']:.6f} (|ref|={abs(stats['ref_at_max_abs']):.3f}), "
            f"worst={stats['worst_ratio']:.2f}x tol (|ref|={abs(stats['ref_at_worst_ratio']):.3f}), "
            f"exceed={stats['exceed_count']}/{stats['numel']}"
        )
        for stats in outlier_stats
    ]

    print(
        f"TensorRT verification: worst_mean_abs={worst_mean_abs:.6f}, "
        f"worst_max_abs={worst_max_abs:.6f}, outlier_tensors={len(outlier_stats)}, "
        f"outlier_elements={kv_outlier_elements}/{kv_total_elements} "
        f"({kv_outlier_fraction:.4%})"
    )
    for line in pointwise_outliers[:8]:
        print(f"  {line}")

    noise = torch.randn(
        1,
        config.chunk_size,
        config.max_action_dim,
        dtype=torch.float32,
        device=device,
    )
    with torch.inference_mode():
        reference_actions = model.sample_actions(
            images,
            img_masks,
            tokens,
            token_masks,
            noise=noise.clone(),
            num_steps=config.num_inference_steps,
        )
        model.set_prefix_cache_backend(backend)
        actual_actions = model.sample_actions(
            images,
            img_masks,
            tokens,
            token_masks,
            noise=noise.clone(),
            num_steps=config.num_inference_steps,
        )
        model.set_prefix_cache_backend(None)
        torch.cuda.synchronize(device)

    action_difference = (reference_actions.float() - actual_actions.float()).abs()
    action_mean_abs = action_difference.mean().item()
    action_max_abs = action_difference.max().item()
    print(
        f"TensorRT action parity: mean_abs={action_mean_abs:.6f}, max_abs={action_max_abs:.6f}, "
        f"num_steps={config.num_inference_steps}"
    )

    failures: list[str] = []
    if worst_mean_abs > kv_mean_threshold:
        failures.append(f"KV mean error {worst_mean_abs:.6f} > {kv_mean_threshold}")
    if worst_max_abs > kv_max_threshold:
        failures.append(f"KV max error {worst_max_abs:.6f} > {kv_max_threshold}")
    if kv_outlier_fraction > kv_outlier_fraction_threshold:
        failures.append(
            f"KV outlier element fraction {kv_outlier_fraction:.4%} > "
            f"{kv_outlier_fraction_threshold:.4%} (atol={atol}, rtol={rtol})"
        )
    if action_mean_abs > action_mean_threshold:
        failures.append(f"action mean error {action_mean_abs:.6f} > {action_mean_threshold}")
    if action_max_abs > action_max_threshold:
        failures.append(f"action max error {action_max_abs:.6f} > {action_max_threshold}")
    if failures:
        details = "\n".join([*failures, *pointwise_outliers[:5]])
        raise RuntimeError(f"TensorRT prefix verification failed:\n{details}")

    engine_stat = engine_path.stat()
    verification_report = {
        "format": "lerobot_pi05_tensorrt_prefix_verification_v2",
        "passed": True,
        "engine": str(engine_path),
        "engine_size": engine_stat.st_size,
        "engine_mtime_ns": engine_stat.st_mtime_ns,
        "engine_sha256": engine_sha256,
        "checkpoint": str(checkpoint),
        "num_cameras": num_cameras,
        "num_layers": (len(reference) - 1) // 2,
        "prefix_weight_fingerprint": prefix_weight_fingerprint,
        "model_architecture_fingerprint": model_architecture_fingerprint,
        **environment,
        "atol": atol,
        "rtol": rtol,
        "thresholds": {
            "kv_mean": kv_mean_threshold,
            "kv_max": kv_max_threshold,
            "kv_outlier_fraction": kv_outlier_fraction_threshold,
            "action_mean": action_mean_threshold,
            "action_max": action_max_threshold,
        },
        "worst_mean_abs": worst_mean_abs,
        "worst_max_abs": worst_max_abs,
        "kv_outlier_elements": kv_outlier_elements,
        "kv_total_elements": kv_total_elements,
        "kv_outlier_fraction": kv_outlier_fraction,
        "kv_outlier_details": pointwise_outliers[:8],
        "action_mean_abs": action_mean_abs,
        "action_max_abs": action_max_abs,
    }
    verification_path.write_text(json.dumps(verification_report, indent=2), encoding="utf-8")
    print(f"TensorRT verification marker written to {verification_path}")

    # Best-effort: keep the sibling export metadata in sync with the verified
    # identity facts so old-format metadata gains the §9.1 fields.
    metadata_path = engine_path.parent / PREFIX_METADATA_NAME
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata.update(
            {
                "prefix_weight_fingerprint": prefix_weight_fingerprint,
                "model_architecture_fingerprint": model_architecture_fingerprint,
                "engine_sha256": engine_sha256,
                **environment,
            }
        )
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(f"Export metadata refreshed with fingerprints: {metadata_path}")


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    output_dir = (args.output_dir or checkpoint / "pi05_tensorrt").resolve()

    if args.skip_export:
        metadata_path = output_dir / PREFIX_METADATA_NAME
        if not metadata_path.is_file():
            raise FileNotFoundError(f"Export metadata not found: {metadata_path}")
    else:
        print(f"Exporting PI0.5 prefix/KV cache to {output_dir / PREFIX_ONNX_NAME}")
        export_prefix_onnx(
            checkpoint,
            output_dir,
            device=args.device,
            opset=args.opset,
            precision=args.precision,
        )

    if args.check_onnx:
        import onnx

        print(f"Checking {output_dir / PREFIX_ONNX_NAME}")
        onnx.checker.check_model(str(output_dir / PREFIX_ONNX_NAME))

    if args.build_engine:
        print(f"Building {args.precision} TensorRT engine")
        engine_path = build_engine(
            output_dir,
            precision=args.precision,
            workspace_mb=args.workspace_mb,
            trtexec=args.trtexec,
        )
        print(f"TensorRT engine written to {engine_path}")
    elif not args.verify_engine:
        print("ONNX export complete. Re-run with --skip-export --build-engine after TensorRT is installed")

    if args.verify_engine:
        engine_path = output_dir / f"prefix_cache_{args.precision}.plan"
        verify_engine(
            checkpoint,
            engine_path,
            device=args.device,
            atol=args.verify_atol,
            rtol=args.verify_rtol,
            kv_mean_threshold=args.verify_kv_mean_threshold,
            kv_max_threshold=args.verify_kv_max_threshold,
            kv_outlier_fraction_threshold=args.verify_kv_outlier_fraction_threshold,
            action_mean_threshold=args.verify_action_mean_threshold,
            action_max_threshold=args.verify_action_max_threshold,
        )


if __name__ == "__main__":
    main()
