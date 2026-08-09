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

"""Serial fixed-noise parity for PI0.5 PyTorch and full-model Triton.

Run ``make-request``, ``run-pytorch``, and ``run-triton`` as separate
processes. This prevents both 9+ GiB model representations from occupying the
GPU together. Every CUDA command checks free memory before model creation.
"""

from __future__ import annotations

import argparse
import gc
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from lerobot.policies.pi05.realtime_vla_v2_triton import (
    FixedNoiseParityHarness,
    PI05PolicyFixedNoiseReference,
    PI05RealtimeVLATritonBackend,
    PI05RealtimeVLATritonConfig,
    PI05RealtimeVLATritonOutput,
    make_fixed_diffusion_noise,
    require_cuda_free_memory,
)
from lerobot.rollout.pi05_parity import (
    build_pi05_parity_report,
    compute_pi05_parity_request_fingerprint,
    load_pi05_parity_request,
    write_pi05_parity_report_atomic,
)

_REQUEST_SCHEMA = 1
_RESULT_SCHEMA = 1


@dataclass(frozen=True)
class ParityRequest:
    images: dict[str, np.ndarray]
    normalized_state: np.ndarray
    normalized_prefill_actions: np.ndarray
    prompt: str
    camera_keys: tuple[str, str]
    image_value_range: str
    fingerprint: str


@dataclass(frozen=True)
class ParityResult:
    actions: np.ndarray
    noise: np.ndarray
    prefill_length: int
    implementation: str
    request_fingerprint: str
    seed: int


request_fingerprint = compute_pi05_parity_request_fingerprint


def save_request(path: str | Path, request: ParityRequest) -> None:
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "schema": _REQUEST_SCHEMA,
        "prompt": request.prompt,
        "camera_keys": list(request.camera_keys),
        "image_value_range": request.image_value_range,
        "fingerprint": request.fingerprint,
    }
    np.savez_compressed(
        path,
        metadata=np.asarray(json.dumps(metadata)),
        camera_0=request.images[request.camera_keys[0]],
        camera_1=request.images[request.camera_keys[1]],
        normalized_state=request.normalized_state,
        normalized_prefill_actions=request.normalized_prefill_actions,
    )


def load_request(path: str | Path) -> ParityRequest:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Parity request not found: {path}")
    validated = load_pi05_parity_request(path)
    return ParityRequest(
        images=validated.images,
        normalized_state=validated.normalized_state,
        normalized_prefill_actions=validated.normalized_prefill_actions,
        prompt=validated.prompt,
        camera_keys=validated.camera_keys,
        image_value_range=validated.image_value_range,
        fingerprint=validated.fingerprint,
    )


def save_result(
    path: str | Path,
    output: PI05RealtimeVLATritonOutput,
    noise: np.ndarray,
    *,
    implementation: str,
    request: ParityRequest,
    seed: int,
) -> None:
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "schema": _RESULT_SCHEMA,
        "implementation": implementation,
        "request_fingerprint": request.fingerprint,
        "prefill_length": output.prefill_length,
        "seed": seed,
    }
    np.savez_compressed(
        path,
        metadata=np.asarray(json.dumps(metadata)),
        actions=output.actions,
        noise=noise,
    )


def load_result(path: str | Path) -> ParityResult:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Parity result not found: {path}")
    with np.load(path, allow_pickle=False) as archive:
        if len(archive.files) != 3 or set(archive.files) != {"metadata", "actions", "noise"}:
            raise ValueError(f"Invalid parity result archive keys: {sorted(archive.files)}")
        metadata_raw = archive["metadata"]
        if metadata_raw.shape != () or not isinstance(metadata_raw.item(), str):
            raise ValueError("Parity result metadata must be a scalar JSON string")
        metadata = json.loads(metadata_raw.item())
        actions = np.asarray(archive["actions"])
        noise = np.asarray(archive["noise"])
    expected_metadata_keys = {
        "schema",
        "implementation",
        "request_fingerprint",
        "prefill_length",
        "seed",
    }
    if not isinstance(metadata, dict) or set(metadata) != expected_metadata_keys:
        raise ValueError("Invalid parity result metadata keys")
    if metadata["schema"] != _RESULT_SCHEMA:
        raise ValueError(f"Unsupported parity result schema: {metadata['schema']!r}")
    if actions.dtype != np.float32 or noise.dtype != np.float32:
        raise ValueError(
            f"Parity result actions and noise must be float32, got {actions.dtype} and {noise.dtype}"
        )
    if actions.shape != (50, 6) or noise.shape != (50, 32):
        raise ValueError(f"Invalid parity result shapes: actions={actions.shape}, noise={noise.shape}")
    if not np.isfinite(actions).all() or not np.isfinite(noise).all():
        raise ValueError("Parity result contains NaN or infinity")
    implementation = metadata["implementation"]
    fingerprint = metadata["request_fingerprint"]
    prefill_length = metadata["prefill_length"]
    seed = metadata["seed"]
    if not isinstance(implementation, str) or not implementation:
        raise ValueError("Parity result implementation must be a non-empty string")
    if (
        not isinstance(fingerprint, str)
        or len(fingerprint) != 64
        or any(character not in "0123456789abcdefABCDEF" for character in fingerprint)
    ):
        raise ValueError("Parity result request_fingerprint must be 64 hexadecimal characters")
    if (
        isinstance(prefill_length, bool)
        or not isinstance(prefill_length, int)
        or not 0 <= prefill_length < 50
    ):
        raise ValueError("Parity result prefill_length must be an integer in [0,50)")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("Parity result seed must be a non-negative integer")
    return ParityResult(
        actions=actions,
        noise=noise,
        prefill_length=prefill_length,
        implementation=implementation,
        request_fingerprint=fingerprint.lower(),
        seed=seed,
    )


def make_synthetic_request(args: argparse.Namespace) -> None:
    if not 0 <= args.prefill_length < 50:
        raise ValueError("prefill-length must be in [0,50)")
    rng = np.random.default_rng(args.request_seed)
    camera_keys = tuple(args.camera_keys)
    images = {
        key: rng.random((args.image_height, args.image_width, 3), dtype=np.float32) for key in camera_keys
    }
    state = rng.uniform(-0.75, 0.75, size=(6,)).astype(np.float32)
    prefix = rng.uniform(-0.75, 0.75, size=(args.prefill_length, 6)).astype(np.float32)
    fingerprint = request_fingerprint(
        images=images,
        normalized_state=state,
        normalized_prefill_actions=prefix,
        prompt=args.prompt,
        camera_keys=camera_keys,
        image_value_range="zero_one",
    )
    save_request(
        args.output,
        ParityRequest(
            images=images,
            normalized_state=state,
            normalized_prefill_actions=prefix,
            prompt=args.prompt,
            camera_keys=camera_keys,
            image_value_range="zero_one",
            fingerprint=fingerprint,
        ),
    )


def _runtime_config(
    args: argparse.Namespace, request: ParityRequest, **kwargs: Any
) -> PI05RealtimeVLATritonConfig:
    return PI05RealtimeVLATritonConfig(
        prompt=request.prompt,
        tokenizer_path=args.tokenizer_path,
        camera_keys=request.camera_keys,
        device=args.device,
        image_value_range=request.image_value_range,
        min_free_cuda_gib=args.min_free_gib,
        **kwargs,
    )


def _strict_load_pi05_policy(checkpoint: Path, device: str):
    from safetensors.torch import load_file

    from lerobot.configs import PreTrainedConfig
    from lerobot.policies.pi05 import PI05Policy

    config = PreTrainedConfig.from_pretrained(checkpoint)
    config.device = device
    config.gradient_checkpointing = False
    config.compile_model = False
    policy = PI05Policy(config)
    state_dict = load_file(checkpoint / "model.safetensors", device="cpu")
    fixed = policy._fix_pytorch_state_dict_keys(state_dict, policy.config)
    remapped = {key if key.startswith("model.") else f"model.{key}": value for key, value in fixed.items()}
    policy.load_state_dict(remapped, strict=True)
    del remapped, fixed, state_dict
    policy.eval()
    return policy


def run_pytorch(args: argparse.Namespace) -> None:
    request = load_request(args.request)
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    if not (checkpoint / "model.safetensors").is_file() or not (checkpoint / "config.json").is_file():
        raise FileNotFoundError(f"PI0.5 pretrained_model directory is incomplete: {checkpoint}")
    require_cuda_free_memory(args.device, args.min_free_gib)
    config = _runtime_config(args, request)
    policy = _strict_load_pi05_policy(checkpoint, args.device)
    noise = make_fixed_diffusion_noise(args.seed)
    try:
        reference = PI05PolicyFixedNoiseReference(policy, config)
        output = reference.infer(
            images=request.images,
            normalized_state=request.normalized_state,
            normalized_prefill_actions=request.normalized_prefill_actions,
            noise=noise,
            prompt=request.prompt,
        )
        save_result(
            args.output,
            output,
            noise,
            implementation="lerobot-pytorch",
            request=request,
            seed=args.seed,
        )
    finally:
        del policy
        gc.collect()
        torch.cuda.empty_cache()


def run_triton(args: argparse.Namespace) -> None:
    request = load_request(args.request)
    config = _runtime_config(
        args,
        request,
        weights_path=args.weights,
        model_config_path=args.model_config,
        weights_sha256=args.weights_sha256,
    )
    noise = make_fixed_diffusion_noise(args.seed)
    backend = PI05RealtimeVLATritonBackend.from_export(config)
    output = backend.infer(
        images=request.images,
        normalized_state=request.normalized_state,
        normalized_prefill_actions=request.normalized_prefill_actions,
        noise=noise,
        prompt=request.prompt,
    )
    save_result(
        args.output,
        output,
        noise,
        implementation="realtime-vla-v2-triton",
        request=request,
        seed=args.seed,
    )


def _compare_result_pair(
    pytorch_path: str | Path,
    triton_path: str | Path,
    *,
    atol: float,
    rtol: float,
    request_path: str | Path | None = None,
) -> dict[str, Any]:
    pytorch_path = Path(pytorch_path).expanduser().resolve()
    triton_path = Path(triton_path).expanduser().resolve()
    pytorch = load_result(pytorch_path)
    triton = load_result(triton_path)
    if pytorch.implementation != "lerobot-pytorch":
        raise ValueError(f"Expected a lerobot-pytorch result, got {pytorch.implementation!r}: {pytorch_path}")
    if triton.implementation != "realtime-vla-v2-triton":
        raise ValueError(
            f"Expected a realtime-vla-v2-triton result, got {triton.implementation!r}: {triton_path}"
        )
    if pytorch.request_fingerprint != triton.request_fingerprint:
        raise ValueError("Parity results were produced from different requests")
    if pytorch.seed != triton.seed:
        raise ValueError("Parity results disagree on fixed diffusion noise seed")
    expected_noise = np.random.default_rng(pytorch.seed).standard_normal((50, 32)).astype(np.float32)
    if not np.array_equal(pytorch.noise, expected_noise):
        raise ValueError("PyTorch result noise does not match its fixed diffusion noise seed")
    if not np.array_equal(triton.noise, expected_noise):
        raise ValueError("Triton result noise does not match its fixed diffusion noise seed")
    if pytorch.prefill_length != triton.prefill_length:
        raise ValueError("Parity results disagree on prefill length")
    request = load_request(request_path) if request_path is not None else None
    if request is not None:
        if pytorch.request_fingerprint != request.fingerprint:
            raise ValueError("Parity results are not bound to the declared request")
        if pytorch.prefill_length != request.normalized_prefill_actions.shape[0]:
            raise ValueError("Parity result prefill length does not match the declared request")
    report = FixedNoiseParityHarness(atol=atol, rtol=rtol).compare(triton.actions, pytorch.actions)
    result = {
        "prefix_length": pytorch.prefill_length,
        "passed": report.passed,
        "max_abs_error": report.max_abs_error,
        "mean_abs_error": report.mean_abs_error,
        "shape": report.shape,
        "seed": pytorch.seed,
        "request_fingerprint": pytorch.request_fingerprint,
        "pytorch_output_path": str(pytorch_path),
        "triton_output_path": str(triton_path),
    }
    if request_path is not None:
        result["request_path"] = str(Path(request_path).expanduser().resolve())
    return result


def compare_results(args: argparse.Namespace) -> None:
    result = _compare_result_pair(
        args.pytorch_output,
        args.triton_output,
        atol=args.atol,
        rtol=args.rtol,
    )
    printable = {
        key: value
        for key, value in result.items()
        if key not in {"pytorch_output_path", "triton_output_path"}
    }
    print(json.dumps(printable, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


def aggregate_results(args: argparse.Namespace) -> None:
    comparisons: list[dict[str, Any]] = []
    for prefix_text, request_path, pytorch_path, triton_path in args.pair:
        try:
            expected_prefix = int(prefix_text)
        except ValueError as exc:
            raise ValueError(f"Aggregate pair prefix must be an integer, got {prefix_text!r}") from exc
        comparison = _compare_result_pair(
            pytorch_path,
            triton_path,
            atol=args.atol,
            rtol=args.rtol,
            request_path=request_path,
        )
        if comparison["prefix_length"] != expected_prefix:
            raise ValueError(
                f"Aggregate pair declared prefix {expected_prefix}, but artifacts contain "
                f"prefix {comparison['prefix_length']}"
            )
        comparisons.append(comparison)

    report = build_pi05_parity_report(
        checkpoint_path=args.checkpoint,
        triton_weights_path=args.weights,
        triton_weights_sha256=args.weights_sha256,
        triton_model_config_path=args.model_config,
        training_max_delay=args.training_max_delay,
        atol=args.atol,
        rtol=args.rtol,
        prefix_results=comparisons,
    )
    report_sha256 = write_pi05_parity_report_atomic(args.output, report)
    print(
        json.dumps(
            {
                "output": str(Path(args.output).expanduser().resolve()),
                "sha256": report_sha256,
                "passed": report["passed"],
                "prefixes": [result["prefix_length"] for result in report["prefix_results"]],
                "thresholds": report["thresholds"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    if not report["passed"]:
        raise SystemExit(1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    make = subparsers.add_parser("make-request", help="Write a deterministic CPU parity request")
    make.add_argument("--output", type=Path, required=True)
    make.add_argument("--prompt", required=True)
    make.add_argument(
        "--camera-keys", nargs=2, default=("observation.images.top", "observation.images.wrist")
    )
    make.add_argument("--prefill-length", type=int, default=6)
    make.add_argument("--request-seed", type=int, default=1000)
    make.add_argument("--image-height", type=int, default=480)
    make.add_argument("--image-width", type=int, default=640)
    make.set_defaults(handler=make_synthetic_request)

    def add_run_common(run_parser: argparse.ArgumentParser) -> None:
        run_parser.add_argument("--request", type=Path, required=True)
        run_parser.add_argument("--output", type=Path, required=True)
        run_parser.add_argument("--tokenizer-path", default="google/paligemma-3b-pt-224")
        run_parser.add_argument("--device", default="cuda")
        run_parser.add_argument("--min-free-gib", type=float, default=18.0)
        run_parser.add_argument("--seed", type=int, default=0)

    pytorch = subparsers.add_parser("run-pytorch", help="Run and save the LeRobot PyTorch reference")
    add_run_common(pytorch)
    pytorch.add_argument("--checkpoint", type=Path, required=True)
    pytorch.set_defaults(handler=run_pytorch)

    triton = subparsers.add_parser("run-triton", help="Run and save the full Triton backend")
    add_run_common(triton)
    triton.add_argument("--weights", type=Path, required=True)
    triton.add_argument("--model-config", type=Path, required=True)
    triton.add_argument("--weights-sha256")
    triton.set_defaults(handler=run_triton)

    compare = subparsers.add_parser("compare", help="Compare two serialized outputs on CPU")
    compare.add_argument("--pytorch-output", type=Path, required=True)
    compare.add_argument("--triton-output", type=Path, required=True)
    compare.add_argument("--atol", type=float, default=8e-2)
    compare.add_argument("--rtol", type=float, default=2e-2)
    compare.set_defaults(handler=compare_results)

    aggregate = subparsers.add_parser(
        "aggregate",
        help="Atomically aggregate complete 0..training_max_delay CPU compare evidence",
    )
    aggregate.add_argument("--checkpoint", type=Path, required=True)
    aggregate.add_argument("--weights", type=Path, required=True)
    aggregate.add_argument("--weights-sha256", required=True)
    aggregate.add_argument("--model-config", type=Path, required=True)
    aggregate.add_argument("--training-max-delay", type=int, required=True)
    aggregate.add_argument("--atol", type=float, default=8e-2)
    aggregate.add_argument("--rtol", type=float, default=2e-2)
    aggregate.add_argument("--output", type=Path, required=True)
    aggregate.add_argument(
        "--pair",
        action="append",
        nargs=4,
        required=True,
        metavar=("PREFIX", "REQUEST", "PYTORCH_OUTPUT", "TRITON_OUTPUT"),
        help="Repeat once for every trained prefix length",
    )
    aggregate.set_defaults(handler=aggregate_results)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.handler(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
