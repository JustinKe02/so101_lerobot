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

"""TensorRT prefix/KV-cache acceleration for PI0.5 inference.

Only the image/language prefix is executed by TensorRT. The action expert stays
in PyTorch so RTC can continue to compute its VJP with ``torch.autograd``.
"""

from __future__ import annotations

import gc
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from lerobot.policies.common.vla_utils import (
    make_att_2d_masks,
    prepare_attention_masks_4d,
)

PREFIX_ONNX_NAME = "prefix_cache.onnx"
PREFIX_METADATA_NAME = "prefix_cache_metadata.json"

# Fields every verification report must carry before an engine may be attached.
# Reports produced by older exporter revisions lack these and are rejected —
# re-run ``lerobot-export-pi05-tensorrt --skip-export --verify-engine``.
REQUIRED_VERIFICATION_FIELDS = (
    "passed",
    "engine_sha256",
    "prefix_weight_fingerprint",
    "model_architecture_fingerprint",
    "tensorrt_version",
    "gpu_compute_capability",
    "num_cameras",
)


class PI05TensorRTError(RuntimeError):
    """Raised when the PI0.5 TensorRT prefix backend cannot be used."""


def engine_verification_path(engine_path: str | Path) -> Path:
    """Return the verification-report path next to *engine_path* (``<engine>.plan.verified.json``)."""
    engine_path = Path(engine_path)
    return engine_path.with_suffix(f"{engine_path.suffix}.verified.json")


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def compute_prefix_weight_fingerprint(model: nn.Module) -> str:
    """SHA-256 over the live PaliGemma prefix weights (name, shape, dtype, raw bytes).

    Computed identically at export, verification, and attach time so an engine
    can be tied to the exact prefix weights it was built from — comparing
    checkpoint paths is not sufficient (plan §9.1).  Must run while the torch
    prefix modules are still resident (before ``release_torch_prefix_modules``).
    """
    paligemma_with_expert = getattr(model, "paligemma_with_expert", None)
    paligemma = getattr(paligemma_with_expert, "paligemma", None)
    if paligemma is None:
        raise PI05TensorRTError(
            "Prefix weight fingerprint requires the torch prefix modules to be resident "
            "(compute it before release_torch_prefix_modules)"
        )
    state = paligemma.state_dict()
    if not state:
        raise PI05TensorRTError("PaliGemma prefix state_dict is empty")
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().to("cpu").contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("utf-8"))
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy())
    return digest.hexdigest()


def compute_model_architecture_fingerprint(policy: Any) -> str:
    """SHA-256 over the architecture facts the engine bakes in (shapes, cameras, layers)."""
    config = policy.config
    model = policy.model
    num_layers = len(model.paligemma_with_expert.paligemma.model.language_model.layers)
    payload = {
        "policy_type": str(config.type),
        "num_cameras": len(config.image_features),
        "image_resolution": [int(value) for value in config.image_resolution],
        "tokenizer_max_length": int(config.tokenizer_max_length),
        "chunk_size": int(config.chunk_size),
        "max_action_dim": int(config.max_action_dim),
        "num_prefix_layers": num_layers,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def runtime_environment_info(device: str | torch.device) -> dict[str, str | None]:
    """TensorRT/CUDA/GPU facts an engine is only valid for (plan §9.1)."""
    try:
        import tensorrt as trt
    except ImportError as exc:
        raise PI05TensorRTError("TensorRT Python bindings are not installed in this environment") from exc
    device = torch.device(device)
    if device.type != "cuda":
        raise PI05TensorRTError("PI0.5 TensorRT inference requires a CUDA device")
    capability = torch.cuda.get_device_capability(device)
    return {
        "tensorrt_version": str(trt.__version__),
        "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(device),
        "gpu_compute_capability": f"{capability[0]}.{capability[1]}",
    }


def validate_prefix_engine_verification(
    engine_path: str | Path,
    policy: Any,
    device: str | torch.device,
) -> dict[str, Any]:
    """Fail-closed attach gate: the engine must carry a qualified verification report.

    Checks, in order: report exists → report schema → verification passed →
    engine bytes unchanged (sha256) → live prefix weights match the verified
    fingerprint → live architecture matches → TensorRT version and GPU compute
    capability match the verification environment.  Any mismatch raises
    :class:`PI05TensorRTError`; there is no silent PyTorch fallback (plan
    line 129).  Hashing the multi-GB engine and prefix weights costs tens of
    seconds once at startup — that is the price of the gate.
    """
    engine_path = Path(engine_path)
    report_path = engine_verification_path(engine_path)
    if not report_path.is_file():
        raise PI05TensorRTError(
            f"TensorRT engine has no verification report: {report_path}. "
            "Run lerobot-export-pi05-tensorrt --skip-export --verify-engine first"
        )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    missing = [field for field in REQUIRED_VERIFICATION_FIELDS if field not in report]
    if missing:
        raise PI05TensorRTError(
            f"Verification report {report_path} lacks required fields {missing}; "
            "re-run verification with the updated exporter"
        )
    if not report["passed"]:
        raise PI05TensorRTError(f"Verification report {report_path} records a failed verification")
    engine_sha256 = file_sha256(engine_path)
    if engine_sha256 != report["engine_sha256"]:
        raise PI05TensorRTError(
            f"TensorRT engine bytes changed since verification: {engine_path} "
            f"(sha256 {engine_sha256[:16]}… != verified {str(report['engine_sha256'])[:16]}…)"
        )
    expected_cameras = len(policy.config.image_features)
    if int(report["num_cameras"]) != expected_cameras:
        raise PI05TensorRTError(
            f"TensorRT engine was verified for {report['num_cameras']} cameras, "
            f"policy expects {expected_cameras}"
        )
    live_weights = compute_prefix_weight_fingerprint(policy.model)
    if live_weights != report["prefix_weight_fingerprint"]:
        raise PI05TensorRTError(
            "TensorRT engine was verified against different prefix weights than the loaded policy "
            f"(live {live_weights[:16]}… != verified {str(report['prefix_weight_fingerprint'])[:16]}…). "
            "Engines are not transferable across checkpoints — re-export for this checkpoint"
        )
    live_architecture = compute_model_architecture_fingerprint(policy)
    if live_architecture != report["model_architecture_fingerprint"]:
        raise PI05TensorRTError(
            "TensorRT engine was verified against a different model architecture than the loaded policy"
        )
    environment = runtime_environment_info(device)
    if str(report["tensorrt_version"]) != environment["tensorrt_version"]:
        raise PI05TensorRTError(
            f"TensorRT version mismatch: engine verified with {report['tensorrt_version']}, "
            f"runtime has {environment['tensorrt_version']}"
        )
    if str(report["gpu_compute_capability"]) != environment["gpu_compute_capability"]:
        raise PI05TensorRTError(
            f"GPU compute capability mismatch: engine verified on {report['gpu_compute_capability']}, "
            f"runtime is {environment['gpu_compute_capability']}"
        )
    return report


def prefix_cache_input_names(num_cameras: int) -> list[str]:
    return [
        *[f"image_{idx}" for idx in range(num_cameras)],
        *[f"img_mask_{idx}" for idx in range(num_cameras)],
        "tokens",
        "token_masks",
    ]


def prefix_cache_output_names(num_layers: int) -> list[str]:
    names = ["prefix_pad_masks"]
    for layer_idx in range(num_layers):
        names.extend((f"key_{layer_idx}", f"value_{layer_idx}"))
    return names


def flatten_dynamic_cache(cache: Any) -> tuple[Tensor, ...]:
    """Flatten a Transformers ``DynamicCache`` into explicit key/value tensors."""
    flat: list[Tensor] = []
    for layer_idx, (keys, values, _sliding_window) in enumerate(cache):
        if keys is None or values is None:
            raise PI05TensorRTError(f"Prefix cache layer {layer_idx} is empty")
        flat.extend((keys, values))
    if not flat:
        raise PI05TensorRTError("Prefix model returned an empty KV cache")
    return tuple(flat)


def dynamic_cache_from_flat(flat_cache: tuple[Tensor, ...] | list[Tensor]):
    """Rebuild the cache object expected by the Hugging Face Gemma expert."""
    if len(flat_cache) == 0 or len(flat_cache) % 2 != 0:
        raise ValueError(f"Expected a non-empty key/value tensor sequence, got {len(flat_cache)} tensors")

    from transformers import DynamicCache

    return DynamicCache(
        tuple((flat_cache[idx], flat_cache[idx + 1], None) for idx in range(0, len(flat_cache), 2))
    )


class PI05PrefixCacheExport(nn.Module):
    """ONNX export boundary for vision, language embedding, and prefix prefill."""

    def __init__(self, model: nn.Module, num_cameras: int):
        super().__init__()
        self.model = model
        self.num_cameras = num_cameras

    def forward(self, *inputs: Tensor) -> tuple[Tensor, ...]:
        expected = self.num_cameras * 2 + 2
        if len(inputs) != expected:
            raise ValueError(f"Expected {expected} prefix inputs, got {len(inputs)}")

        images = list(inputs[: self.num_cameras])
        img_masks = list(inputs[self.num_cameras : self.num_cameras * 2])
        tokens = inputs[-2]
        token_masks = inputs[-1]

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.model.embed_prefix(
            images, img_masks, tokens, token_masks
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks.to(dtype=torch.int32))
        prefix_position_ids = torch.cumsum(prefix_pad_masks.to(dtype=torch.int32), dim=1) - 1
        prefix_att_2d_masks_4d = prepare_attention_masks_4d(prefix_att_2d_masks)
        self.model.paligemma_with_expert.paligemma.model.language_model.config._attn_implementation = (  # noqa: SLF001
            "eager"
        )

        _, past_key_values = self.model.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )
        return (prefix_pad_masks, *flatten_dynamic_cache(past_key_values))


def _trt_dtype_to_torch(trt: Any, dtype: Any) -> torch.dtype:
    mapping = {
        trt.float32: torch.float32,
        trt.float16: torch.float16,
        trt.int32: torch.int32,
        trt.int8: torch.int8,
        trt.bool: torch.bool,
    }
    if hasattr(trt, "bfloat16"):
        mapping[trt.bfloat16] = torch.bfloat16
    if hasattr(trt, "int64"):
        mapping[trt.int64] = torch.int64
    try:
        return mapping[dtype]
    except KeyError as exc:
        raise PI05TensorRTError(f"Unsupported TensorRT tensor dtype: {dtype}") from exc


class TensorRTEngineRunner:
    """TensorRT 10 named-tensor runner backed by PyTorch CUDA allocations."""

    def __init__(self, engine_path: str | Path, device: torch.device):
        if device.type != "cuda":
            raise PI05TensorRTError("PI0.5 TensorRT inference requires a CUDA device")

        try:
            import tensorrt as trt
        except ImportError as exc:
            raise PI05TensorRTError("TensorRT Python bindings are not installed in this environment") from exc

        self.trt = trt
        self.device = device
        self.engine_path = Path(engine_path)
        if not self.engine_path.is_file():
            raise FileNotFoundError(f"TensorRT engine not found: {self.engine_path}")

        logger = trt.Logger(trt.Logger.ERROR)
        runtime = trt.Runtime(logger)
        self.engine = runtime.deserialize_cuda_engine(self.engine_path.read_bytes())
        if self.engine is None:
            raise PI05TensorRTError(f"Failed to deserialize TensorRT engine: {self.engine_path}")
        self.context = self.engine.create_execution_context()
        if self.context is None or not hasattr(self.context, "execute_async_v3"):
            raise PI05TensorRTError("The PI0.5 prefix backend requires TensorRT 10 execute_async_v3")

        self.input_names: list[str] = []
        self.output_names: list[str] = []
        for idx in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(idx)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.input_names.append(name)
            else:
                self.output_names.append(name)

        # Keep externally enqueued tensors alive until the following invocation.
        self._bound_inputs: dict[str, Tensor] = {}
        self._bound_outputs: dict[str, Tensor] = {}

    def __call__(self, inputs: dict[str, Tensor]) -> dict[str, Tensor]:
        bound_inputs: dict[str, Tensor] = {}
        for name in self.input_names:
            if name not in inputs:
                raise KeyError(f"Missing TensorRT input {name!r} for {self.engine_path}")
            dtype = _trt_dtype_to_torch(self.trt, self.engine.get_tensor_dtype(name))
            tensor = inputs[name].to(device=self.device, dtype=dtype).contiguous()
            bound_inputs[name] = tensor
            if not self.context.set_input_shape(name, tuple(tensor.shape)):
                raise PI05TensorRTError(
                    f"Failed to set TensorRT input shape for {name}: {tuple(tensor.shape)}"
                )
            if not self.context.set_tensor_address(name, tensor.data_ptr()):
                raise PI05TensorRTError(f"Failed to bind TensorRT input {name}")

        bound_outputs: dict[str, Tensor] = {}
        for name in self.output_names:
            shape = tuple(int(dim) for dim in self.context.get_tensor_shape(name))
            if any(dim < 0 for dim in shape):
                raise PI05TensorRTError(f"TensorRT output shape is unresolved for {name}: {shape}")
            dtype = _trt_dtype_to_torch(self.trt, self.engine.get_tensor_dtype(name))
            output = self._bound_outputs.get(name)
            if output is None or tuple(output.shape) != shape or output.dtype != dtype:
                output = torch.empty(shape, dtype=dtype, device=self.device)
            if not self.context.set_tensor_address(name, output.data_ptr()):
                raise PI05TensorRTError(f"Failed to bind TensorRT output {name}")
            bound_outputs[name] = output

        stream = torch.cuda.current_stream(device=self.device).cuda_stream
        if not self.context.execute_async_v3(stream_handle=stream):
            raise PI05TensorRTError(f"TensorRT execution failed for {self.engine_path}")

        self._bound_inputs = bound_inputs
        self._bound_outputs = bound_outputs
        return bound_outputs


class PI05TensorRTPrefixCache:
    """Callable prefix backend returning the cache consumed by PI05 ``denoise_step``."""

    _KEY_PATTERN = re.compile(r"^key_(\d+)$")

    def __init__(
        self,
        engine_path: str | Path,
        *,
        device: str | torch.device,
        cache_dtype: torch.dtype,
    ):
        self.device = torch.device(device)
        self.cache_dtype = cache_dtype
        self.runner = TensorRTEngineRunner(engine_path, self.device)
        self.num_cameras = sum(name.startswith("image_") for name in self.runner.input_names)
        layer_indices = sorted(
            int(match.group(1))
            for name in self.runner.output_names
            if (match := self._KEY_PATTERN.match(name)) is not None
        )
        if layer_indices != list(range(len(layer_indices))):
            raise PI05TensorRTError(f"Invalid TensorRT cache layer outputs: {layer_indices}")
        if not layer_indices:
            raise PI05TensorRTError("TensorRT prefix engine exposes no key_N outputs")
        self.num_layers = len(layer_indices)
        for layer_idx in layer_indices:
            if f"value_{layer_idx}" not in self.runner.output_names:
                raise PI05TensorRTError(f"TensorRT prefix engine is missing value_{layer_idx}")
        if "prefix_pad_masks" not in self.runner.output_names:
            raise PI05TensorRTError("TensorRT prefix engine is missing prefix_pad_masks")

    def __call__(
        self,
        images: list[Tensor],
        img_masks: list[Tensor],
        tokens: Tensor,
        token_masks: Tensor,
    ):
        if len(images) != self.num_cameras or len(img_masks) != self.num_cameras:
            raise PI05TensorRTError(
                f"TensorRT engine expects {self.num_cameras} cameras, got "
                f"{len(images)} images and {len(img_masks)} masks"
            )

        inputs: dict[str, Tensor] = {"tokens": tokens, "token_masks": token_masks}
        for idx, image in enumerate(images):
            inputs[f"image_{idx}"] = image
        for idx, img_mask in enumerate(img_masks):
            inputs[f"img_mask_{idx}"] = img_mask

        outputs = self.runner(inputs)
        flat_cache: list[Tensor] = []
        for layer_idx in range(self.num_layers):
            flat_cache.extend(
                (
                    outputs[f"key_{layer_idx}"].to(dtype=self.cache_dtype),
                    outputs[f"value_{layer_idx}"].to(dtype=self.cache_dtype),
                )
            )
        return outputs["prefix_pad_masks"].to(dtype=torch.bool), dynamic_cache_from_flat(flat_cache)


def enable_pi05_tensorrt_prefix(
    policy: nn.Module,
    engine_path: str | Path,
    *,
    device: str | torch.device,
    release_torch_prefix: bool = True,
    require_verification: bool = True,
) -> PI05TensorRTPrefixCache:
    """Attach a TensorRT prefix engine to a loaded PI0.5 policy.

    With ``require_verification`` (the default, and mandatory on the rollout
    path) the engine must carry a qualified verification report whose engine
    sha256, prefix weight fingerprint, architecture fingerprint, and runtime
    environment all match — otherwise this raises instead of silently falling
    back to PyTorch.  ``require_verification=False`` exists for unit tests
    only.
    """
    model = getattr(policy, "model", None)
    if model is None or not hasattr(model, "set_prefix_cache_backend"):
        raise TypeError("TensorRT prefix acceleration requires a PI0.5 policy")

    engine_path = Path(engine_path)
    if not engine_path.is_file():
        raise FileNotFoundError(f"TensorRT engine not found: {engine_path}")
    try:
        import tensorrt  # noqa: F401
    except ImportError as exc:
        raise PI05TensorRTError("TensorRT Python bindings are not installed in this environment") from exc

    # The fingerprint gate must run while the torch prefix modules are still
    # resident: it hashes the live PaliGemma weights the engine claims to match.
    if require_verification:
        validate_prefix_engine_verification(engine_path, policy, device)

    paligemma = model.paligemma_with_expert.paligemma
    cache_dtype = paligemma.model.language_model.layers[0].self_attn.k_proj.weight.dtype

    # Releasing before engine deserialization avoids holding two copies of the
    # large vision/language prefix model on the GPU at peak memory.
    if release_torch_prefix:
        model.release_torch_prefix_modules()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    backend = PI05TensorRTPrefixCache(engine_path, device=device, cache_dtype=cache_dtype)
    model.set_prefix_cache_backend(backend)
    return backend
