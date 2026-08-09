from pathlib import Path

import pytest
import torch

from lerobot.scripts.lerobot_export_pi05_realtime_vla_v2 import (
    SafeTensorSource,
    _fused_qkv_kernel,
    _linear_kernel,
    _source_shapes,
    _vision_patch_kernel,
    parse_args,
    prepare_time_embeddings,
    validate_pi05_config,
    validate_source_shapes,
)


def _valid_config() -> dict:
    return {
        "type": "pi05",
        "paligemma_variant": "gemma_2b",
        "action_expert_variant": "gemma_300m",
        "n_obs_steps": 1,
        "chunk_size": 50,
        "n_action_steps": 50,
        "max_state_dim": 32,
        "max_action_dim": 32,
        "num_inference_steps": 10,
        "use_relative_actions": False,
        "image_resolution": [224, 224],
        "train_expert_only": False,
        "freeze_vision_encoder": False,
        "rtc_training_max_delay": 6,
        "input_features": {
            "observation.state": {"type": "STATE", "shape": [6]},
            "observation.images.top": {"type": "VISUAL", "shape": [3, 480, 640]},
            "observation.images.wrist": {"type": "VISUAL", "shape": [3, 480, 640]},
        },
        "output_features": {"action": {"type": "ACTION", "shape": [6]}},
    }


def test_linear_kernel_transposes_and_folds_pi_gemma_rms_delta() -> None:
    weight = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    rms_delta = torch.tensor([0.5, -0.5])

    plain = _linear_kernel(weight)
    folded = _linear_kernel(weight, rms_delta)

    torch.testing.assert_close(plain, torch.tensor([[1.0, 3.0], [2.0, 4.0]]))
    torch.testing.assert_close(folded, torch.tensor([[1.5, 4.5], [1.0, 2.0]]))


def test_fused_qkv_transposes_concatenates_and_interleaves_rope_pairs() -> None:
    q_weight = torch.arange(16, dtype=torch.float32).reshape(8, 2)
    k_weight = 100 + torch.arange(8, dtype=torch.float32).reshape(4, 2)
    v_weight = 200 + torch.arange(8, dtype=torch.float32).reshape(4, 2)

    result = _fused_qkv_kernel(
        q_weight,
        k_weight,
        v_weight,
        num_heads=2,
        head_dim=4,
    )

    expected_first_input = torch.tensor(
        [
            0,
            4,
            2,
            6,
            8,
            12,
            10,
            14,
            100,
            104,
            102,
            106,
            200,
            202,
            204,
            206,
        ],
        dtype=torch.float32,
    )
    assert result.shape == (2, 16)
    torch.testing.assert_close(result[0], expected_first_input)


def test_vision_patch_kernel_uses_jax_hwio_layout() -> None:
    weight = torch.arange(2 * 3 * 2 * 4).reshape(2, 3, 2, 4)

    result = _vision_patch_kernel(weight)

    assert result.shape == (2, 4, 3, 2)
    assert result[1, 3, 2, 1] == weight[1, 2, 1, 3]


def test_time_embeddings_match_pi05_direct_euler_timestep_grid() -> None:
    from lerobot.policies.pi05.modeling_pi05 import create_sinusoidal_pos_embedding

    num_steps = 10
    dimension = 8
    dt = -1.0 / num_steps
    expected = torch.stack(
        [
            create_sinusoidal_pos_embedding(
                torch.tensor([1.0 + step * dt], dtype=torch.float32),
                dimension,
                min_period=0.004,
                max_period=4.0,
                device=torch.device("cpu"),
            )[0]
            for step in range(num_steps)
        ]
    ).to(torch.bfloat16)

    actual = prepare_time_embeddings(num_steps=num_steps, dimension=dimension)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_config_accepts_two_camera_six_axis_full_rtc_with_padded_width_32() -> None:
    validate_pi05_config(_valid_config())


@pytest.mark.parametrize(
    ("update", "message"),
    [
        ({"max_action_dim": 6}, "max_action_dim"),
        ({"train_expert_only": True}, "full-model"),
        ({"rtc_training_max_delay": 0}, "trained-prefix RTC"),
    ],
)
def test_config_rejects_incompatible_triton_architecture(update: dict, message: str) -> None:
    config = _valid_config()
    config.update(update)

    with pytest.raises(ValueError, match=message):
        validate_pi05_config(config)


def test_source_shape_validation_reports_mismatch_before_loading_tensors() -> None:
    source = SafeTensorSource(Path("unused.safetensors"))
    source.shapes = _source_shapes()
    source.shapes["action_in_proj.weight"] = (1024, 6)

    with pytest.raises(ValueError, match=r"action_in_proj.weight.*got \(1024, 6\).+expected \(1024, 32\)"):
        validate_source_shapes(source)


def test_cli_validate_only_does_not_require_output_or_tokenizer() -> None:
    args = parse_args(["--checkpoint", "checkpoint", "--validate-only"])

    assert args.checkpoint == Path("checkpoint")
    assert args.output is None
    assert args.tokenizer_path is None


def test_cli_export_requires_output_prompt_and_tokenizer(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit, match="2"):
        parse_args(["--checkpoint", "checkpoint"])

    assert "output, prompt, tokenizer_path" in capsys.readouterr().err


def test_cli_parses_complete_export_arguments() -> None:
    args = parse_args(
        [
            "--checkpoint",
            "checkpoint",
            "--output",
            "weights.pkl",
            "--prompt",
            "pick_up_cube",
            "--tokenizer-path",
            "google/paligemma-3b-pt-224",
            "--tokenizer-max-length",
            "64",
            "--overwrite",
        ]
    )

    assert args.output == Path("weights.pkl")
    assert args.prompt == "pick_up_cube"
    assert args.tokenizer_max_length == 64
    assert args.overwrite is True
