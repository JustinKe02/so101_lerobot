#!/usr/bin/env python

# Copyright 2025 Physical Intelligence and The HuggingFace Inc. team. All rights reserved.
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

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from lerobot.configs import PipelineFeatureType, PolicyFeature
from lerobot.processor import (
    AbsoluteActionsProcessorStep,
    AddBatchDimensionProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStep,
    ProcessorStepRegistry,
    RelativeActionsProcessorStep,
    TokenizerProcessorStep,
    make_default_policy_processor_steps,
    make_policy_processor_pipelines,
)
from lerobot.types import EnvTransition, TransitionKey
from lerobot.utils.constants import OBS_STATE

from .configuration_pi05 import PI05Config


@ProcessorStepRegistry.register(name="pi05_temporal_offset_processor_step")
@dataclass
class Pi05TemporalOffsetProcessorStep(ProcessorStep):
    """Select matching future state/action windows while keeping current images."""

    max_offset_steps: int = 0
    chunk_size: int = 50

    def get_config(self) -> dict[str, int]:
        return {"max_offset_steps": self.max_offset_steps, "chunk_size": self.chunk_size}

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        if self.max_offset_steps <= 0:
            return transition

        action = transition.get(TransitionKey.ACTION)
        observation = transition.get(TransitionKey.OBSERVATION) or {}
        state = observation.get(OBS_STATE)
        if action is None:
            # Inference checkpoints retain this processor, but inference has no
            # action target or future-state window to select from.
            return transition
        if state is None:
            raise ValueError("PI05 temporal offset augmentation requires observation.state")
        if action.ndim != 3 or state.ndim != 3:
            raise ValueError(
                "PI05 temporal offset augmentation expects batched action/state windows "
                f"with rank 3, got action={tuple(action.shape)}, state={tuple(state.shape)}"
            )

        batch_size = action.shape[0]
        required_action_steps = self.chunk_size + self.max_offset_steps
        required_state_steps = self.max_offset_steps + 1
        if action.shape[1] < required_action_steps or state.shape[1] < required_state_steps:
            raise ValueError(
                "PI05 temporal offset windows are shorter than configured: "
                f"action_steps={action.shape[1]} (need {required_action_steps}), "
                f"state_steps={state.shape[1]} (need {required_state_steps})"
            )

        offsets = torch.randint(
            0,
            self.max_offset_steps + 1,
            (batch_size,),
            device=action.device,
        )
        batch_indices = torch.arange(batch_size, device=action.device)
        action_indices = offsets[:, None] + torch.arange(self.chunk_size, device=action.device)[None, :]

        new_transition = transition.copy()
        new_observation = observation.copy()
        new_observation[OBS_STATE] = state[batch_indices, offsets]
        new_transition[TransitionKey.OBSERVATION] = new_observation
        new_transition[TransitionKey.ACTION] = action[batch_indices[:, None], action_indices]

        complementary = (transition.get(TransitionKey.COMPLEMENTARY_DATA) or {}).copy()
        action_pad = complementary.get("action_is_pad")
        if isinstance(action_pad, torch.Tensor) and action_pad.ndim == 2:
            complementary["action_is_pad"] = action_pad[batch_indices[:, None], action_indices]
        state_pad = complementary.get(f"{OBS_STATE}_is_pad")
        if isinstance(state_pad, torch.Tensor) and state_pad.ndim == 2:
            complementary[f"{OBS_STATE}_is_pad"] = state_pad[batch_indices, offsets]
        complementary["vlash_offset"] = offsets
        new_transition[TransitionKey.COMPLEMENTARY_DATA] = complementary
        return new_transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


def reconcile_pi05_processors(
    config: PI05Config,
    preprocessor: PolicyProcessorPipeline,
    postprocessor: PolicyProcessorPipeline,
) -> tuple[PolicyProcessorPipeline, PolicyProcessorPipeline]:
    """Apply current temporal-offset config to a checkpoint-loaded pipeline."""
    temporal_step = Pi05TemporalOffsetProcessorStep(
        max_offset_steps=config.temporal_offset_max_steps,
        chunk_size=config.chunk_size,
    )
    steps = list(preprocessor.steps)
    temporal_idx = next(
        (idx for idx, step in enumerate(steps) if isinstance(step, Pi05TemporalOffsetProcessorStep)),
        None,
    )
    if temporal_idx is None:
        insert_idx = next(
            (idx for idx, step in enumerate(steps) if isinstance(step, RelativeActionsProcessorStep)),
            next(
                (
                    idx + 1
                    for idx, step in enumerate(steps)
                    if isinstance(step, AddBatchDimensionProcessorStep)
                ),
                0,
            ),
        )
        steps.insert(insert_idx, temporal_step)
    else:
        steps[temporal_idx] = temporal_step
    preprocessor.steps = steps
    return preprocessor, postprocessor


@ProcessorStepRegistry.register(name="pi05_prepare_state_tokenizer_processor_step")
@dataclass
class Pi05PrepareStateTokenizerProcessorStep(ProcessorStep):
    """
    Processor step to prepare the state and tokenize the language input.
    """

    max_state_dim: int = 32
    task_key: str = "task"

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        transition = transition.copy()

        state = transition.get(TransitionKey.OBSERVATION, {}).get(OBS_STATE)
        if state is None:
            raise ValueError("State is required for PI05")
        tasks = transition.get(TransitionKey.COMPLEMENTARY_DATA, {}).get(self.task_key)
        if tasks is None:
            raise ValueError("No task found in complementary data")

        # TODO: check if this necessary
        state = deepcopy(state)

        # State should already be normalized to [-1, 1] by the NormalizerProcessorStep that runs before this step
        # Discretize into 256 bins (see openpi `PaligemmaTokenizer.tokenize()`)
        state_np = state.cpu().numpy()
        discretized_states = np.digitize(state_np, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1

        full_prompts = []
        for i, task in enumerate(tasks):
            cleaned_text = task.strip().replace("_", " ").replace("\n", " ")
            state_str = " ".join(map(str, discretized_states[i]))
            full_prompt = f"Task: {cleaned_text}, State: {state_str};\nAction: "
            full_prompts.append(full_prompt)

        transition[TransitionKey.COMPLEMENTARY_DATA][self.task_key] = full_prompts
        # Normalize state to [-1, 1] range if needed (assuming it's already normalized by normalizer processor step!!)
        # Discretize into 256 bins (see openpi `PaligemmaTokenizer.tokenize()`)
        return transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """
        This step does not alter the feature definitions.
        """
        return features


def make_pi05_pre_post_processors(
    config: PI05Config,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """
    Constructs pre-processor and post-processor pipelines for the PI0 policy.

    The pre-processing pipeline prepares input data for the model by:
    1. Renaming features to match pretrained configurations.
    2. Normalizing input and output features based on dataset statistics.
    3. Adding a batch dimension.
    4. Appending a newline character to the task description for tokenizer compatibility.
    5. Tokenizing the text prompt using the PaliGemma tokenizer.
    6. Moving all data to the specified device.

    The post-processing pipeline handles the model's output by:
    1. Moving data to the CPU.
    2. Unnormalizing the output features to their original scale.

    Args:
        config: The configuration object for the PI0 policy.
        dataset_stats: A dictionary of statistics for normalization.
        preprocessor_kwargs: Additional arguments for the pre-processor pipeline.
        postprocessor_kwargs: Additional arguments for the post-processor pipeline.

    Returns:
        A tuple containing the configured pre-processor and post-processor pipelines.
    """

    relative_step = RelativeActionsProcessorStep(
        enabled=config.use_relative_actions,
        exclude_joints=getattr(config, "relative_exclude_joints", []),
        action_names=getattr(config, "action_feature_names", None),
    )

    steps = make_default_policy_processor_steps(config, dataset_stats)

    # OpenPI order: raw → relative → normalize → model → unnormalize → absolute
    input_steps: list[ProcessorStep] = [
        steps.rename_observations,  # To mimic the same processor as pretrained one
        steps.add_batch_dim,
        Pi05TemporalOffsetProcessorStep(
            max_offset_steps=config.temporal_offset_max_steps,
            chunk_size=config.chunk_size,
        ),
        relative_step,
        # NOTE: NormalizerProcessorStep MUST come before Pi05PrepareStateTokenizerProcessorStep
        # because the tokenizer step expects normalized state in [-1, 1] range for discretization
        steps.normalize,
        Pi05PrepareStateTokenizerProcessorStep(max_state_dim=config.max_state_dim),
        TokenizerProcessorStep(
            tokenizer_name="google/paligemma-3b-pt-224",
            max_length=config.tokenizer_max_length,
            padding_side="right",
            padding="max_length",
        ),
        steps.to_device,
    ]

    output_steps: list[ProcessorStep] = [
        steps.unnormalize,
        AbsoluteActionsProcessorStep(enabled=config.use_relative_actions, relative_step=relative_step),
        steps.to_cpu,
    ]

    return make_policy_processor_pipelines(input_steps=input_steps, output_steps=output_steps)
