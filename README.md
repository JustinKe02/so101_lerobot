<p align="center">
  <img alt="LeRobot, Hugging Face Robotics Library" src="./media/readme/lerobot-logo-thumbnail.png" width="100%">
</p>

<div align="center">

[![Tests](https://github.com/huggingface/lerobot/actions/workflows/latest_deps_tests.yml/badge.svg?branch=main)](https://github.com/huggingface/lerobot/actions/workflows/latest_deps_tests.yml?query=branch%3Amain)
[![Tests](https://github.com/huggingface/lerobot/actions/workflows/docker_publish.yml/badge.svg?branch=main)](https://github.com/huggingface/lerobot/actions/workflows/docker_publish.yml?query=branch%3Amain)
[![Python versions](https://img.shields.io/pypi/pyversions/lerobot)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/huggingface/lerobot/blob/main/LICENSE)
[![Status](https://img.shields.io/pypi/status/lerobot)](https://pypi.org/project/lerobot/)
[![Version](https://img.shields.io/pypi/v/lerobot)](https://pypi.org/project/lerobot/)
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-v2.1-ff69b4.svg)](https://github.com/huggingface/lerobot/blob/main/CODE_OF_CONDUCT.md)
[![Discord](https://img.shields.io/badge/Discord-Join_Us-5865F2?style=flat&logo=discord&logoColor=white)](https://discord.gg/q8Dzzpym3f)

</div>

> [!NOTE]
> 当前 GitHub 派生仓库为 [JustinKe02/so101_lerobot](https://github.com/JustinKe02/so101_lerobot)，
> 仓库所有者为 [JustinKe02](https://github.com/JustinKe02)。PI0.5 + VLASH 的当前开发分支是
> [`codex/pi05-vlash`](https://github.com/JustinKe02/so101_lerobot/tree/codex/pi05-vlash)。项目基于
> [Hugging Face LeRobot](https://github.com/huggingface/lerobot) 开发，上游版权、许可证和引用信息保持不变。

**LeRobot** aims to provide models, datasets, and tools for real-world robotics in PyTorch. The goal is to lower the barrier to entry so that everyone can contribute to and benefit from shared datasets and pretrained models.

🤗 A hardware-agnostic, Python-native interface that standardizes control across diverse platforms, from low-cost arms (SO-100) to humanoids.

🤗 A standardized, scalable LeRobotDataset format (Parquet + MP4 or images) hosted on the Hugging Face Hub, enabling efficient storage, streaming and visualization of massive robotic datasets.

🤗 State-of-the-art policies that have been shown to transfer to the real-world ready for training and deployment.

🤗 Comprehensive support for the open-source ecosystem to democratize physical AI.

## Quick Start

LeRobot can be installed directly from PyPI.

```bash
pip install lerobot
lerobot-info
```

> [!IMPORTANT]
> For detailed installation guide, please see the [Installation Documentation](https://huggingface.co/docs/lerobot/installation).

## PI0.5 + VLASH SO-101 抓取

本分支在 LeRobot PI0.5 上集成了受
[MIT HAN Lab VLASH](https://github.com/mit-han-lab/vlash) 启发的异步推理链路，并完成了
SO-101 双相机抓取任务的训练、离线回放和真机闭环验证。该实现为 LeRobot 原生后端，不依赖
上游 VLASH 运行环境。以下结果最后更新于 2026-08-04。

主要改动包括：

- 为 PI0.5 增加连续关节状态条件和 temporal offset 数据增强。
- 在执行当前 action chunk 时，预测 chunk 边界的未来机械臂状态并异步生成下一 chunk。
- 使用策略动作、过滤后动作、机器人实际下发动作和关节观测估计真实跟踪增益。
- 保留 LeRobot 的动作后处理、机器人侧限幅和 deadline fail-closed 机制。
- 提供训练脚本、数据集回放评估脚本和通用真机启动脚本。

### 当前实验结果

| 项目            | 结果                                                           |
| --------------- | -------------------------------------------------------------- |
| 数据集          | 40 episodes，17,960 帧，top + wrist 双相机，30 Hz              |
| 数据划分        | 全部用于训练，无独立验证集                                     |
| 第一阶段        | PI0.5 全参数任务适配，10 epochs，5,613 steps，学习率 2.5e-5    |
| 第二阶段        | VLASH 专家分支微调，10 epochs，5,613 steps，学习率 5e-6        |
| VLASH 训练配置  | `state_cond=true`，`temporal_offset_max_steps=8`               |
| VLASH 最终 loss | `0.012`，epoch 6 后逐渐进入平台期                              |
| 真机推理配置    | 30 Hz，horizon 10，overlap 5，future-state delta 5.0           |
| 60 秒真机控制   | 1,778 次动作反馈，约 29.6 Hz                                   |
| 稳态推理时延    | P95 134.76 ms，最大 157.18 ms，deadline miss 0                 |
| 当前效果        | 已完成物体接近、夹取和搬运的真机闭环验证，但速度与泛化仍需优化 |

RTC 与 VLASH 是两个可选推理后端，不会在当前链路中叠加运行。RTC 使用旧动作 prefix guidance
和实际消耗步数融合新旧 chunk；VLASH 使用预测的未来状态生成下一完整 chunk，并在边界处整块
切换。当前权重经过状态条件和 temporal offset 训练，真机运行使用 `--inference.type=vlash`。

### 训练与运行

当前 SO-101 实验训练脚本执行的是第二阶段 VLASH 专家分支微调，并提供 smoke 和完整 10-epoch
两种模式。运行前需要先准备第一阶段的全参数任务权重，并按实际机器调整脚本中的数据、基础权重和
环境路径：

```bash
bash src/lerobot/scripts/train_pi05_so101_vlash_10epochs.sh smoke
bash src/lerobot/scripts/train_pi05_so101_vlash_10epochs.sh full
```

使用最终权重启动 VLASH 真机推理。机器人端口、标定 ID 和相机配置必须替换为当前设备的实际值：

```bash
MODEL=outputs/train/pi05_so101_vlash_expert_only_10epochs_bs32_seed1000/checkpoints/005613/pretrained_model

VLASH_DURATION_S=60 bash examples/inference/run_pi05_vlash.sh "$MODEL" \
  --robot.type=so101_follower \
  --robot.port=<FOLLOWER_PORT> \
  --robot.id=<ROBOT_ID> \
  --robot.cameras='<CAMERA_CONFIG>' \
  --robot.max_relative_target=5.0 \
  --task="Put the block in the bin" \
  --fps=30 \
  --device=cuda
```

开始真机运行前，应先核对标定、相机视角、初始姿态、设备占用和机械臂工作空间。当前数据没有
独立验证集，现有结果不能替代多初始位置、多光照和多物体布局下的成功率评估。机械臂速度偏慢
主要与动作滤波持续改写、硬件 clamp 和关节跟踪增益偏低有关，不是控制循环未达到 30 Hz。

详细资料：

- [PI0.5 VLASH 使用说明](./docs/source/pi05.mdx#vlash-style-asynchronous-inference)
- [训练与推理链路报告](./PI05_VLASH_TRAINING_INFERENCE_REPORT_20260730.md)
- [当前版本与速度分析](./PI05_VLASH_CURRENT_STATUS_SPEED_REPORT_20260730.md)
- [本周工作总结](./WEEKLY_REPORT_20260727_20260731.md)
- [VLASH 通用启动脚本](./examples/inference/run_pi05_vlash.sh)
- [VLASH 数据集回放评估](./examples/inference/evaluate_pi05_vlash_replay.py)

## Robots & Control

<div align="center">
  <img src="./media/readme/robots_control_video.webp" width="640px" alt="Reachy 2 Demo">
</div>

LeRobot provides a unified `Robot` class interface that decouples control logic from hardware specifics. It supports a wide range of robots and teleoperation devices.

```python
from lerobot.robots.myrobot import MyRobot

# Connect to a robot
robot = MyRobot(config=...)
robot.connect()

# Read observation and send action
obs = robot.get_observation()
action = model.select_action(obs)
robot.send_action(action)
```

**Supported Hardware:** SO100, LeKiwi, Koch, HopeJR, OMX, EarthRover, Reachy2, Gamepads, Keyboards, Phones, OpenARM, Unitree G1, reBot B601.

While these devices are natively integrated into the LeRobot codebase, the library is designed to be extensible. You can easily implement the Robot interface to utilize LeRobot's data collection, training, and visualization tools for your own custom robot.

For detailed hardware setup guides, see the [Hardware Documentation](https://huggingface.co/docs/lerobot/integrate_hardware).

## LeRobot Dataset

To solve the data fragmentation problem in robotics, we utilize the **LeRobotDataset** format.

- **Structure:** Synchronized MP4 videos (or images) for vision and Parquet files for state/action data.
- **HF Hub Integration:** Explore thousands of robotics datasets on the [Hugging Face Hub](https://huggingface.co/lerobot).
- **Tools:** Seamlessly delete episodes, split by indices/fractions, add/remove features, and merge multiple datasets.

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# Load a dataset from the Hub
dataset = LeRobotDataset("lerobot/aloha_mobile_cabinet")

# Access data (automatically handles video decoding)
episode_index=0
print(f"{dataset[episode_index]['action'].shape=}\n")
```

Learn more about it in the [LeRobotDataset Documentation](https://huggingface.co/docs/lerobot/lerobot-dataset-v3)

## SoTA Models

LeRobot implements state-of-the-art policies in pure PyTorch, covering Imitation Learning, Reinforcement Learning, Vision-Language-Action (VLA) models, World Models, and Reward Models, with more coming soon. It also provides you with the tools to instrument and inspect your training process.

<p align="center">
  <img alt="Gr00t Architecture" src="./media/readme/VLA_architecture.jpg" width="640px">
</p>

Training a policy is as simple as running a script configuration:

```bash
lerobot-train \
  --policy.type=act \
  --dataset.repo_id=lerobot/aloha_mobile_cabinet
```

| Category                   | Models                                                                                                                                                                                                                                                                                                                                                                                     |
| -------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Imitation Learning**     | [ACT](./docs/source/policy_act_README.md), [Diffusion](./docs/source/policy_diffusion_README.md), [VQ-BeT](./docs/source/policy_vqbet_README.md), [Multitask DiT Policy](./docs/source/policy_multi_task_dit_README.md)                                                                                                                                                                    |
| **Reinforcement Learning** | [HIL-SERL](./docs/source/hilserl.mdx), [TDMPC](./docs/source/policy_tdmpc_README.md) & QC-FQL (coming soon)                                                                                                                                                                                                                                                                                |
| **VLAs Models**            | [Pi0](./docs/source/pi0.mdx), [Pi0Fast](./docs/source/pi0fast.mdx), [Pi0.5](./docs/source/pi05.mdx), [GR00T N1.7](./docs/source/policy_groot_README.md), [SmolVLA](./docs/source/policy_smolvla_README.md), [XVLA](./docs/source/xvla.mdx), [EO-1](./docs/source/eo1.mdx), [MolmoAct2](./docs/source/molmoact2.mdx), [WALL-OSS](./docs/source/walloss.mdx), [EVO1](./docs/source/evo1.mdx) |
| **World Models**           | [VLA-JEPA](./docs/source/vla_jepa.mdx), [LingBot-VA](./docs/source/lingbot_va.mdx), [FastWAM](./docs/source/fastwam.mdx)                                                                                                                                                                                                                                                                   |
| **Reward Models**          | [SARM](./docs/source/sarm.mdx), [TOPReward](./docs/source/topreward.mdx), [Robometer](./docs/source/robometer.mdx)                                                                                                                                                                                                                                                                         |

Similarly to the hardware, you can easily implement your own policy & leverage LeRobot's data collection, training, and visualization tools, and share your model to the HF Hub

For detailed policy setup guides, see the [Policy Documentation](https://huggingface.co/docs/lerobot/bring_your_own_policies). For GPU/RAM requirements and expected training time per policy, see the [Compute Hardware Guide](https://huggingface.co/docs/lerobot/hardware_guide).

## Inference & Evaluation

Evaluate your policies in simulation or on real hardware using the unified evaluation script. LeRobot supports standard benchmarks like **LIBERO**, **MetaWorld** and more to come.

```bash
# Evaluate a policy on the LIBERO benchmark
lerobot-eval \
  --policy.path=lerobot/pi0_libero_finetuned \
  --env.type=libero \
  --env.task=libero_object \
  --eval.n_episodes=10
```

Learn how to implement your own simulation environment or benchmark and distribute it from the HF Hub by following the [EnvHub Documentation](https://huggingface.co/docs/lerobot/envhub)

## Resources

- **[Documentation](https://huggingface.co/docs/lerobot/index):** The complete guide to tutorials & API.
- **[Chinese Tutorials: LeRobot+SO-ARM101中文教程-同济子豪兄](https://zihao-ai.feishu.cn/wiki/space/7589642043471924447)** Detailed doc for assembling, teleoperate, dataset, train, deploy. Verified by Seed Studio and 5 global hackathon players.
- **[Discord](https://discord.gg/q8Dzzpym3f):** Join the `LeRobot` server to discuss with the community.
- **[X](https://x.com/LeRobotHF):** Follow us on X to stay up-to-date with the latest developments.
- **[Robot Learning Tutorial](https://huggingface.co/spaces/lerobot/robot-learning-tutorial):** A free, hands-on course to learn robot learning using LeRobot.
- **[T-Shirt Folding Experiment](https://huggingface.co/spaces/lerobot/robot-folding):** An end-to-end demonstration of folding t-shirts with LeRobot.
- **[LeLab](https://github.com/huggingface/leLab):** A web interface for LeRobot — teleoperate, calibrate, record datasets, replay, and train your SO arm from the browser, no CLI required.

## Citation

If you use LeRobot in your project, please cite the GitHub repository to acknowledge the ongoing development and contributors:

```bibtex
@misc{cadene2024lerobot,
    author = {Cadene, Remi and Alibert, Simon and Soare, Alexander and Gallouedec, Quentin and Zouitine, Adil and Palma, Steven and Kooijmans, Pepijn and Aractingi, Michel and Shukor, Mustafa and Aubakirova, Dana and Russi, Martino and Capuano, Francesco and Pascal, Caroline and Choghari, Jade and Meftah, Khalil and Ellerbach, Maxime and Moss, Jess and Wolf, Thomas},
    title = {LeRobot: State-of-the-art Machine Learning for Real-World Robotics in Pytorch},
    howpublished = "\url{https://github.com/huggingface/lerobot}",
    year = {2024}
}
```

If you are referencing our research or the academic paper, please also cite our ICLR publication:

<details>
<summary><b>ICLR 2026 Paper</b></summary>

```bibtex
@inproceedings{cadenelerobot,
  title={LeRobot: An Open-Source Library for End-to-End Robot Learning},
  author={Cadene, Remi and Alibert, Simon and Capuano, Francesco and Aractingi, Michel and Zouitine, Adil and Kooijmans, Pepijn and Choghari, Jade and Russi, Martino and Pascal, Caroline and Palma, Steven and Shukor, Mustafa and Moss, Jess and Soare, Alexander and Aubakirova, Dana and Lhoest, Quentin and Gallou\'edec, Quentin and Wolf, Thomas},
  booktitle={The Fourteenth International Conference on Learning Representations},
  year={2026},
  url={https://arxiv.org/abs/2602.22818}
}
```

</details>

## Contribute

We welcome contributions from everyone in the community! To get started, please read our [CONTRIBUTING.md](https://github.com/huggingface/lerobot/blob/main/CONTRIBUTING.md) guide. Whether you're adding a new feature, improving documentation, or fixing a bug, your help and feedback are invaluable. We're incredibly excited about the future of open-source robotics and can't wait to work with you on what's next—thank you for your support!

<p align="center">
  <img alt="SO101 Video" src="./media/readme/so100_video.webp" width="640px">
</p>

<div align="center">
<sub>Built by the <a href="https://huggingface.co/lerobot">LeRobot</a> team at <a href="https://huggingface.co">Hugging Face</a> with ❤️</sub>
</div>
