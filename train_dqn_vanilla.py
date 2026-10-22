"""The smallest practical Stable-Baselines3 DQN starter for Atari Pong.

Player 1 (left/orange) trains against the fixed visual ball-tracking
baseline. Apart from the necessary Pong adapter and ONNX export, this uses
SB3's unmodified ``DQN`` and its standard ``CnnPolicy``.

Example:
    python train_dqn_vanilla.py --timesteps 500000
"""

from __future__ import annotations

import argparse
import copy
import inspect
from pathlib import Path

import gymnasium as gym
import torch
from stable_baselines3 import DQN
from torch import nn
from torch.nn import functional as F

from train_ppo_sb3 import Player1VsPongBaselineEnv, PlayingFieldObservation


class ExportedPolicy(nn.Module):
    """Add the training preprocessing to the fixed public ONNX interface."""

    def __init__(self, q_net: nn.Module) -> None:
        super().__init__()
        self.q_net = copy.deepcopy(q_net).cpu()

    def forward(self, image: torch.Tensor, player_id: torch.Tensor) -> torch.Tensor:
        # The runner supplies normalized RGB. SB3 receives uint8 observations
        # and normalizes them inside CnnPolicy, therefore convert back to its
        # expected 0..255 scale before calling its untouched Q-network.
        image = image[:, :, 34:194, :] * 255.0
        image = F.interpolate(image, size=(84, 84), mode="bilinear", align_corners=False)
        action = torch.argmax(self.q_net(image), dim=1).to(torch.int64)
        return action + player_id * 0


def make_env() -> gym.Env:
    """Crop Atari's static header and use the standard 84 x 84 DQN input."""
    env = PlayingFieldObservation(Player1VsPongBaselineEnv())
    return gym.wrappers.ResizeObservation(env, (84, 84))


def export_policy(model: DQN, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    policy = ExportedPolicy(model.policy.q_net).eval()
    image = torch.zeros((1, 3, 210, 160), dtype=torch.float32)
    player_id = torch.ones((1,), dtype=torch.int64)
    export_kwargs = {
        "input_names": ["image", "player_id"],
        "output_names": ["action"],
        "opset_version": 17,
    }
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        export_kwargs["dynamo"] = False
    torch.onnx.export(policy, (image, player_id), str(output), **export_kwargs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train vanilla Stable-Baselines3 DQN for Atari Pong.")
    parser.add_argument("--timesteps", type=int, default=500_000, help="Training steps (default: 500000).")
    parser.add_argument("--output", default="baselines/baseline_pong_dqn.onnx", help="Exported ONNX path.")
    parser.add_argument("--device", default="auto", help="SB3 device, e.g. auto, cuda, mps or cpu.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.timesteps <= 0:
        raise ValueError("--timesteps muss positiv sein.")
    env = make_env()
    try:
        # Deliberately standard SB3 DQN: no custom network, Double-DQN
        # subclass, reward shaping or special callback.
        model = DQN(
            "CnnPolicy",
            env,
            learning_rate=1e-4,
            # 10k Atari frames keep the replay buffer below roughly 500 MB.
            buffer_size=10_000,
            learning_starts=5_000,
            batch_size=32,
            train_freq=4,
            verbose=1,
            seed=42,
            device=args.device,
        )
        model.learn(total_timesteps=args.timesteps, progress_bar=False)
        output = Path(args.output)
        export_policy(model, output)
        print(f"Vanilla-DQN-Baseline exportiert nach {output}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
