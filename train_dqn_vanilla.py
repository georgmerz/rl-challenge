"""An Atari-style Stable-Baselines3 DQN starter for Atari Pong.

Player 1 (left/orange) trains against the fixed visual ball-tracking
baseline. Apart from the necessary Pong adapter and ONNX export, this uses
SB3's unmodified ``DQN`` and its standard ``CnnPolicy``.

Example:
    python train_dqn_vanilla.py --timesteps 1000000
"""

from __future__ import annotations

import argparse
import copy
import inspect
from pathlib import Path

import gymnasium as gym
import torch
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from torch import nn
from torch.nn import functional as F

from train_ppo_sb3 import FRAME_STACK, Player1VsPongBaselineEnv, PlayingFieldObservation, StackFramesAsChannels


class ExportedPolicy(nn.Module):
    """Add the training preprocessing to the fixed public ONNX interface."""

    def __init__(self, q_net: nn.Module) -> None:
        super().__init__()
        self.q_net = copy.deepcopy(q_net).cpu()

    def forward(self, image: torch.Tensor, player_id: torch.Tensor) -> torch.Tensor:
        # Four RGB frames arrive as twelve channels (oldest to newest). Match
        # CleanRL's Atari-style crop, grayscale and resize preprocessing.
        batch = image.shape[0]
        frames = image.reshape(batch, FRAME_STACK, 3, 210, 160)[:, :, :, 34:194, :]
        image = (0.299 * frames[:, :, 0] + 0.587 * frames[:, :, 1] + 0.114 * frames[:, :, 2]) * 255.0
        image = F.interpolate(image, size=(84, 84), mode="bilinear", align_corners=False)
        action = torch.argmax(self.q_net(image), dim=1).to(torch.int64)
        return action + player_id * 0


def make_env() -> gym.Env:
    """Crop Atari's static header and use the standard 84 x 84 DQN input."""
    env = PlayingFieldObservation(Player1VsPongBaselineEnv())
    env = gym.wrappers.ResizeObservation(env, (84, 84))
    env = gym.wrappers.GrayscaleObservation(env, keep_dim=True)
    env = gym.wrappers.FrameStackObservation(env, FRAME_STACK)
    return StackFramesAsChannels(env)


def export_policy(model: DQN, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    policy = ExportedPolicy(model.policy.q_net).eval()
    image = torch.zeros((1, 12, 210, 160), dtype=torch.float32)
    player_id = torch.ones((1,), dtype=torch.int64)
    export_kwargs = {
        "input_names": ["image", "player_id"],
        "output_names": ["action"],
        "opset_version": 17,
    }
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        export_kwargs["dynamo"] = False
    torch.onnx.export(policy, (image, player_id), str(output), **export_kwargs)


class ServerCheckpointCallback(BaseCallback):
    """Save SB3 weight and portable ONNX checkpoints during long runs."""

    def __init__(self, output: Path, every_steps: int) -> None:
        super().__init__()
        self.output = output
        self.every_steps = every_steps
        self.next_checkpoint = every_steps

    def _on_step(self) -> bool:
        if self.num_timesteps >= self.next_checkpoint:
            checkpoint = self.output.with_name(f"{self.output.stem}_step_{self.num_timesteps}")
            self.model.save(checkpoint)
            export_policy(self.model, checkpoint.with_suffix(".onnx"))
            print(f"Checkpoint nach {self.num_timesteps} Schritten: {checkpoint}.zip und .onnx")
            self.next_checkpoint += self.every_steps
        return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train vanilla Stable-Baselines3 DQN for Atari Pong.")
    parser.add_argument("--timesteps", type=int, default=1_000_000, help="Training steps (default: 1000000).")
    parser.add_argument("--output", default="baselines/baseline_pong_dqn.onnx", help="Exported ONNX path.")
    parser.add_argument("--device", default="auto", help="SB3 device, e.g. auto, cuda, mps or cpu.")
    parser.add_argument("--buffer-size", type=int, default=100_000, help="Replay-buffer size (default: 100000).")
    parser.add_argument("--learning-starts", type=int, default=80_000, help="Random steps before updates (default: 80000).")
    parser.add_argument("--export-every", type=int, default=100_000, help="Checkpoint interval in steps (default: 100000).")
    parser.add_argument("--resume", type=Path, help="Optional .zip checkpoint to continue; --timesteps are added training steps.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if min(args.timesteps, args.buffer_size, args.learning_starts, args.export_every) <= 0:
        raise ValueError("--timesteps, --buffer-size, --learning-starts und --export-every müssen positiv sein.")
    env = make_env()
    try:
        # Deliberately standard SB3 DQN: no custom network, Double-DQN
        # subclass, reward shaping or special callback.
        if args.resume:
            model = DQN.load(args.resume, env=env, device=args.device)
            print(f"Training wird fortgesetzt: {args.resume}")
        else:
            model = DQN(
                "CnnPolicy",
                env,
                learning_rate=1e-4,
                # CleanRL-inspired Atari settings, reduced from its 1M buffer so
                # that a free Colab runtime remains practical.
                buffer_size=args.buffer_size,
                optimize_memory_usage=True,
                replay_buffer_kwargs={"handle_timeout_termination": False},
                learning_starts=args.learning_starts,
                batch_size=32,
                train_freq=4,
                target_update_interval=1_000,
                exploration_fraction=0.10,
                exploration_final_eps=0.01,
                verbose=1,
                seed=42,
                device=args.device,
            )
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        model.learn(
            total_timesteps=args.timesteps,
            reset_num_timesteps=not bool(args.resume),
            callback=ServerCheckpointCallback(output, args.export_every),
            progress_bar=False,
        )
        export_policy(model, output)
        model.save(output.with_suffix(".zip"))
        print(f"Vanilla-DQN-Baseline exportiert nach {output}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
