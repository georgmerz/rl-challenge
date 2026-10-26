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
import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, NatureCNN
from torch import nn
from torch.nn import functional as F

from train_ppo_sb3 import FRAME_STACK, Player1VsPongBaselineEnv

# These are deliberately fixed so that the usual student command only needs
# --timesteps and --output. They are ordinary Atari-DQN defaults, not rules of
# the challenge; changing them is an optional experiment.
LEARNING_RATE = 1e-4
REPLAY_BUFFER_SIZE = 100_000
RANDOM_START_STEPS = 80_000
FINAL_EXPLORATION_RATE = 0.01


class PublicFrameStack(gym.ObservationWrapper):
    """Expose four raw frames exactly as the public ONNX input image."""

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.observation_space = spaces.Box(0.0, 1.0, shape=(12, 210, 160), dtype=np.float32)

    def observation(self, frames: np.ndarray) -> np.ndarray:
        frames = np.asarray(frames)
        image = frames.transpose(0, 3, 1, 2).reshape(12, 210, 160)
        return np.ascontiguousarray(image, dtype=np.float32) / 255.0


class AtariPreprocessorCNN(BaseFeaturesExtractor):
    """The one shared preprocessing path for training and ONNX inference."""

    def __init__(
        self,
        observation_space: spaces.Box,
        features_dim: int = 512,
        normalized_image: bool = True,
    ) -> None:
        super().__init__(observation_space, features_dim)
        processed_space = spaces.Box(0.0, 1.0, shape=(FRAME_STACK, 84, 84), dtype=np.float32)
        self.encoder = NatureCNN(processed_space, features_dim=features_dim, normalized_image=normalized_image)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        # Public input: four normalized RGB frames, oldest to newest.
        batch = image.shape[0]
        frames = image.reshape(batch, FRAME_STACK, 3, 210, 160)[:, :, :, 34:194, :]
        grayscale = 0.299 * frames[:, :, 0] + 0.587 * frames[:, :, 1] + 0.114 * frames[:, :, 2]
        resized = F.interpolate(grayscale, size=(84, 84), mode="bilinear", align_corners=False)
        return self.encoder(resized)


class ExportedPolicy(nn.Module):
    """Expose the trained Q-network through the fixed public ONNX interface."""

    def __init__(self, q_net: nn.Module) -> None:
        super().__init__()
        self.q_net = copy.deepcopy(q_net).cpu()

    def forward(self, image: torch.Tensor, player_id: torch.Tensor) -> torch.Tensor:
        # The Q-network already contains AtariPreprocessorCNN. Consequently
        # this path is identical during SB3 training and ONNX inference.
        action = torch.argmax(self.q_net(image), dim=1).to(torch.int64)
        return action + player_id * 0


def make_env() -> gym.Env:
    """Stack raw frames; the model performs all visual preprocessing itself."""
    env = Player1VsPongBaselineEnv()
    env = gym.wrappers.FrameStackObservation(env, FRAME_STACK)
    return PublicFrameStack(env)


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
    advanced = parser.add_argument_group("optional settings for long runs")
    advanced.add_argument(
        "--exploration-fraction",
        type=float,
        default=0.10,
        help="Share of the run over which epsilon falls from 1.0 to 0.01 (default: 0.10).",
    )
    advanced.add_argument("--export-every", type=int, default=100_000, help="Checkpoint interval in steps (default: 100000).")
    advanced.add_argument("--resume", type=Path, help="Optional .zip checkpoint to continue; --timesteps are added training steps.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if min(args.timesteps, args.export_every) <= 0:
        raise ValueError("--timesteps und --export-every müssen positiv sein.")
    if not 0 < args.exploration_fraction <= 1:
        raise ValueError("--exploration-fraction muss zwischen 0 (exklusiv) und 1 liegen.")
    env = make_env()
    try:
        # Standard SB3 DQN. The feature extractor contains only the shared
        # image preprocessing plus SB3's standard NatureCNN encoder.
        if args.resume:
            model = DQN.load(args.resume, env=env, device=args.device)
            print(f"Training wird fortgesetzt: {args.resume}")
        else:
            model = DQN(
                "CnnPolicy",
                env,
                learning_rate=LEARNING_RATE,
                # The fixed settings above keep the example readable and
                # memory-friendly for Colab.
                buffer_size=REPLAY_BUFFER_SIZE,
                optimize_memory_usage=True,
                replay_buffer_kwargs={"handle_timeout_termination": False},
                learning_starts=RANDOM_START_STEPS,
                batch_size=32,
                train_freq=4,
                target_update_interval=1_000,
                exploration_fraction=args.exploration_fraction,
                exploration_final_eps=FINAL_EXPLORATION_RATE,
                policy_kwargs={
                    "features_extractor_class": AtariPreprocessorCNN,
                    "normalize_images": False,
                },
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
