"""Vanilla-DQN mit einfachem Self-Play fuer die Pong-Challenge.

Der trainierte Agent spielt pro Episode abwechselnd links/orange und
rechts/gruen gegen seine aktuelle (greedy ausgefuehrte) Policy. Dieses Skript
ist als ein vollstaendiger, veränderbarer Einstieg gedacht.

Beispiel:
    python train_dqn_vanilla.py --timesteps 1000000
"""

from __future__ import annotations

import argparse
import copy
import inspect
from collections import deque
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from pettingzoo.atari import pong_v3
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, NatureCNN
from torch import nn
from torch.nn import functional as F


FRAME_STACK = 4
MAX_EPISODE_STEPS = 1_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train vanilla DQN with self-play for Atari Pong.")
    parser.add_argument("--timesteps", type=int, default=1_000_000, help="Anzahl Trainingsschritte.")
    parser.add_argument("--output", default="baselines/baseline_pong_dqn.onnx", help="Ziel für die ONNX-Datei.")
    parser.add_argument("--device", default="auto", help="auto, cuda, mps oder cpu.")
    parser.add_argument("--export-every", type=int, default=100_000, help="Checkpoint-Abstand in Schritten.")
    return parser.parse_args()


def main() -> None:
    """Vier sichtbare Schritte des Minimalbeispiels."""
    args = parse_args()
    if min(args.timesteps, args.export_every) <= 0:
        raise ValueError("--timesteps und --export-every müssen positiv sein.")

    # 1. Pong als Einzelagenten-Umgebung erzeugen.
    env = SelfPlayPong()
    try:
        # 2. Standard-DQN erstellen. Die festen Werte sind übliche Atari-Startwerte.
        model = DQN(
            "CnnPolicy",
            env,
            learning_rate=1e-4,
            buffer_size=100_000,
            learning_starts=80_000,
            batch_size=32,
            train_freq=4,
            target_update_interval=1_000,
            exploration_fraction=0.4,
            exploration_final_eps=0.01,
            policy_kwargs={"features_extractor_class": PongCNN, "normalize_images": False},
            verbose=1,
            seed=42,
            device=args.device,
        )
        env.set_opponent(model)

        # 3. Lernen. Der Gegner verwendet dabei dieselbe, laufend besser werdende Policy.
        output = Path(args.output)
        model.learn(
            total_timesteps=args.timesteps,
            callback=CheckpointCallback(output, args.export_every),
            progress_bar=False,
        )

        # 4. Finale Policy im Turnierformat und als SB3-Checkpoint sichern.
        export_onnx(model, output)
        model.save(output.with_suffix(".zip"))
        print(f"Vanilla-DQN-Policy exportiert nach {output}")
    finally:
        env.close()


def stack_frames(frames: np.ndarray) -> np.ndarray:
    """Vier RGB-Frames in das öffentliche Format [12, 210, 160] umwandeln."""
    image = np.asarray(frames).transpose(0, 3, 1, 2).reshape(12, 210, 160)
    return np.ascontiguousarray(image, dtype=np.float32) / 255.0


class SelfPlayPong(gym.Env):
    """Macht aus dem Zwei-Spieler-Pong eine Gym-Umgebung für einen DQN-Agenten."""

    metadata = {"render_modes": []}

    def __init__(self) -> None:
        self.game = pong_v3.parallel_env(render_mode=None, num_players=2)
        self.observation_space = spaces.Box(0.0, 1.0, shape=(12, 210, 160), dtype=np.float32)
        self.action_space = spaces.Discrete(6)
        self.model: DQN | None = None
        self.controlled_player: str | None = None
        self.opponent_player: str | None = None
        self.frames: dict[str, deque[np.ndarray]] | None = None
        self.steps = 0

    def set_opponent(self, model: DQN) -> None:
        """Die aktuelle DQN-Policy liefert im Spiel die Gegneraktionen."""
        self.model = model

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        observations, _ = self.game.reset(seed=seed)
        right_player, left_player = self.game.possible_agents

        # Beide Seiten kommen im Replay-Buffer vor; die Farbe ist im Bild sichtbar.
        if self.np_random.integers(2):
            self.controlled_player, self.opponent_player = left_player, right_player
        else:
            self.controlled_player, self.opponent_player = right_player, left_player
        self.frames = {
            player: deque([observations[player].copy()] * FRAME_STACK, maxlen=FRAME_STACK)
            for player in self.game.possible_agents
        }
        self.steps = 0
        return stack_frames(np.stack(self.frames[self.controlled_player])), {}

    def step(self, action: int):
        if self.model is None or self.frames is None:
            raise RuntimeError("Vor dem Training muss set_opponent(model) aufgerufen werden.")
        if self.controlled_player is None or self.opponent_player is None:
            raise RuntimeError("reset() muss vor step() aufgerufen werden.")

        opponent_image = stack_frames(np.stack(self.frames[self.opponent_player]))
        # Der Gegner spielt die aktuelle Policy ohne zusätzliche Exploration.
        opponent_action, _ = self.model.predict(opponent_image, deterministic=True)
        observations, rewards, terminations, truncations, _ = self.game.step(
            {
                self.controlled_player: int(action),
                self.opponent_player: int(opponent_action),
            }
        )
        for player, image in observations.items():
            self.frames[player].append(image)
        self.steps += 1

        point_scored = any(reward != 0 for reward in rewards.values())
        naturally_finished = any(terminations.values()) or any(truncations.values())
        truncated = not (point_scored or naturally_finished) and self.steps >= MAX_EPISODE_STEPS
        terminated = point_scored or naturally_finished
        reward = float(rewards.get(self.controlled_player, 0.0))
        if terminated or truncated:
            observation = np.zeros(self.observation_space.shape, dtype=np.float32)
        else:
            observation = stack_frames(np.stack(self.frames[self.controlled_player]))
        return observation, reward, terminated, truncated, {}

    def close(self) -> None:
        self.game.close()


class PongCNN(BaseFeaturesExtractor):
    """Vorverarbeitung im Modell: identisch beim Training und im ONNX-Export."""

    def __init__(self, observation_space: spaces.Box, features_dim: int = 512, normalized_image: bool = True) -> None:
        super().__init__(observation_space, features_dim)
        processed_space = spaces.Box(0.0, 1.0, shape=(FRAME_STACK, 84, 84), dtype=np.float32)
        self.cnn = NatureCNN(processed_space, features_dim=features_dim, normalized_image=normalized_image)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        # [B, 12, 210, 160] -> vier Graustufenbilder [B, 4, 84, 84].
        batch_size = image.shape[0]
        rgb = image.reshape(batch_size, FRAME_STACK, 3, 210, 160)[:, :, :, 34:194, :]
        gray = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
        return self.cnn(F.interpolate(gray, size=(84, 84), mode="bilinear", align_corners=False))


class OnnxPolicy(nn.Module):
    """Ergänzt das DQN-Q-Netz um die verpflichtende player_id-ONNX-Eingabe."""

    def __init__(self, q_net: nn.Module) -> None:
        super().__init__()
        self.q_net = copy.deepcopy(q_net).cpu()

    def forward(self, image: torch.Tensor, player_id: torch.Tensor) -> torch.Tensor:
        action = torch.argmax(self.q_net(image), dim=1).to(torch.int64)
        return action + player_id * 0  # player_id bleibt Teil der ONNX-Schnittstelle.


def export_onnx(model: DQN, output: Path) -> None:
    """Exportiert die gelernte greedy Policy für den Browser-Runner."""
    output.parent.mkdir(parents=True, exist_ok=True)
    policy = OnnxPolicy(model.policy.q_net).eval()
    image = torch.zeros((1, 12, 210, 160), dtype=torch.float32)
    player_id = torch.ones((1,), dtype=torch.int64)
    arguments = {
        "input_names": ["image", "player_id"],
        "output_names": ["action"],
        "opset_version": 17,
    }
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        arguments["dynamo"] = False
    torch.onnx.export(policy, (image, player_id), str(output), **arguments)


class CheckpointCallback(BaseCallback):
    """Schreibt regelmäßig ein SB3- und ein ONNX-Checkpoint."""

    def __init__(self, output: Path, every_steps: int) -> None:
        super().__init__()
        self.output = output
        self.every_steps = every_steps
        self.next_checkpoint = every_steps

    def _on_step(self) -> bool:
        if self.num_timesteps >= self.next_checkpoint:
            checkpoint = self.output.with_name(f"{self.output.stem}_step_{self.num_timesteps}")
            self.model.save(checkpoint)
            export_onnx(self.model, checkpoint.with_suffix(".onnx"))
            print(f"Checkpoint nach {self.num_timesteps} Schritten: {checkpoint}.zip und .onnx")
            self.next_checkpoint += self.every_steps
        return True


if __name__ == "__main__":
    main()
