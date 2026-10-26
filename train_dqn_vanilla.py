"""Vanilla-DQN für Atari Pong gegen ein wählbares ONNX-Gegnermodell.

Beispiel:
    python create_baselines.py
    python train_dqn_vanilla.py --opponent baselines/baseline_pong_track_ball.onnx
"""

from __future__ import annotations

import argparse
import copy
import inspect
from pathlib import Path

import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, NatureCNN
from torch import nn
from torch.nn import functional as F

from pong_opponent_env import FRAME_STACK, PongVsOnnxEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train vanilla DQN against an ONNX Pong opponent.")
    parser.add_argument(
        "--opponent",
        type=Path,
        default=Path("baselines/baseline_pong_track_ball.onnx"),
        help="ONNX-Datei des Gegners (Standard: Ball-Tracking-Baseline).",
    )
    parser.add_argument("--player-id", choices=(1, 2), type=int, default=1, help="Trainierte Seite: 1 links, 2 rechts.")
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

    # 1. Pong-Umgebung gegen den gewählten ONNX-Gegner erzeugen.
    env = PongVsOnnxEnv(args.opponent, player_id=args.player_id)
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

        # 3. Lernen und regelmäßig portable ONNX-Checkpoints schreiben.
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


if __name__ == "__main__":
    main()
