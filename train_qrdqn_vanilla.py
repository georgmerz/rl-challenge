"""Vanilla-QR-DQN für Atari Pong gegen ein wählbares ONNX-Modell.

QR-DQN schätzt für jede Aktion mehrere mögliche Returns (Quantile). Für die
greedy Aktion wird deren Mittelwert verwendet. Das Skript entspricht ansonsten
bewusst dem einfachen DQN-Beispiel in ``train_dqn_vanilla.py``.

Beispiel:
    python create_baselines.py
    python train_qrdqn_vanilla.py --opponent baselines/baseline_pong_reactive.onnx
"""

from __future__ import annotations

import argparse
import copy
import inspect
from pathlib import Path

import numpy as np
import torch
from gymnasium import spaces
from sb3_contrib import QRDQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, NatureCNN
from torch import nn

from pong_opponent_env import FRAME_STACK, PongPreprocessor, PongVsOnnxEnv


def parse_args() -> argparse.Namespace:
    """Nur die Einstellungen, die für einen ersten Lauf wichtig sind."""
    parser = argparse.ArgumentParser(description="Train vanilla QR-DQN against an ONNX Pong opponent.")
    parser.add_argument(
        "--opponent",
        type=Path,
        default=Path("baselines/baseline_pong_reactive.onnx"),
        help="ONNX-Datei des Gegners (Standard: reaktive Baseline).",
    )
    parser.add_argument(
        "--player-id",
        choices=(0, 1, 2),
        type=int,
        default=0,
        help="0: Seite pro Episode zufällig wählen (Standard); 1: links; 2: rechts.",
    )
    parser.add_argument("--timesteps", type=int, default=1_000_000, help="Anzahl Trainingsschritte.")
    parser.add_argument("--output", default="baselines/baseline_pong_qrdqn.onnx", help="Ziel für ONNX.")
    parser.add_argument("--device", default="auto", help="auto, cuda, mps oder cpu.")
    parser.add_argument("--export-every", type=int, default=100_000, help="Checkpoint-Abstand in Schritten.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if min(args.timesteps, args.export_every) <= 0:
        raise ValueError("--timesteps und --export-every müssen positiv sein.")

    # Wie beim DQN-Beispiel lernt genau eine, pro Episode zufällig gewählte
    # Seite gegen einen festen ONNX-Gegner.
    env = PongVsOnnxEnv(args.opponent, player_id=None if args.player_id == 0 else args.player_id)
    try:
        buffer_size = min(10_000, max(256, args.timesteps))
        model = QRDQN(
            "CnnPolicy",
            env,
            learning_rate=5e-5,
            buffer_size=buffer_size,
            learning_starts=5_000,
            batch_size=32,
            train_freq=4,
            target_update_interval=1_000,
            exploration_fraction=0.4,
            exploration_final_eps=0.01,
            policy_kwargs={"features_extractor_class": PongCNN},
            verbose=1,
            seed=42,
            device=args.device,
        )

        output = Path(args.output)
        model.learn(
            total_timesteps=args.timesteps,
            callback=CheckpointCallback(output, args.export_every),
            progress_bar=False,
        )

        export_onnx(model, output)
        model.save(output.with_suffix(".zip"))
        print(f"Vanilla-QR-DQN-Policy exportiert nach {output}")
    finally:
        env.close()


class PongCNN(BaseFeaturesExtractor):
    """Dasselbe kleine CNN wie im DQN-Beispiel."""

    def __init__(self, observation_space: spaces.Box, features_dim: int = 512, normalized_image: bool = True) -> None:
        super().__init__(observation_space, features_dim + 1)
        processed_space = spaces.Box(0.0, 1.0, shape=(FRAME_STACK, 84, 84), dtype=np.float32)
        self.cnn = NatureCNN(processed_space, features_dim=features_dim, normalized_image=normalized_image)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        visual_features = self.cnn(image[:, :FRAME_STACK])
        side_feature = image[:, FRAME_STACK, 0, 0].unsqueeze(1)
        return torch.cat((visual_features, side_feature), dim=1)


class OnnxPolicy(nn.Module):
    """Verbindet Vorverarbeitung und greedy QR-DQN-Entscheidung."""

    def __init__(self, quantile_net: nn.Module) -> None:
        super().__init__()
        self.preprocessor = PongPreprocessor()
        self.quantile_net = copy.deepcopy(quantile_net).cpu()

    def forward(self, image: torch.Tensor, player_id: torch.Tensor) -> torch.Tensor:
        network_input = self.preprocessor(image, player_id)
        # Form: [Batch, Quantil, Aktion]. Ihr Mittelwert entspricht dem von
        # QR-DQN für die greedy Entscheidung verwendeten erwarteten Q-Wert.
        quantiles = self.quantile_net(network_input)
        expected_q_values = quantiles.mean(dim=1)
        return torch.argmax(expected_q_values, dim=1).to(torch.int64)


def export_onnx(model: QRDQN, output: Path) -> None:
    """Exportiert QR-DQN im festen Browser- und Turnierformat."""
    output.parent.mkdir(parents=True, exist_ok=True)
    policy = OnnxPolicy(model.policy.quantile_net).eval()
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
    """Schreibt regelmäßig SB3- und ONNX-Checkpoints."""

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
