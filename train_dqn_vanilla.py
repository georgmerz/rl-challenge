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


# Diese Optionen sind die wenigen Einstellungen, die man für einen ersten
# Lauf normalerweise verändern möchte. Die DQN-Details stehen bewusst unten.
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train vanilla DQN against an ONNX Pong opponent.")
    parser.add_argument(
        "--opponent",
        type=Path,
        default=Path("baselines/baseline_pong_track_ball.onnx"),
        help="ONNX-Datei des Gegners (Standard: Ball-Tracking-Baseline).",
    )
    parser.add_argument(
        "--player-id",
        choices=(0, 1, 2),
        type=int,
        default=0,
        help="0: Seiten pro Episode wechseln (Standard); 1: links; 2: rechts.",
    )
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
    #    Bei player_id=0 wechselt sie pro Episode die trainierte Seite.
    env = PongVsOnnxEnv(args.opponent, player_id=None if args.player_id == 0 else args.player_id)
    try:
        # 2. Standard-DQN erstellen. Die festen Werte sind übliche Atari-
        #    Startwerte und kein Teil der Challenge-Regeln.
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

        # 3. Lernen. Ein Checkpoint enthält sowohl SB3-Gewichte (.zip) als
        #    auch eine im Browser lauffähige ONNX-Datei (.onnx).
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
    """Bildvorverarbeitung plus ein tatsächliches Seitenmerkmal für das Q-Netz."""

    def __init__(self, observation_space: spaces.Box, features_dim: int = 512, normalized_image: bool = True) -> None:
        # NatureCNN liefert 512 Bildmerkmale; das 513. Merkmal ist die Seite.
        super().__init__(observation_space, features_dim + 1)
        processed_space = spaces.Box(0.0, 1.0, shape=(FRAME_STACK, 84, 84), dtype=np.float32)
        self.cnn = NatureCNN(processed_space, features_dim=features_dim, normalized_image=normalized_image)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        # Internes Trainingsformat: [B, 13, 210, 160]. Die Kanäle 0..11
        # enthalten vier RGB-Frames, Kanal 12 kodiert die Spielerseite.
        batch_size = image.shape[0]
        rgb = image[:, :12].reshape(batch_size, FRAME_STACK, 3, 210, 160)[:, :, :, 34:194, :]
        # Vom Originalbild bleibt nur das Spielfeld. Anschließend folgen
        # Graustufen und 84x84 Skalierung wie bei klassischen Atari-DQNs.
        gray = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
        visual_features = self.cnn(F.interpolate(gray, size=(84, 84), mode="bilinear", align_corners=False))
        # Das Q-Netz bekommt 512 Bildmerkmale PLUS dieses eine Seitenmerkmal.
        side_feature = image[:, 12, 0, 0].unsqueeze(1)
        return torch.cat((visual_features, side_feature), dim=1)


class OnnxPolicy(nn.Module):
    """Übersetzt den öffentlichen ONNX-Vertrag in das interne DQN-Format.

    Außen: image mit 12 Kanälen und player_id aus dem Runner.
    Innen: image mit 13 Kanälen, weil DQN ein Seitenmerkmal mitlernt.
    """

    def __init__(self, q_net: nn.Module) -> None:
        super().__init__()
        self.q_net = copy.deepcopy(q_net).cpu()

    def forward(self, image: torch.Tensor, player_id: torch.Tensor) -> torch.Tensor:
        # Der Runner liefert 1 für links und 2 für rechts. Das Training kennt
        # dagegen 0 bzw. 1. Der so erzeugte Kanal macht player_id zu einer
        # echten, entscheidungsrelevanten Eingabe des Q-Netzes.
        side_channel = (player_id - 1).to(image.dtype).reshape(-1, 1, 1, 1)
        side_channel = side_channel.expand(-1, 1, 210, 160)
        network_input = torch.cat((image, side_channel), dim=1)
        # DQN liefert sechs Q-Werte. argmax wählt die beste Atari-Aktion 0..5.
        return torch.argmax(self.q_net(network_input), dim=1).to(torch.int64)


def export_onnx(model: DQN, output: Path) -> None:
    """Exportiert die gelernte greedy Policy im festen Browser-Format.

    Die Beispieltensoren beschreiben den öffentlichen Vertrag, nicht das
    13-Kanal-Trainingsformat. OnnxPolicy ergänzt den dreizehnten Kanal selbst.
    """
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
            # .zip ist zum Fortsetzen des Trainings; .onnx ist zum Spielen
            # im Browser und für die spätere Abgabe.
            self.model.save(checkpoint)
            export_onnx(self.model, checkpoint.with_suffix(".onnx"))
            print(f"Checkpoint nach {self.num_timesteps} Schritten: {checkpoint}.zip und .onnx")
            self.next_checkpoint += self.every_steps


if __name__ == "__main__":
    main()
