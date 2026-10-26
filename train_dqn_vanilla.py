"""Vanilla-DQN für Atari Pong gegen ein wählbares ONNX-Gegnermodell.

Beispiel:
    python create_baselines.py
    python train_dqn_vanilla.py --opponent baselines/baseline_pong_reactive.onnx
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

from pong_opponent_env import FRAME_STACK, PongPreprocessor, PongVsOnnxEnv


# Diese Optionen sind die wenigen Einstellungen, die man für einen ersten
# Lauf normalerweise verändern möchte. Die DQN-Details stehen bewusst unten.
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train vanilla DQN against an ONNX Pong opponent.")
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
    #    Bei player_id=0 wählt sie die trainierte Seite pro Episode zufällig.
    env = PongVsOnnxEnv(args.opponent, player_id=None if args.player_id == 0 else args.player_id)
    try:
        # 2. Standard-DQN erstellen. Die festen Werte sind übliche Atari-
        #    Startwerte und kein Teil der Challenge-Regeln.
        # Der Buffer wächst bei kurzen Smoke-Tests nicht unnötig. Bei langen
        # Läufen bleibt er auf 10.000 komprimierte Beobachtungen begrenzt.
        buffer_size = min(10_000, max(256, args.timesteps))
        model = DQN(
            "CnnPolicy",
            env,
            learning_rate=1e-4,
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
    """Kleines Q-Netz für vier Graustufenbilder plus ein Seitenmerkmal."""

    def __init__(self, observation_space: spaces.Box, features_dim: int = 512, normalized_image: bool = True) -> None:
        # NatureCNN liefert 512 Bildmerkmale; das 513. Merkmal ist die Seite.
        super().__init__(observation_space, features_dim + 1)
        processed_space = spaces.Box(0.0, 1.0, shape=(FRAME_STACK, 84, 84), dtype=np.float32)
        self.cnn = NatureCNN(processed_space, features_dim=features_dim, normalized_image=normalized_image)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        # Stable-Baselines3 hat die gespeicherten uint8-Werte hier bereits auf
        # 0..1 normiert. Kanal 0..3 sind Bilder, Kanal 4 ist die Seite.
        visual_features = self.cnn(image[:, :FRAME_STACK])
        side_feature = image[:, FRAME_STACK, 0, 0].unsqueeze(1)
        return torch.cat((visual_features, side_feature), dim=1)


class OnnxPolicy(nn.Module):
    """Übersetzt den öffentlichen ONNX-Vertrag in das interne DQN-Format.

    Außen: vier RGB-Frames und player_id aus dem Runner.
    Innen: vier kleine Graustufenbilder und ein Seitenkanal wie im Training.
    """

    def __init__(self, q_net: nn.Module) -> None:
        super().__init__()
        self.preprocessor = PongPreprocessor()
        self.q_net = copy.deepcopy(q_net).cpu()

    def forward(self, image: torch.Tensor, player_id: torch.Tensor) -> torch.Tensor:
        # Dasselbe Modul erzeugt beim Export exakt die Trainingsbeobachtung.
        network_input = self.preprocessor(image, player_id)
        # DQN liefert sechs Q-Werte. argmax wählt die beste Atari-Aktion 0..5.
        return torch.argmax(self.q_net(network_input), dim=1).to(torch.int64)


def export_onnx(model: DQN, output: Path) -> None:
    """Exportiert die gelernte greedy Policy im festen Browser-Format.

    Die Beispieltensoren beschreiben den öffentlichen Vertrag. OnnxPolicy
    bettet die im Training verwendete Vorverarbeitung in die ONNX-Datei ein.
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
        # Stable-Baselines3 beendet das Training, sobald ein Callback nicht
        # True zurückgibt. Deshalb muss auch zwischen Checkpoints True folgen.
        return True


if __name__ == "__main__":
    main()
