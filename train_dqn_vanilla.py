"""Minimaler Startpunkt: Vanilla-DQN fuer Atari Pong trainieren.

Dieses Beispiel trainiert Spieler 1 (links/orange) gegen die feste
Ball-Tracking-Baseline. Die technische Umgebung und der ONNX-Export stecken
in ``_training_support.py``; hier steht nur der Trainingsablauf.

Beispiel:
    python train_dqn_vanilla.py --timesteps 1000000
"""

from __future__ import annotations

import argparse
from pathlib import Path

from _training_support import OnnxCheckpointCallback, export_policy, make_dqn, make_env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train vanilla DQN for the Pong challenge.")
    parser.add_argument("--timesteps", type=int, default=1_000_000, help="Anzahl Trainingsschritte.")
    parser.add_argument("--output", default="baselines/baseline_pong_dqn.onnx", help="Ziel fuer die ONNX-Datei.")
    parser.add_argument("--device", default="auto", help="auto, cuda, mps oder cpu.")
    parser.add_argument(
        "--exploration-fraction",
        type=float,
        default=0.10,
        help="Anteil des Laufs mit fallendem Epsilon von 1.0 auf 0.01.",
    )
    parser.add_argument("--export-every", type=int, default=100_000, help="Checkpoint-Abstand in Schritten.")
    parser.add_argument("--resume", type=Path, help="Optionales .zip-Checkpoint zum Fortsetzen.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if min(args.timesteps, args.export_every) <= 0:
        raise ValueError("--timesteps und --export-every muessen positiv sein.")
    if not 0 < args.exploration_fraction <= 1:
        raise ValueError("--exploration-fraction muss zwischen 0 (exklusiv) und 1 liegen.")

    # 1. Umgebung: linker Spieler gegen feste Ball-Tracking-Baseline.
    env = make_env()
    try:
        # 2. Standard-DQN anlegen (oder ein gespeichertes Training laden).
        model = make_dqn(env, args.device, args.exploration_fraction, args.resume)
        if args.resume:
            print(f"Training wird fortgesetzt: {args.resume}")

        # 3. Lernen; unterwegs entstehen .zip- und .onnx-Checkpoints.
        output = Path(args.output)
        model.learn(
            total_timesteps=args.timesteps,
            reset_num_timesteps=not bool(args.resume),
            callback=OnnxCheckpointCallback(output, args.export_every),
            progress_bar=False,
        )

        # 4. Finale Policy exportieren und auch das SB3-Checkpoint behalten.
        export_policy(model, output)
        model.save(output.with_suffix(".zip"))
        print(f"Vanilla-DQN-Baseline exportiert nach {output}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
