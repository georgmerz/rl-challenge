"""Validate an ONNX submission against the tournament's CPU model contract.

Usage:
    python validate_submission.py path/to/model.onnx
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from app import ModelAgent


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate an RL Challenge ONNX model on CPU.")
    parser.add_argument("model", type=Path, help="Path to the submitted .onnx model")
    args = parser.parse_args()

    agent = ModelAgent(args.model)
    image = np.zeros((210, 160, 3), dtype=np.uint8)
    actions = [agent.act(image, player_id) for player_id in (1, 2)]
    print(f"OK: {args.model} ist turnierkompatibel auf CPU (Testaktionen: {actions}).")


if __name__ == "__main__":
    main()
