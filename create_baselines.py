"""Create small ONNX baseline policies for the Pong runner.

Run after installing requirements.txt:
    python create_baselines.py
"""

import inspect
from pathlib import Path

import torch


OUT_DIR = Path(__file__).resolve().parent / "baselines"
# Four RGB frames, oldest to newest. The runner always supplies this history.
IMAGE_SHAPE = (1, 12, 210, 160)

NOOP = 0
FIRE = 1
# PettingZoo nennt 2/3 offiziell RIGHT/LEFT. In der gedrehten Pong-
# Darstellung bewegen diese Controller-Richtungen den Schläger auf/ab.
PADDLE_UP = 2
PADDLE_DOWN = 3


class PongStaticPolicy(torch.nn.Module):
    """Always press fire.

    This intentionally weak policy is useful to check upload, ONNX export and
    the tournament runner. Pong's serve timer also prevents it from stalling
    a game indefinitely.
    """

    def forward(self, image: torch.Tensor, player_id: torch.Tensor) -> torch.Tensor:
        action = torch.full((image.shape[0],), FIRE, dtype=torch.int64, device=image.device)
        return action + player_id * 0 + image[:, 0, 0, 0].to(torch.int64) * 0


class PongTrackBallPolicy(torch.nn.Module):
    """A small visual Pong policy that moves its paddle towards the ball.

    The left/orange paddle is player 1 and the right/green paddle is player 2.
    It only uses the current image and therefore demonstrates the exact public
    ONNX interface used by student models.
    """

    def forward(self, image: torch.Tensor, player_id: torch.Tensor) -> torch.Tensor:
        # The newest RGB frame is the last group of three channels.
        image = image[:, 9:12]
        red, green, blue = image[:, 0], image[:, 1], image[:, 2]
        height = image.shape[2]
        y = torch.arange(height, device=image.device, dtype=torch.float32).reshape(1, height, 1)
        playing_field = ((y >= 34) & (y <= 193)).to(torch.float32)

        # Ball: the only bright white object inside the actual field.
        ball = ((red > 0.9) & (green > 0.9) & (blue > 0.9)).to(torch.float32) * playing_field
        ball_count = ball.sum(dim=(1, 2)).clamp_min(1.0)
        ball_y = (ball * y).sum(dim=(1, 2)) / ball_count

        # Pong's left paddle is orange and its right paddle green.
        orange = ((red > 0.7) & (green > 0.3) & (green < 0.75) & (blue < 0.5)).to(torch.float32)
        green_paddle = ((red < 0.55) & (green > 0.55) & (blue < 0.55)).to(torch.float32)
        player_pixels = torch.where((player_id == 1).reshape(-1, 1, 1), orange, green_paddle) * playing_field
        paddle_count = player_pixels.sum(dim=(1, 2)).clamp_min(1.0)
        paddle_y = (player_pixels * y).sum(dim=(1, 2)) / paddle_count

        move_up = torch.full_like(player_id, PADDLE_UP)
        move_down = torch.full_like(player_id, PADDLE_DOWN)
        noop = torch.full_like(player_id, NOOP)
        fire = torch.full_like(player_id, FIRE)
        tracking_action = torch.where(
            ball_y < paddle_y - 4.0,
            move_up,
            torch.where(ball_y > paddle_y + 4.0, move_down, noop),
        )
        # When no ball is visible yet, request a serve.
        return torch.where(ball.sum(dim=(1, 2)) > 0, tracking_action, fire)


def export_policy(policy: torch.nn.Module, filename: str) -> None:
    policy = policy.eval()
    image = torch.zeros(IMAGE_SHAPE, dtype=torch.float32)
    player_id = torch.ones((1,), dtype=torch.int64)
    export_kwargs = {
        "input_names": ["image", "player_id"],
        "output_names": ["action"],
        "opset_version": 17,
    }
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        export_kwargs["dynamo"] = False
    torch.onnx.export(policy, (image, player_id), str(OUT_DIR / filename), **export_kwargs)


if __name__ == "__main__":
    OUT_DIR.mkdir(exist_ok=True)
    for stale_filename in (
        "baseline_straight.onnx",
        "baseline_wall_avoid.onnx",
        "baseline_wall_counter.onnx",
        "baseline_noop.onnx",
        "baseline_follow_ball.onnx",
        "baseline_wide_loop.onnx",
    ):
        (OUT_DIR / stale_filename).unlink(missing_ok=True)
    export_policy(PongStaticPolicy(), "baseline_pong_static.onnx")
    export_policy(PongTrackBallPolicy(), "baseline_pong_track_ball.onnx")
    print(f"Pong-Baselines geschrieben nach {OUT_DIR}")
