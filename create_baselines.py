"""Create small ONNX baseline policies for the Surround runner.

Run after installing requirements.txt:
    python create_baselines.py
"""

import inspect
from pathlib import Path

import torch


OUT_DIR = Path(__file__).resolve().parent / "baselines"
IMAGE_SHAPE = (1, 3, 210, 160)

UP = 1
RIGHT = 2
LEFT = 3
DOWN = 4


class StraightPolicy(torch.nn.Module):
    """Keeps moving away from the initial side.

    The public runner uses player_id 1 for the left/green player and player_id
    2 for the right/blue player.
    """

    def forward(self, image: torch.Tensor, player_id: torch.Tensor) -> torch.Tensor:
        left_player_action = torch.full((image.shape[0],), RIGHT, dtype=torch.int64, device=image.device)
        right_player_action = torch.full((image.shape[0],), LEFT, dtype=torch.int64, device=image.device)
        action = torch.where(player_id == 1, left_player_action, right_player_action)
        # Keep both public inputs in the ONNX graph, even though this simple
        # baseline does not need the image to choose its action.
        return action + image[:, 0, 0, 0].to(torch.int64) * 0


class WallAvoidPolicy(torch.nn.Module):
    """A tiny color-based policy that drives a conservative loop.

    It identifies the controlled player by color: player 1 is the left/green
    player and player 2 is the right/blue player. The policy is intentionally
    simple, but it works from either upload slot and makes Surround visibly
    more interesting than a static baseline.
    """

    def forward(self, image: torch.Tensor, player_id: torch.Tensor) -> torch.Tensor:
        # The runner supplies float32 NCHW images in [0, 1].
        red = image[:, 0, :, :]
        green = image[:, 1, :, :]
        blue = image[:, 2, :, :]
        height = image.shape[2]
        width = image.shape[3]
        x = torch.arange(width, device=image.device).reshape(1, 1, width).to(torch.float32)
        y = torch.arange(height, device=image.device).reshape(1, height, 1).to(torch.float32)
        playing_field = ((y >= 34) & (y <= 193)).to(torch.float32)

        green_player = ((red < 0.5) & (green > 0.6) & (blue < 0.5)).to(torch.float32)
        blue_player = ((red < 0.35) & (green < 0.35) & (blue > 0.55)).to(torch.float32)
        player_is_left = (player_id == 1).reshape(-1, 1, 1)
        player_pixels = torch.where(player_is_left, green_player, blue_player) * playing_field

        pixel_count = player_pixels.sum(dim=(1, 2)).clamp_min(1.0)
        x_center = (player_pixels * x).sum(dim=(1, 2)) / pixel_count
        y_center = (player_pixels * y).sum(dim=(1, 2)) / pixel_count

        up = torch.full_like(player_id, UP)
        right = torch.full_like(player_id, RIGHT)
        left = torch.full_like(player_id, LEFT)
        down = torch.full_like(player_id, DOWN)

        # Player 1 starts left and drives clockwise, turning before the boundary.
        p1_action = torch.where(
            (x_center < 122.0) & (y_center < 155.0),
            down,
            torch.where(
                (x_center < 122.0) & (y_center >= 155.0),
                right,
                torch.where(
                    (x_center >= 122.0) & (y_center > 58.0),
                    up,
                    torch.where((x_center > 38.0) & (y_center <= 58.0), left, down),
                ),
            ),
        )

        # Player 2 starts right and drives counter-clockwise, also turning early.
        p2_action = torch.where(
            (x_center > 38.0) & (y_center > 65.0),
            up,
            torch.where(
                (x_center > 38.0) & (y_center <= 65.0),
                left,
                torch.where(
                    (x_center <= 38.0) & (y_center < 155.0),
                    down,
                    torch.where((x_center < 122.0) & (y_center >= 155.0), right, up),
                ),
            ),
        )

        return torch.where(player_id == 1, p1_action, p2_action)

def export_policy(policy: torch.nn.Module, filename: str) -> None:
    policy.eval()
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
    for stale_filename in ("baseline_noop.onnx", "baseline_follow_ball.onnx", "baseline_wide_loop.onnx"):
        (OUT_DIR / stale_filename).unlink(missing_ok=True)
    export_policy(StraightPolicy(), "baseline_straight.onnx")
    export_policy(WallAvoidPolicy(), "baseline_wall_avoid.onnx")
    print(f"Baselines geschrieben nach {OUT_DIR}")
