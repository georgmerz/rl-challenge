"""Gym-Adapter: ein trainierter Pong-Spieler gegen ein beliebiges ONNX-Modell.

Das Modul verwendet ``ModelAgent`` aus ``app.py``. Der Gegner erhält damit
exakt dieselben vier Frames, player_id und CPU-ONNX-Laufzeit wie im Turnier.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from pettingzoo.atari import pong_v3

from app import ModelAgent


FRAME_STACK = 4


def public_image(frames: np.ndarray) -> np.ndarray:
    """Vier RGB-Frames in das öffentliche Format [12, 210, 160] umwandeln."""
    image = np.asarray(frames).transpose(0, 3, 1, 2).reshape(12, 210, 160)
    return np.ascontiguousarray(image, dtype=np.float32) / 255.0


class PongVsOnnxEnv(gym.Env):
    """Trainiert eine Seite von Pong gegen ein wählbares ONNX-Gegnermodell.

    ``player_id=1`` trainiert links/orange, ``player_id=2`` rechts/grün.
    Die andere Seite wird bei jedem Schritt vom ONNX-Modell gesteuert.
    """

    metadata = {"render_modes": []}

    def __init__(self, opponent_model: str | Path, player_id: int = 1, max_episode_steps: int = 1_000) -> None:
        if player_id not in (1, 2):
            raise ValueError("player_id muss 1 (links/orange) oder 2 (rechts/grün) sein.")
        if max_episode_steps <= 0:
            raise ValueError("max_episode_steps muss positiv sein.")

        self.game = pong_v3.parallel_env(render_mode=None, num_players=2)
        self.opponent = ModelAgent(Path(opponent_model))
        self.player_id = player_id
        self.max_episode_steps = max_episode_steps
        self.observation_space = spaces.Box(0.0, 1.0, shape=(12, 210, 160), dtype=np.float32)
        self.action_space = spaces.Discrete(6)
        self.controlled_player: str | None = None
        self.opponent_player: str | None = None
        self.opponent_id: int | None = None
        self.frames: dict[str, deque[np.ndarray]] | None = None
        self.steps = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        observations, _ = self.game.reset(seed=seed)
        right_player, left_player = self.game.possible_agents
        if self.player_id == 1:
            self.controlled_player, self.opponent_player, self.opponent_id = left_player, right_player, 2
        else:
            self.controlled_player, self.opponent_player, self.opponent_id = right_player, left_player, 1
        self.frames = {
            player: deque([observations[player].copy()] * FRAME_STACK, maxlen=FRAME_STACK)
            for player in self.game.possible_agents
        }
        self.steps = 0
        return public_image(np.stack(self.frames[self.controlled_player])), {}

    def step(self, action: int):
        if self.frames is None or self.controlled_player is None or self.opponent_player is None or self.opponent_id is None:
            raise RuntimeError("reset() muss vor step() aufgerufen werden.")

        opponent_action = self.opponent.act(np.stack(self.frames[self.opponent_player]), self.opponent_id)
        observations, rewards, terminations, truncations, _ = self.game.step(
            {self.controlled_player: int(action), self.opponent_player: opponent_action}
        )
        for player, image in observations.items():
            self.frames[player].append(image)
        self.steps += 1

        point_scored = any(reward != 0 for reward in rewards.values())
        naturally_finished = any(terminations.values()) or any(truncations.values())
        truncated = not (point_scored or naturally_finished) and self.steps >= self.max_episode_steps
        terminated = point_scored or naturally_finished
        reward = float(rewards.get(self.controlled_player, 0.0))
        if terminated or truncated:
            observation = np.zeros(self.observation_space.shape, dtype=np.float32)
        else:
            observation = public_image(np.stack(self.frames[self.controlled_player]))
        return observation, reward, terminated, truncated, {}

    def close(self) -> None:
        self.game.close()
