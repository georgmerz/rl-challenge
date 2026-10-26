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
    Mit ``player_id=None`` wechselt die trainierte Seite pro Episode. Die
    Beobachtung hat intern 13 Kanäle: zwölf öffentliche Bildkanäle und einen
    zusätzlichen konstanten Seitenkanal (0 für links, 1 für rechts).
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        opponent_model: str | Path,
        player_id: int | None = None,
        max_episode_steps: int = 1_000,
    ) -> None:
        if player_id not in (None, 1, 2):
            raise ValueError("player_id muss None, 1 (links/orange) oder 2 (rechts/grün) sein.")
        if max_episode_steps <= 0:
            raise ValueError("max_episode_steps muss positiv sein.")

        self.game = pong_v3.parallel_env(render_mode=None, num_players=2)
        self.opponent = ModelAgent(Path(opponent_model))
        self.training_player_id = player_id
        self.current_player_id: int | None = None
        self.max_episode_steps = max_episode_steps
        self.observation_space = spaces.Box(0.0, 1.0, shape=(13, 210, 160), dtype=np.float32)
        self.action_space = spaces.Discrete(6)
        self.controlled_player: str | None = None
        self.opponent_player: str | None = None
        self.opponent_id: int | None = None
        self.frames: dict[str, deque[np.ndarray]] | None = None
        self.steps = 0

    def _observation(self) -> np.ndarray:
        if self.frames is None or self.controlled_player is None or self.current_player_id is None:
            raise RuntimeError("reset() muss vor der Beobachtung aufgerufen werden.")
        image = public_image(np.stack(self.frames[self.controlled_player]))
        side_channel = np.full((1, 210, 160), self.current_player_id - 1, dtype=np.float32)
        return np.concatenate((image, side_channel), axis=0)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        observations, _ = self.game.reset(seed=seed)
        right_player, left_player = self.game.possible_agents
        self.current_player_id = self.training_player_id or (1 + int(self.np_random.integers(2)))
        if self.current_player_id == 1:
            self.controlled_player, self.opponent_player, self.opponent_id = left_player, right_player, 2
        else:
            self.controlled_player, self.opponent_player, self.opponent_id = right_player, left_player, 1
        self.frames = {
            player: deque([observations[player].copy()] * FRAME_STACK, maxlen=FRAME_STACK)
            for player in self.game.possible_agents
        }
        self.steps = 0
        return self._observation(), {}

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
            observation = self._observation()
        return observation, reward, terminated, truncated, {}

    def close(self) -> None:
        self.game.close()
