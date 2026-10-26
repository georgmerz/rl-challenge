"""Gym-Adapter: ein trainierter Pong-Spieler gegen ein beliebiges ONNX-Modell.

Das Modul verwendet ``ModelAgent`` aus ``app.py``. Der Gegner erhält damit
exakt dieselben vier Frames, player_id und CPU-ONNX-Laufzeit wie im Turnier.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from pettingzoo.atari import pong_v3
from torch import nn
from torch.nn import functional as F

from app import ModelAgent


FRAME_STACK = 4


def public_image(frames: np.ndarray) -> np.ndarray:
    """Vier RGB-Frames in das öffentliche Modellformat [12, 210, 160] wandeln.

    Diese Funktion entspricht der Vorverarbeitung des Turnier-Runners:
    ältester Frame zuerst, HWC -> NCHW und Pixelwerte von 0..255 nach 0..1.
    """
    image = np.asarray(frames).transpose(0, 3, 1, 2).reshape(12, 210, 160)
    return np.ascontiguousarray(image, dtype=np.float32) / 255.0


class PongPreprocessor(nn.Module):
    """Komprimiert den öffentlichen State für DQN auf fünf kleine Kanäle.

    Vier Kanäle enthalten die zugeschnittenen 84x84-Graustufenbilder. Der
    fünfte Kanal enthält 0 für links und 255 für rechts. Die Ausgabe bleibt
    absichtlich im Wertebereich 0..255, damit Stable-Baselines3 sie wie ein
    übliches Atari-Bild speichert und erst vor dem CNN auf 0..1 normiert.
    """

    def forward(self, image: torch.Tensor, player_id: torch.Tensor) -> torch.Tensor:
        batch_size = image.shape[0]
        rgb = image.reshape(batch_size, FRAME_STACK, 3, 210, 160)[:, :, :, 34:194, :]
        gray = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
        gray = F.interpolate(gray, size=(84, 84), mode="bilinear", align_corners=False) * 255.0
        side = (player_id - 1).to(image.dtype).reshape(-1, 1, 1, 1) * 255.0
        side = side.expand(-1, 1, 84, 84)
        return torch.cat((gray, side), dim=1)


class PongVsOnnxEnv(gym.Env):
    """Trainiert eine Seite von Pong gegen ein wählbares ONNX-Gegnermodell.

    ``player_id=1`` trainiert links/orange, ``player_id=2`` rechts/grün.
    Mit ``player_id=None`` wird die trainierte Seite pro Episode zufällig
    gewählt. Die Beobachtung hat intern fünf kleine ``uint8``-Kanäle: vier
    vorverarbeitete Bilder und einen Seitenkanal. Der öffentliche
    ONNX-Vertrag bleibt davon unberührt.
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
        # ModelAgent prüft den ONNX-Vertrag und ruft den Gegner mit CPU-ONNX
        # auf. Der Gegner sieht damit exakt die Turnier-Eingabe, nicht den
        # zusätzlichen Trainingskanal weiter unten.
        self.opponent = ModelAgent(Path(opponent_model))
        self.preprocessor = PongPreprocessor().eval()
        self.training_player_id = player_id
        self.current_player_id: int | None = None
        self.max_episode_steps = max_episode_steps
        self.observation_space = spaces.Box(0, 255, shape=(5, 84, 84), dtype=np.uint8)
        self.action_space = spaces.Discrete(6)
        self.controlled_player: str | None = None
        self.opponent_player: str | None = None
        self.opponent_id: int | None = None
        self.frames: dict[str, deque[np.ndarray]] | None = None
        self.steps = 0

    def _observation(self) -> np.ndarray:
        if self.frames is None or self.controlled_player is None or self.current_player_id is None:
            raise RuntimeError("reset() muss vor der Beobachtung aufgerufen werden.")
        image = torch.from_numpy(public_image(np.stack(self.frames[self.controlled_player]))).unsqueeze(0)
        player_id = torch.tensor([self.current_player_id], dtype=torch.int64)
        with torch.no_grad():
            observation = self.preprocessor(image, player_id)[0].round().clamp(0, 255).to(torch.uint8)
        return observation.numpy()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        observations, _ = self.game.reset(seed=seed)
        right_player, left_player = self.game.possible_agents
        # Ohne feste Vorgabe werden beide Rollen trainiert. Das ist wichtig:
        # Nur dann sieht das Q-Netz sowohl Seitenwerte 0 als auch 1.
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

        # Der ONNX-Gegner erhält seine eigene Vier-Frame-History und seine
        # physische Seite (1 links/orange, 2 rechts/grün).
        opponent_action = self.opponent.act(np.stack(self.frames[self.opponent_player]), self.opponent_id)
        observations, rewards, terminations, truncations, _ = self.game.step(
            {self.controlled_player: int(action), self.opponent_player: opponent_action}
        )
        for player, image in observations.items():
            self.frames[player].append(image)
        self.steps += 1

        # Eine Episode entspricht einem Ballwechsel. Nach einem Punkt oder
        # nach dem Trainings-Timeout startet DQN eine neue Episode.
        point_scored = any(reward != 0 for reward in rewards.values())
        naturally_finished = any(terminations.values()) or any(truncations.values())
        truncated = not (point_scored or naturally_finished) and self.steps >= self.max_episode_steps
        terminated = point_scored or naturally_finished
        reward = float(rewards.get(self.controlled_player, 0.0))
        if terminated or truncated:
            observation = np.zeros(self.observation_space.shape, dtype=np.uint8)
        else:
            observation = self._observation()
        return observation, reward, terminated, truncated, {}

    def close(self) -> None:
        self.game.close()
