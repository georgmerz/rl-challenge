"""Technische Bausteine fuer ``train_dqn_vanilla.py``.

Dieses Modul ist kein weiterer Trainingsauftrag. Es kapselt nur den Adapter
zwischen PettingZoo (zwei Spieler), Stable-Baselines3 (ein Agent) und dem
oeffentlichen ONNX-Format der Challenge. Im Einstiegsskript steht dadurch der
eigentliche DQN-Ablauf ohne diese Details.
"""

from __future__ import annotations

import copy
import inspect
from collections import deque
from pathlib import Path

import gymnasium as gym
import numpy as np
import onnxruntime as ort
import torch
from gymnasium import spaces
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, NatureCNN
from torch import nn
from torch.nn import functional as F


# Gewoehnliche Atari-DQN-Startwerte, keine Regeln der Challenge.
LEARNING_RATE = 1e-4
REPLAY_BUFFER_SIZE = 100_000
RANDOM_START_STEPS = 80_000
FINAL_EXPLORATION_RATE = 0.01
FRAME_STACK = 4
MAX_TRAINING_STEPS = 1_000


def public_image(frames: np.ndarray) -> np.ndarray:
    """Wandelt vier HWC-Frames in das oeffentliche 12-Kanal-Format um."""
    frames = np.asarray(frames)
    image = frames.transpose(0, 3, 1, 2).reshape(12, 210, 160)
    return np.ascontiguousarray(image, dtype=np.float32) / 255.0


class Player1VsBaselineEnv(gym.Env):
    """Einspieler-Sicht: links/orange trainiert gegen die Tracking-Baseline."""

    metadata = {"render_modes": []}

    def __init__(self) -> None:
        from pettingzoo.atari import pong_v3

        self.env = pong_v3.parallel_env(render_mode=None, num_players=2)
        opponent_path = Path(__file__).resolve().parent / "baselines" / "baseline_pong_track_ball.onnx"
        if not opponent_path.is_file():
            raise FileNotFoundError("Bitte zuerst `python create_baselines.py` ausfuehren.")
        self.opponent = ort.InferenceSession(str(opponent_path), providers=["CPUExecutionProvider"])
        self.observation_space = spaces.Box(0, 255, shape=(210, 160, 3), dtype=np.uint8)
        self.action_space = spaces.Discrete(6)
        self.player_1: str | None = None
        self.player_2: str | None = None
        self.frame_history: dict[str, deque[np.ndarray]] | None = None
        self.episode_steps = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        observations, _ = self.env.reset(seed=seed)
        # pong_v3: first_0 ist rechts/gruen, second_0 links/orange.
        self.player_2, self.player_1 = self.env.possible_agents
        self.frame_history = {
            name: deque([observations[name].copy()] * FRAME_STACK, maxlen=FRAME_STACK)
            for name in self.env.possible_agents
        }
        self.episode_steps = 0
        return observations[self.player_1], {}

    def step(self, action: int):
        if self.player_1 is None or self.player_2 is None or self.frame_history is None:
            raise RuntimeError("reset() muss vor step() aufgerufen werden.")
        opponent_action = int(
            np.asarray(
                self.opponent.run(
                    ["action"],
                    {
                        "image": public_image(np.stack(self.frame_history[self.player_2]))[None],
                        "player_id": np.asarray([2], dtype=np.int64),
                    },
                )[0]
            ).reshape(-1)[0]
        )
        observations, rewards, terminations, truncations, _ = self.env.step(
            {self.player_1: int(action), self.player_2: opponent_action}
        )
        for name, observation in observations.items():
            self.frame_history[name].append(observation)
        self.episode_steps += 1

        point_scored = any(reward != 0 for reward in rewards.values())
        natural_end = bool(terminations.get(self.player_1, False)) or bool(truncations.get(self.player_1, False))
        timed_out = not (point_scored or natural_end) and self.episode_steps >= MAX_TRAINING_STEPS
        terminated = point_scored or natural_end
        truncated = timed_out
        reward = float(rewards.get(self.player_1, 0.0))
        if terminated or truncated:
            observation = np.zeros(self.observation_space.shape, dtype=np.uint8)
        else:
            observation = observations[self.player_1]
        return observation, reward, terminated, truncated, {}

    def close(self) -> None:
        self.env.close()


class PublicFrameStack(gym.ObservationWrapper):
    """Macht vier Rohframes genau als oeffentliches ONNX-Eingabebild sichtbar."""

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.observation_space = spaces.Box(0.0, 1.0, shape=(12, 210, 160), dtype=np.float32)

    def observation(self, frames: np.ndarray) -> np.ndarray:
        return public_image(frames)


class AtariPreprocessorCNN(BaseFeaturesExtractor):
    """Vorverarbeitung im Netz: deshalb gleich beim Training und in ONNX."""

    def __init__(
        self,
        observation_space: spaces.Box,
        features_dim: int = 512,
        normalized_image: bool = True,
    ) -> None:
        super().__init__(observation_space, features_dim)
        processed_space = spaces.Box(0.0, 1.0, shape=(FRAME_STACK, 84, 84), dtype=np.float32)
        self.encoder = NatureCNN(processed_space, features_dim=features_dim, normalized_image=normalized_image)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        batch = image.shape[0]
        frames = image.reshape(batch, FRAME_STACK, 3, 210, 160)[:, :, :, 34:194, :]
        grayscale = 0.299 * frames[:, :, 0] + 0.587 * frames[:, :, 1] + 0.114 * frames[:, :, 2]
        resized = F.interpolate(grayscale, size=(84, 84), mode="bilinear", align_corners=False)
        return self.encoder(resized)


class ExportedPolicy(nn.Module):
    """Legt das Q-Netz auf die feste ONNX-Schnittstelle der Challenge."""

    def __init__(self, q_net: nn.Module) -> None:
        super().__init__()
        self.q_net = copy.deepcopy(q_net).cpu()

    def forward(self, image: torch.Tensor, player_id: torch.Tensor) -> torch.Tensor:
        action = torch.argmax(self.q_net(image), dim=1).to(torch.int64)
        # player_id muss im Exportgraphen bleiben, kann hier aber ungenutzt sein.
        return action + player_id * 0


def make_env() -> gym.Env:
    """Erzeugt die einspielerige Trainingsumgebung im oeffentlichen Format."""
    env = Player1VsBaselineEnv()
    env = gym.wrappers.FrameStackObservation(env, FRAME_STACK)
    return PublicFrameStack(env)


def make_dqn(env: gym.Env, device: str, exploration_fraction: float, resume: Path | None = None) -> DQN:
    """Erzeugt das Vanilla-DQN oder setzt ein vorhandenes SB3-Checkpoint fort."""
    if resume:
        return DQN.load(resume, env=env, device=device)
    return DQN(
        "CnnPolicy",
        env,
        learning_rate=LEARNING_RATE,
        buffer_size=REPLAY_BUFFER_SIZE,
        optimize_memory_usage=True,
        replay_buffer_kwargs={"handle_timeout_termination": False},
        learning_starts=RANDOM_START_STEPS,
        batch_size=32,
        train_freq=4,
        target_update_interval=1_000,
        exploration_fraction=exploration_fraction,
        exploration_final_eps=FINAL_EXPLORATION_RATE,
        policy_kwargs={
            "features_extractor_class": AtariPreprocessorCNN,
            "normalize_images": False,
        },
        verbose=1,
        seed=42,
        device=device,
    )


def export_policy(model: DQN, output: Path) -> None:
    """Exportiert die Policy in das ONNX-Format des Turnier-Runners."""
    output.parent.mkdir(parents=True, exist_ok=True)
    policy = ExportedPolicy(model.policy.q_net).eval()
    image = torch.zeros((1, 12, 210, 160), dtype=torch.float32)
    player_id = torch.ones((1,), dtype=torch.int64)
    export_kwargs = {
        "input_names": ["image", "player_id"],
        "output_names": ["action"],
        "opset_version": 17,
    }
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        export_kwargs["dynamo"] = False
    torch.onnx.export(policy, (image, player_id), str(output), **export_kwargs)


class OnnxCheckpointCallback(BaseCallback):
    """Sichert SB3- und ONNX-Checkpoints bei langen Laeufen."""

    def __init__(self, output: Path, every_steps: int) -> None:
        super().__init__()
        self.output = output
        self.every_steps = every_steps
        self.next_checkpoint = every_steps

    def _on_step(self) -> bool:
        if self.num_timesteps >= self.next_checkpoint:
            checkpoint = self.output.with_name(f"{self.output.stem}_step_{self.num_timesteps}")
            self.model.save(checkpoint)
            export_policy(self.model, checkpoint.with_suffix(".onnx"))
            print(f"Checkpoint nach {self.num_timesteps} Schritten: {checkpoint}.zip und .onnx")
            self.next_checkpoint += self.every_steps
