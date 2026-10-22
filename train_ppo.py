"""Train a small PPO self-play baseline for the Atari Surround challenge.

The resulting ``.onnx`` file implements the public challenge interface:
    image:     float32 [1, 3, 210, 160] in [0, 1]
    player_id: int64 [1]
    action:    int64 tensor containing one action in {0, ..., 4}

Example:
    python train_ppo.py --timesteps 100000 --output baselines/baseline_ppo.onnx

This is deliberately a compact educational baseline, not a tuned training
pipeline. It uses one shared policy for both players and trains it in self-play.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Categorical


ACTION_COUNT = 5
OBSERVATION_SHAPE = (3, 210, 160)


class ActorCritic(nn.Module):
    """A small CNN producing an action distribution and a state value."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
        )
        self.policy_head = nn.Linear(32 * 7 * 7, ACTION_COUNT)
        self.value_head = nn.Linear(32 * 7 * 7, 1)

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Reducing the Atari image makes this baseline practical on a CPU.
        image = F.interpolate(image, size=(84, 84), mode="bilinear", align_corners=False)
        features = self.encoder(image)
        return self.policy_head(features), self.value_head(features).squeeze(-1)


class ExportedPolicy(nn.Module):
    """Expose ActorCritic through the challenge's required model signature."""

    def __init__(self, actor_critic: ActorCritic) -> None:
        super().__init__()
        self.actor_critic = actor_critic

    def forward(self, image: torch.Tensor, player_id: torch.Tensor) -> torch.Tensor:
        # The RGB image contains both players. player_id remains an accepted part
        # of the public API even though this shared policy does not need it.
        logits, _ = self.actor_critic(image)
        # Keep player_id as a graph input for the fixed ONNX challenge API.
        return torch.argmax(logits, dim=1).to(torch.int64) + player_id * 0


def observation_batch(observations: dict[str, np.ndarray], agent_names: list[str], device: torch.device) -> torch.Tensor:
    """Convert PettingZoo uint8 HWC observations to normalized NCHW tensors."""
    images = np.stack([observations[name] for name in agent_names])
    return torch.from_numpy(images).permute(0, 3, 1, 2).contiguous().to(device=device, dtype=torch.float32) / 255.0


def export_policy(model: ActorCritic, output: Path) -> None:
    """Export a deterministic (argmax) challenge policy."""
    output.parent.mkdir(parents=True, exist_ok=True)
    policy = ExportedPolicy(model.cpu()).eval()
    example_image = torch.zeros((1, *OBSERVATION_SHAPE), dtype=torch.float32)
    example_player_id = torch.ones((1,), dtype=torch.int64)
    torch.onnx.export(
        policy,
        (example_image, example_player_id),
        str(output),
        input_names=["image", "player_id"],
        output_names=["action"],
        opset_version=17,
    )


def train(args: argparse.Namespace) -> Path:
    from pettingzoo.atari import surround_v2

    if args.timesteps <= 0 or args.rollout_steps <= 0:
        raise ValueError("--timesteps und --rollout-steps müssen positiv sein.")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    model = ActorCritic().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, eps=1e-5)

    env = surround_v2.parallel_env(render_mode=None)
    observations, _ = env.reset(seed=args.seed)
    agent_names = list(env.possible_agents)
    if len(agent_names) != 2:
        raise RuntimeError(f"Surround erwartet zwei Agents, erhalten: {agent_names}")

    env_steps = 0
    episode = 1
    score = np.zeros(2, dtype=np.float32)
    try:
        while env_steps < args.timesteps:
            steps = min(args.rollout_steps, args.timesteps - env_steps)
            observations_buffer = torch.empty((steps, 2, *OBSERVATION_SHAPE), dtype=torch.uint8)
            actions_buffer = torch.empty((steps, 2), dtype=torch.int64)
            log_probs_buffer = torch.empty((steps, 2), dtype=torch.float32)
            rewards_buffer = torch.empty((steps, 2), dtype=torch.float32)
            dones_buffer = torch.empty((steps, 2), dtype=torch.float32)
            values_buffer = torch.empty((steps, 2), dtype=torch.float32)

            for step in range(steps):
                obs = observation_batch(observations, agent_names, device)
                with torch.no_grad():
                    logits, values = model(obs)
                    distribution = Categorical(logits=logits)
                    actions = distribution.sample()

                actions_by_agent = {name: int(action) for name, action in zip(agent_names, actions.cpu().tolist())}
                next_observations, rewards, terminations, truncations, _ = env.step(actions_by_agent)
                dones = np.asarray(
                    [terminations.get(name, False) or truncations.get(name, False) for name in agent_names],
                    dtype=np.float32,
                )
                rewards_array = np.asarray([rewards.get(name, 0.0) for name in agent_names], dtype=np.float32)

                observations_buffer[step] = (obs * 255.0).to(torch.uint8).cpu()
                actions_buffer[step] = actions.cpu()
                log_probs_buffer[step] = distribution.log_prob(actions).cpu()
                rewards_buffer[step] = torch.from_numpy(rewards_array)
                dones_buffer[step] = torch.from_numpy(dones)
                values_buffer[step] = values.cpu()
                score += rewards_array
                env_steps += 1

                if bool(dones.all()):
                    if episode % args.log_every == 0:
                        print(f"Episode {episode:5d} | Schritte {env_steps:7d} | Rückgaben: {score.tolist()}")
                    episode += 1
                    score.fill(0.0)
                    observations, _ = env.reset(seed=args.seed + episode)
                else:
                    observations = next_observations

            with torch.no_grad():
                next_obs = observation_batch(observations, agent_names, device)
                _, next_values = model(next_obs)

            advantages = torch.zeros_like(rewards_buffer)
            last_advantage = torch.zeros(2)
            for step in reversed(range(steps)):
                if step == steps - 1:
                    following_values = next_values.cpu()
                else:
                    following_values = values_buffer[step + 1]
                non_terminal = 1.0 - dones_buffer[step]
                delta = rewards_buffer[step] + args.gamma * following_values * non_terminal - values_buffer[step]
                last_advantage = delta + args.gamma * args.gae_lambda * non_terminal * last_advantage
                advantages[step] = last_advantage
            returns = advantages + values_buffer

            flat_observations = observations_buffer.flatten(0, 1)
            flat_actions = actions_buffer.flatten()
            flat_log_probs = log_probs_buffer.flatten()
            flat_advantages = advantages.flatten()
            flat_returns = returns.flatten()
            sample_count = flat_actions.numel()
            flat_advantages = (flat_advantages - flat_advantages.mean()) / (flat_advantages.std() + 1e-8)

            model.train()
            for _ in range(args.update_epochs):
                indices = torch.randperm(sample_count)
                for start in range(0, sample_count, args.minibatch_size):
                    batch = indices[start : start + args.minibatch_size]
                    batch_obs = flat_observations[batch].to(device=device, dtype=torch.float32) / 255.0
                    batch_actions = flat_actions[batch].to(device)
                    logits, values = model(batch_obs)
                    distribution = Categorical(logits=logits)
                    log_ratio = distribution.log_prob(batch_actions) - flat_log_probs[batch].to(device)
                    ratio = log_ratio.exp()
                    batch_advantages = flat_advantages[batch].to(device)
                    unclipped = ratio * batch_advantages
                    clipped = torch.clamp(ratio, 1.0 - args.clip_coef, 1.0 + args.clip_coef) * batch_advantages
                    policy_loss = -torch.minimum(unclipped, clipped).mean()
                    value_loss = F.mse_loss(values, flat_returns[batch].to(device))
                    entropy = distribution.entropy().mean()
                    loss = policy_loss + args.value_coef * value_loss - args.entropy_coef * entropy

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()

        output = Path(args.output)
        export_policy(model, output)
        print(f"PPO-Baseline exportiert nach {output}")
        return output
    finally:
        env.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a PPO self-play policy for Atari Surround.")
    parser.add_argument("--timesteps", type=int, default=100_000, help="Anzahl der Environment-Schritte (Standard: 100000).")
    parser.add_argument("--output", default="baselines/baseline_ppo.onnx", help="Zielpfad für das exportierte ONNX-Modell.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu", help="z. B. cpu, mps oder cuda")
    parser.add_argument("--rollout-steps", type=int, default=128)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=2.5e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--log-every", type=int, default=10)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
