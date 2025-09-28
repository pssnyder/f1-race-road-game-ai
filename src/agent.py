"""
Clean DQN agent used by the trainer and environment under src/.

Exports:
- class DQNAgent
	- choose_action(state, training_mode=True)
	- store_experience(state, action, reward, next_state, done)
	- train_from_experience()
	- copy_to_target_network()
	- save_agent(path)
	- load_agent(path)
	- create_training_charts()
"""

from __future__ import annotations

from collections import deque
import random
from typing import Deque, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class DQN(nn.Module):
	def __init__(self, state_size: int, action_size: int, hidden_size: int = 128) -> None:
		super().__init__()
		self.net = nn.Sequential(
			nn.Linear(state_size, hidden_size),
			nn.ReLU(),
			nn.Linear(hidden_size, hidden_size),
			nn.ReLU(),
			nn.Linear(hidden_size, 64),
			nn.ReLU(),
			nn.Linear(64, action_size),
		)

	def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
		return self.net(x)


class ReplayBuffer:
	def __init__(self, capacity: int) -> None:
		self.buffer: Deque[Tuple[np.ndarray, int, float, np.ndarray, bool]] = deque(maxlen=capacity)

	def push(self, state, action, reward, next_state, done) -> None:
		self.buffer.append((state, action, reward, next_state, done))

	def sample(self, batch_size: int):
		batch = random.sample(self.buffer, batch_size)
		states, actions, rewards, next_states, dones = zip(*batch)
		states = torch.from_numpy(np.array(states, dtype=np.float32))
		actions = torch.from_numpy(np.array(actions, dtype=np.int64))
		rewards = torch.from_numpy(np.array(rewards, dtype=np.float32))
		next_states = torch.from_numpy(np.array(next_states, dtype=np.float32))
		dones = torch.from_numpy(np.array(dones, dtype=bool))
		return states, actions, rewards, next_states, dones

	def __len__(self) -> int:  # pragma: no cover - trivial
		return len(self.buffer)


class DQNAgent:
	def __init__(
		self,
		state_size: int,
		action_size: int,
		learning_rate: float = 0.001,
		gamma: float = 0.99,
		epsilon_start: float = 1.0,
		epsilon_end: float = 0.01,
		epsilon_decay: float = 0.995,
		memory_size: int = 10000,
		batch_size: int = 32,
	) -> None:
		self.state_size = state_size
		self.action_size = action_size
		self.LEARNING_RATE = learning_rate
		self.GAMMA = gamma
		self.batch_size = batch_size
		self.epsilon = epsilon_start
		self.EPSILON_END = epsilon_end
		self.EPSILON_DECAY = epsilon_decay

		self.main_network = DQN(state_size, action_size)
		self.target_network = DQN(state_size, action_size)
		self.optimizer = optim.Adam(self.main_network.parameters(), lr=learning_rate)

		self.memory = ReplayBuffer(memory_size)
		self.copy_to_target_network()

		# Tracking
		self.training_losses: list[float] = []
		self.episode_scores: list[int] = []
		self.exploration_rates: list[float] = []

	def copy_to_target_network(self) -> None:
		self.target_network.load_state_dict(self.main_network.state_dict())

	def choose_action(self, state: np.ndarray, training_mode: bool = True) -> int:
		if training_mode and random.random() < self.epsilon:
			return random.randrange(self.action_size)
		with torch.no_grad():
			state_tensor = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
			q_values = self.main_network(state_tensor)
			return int(q_values.argmax().item())

	def store_experience(self, state, action, reward, next_state, done) -> None:
		self.memory.push(state, action, reward, next_state, done)

	def train_from_experience(self) -> None:
		if len(self.memory) < self.batch_size:
			return
		states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
		current_q = self.main_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
		with torch.no_grad():
			next_q = self.target_network(next_states).max(1)[0]
			target_q = rewards + (self.GAMMA * next_q * (~dones))
		loss = nn.MSELoss()(current_q, target_q)
		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()
		self.training_losses.append(float(loss.item()))
		if self.epsilon > self.EPSILON_END:
			self.epsilon *= self.EPSILON_DECAY
		self.exploration_rates.append(self.epsilon)

	def save_agent(self, filepath: str) -> None:
		checkpoint = {
			"main_network_state": self.main_network.state_dict(),
			"target_network_state": self.target_network.state_dict(),
			"optimizer_state": self.optimizer.state_dict(),
			"current_epsilon": self.epsilon,
			"training_losses": self.training_losses,
			"episode_scores": self.episode_scores,
			"exploration_rates": self.exploration_rates,
		}
		torch.save(checkpoint, filepath)
		print(f"Saved agent to {filepath}")

	def load_agent(self, filepath: str) -> None:
		checkpoint = torch.load(filepath, weights_only=False)
		self.main_network.load_state_dict(checkpoint["main_network_state"])
		self.target_network.load_state_dict(checkpoint["target_network_state"])
		self.optimizer.load_state_dict(checkpoint["optimizer_state"])
		self.epsilon = float(checkpoint.get("current_epsilon", self.epsilon))
		self.training_losses = list(checkpoint.get("training_losses", []))
		self.episode_scores = list(checkpoint.get("episode_scores", []))
		self.exploration_rates = list(checkpoint.get("exploration_rates", []))
		print(f"Loaded agent from {filepath}")

	def create_training_charts(self, out_path: str = "results/charts/ai_training_progress.png") -> None:
		# Make sure directories exist; avoid deleting anything
		import os

		os.makedirs(os.path.dirname(out_path), exist_ok=True)
		fig, axes = plt.subplots(2, 2, figsize=(14, 9))

		# Scores
		axes[0, 0].plot(self.episode_scores, color="steelblue")
		axes[0, 0].set_title("Scores per Episode")
		axes[0, 0].set_xlabel("Episode")
		axes[0, 0].set_ylabel("Score")
		axes[0, 0].grid(True, alpha=0.3)

		# Moving average
		if len(self.episode_scores) >= 10:
			w = 10
			mov = [sum(self.episode_scores[i:i+w]) / w for i in range(0, len(self.episode_scores) - w + 1)]
			axes[0, 1].plot(mov, color="seagreen")
			axes[0, 1].set_title(f"{w}-Episode Moving Average")
		else:
			axes[0, 1].plot(self.episode_scores, color="seagreen")
			axes[0, 1].set_title("Scores (short run)")
		axes[0, 1].set_xlabel("Episode")
		axes[0, 1].set_ylabel("Avg Score")
		axes[0, 1].grid(True, alpha=0.3)

		# Loss
		if self.training_losses:
			axes[1, 0].plot(self.training_losses, color="firebrick")
			axes[1, 0].set_title("Training Loss")
			axes[1, 0].set_xlabel("Step")
			axes[1, 0].set_ylabel("MSE Loss")
			axes[1, 0].grid(True, alpha=0.3)
		else:
			axes[1, 0].text(0.5, 0.5, "No training yet", ha="center", va="center")

		# Epsilon
		if self.exploration_rates:
			axes[1, 1].plot(self.exploration_rates, color="orange")
			axes[1, 1].set_title("Exploration Rate (epsilon)")
			axes[1, 1].set_xlabel("Step")
			axes[1, 1].set_ylabel("Epsilon")
			axes[1, 1].grid(True, alpha=0.3)
		else:
			axes[1, 1].text(0.5, 0.5, "No exploration data", ha="center", va="center")

		plt.tight_layout()
		plt.savefig(out_path, dpi=200, bbox_inches="tight")
		plt.close(fig)

