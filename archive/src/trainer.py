"""
Training orchestrator for the F1 Race AI project (src/ version).

Keeps a small, readable API for educational use while delegating
to src.environment.F1RaceEnvironment and src.agent.DQNAgent.
"""

from __future__ import annotations

import os
import time
from typing import List

from .environment import F1RaceEnvironment
from .agent import DQNAgent


def train(
	episodes: int = 2000,
	target_update_frequency: int = 100,
	save_frequency: int = 500,
	show_training: bool = False,
	models_dir: str = "models",
) -> DQNAgent:
	os.makedirs(models_dir, exist_ok=True)
	os.makedirs("results/logs", exist_ok=True)
	os.makedirs("results/charts", exist_ok=True)

	env = F1RaceEnvironment(render=show_training)
	agent = DQNAgent(
		state_size=env.state_space_size,
		action_size=env.action_space_size,
		learning_rate=0.001,
		gamma=0.99,
		epsilon_start=1.0,
		epsilon_end=0.01,
		epsilon_decay=0.995,
		memory_size=10000,
		batch_size=32,
	)

	best = 0
	scores: List[int] = []
	lengths: List[int] = []
	start = time.time()

	for ep in range(episodes):
		state = env.reset()
		total_reward = 0.0
		steps = 0
		while True:
			action = agent.choose_action(state, training_mode=True)
			next_state, reward, done, info = env.step(action)
			agent.store_experience(state, action, reward, next_state, done)
			state = next_state
			total_reward += reward
			steps += 1
			if len(agent.memory) > agent.batch_size:
				agent.train_from_experience()
			if show_training:
				env.render()
				time.sleep(0.01)
			if done:
				break

		score = int(info.get("score", 0))
		scores.append(score)
		lengths.append(steps)
		agent.episode_scores.append(score)

		if ep % target_update_frequency == 0:
			agent.copy_to_target_network()

		if ep % 100 == 0:
			avg_score = sum(scores[-100:]) / max(1, min(100, len(scores)))
			avg_len = sum(lengths[-100:]) / max(1, min(100, len(lengths)))
			print(
				f"Episode {ep:4d} | Score {score:3d} | AvgScore {avg_score:5.2f} | Steps {steps:4d} | Eps {agent.epsilon:0.3f}"
			)
		if score > best:
			best = score
			agent.save_agent(os.path.join(models_dir, f"best_ep_{ep}_score_{best}.pth"))

		if ep > 0 and ep % save_frequency == 0:
			agent.save_agent(os.path.join(models_dir, f"checkpoint_ep_{ep}.pth"))

	elapsed = time.time() - start
	print(f"Training finished in {elapsed/60:.1f} minutes. Best score {best}.")

	final_model_path = os.path.join(models_dir, "f1_race_ai_final_model.pth")
	agent.save_agent(final_model_path)
	agent.create_training_charts("results/charts/ai_training_progress.png")
	env.close()
	return agent


def evaluate(model_path: str, episodes: int = 5, show_games: bool = True) -> list[int]:
	env = F1RaceEnvironment(render=show_games)
	agent = DQNAgent(env.state_space_size, env.action_space_size)
	agent.load_agent(model_path)
	agent.epsilon = 0.0
	scores: List[int] = []
	for ep in range(episodes):
		state = env.reset()
		while True:
			action = agent.choose_action(state, training_mode=False)
			next_state, _, done, info = env.step(action)
			state = next_state
			if show_games:
				env.render()
				time.sleep(0.03)
			if done:
				scores.append(int(info.get("score", 0)))
				break
	env.close()
	return scores


if __name__ == "__main__":  # Small CLI for convenience
	# Minimal interactive entry point mirroring train_ai.py but using src modules
	print("F1 Race AI (src) - choose: train/test")
	choice = input("Enter 'train' or 'test': ").strip().lower()
	if choice == "train":
		show = input("Show training? (y/N): ").strip().lower() == "y"
		eps = input("Episodes (default 2000): ").strip()
		episodes = int(eps) if eps else 2000
		train(episodes=episodes, show_training=show)
	elif choice == "test":
		model = input("Model path (default models/f1_race_ai_final_model.pth): ").strip()
		model = model or "models/f1_race_ai_final_model.pth"
		evaluate(model)
	else:
		print("No action selected.")

