# Technical Guide

This guide explains the internals of the educational F1 Race AI project.

- Environment: `src/environment.py` exposes a small Gym-like interface.
- Agent: `src/agent.py` implements a DQN with replay and target network.
- Trainer: `src/trainer.py` orchestrates training/evaluation and saves artifacts into `models/` and `results/`.

See `README.md` for a friendly introduction. This document focuses on architecture, data flow, and extension points.

## Data Flow
1. Trainer creates Environment and Agent.
2. Loop: choose_action -> env.step -> store_experience -> train_from_experience.
3. Periodically copy main network weights to target network.
4. Save checkpoints to `models/` and progress charts to `results/charts/`.

## Extending
- Swap reward shaping in `F1RaceEnvironment._reward`.
- Try alternative network widths in `DQN`.
- Add Double-DQN or prioritized replay in the Agent class.
