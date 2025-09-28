# F1 Race AI Training

This project trains a Deep Q-Network (DQN) AI agent to play the F1 Race Road Game using reinforcement learning.

## Project Structure

- `f1_race_env.py` - Game environment wrapper for AI training
- `dqn_agent.py` - Deep Q-Network implementation
- `train_ai.py` - Main training and testing script
- `images/` - Game asset images
- `audio/` - Game sound files  
- `textfile/` - High score storage

## Installation

1. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Train a new AI agent:
```bash
python train_ai.py
```
Then select 'train' when prompted.

### Test a trained agent:
```bash
python train_ai.py
```
Then select 'test' when prompted and choose a saved model.

### Run random baseline:
```bash
python train_ai.py
```
Then select 'baseline' when prompted.

## AI Details

### State Space (5 dimensions):
- Car X position (normalized)
- Obstacle X position (normalized)  
- Obstacle Y position (normalized)
- Game speed (normalized)
- Distance to obstacle (normalized)

### Action Space (3 actions):
- 0: No action
- 1: Move left
- 2: Move right

### Reward Structure:
- +0.1 for each frame survived
- +10 for each obstacle dodged
- -100 for crashing
- -0.01 for unnecessary movements

### Network Architecture:
```
Input (5) → Dense(128) → ReLU → Dense(128) → ReLU → Dense(64) → ReLU → Output(3)
```

## Training Parameters

- Learning Rate: 0.001
- Gamma (discount factor): 0.99
- Epsilon decay: 0.995 (1.0 → 0.01)
- Replay buffer size: 10,000
- Batch size: 32
- Target network update frequency: 100 episodes

## Files Generated During Training

- `dqn_model_episode_X.pth` - Model checkpoints
- `dqn_model_final.pth` - Final trained model
- `training_metrics.png` - Training progress plots