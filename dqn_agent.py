import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt

class DQN(nn.Module):
    """Deep Q-Network for F1 Race Game"""
    
    def __init__(self, state_size, action_size, hidden_size=128):
        super(DQN, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, action_size)
        )
    
    def forward(self, x):
        return self.network(x)

class ReplayBuffer:
    """Experience replay buffer for DQN"""
    
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to numpy arrays first to avoid the tensor creation warning
        states_np = np.array(states, dtype=np.float32)
        next_states_np = np.array(next_states, dtype=np.float32)
        actions_np = np.array(actions, dtype=np.int64)
        rewards_np = np.array(rewards, dtype=np.float32)
        dones_np = np.array(dones, dtype=bool)
        
        return (
            torch.from_numpy(states_np),
            torch.from_numpy(actions_np),
            torch.from_numpy(rewards_np),
            torch.from_numpy(next_states_np),
            torch.from_numpy(dones_np)
        )
    
    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    """DQN Agent for F1 Race Game"""
    
    def __init__(self, state_size, action_size, lr=0.001, gamma=0.99, 
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995, 
                 memory_size=10000, batch_size=32):
        
        self.state_size = state_size
        self.action_size = action_size
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        
        # Neural networks
        self.q_network = DQN(state_size, action_size)
        self.target_network = DQN(state_size, action_size)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        # Replay buffer
        self.memory = ReplayBuffer(memory_size)
        
        # Update target network
        self.update_target_network()
        
        # Training metrics
        self.losses = []
        self.scores = []
        self.epsilons = []
    
    def update_target_network(self):
        """Copy weights from main network to target network"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def act(self, state, training=True):
        """Choose action using epsilon-greedy policy"""
        if training and random.random() < self.epsilon:
            return random.choice(range(self.action_size))
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.q_network(state_tensor)
            return q_values.argmax().item()
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.memory.push(state, action, reward, next_state, done)
    
    def replay(self):
        """Train the network on a batch of experiences"""
        if len(self.memory) < self.batch_size:
            return
        
        # Sample batch from memory
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Next Q values from target network
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # Compute loss
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.losses.append(loss.item())
        
        # Decay epsilon
        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay
            
        self.epsilons.append(self.epsilon)
    
    def save(self, filepath):
        """Save the trained model"""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'losses': self.losses,
            'scores': self.scores,
            'epsilons': self.epsilons
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath):
        """Load a trained model"""
        checkpoint = torch.load(filepath)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.losses = checkpoint['losses']
        self.scores = checkpoint['scores']
        self.epsilons = checkpoint['epsilons']
        print(f"Model loaded from {filepath}")
    
    def plot_training_metrics(self):
        """Plot training metrics"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Scores over episodes
        axes[0, 0].plot(self.scores)
        axes[0, 0].set_title('Scores over Episodes')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].grid(True)
        
        # Moving average of scores (or regular scores if not enough data)
        if len(self.scores) >= 100:
            moving_avg = np.convolve(self.scores, np.ones(100)/100, mode='valid')
            axes[0, 1].plot(moving_avg)
            axes[0, 1].set_title('Moving Average Score (100 episodes)')
            axes[0, 1].set_xlabel('Episode')
            axes[0, 1].set_ylabel('Average Score')
        else:
            # Show regular scores with trend line if we don't have enough for moving average
            axes[0, 1].plot(self.scores, 'b-', alpha=0.6, label='Scores')
            if len(self.scores) > 1:
                z = np.polyfit(range(len(self.scores)), self.scores, 1)
                p = np.poly1d(z)
                axes[0, 1].plot(range(len(self.scores)), p(range(len(self.scores))), 'r--', alpha=0.8, label='Trend')
                axes[0, 1].legend()
            axes[0, 1].set_title(f'Scores with Trend ({len(self.scores)} episodes)')
            axes[0, 1].set_xlabel('Episode')
            axes[0, 1].set_ylabel('Score')
        axes[0, 1].grid(True)
        
        # Loss over training steps
        if self.losses:
            axes[1, 0].plot(self.losses)
            axes[1, 0].set_title('Loss over Training Steps')
            axes[1, 0].set_xlabel('Training Step')
            axes[1, 0].set_ylabel('Loss')
            axes[1, 0].grid(True)
        else:
            axes[1, 0].text(0.5, 0.5, 'No Loss Data\n(Training started)', 
                           ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('Loss over Training Steps')
        
        # Epsilon decay
        if self.epsilons:
            axes[1, 1].plot(self.epsilons)
            axes[1, 1].set_title('Epsilon Decay')
            axes[1, 1].set_xlabel('Training Step')
            axes[1, 1].set_ylabel('Epsilon')
            axes[1, 1].grid(True)
        else:
            axes[1, 1].text(0.5, 0.5, 'No Epsilon Data', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Epsilon Decay')
        
        plt.tight_layout()
        plt.savefig('training_metrics.png', dpi=300, bbox_inches='tight')
        print("Training metrics saved to 'training_metrics.png'")
        
        # Don't show the plot interactively to avoid blocking
        plt.close()