"""
🔄 Backward Compatible Agent System
===================================

This allows you to keep using your 20K trained model while gradually
transitioning to the enhanced 7-feature system.

Author: Pat Snyder 💻
"""

import torch
import torch.nn as nn
import numpy as np
from agent import DQNAgent, DQN

class BackwardCompatibleDQN(nn.Module):
    """
    🧠 Neural Network that can handle both old (5) and new (7) input formats
    
    This is a production-grade solution for model evolution!
    """
    
    def __init__(self, state_size, action_size, hidden_size=128):
        super(BackwardCompatibleDQN, self).__init__()
        
        self.state_size = state_size
        
        if state_size == 5:
            # Original 5-input architecture
            self.net = nn.Sequential(
                nn.Linear(5, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, 64),
                nn.ReLU(),
                nn.Linear(64, action_size)
            )
        elif state_size == 7:
            # Enhanced 7-input architecture
            self.net = nn.Sequential(
                nn.Linear(7, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, 64),
                nn.ReLU(),
                nn.Linear(64, action_size)
            )
        else:
            raise ValueError(f"Unsupported state size: {state_size}")
    
    def forward(self, x):
        return self.net(x)

class EnsembleAgent:
    """
    🎭 Agent that combines old and new models for best of both worlds
    
    This is how Netflix, Google, etc. handle model transitions!
    """
    
    def __init__(self, old_model_path=None, state_size=7, action_size=3):
        self.state_size = state_size
        self.action_size = action_size
        
        # Load old model if available
        self.old_model = None
        if old_model_path:
            try:
                self.old_model = DQNAgent(state_size=5, action_size=action_size)
                self.old_model.load_agent(old_model_path)
                print(f"✅ Loaded old model from {old_model_path}")
            except Exception as e:
                print(f"⚠️ Could not load old model: {e}")
        
        # Create new model for enhanced features
        self.new_model = DQNAgent(state_size=7, action_size=action_size)
        
        # Mixing weights (start favoring old model, gradually shift to new)
        self.old_weight = 0.7  # Start with 70% old model
        self.new_weight = 0.3  # Start with 30% new model
        
        self.episode_count = 0
        self.transition_episodes = 1000  # Gradually shift over 1000 episodes
    
    def choose_action(self, state, training_mode=True):
        """Choose action using ensemble of old and new models"""
        
        if self.old_model is None:
            # Only new model available
            return self.new_model.choose_action(state, training_mode)
        
        # Get predictions from both models
        old_state = state[:5]  # Use first 5 features for old model
        old_q_values = self.old_model.main_network(torch.FloatTensor(old_state).unsqueeze(0))
        new_q_values = self.new_model.main_network(torch.FloatTensor(state).unsqueeze(0))
        
        # Combine predictions with current weights
        combined_q_values = (self.old_weight * old_q_values + 
                           self.new_weight * new_q_values)
        
        # Choose action based on combined Q-values
        if training_mode and np.random.random() < self.new_model.epsilon:
            return np.random.choice(self.action_size)
        else:
            return torch.argmax(combined_q_values).item()
    
    def train_new_model(self, state, action, reward, next_state, done):
        """Train only the new model with new experiences"""
        self.new_model.store_experience(state, action, reward, next_state, done)
        
        if len(self.new_model.memory) > self.new_model.batch_size:
            self.new_model.train_from_experience()
    
    def update_weights(self):
        """Gradually shift from old model to new model"""
        self.episode_count += 1
        
        if self.episode_count < self.transition_episodes:
            # Gradually decrease old model weight, increase new model weight
            progress = self.episode_count / self.transition_episodes
            self.old_weight = 0.7 * (1 - progress)  # 0.7 → 0.0
            self.new_weight = 0.3 + 0.7 * progress  # 0.3 → 1.0
        else:
            # Full transition to new model
            self.old_weight = 0.0
            self.new_weight = 1.0
    
    def get_model_status(self):
        """Get current ensemble status"""
        return {
            'episode': self.episode_count,
            'old_weight': self.old_weight,
            'new_weight': self.new_weight,
            'transition_complete': self.old_weight == 0.0
        }

class FeatureAdapter:
    """
    🔧 Converts between old (5) and new (7) feature formats
    
    This handles the state space transformation!
    """
    
    @staticmethod
    def expand_state(old_state):
        """Convert 5-feature state to 7-feature state with defaults"""
        if len(old_state) == 7:
            return old_state  # Already new format
        elif len(old_state) == 5:
            # Add default values for new features
            future_obstacle_pos = old_state[1]  # Same as current obstacle position
            threat_urgency = 0.5  # Default medium urgency
            return np.append(old_state, [future_obstacle_pos, threat_urgency])
        else:
            raise ValueError(f"Unsupported state size: {len(old_state)}")
    
    @staticmethod
    def contract_state(new_state):
        """Convert 7-feature state to 5-feature state by dropping new features"""
        if len(new_state) == 5:
            return new_state  # Already old format
        elif len(new_state) == 7:
            return new_state[:5]  # Drop last 2 features
        else:
            raise ValueError(f"Unsupported state size: {len(new_state)}")

# 🧪 EXAMPLE USAGE
if __name__ == "__main__":
    print("🔄 Testing Backward Compatible System")
    print("=" * 50)
    
    # Simulate old and new states
    old_state = np.array([0.5, 0.3, 0.8, 0.4, 0.6])  # 5 features
    new_state = np.array([0.5, 0.3, 0.8, 0.4, 0.6, 0.3, 0.9])  # 7 features
    
    # Test feature adapter
    adapter = FeatureAdapter()
    
    expanded = adapter.expand_state(old_state)
    print(f"📈 Old state expanded: {old_state} → {expanded}")
    
    contracted = adapter.contract_state(new_state)
    print(f"📉 New state contracted: {new_state} → {contracted}")
    
    print("\n✅ Backward compatibility system ready!")
    print("💡 You can now use your 20K model with the enhanced environment!")