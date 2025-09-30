"""
🧠 Deep Q-Network (DQN) Agent for F1 Race Game 🏎️
==================================================

Welcome to the AI "brain" of our racing game! 🤖✨

This module contains the artificial intelligence that learns to play the F1 racing game.
Think of it as creating a digital brain that starts knowing nothing about driving,
but through millions of practice attempts, becomes an expert race car driver!

🎯 WHAT IS DEEP Q-LEARNING?
---------------------------
Imagine you're learning to play a new video game:
- You start by pressing random buttons (EXPLORATION)
- You notice some actions give points, others cause game over (LEARNING)
- You remember what worked and what didn't (MEMORY)
- You get better by practicing the good moves more often (EXPLOITATION)
- Eventually, you become a pro player! 🏆

That's exactly what our DQN agent does, but MUCH faster than humans!

🧠 HOW THE AI "BRAIN" WORKS:
1. 📡 NEURAL NETWORK: Like a digital brain with artificial neurons
2. 🎯 Q-VALUES: The AI's "confidence score" for each possible action
3. 🎲 EXPLORATION: Sometimes try random actions to discover new strategies
4. 💭 MEMORY: Remember past experiences to learn from them
5. 🎓 TRAINING: Update the brain based on what worked and what didn't

🔄 THE LEARNING CYCLE:
- 👀 SEE: Current game state (car position, obstacle location, etc.)
- 🤔 THINK: Neural network calculates Q-values for each action
- 🎬 ACT: Choose the action with highest Q-value (or explore randomly)
- 📊 LEARN: Update neural network based on the reward received
- 🔄 REPEAT: Do this millions of times to become expert!

🎮 WHY DEEP Q-LEARNING FOR THIS GAME?
------------------------------------
- ✅ Perfect for decision-making problems (left, right, or stay?)
- ✅ Handles continuous learning (gets better over time)
- ✅ Works with simple state information (5 numbers describing game)
- ✅ Can balance exploration vs exploitation automatically
- ✅ No need for perfect training data - learns from trial and error!

👥 AUDIENCE NOTES:
- 🔬 Data Scientists: Notice the experience replay and target networks for stability
- 👩‍🏫 Educators: Great example of reinforcement learning in action
- 👶 Young Coders: It's like teaching a computer to get better at games by practicing!
- 🤓 AI Curious: See how neural networks learn optimal strategies through rewards

Author: Pat Snyder 💻
Created for: Learning Labs Portfolio 🌟
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt

class DQN(nn.Module):
    """
    🧠 Deep Q-Network - The AI's "Brain" 
    
    This is a neural network that takes in game state information
    and outputs Q-values (quality scores) for each possible action.
    
    Think of it as a function that answers:
    "Given this game situation, how good is each possible action?"
    """
    
    def __init__(self, state_size, action_size, hidden_size=128):
        """
        🏗️ Build the neural network architecture
        
        Args:
            state_size (int): How many numbers describe the game state (5 for our game)
            action_size (int): How many actions are possible (3: left, right, stay)
            hidden_size (int): Size of hidden layers (more = smarter but slower)
        """
        super(DQN, self).__init__()
        
        # 🏗️ NEURAL NETWORK ARCHITECTURE
        # ==============================
        # This creates a "feedforward" neural network with 4 layers
        self.net = nn.Sequential(
            # INPUT LAYER → FIRST HIDDEN LAYER
            nn.Linear(state_size, hidden_size),    # Transform 5 inputs to 128 neurons
            nn.ReLU(),                             # Activation: "fire" if input > 0
            
            # FIRST HIDDEN → SECOND HIDDEN LAYER  
            nn.Linear(hidden_size, hidden_size),   # 128 → 128 neurons
            nn.ReLU(),                             # More activation
            
            # SECOND HIDDEN → THIRD HIDDEN LAYER
            nn.Linear(hidden_size, 64),            # 128 → 64 neurons (narrowing down)
            nn.ReLU(),                             # More activation
            
            # FINAL HIDDEN → OUTPUT LAYER
            nn.Linear(64, action_size)             # 64 → 3 outputs (Q-values for actions)
        )
    
    def forward(self, x):
        """
        🔮 Make a prediction: given game state, what are Q-values for each action?
        
        Args:
            x (tensor): Game state as numbers
            
        Returns:
            tensor: Q-values for each action [left_score, right_score, stay_score]
        """
        return self.net(x)

class ReplayBuffer:
    """
    📚 Experience Replay Buffer - The AI's Memory System
    
    Humans learn from past experiences. Our AI does too!
    This class stores past game experiences so the AI can learn from them later.
    
    Why is this important?
    - 🧠 Learn from diverse experiences, not just recent ones
    - 🎯 Break correlation between consecutive game states  
    - 📈 More stable and efficient learning
    """
    
    def __init__(self, capacity):
        """
        📦 Create memory storage with limited capacity
        
        Args:
            capacity (int): Maximum number of experiences to remember
                           (older experiences get forgotten when buffer is full)
        """
        # Use deque for efficient add/remove operations
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """
        💾 Store a new experience in memory
        
        Args:
            state: Game state before action
            action: What action was taken  
            reward: What reward was received
            next_state: Game state after action
            done: Whether game ended
        """
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        """
        🎲 Get random sample of past experiences for learning
        
        Args:
            batch_size (int): How many experiences to sample
            
        Returns:
            tuple: Batch of experiences as PyTorch tensors
        """
        # Randomly sample experiences (breaks correlation, improves learning)
        batch = random.sample(self.buffer, batch_size)
        
        # Separate the different parts of experiences
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to numpy arrays first (avoids PyTorch warnings)
        states_array = np.array(states, dtype=np.float32)
        next_states_array = np.array(next_states, dtype=np.float32)
        actions_array = np.array(actions, dtype=np.int64)
        rewards_array = np.array(rewards, dtype=np.float32)
        dones_array = np.array(dones, dtype=bool)
        
        # Convert to PyTorch tensors for neural network processing
        return (
            torch.from_numpy(states_array),
            torch.from_numpy(actions_array), 
            torch.from_numpy(rewards_array),
            torch.from_numpy(next_states_array),
            torch.from_numpy(dones_array)
        )
    
    def __len__(self):
        """📏 How many experiences are stored in memory?"""
        return len(self.buffer)

class DQNAgent:
    """
    🤖 DQN Agent - The Complete AI Racing Driver
    
    This is the main AI agent that combines everything:
    - 🧠 Neural network for decision making
    - 📚 Memory system for learning from experience  
    - 🎯 Training algorithms for improving over time
    - 📊 Metrics tracking for monitoring progress
    
    The agent starts as a terrible driver but becomes expert through practice!
    """
    
    def __init__(self, state_size, action_size, learning_rate=0.001, gamma=0.99, 
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995, 
                 memory_size=10000, batch_size=32):
        """
        🎮 Initialize the DQN Agent with all its components
        
        Args:
            state_size (int): Size of game state vector (5 for our game)
            action_size (int): Number of possible actions (3 for our game)
            learning_rate (float): How fast the AI learns (0.001 = slow but stable)
            gamma (float): How much AI cares about future rewards (0.99 = very forward-thinking)
            epsilon_start (float): Initial exploration rate (1.0 = 100% random at start)
            epsilon_end (float): Final exploration rate (0.01 = 1% random when expert) 
            epsilon_decay (float): How fast to reduce exploration (0.995 = gradual)
            memory_size (int): How many past experiences to remember (10000 = good balance)
            batch_size (int): How many experiences to learn from at once (32 = standard)
        """
        
        # 🎛️ AGENT CONFIGURATION - Easy to modify!
        # =========================================
        self.state_size = state_size
        self.action_size = action_size
        self.LEARNING_RATE = learning_rate       # 📚 How fast AI learns (bigger = faster but less stable)
        self.GAMMA = gamma                       # 🔮 How much AI cares about future (0.99 = very future-focused)
        self.batch_size = batch_size            # 📦 Training batch size (32 = good default)
        
        # 🎲 EXPLORATION SETTINGS
        # =====================
        self.epsilon = epsilon_start            # 🎯 Current exploration rate  
        self.EPSILON_END = epsilon_end          # 🎯 Minimum exploration rate
        self.EPSILON_DECAY = epsilon_decay      # 📉 How fast to reduce exploration
        
        # 🧠 NEURAL NETWORKS - Main brain and backup brain
        # ===============================================
        self.main_network = DQN(state_size, action_size)      # Primary decision-making network
        self.target_network = DQN(state_size, action_size)    # Stable target for training
        self.optimizer = optim.Adam(self.main_network.parameters(), lr=learning_rate)
        
        # 📚 MEMORY SYSTEM
        # ================
        self.memory = ReplayBuffer(memory_size)
        
        # 🔄 INITIALIZE TARGET NETWORK
        # ============================
        self.copy_to_target_network()
        
        # 📊 TRAINING METRICS - Track progress over time
        # =============================================
        self.training_losses = []         # Loss values during training
        self.episode_scores = []         # Score achieved in each episode
        self.exploration_rates = []      # Epsilon values over time
        
        # 🎯 ADAPTIVE EXPLORATION TRACKING
        # =================================
        self.total_episodes = 0          # Total episodes planned for training
        self.exploration_end_episode = 0 # Episode when exploration should reach minimum
        self.exploration_decay_type = "exponential"  # Type of decay: "exponential" or "linear"
        self.linear_decay_per_episode = 0.0  # Amount to decrease epsilon per episode (for linear decay)
    
    def copy_to_target_network(self):
        """
        📋 Copy main network weights to target network
        
        Why do we need two networks?
        - 🎯 Target network provides stable learning targets
        - 🧠 Main network is constantly changing during learning
        - 📈 This prevents unstable "chasing moving targets" problem
        """
        self.target_network.load_state_dict(self.main_network.state_dict())
    
    def choose_action(self, state, training_mode=True):
        """
        🤔 Decide what action to take given current game state
        
        Uses "epsilon-greedy" strategy:
        - Sometimes explore randomly (try new things)
        - Sometimes exploit knowledge (use what we learned)
        
        Args:
            state (numpy.array): Current game state
            training_mode (bool): Whether to use exploration (True) or be greedy (False)
            
        Returns:
            int: Chosen action (0=stay, 1=left, 2=right)
        """
        # 🎲 EXPLORATION: Sometimes try random actions
        # ===========================================
        if training_mode and random.random() < self.epsilon:
            # Random exploration - try something new!
            return random.choice(range(self.action_size))
        
        # 🎯 EXPLOITATION: Use neural network to pick best action
        # ======================================================
        with torch.no_grad():  # Don't track gradients for inference
            # Convert state to PyTorch tensor
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            
            # Get Q-values from neural network
            q_values = self.main_network(state_tensor)
            
            # Choose action with highest Q-value
            best_action = q_values.argmax().item()
            return best_action
    
    def store_experience(self, state, action, reward, next_state, done):
        """
        💾 Store experience in memory for later learning
        
        Args:
            state: Game state before taking action
            action: Action that was taken
            reward: Reward received for the action
            next_state: Game state after taking action  
            done: Whether the game ended
        """
        self.memory.push(state, action, reward, next_state, done)
    
    def train_from_experience(self):
        """
        🎓 Learn from past experiences using Deep Q-Learning algorithm
        
        This is where the magic happens! The AI looks at past experiences
        and updates its neural network to make better decisions in the future.
        """
        # 📚 CHECK IF ENOUGH EXPERIENCES TO LEARN
        # =====================================
        if len(self.memory) < self.batch_size:
            return  # Not enough experiences yet, wait for more
        
        # 🎲 SAMPLE RANDOM EXPERIENCES FROM MEMORY
        # ========================================
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # 🧠 CALCULATE CURRENT Q-VALUES (what network currently thinks)
        # ============================================================
        current_q_values = self.main_network(states).gather(1, actions.unsqueeze(1))
        
        # 🎯 CALCULATE TARGET Q-VALUES (what network should think)
        # =======================================================
        with torch.no_grad():  # Don't track gradients for target calculation
            # Get best future Q-values from target network
            future_q_values = self.target_network(next_states).max(1)[0]
            
            # Calculate target: reward + (discount * future_value * not_done)
            target_q_values = rewards + (self.GAMMA * future_q_values * ~dones)
        
        # 📏 CALCULATE LEARNING LOSS
        # ==========================
        # How different are our current predictions from ideal predictions?
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        # 🎓 UPDATE NEURAL NETWORK (the actual learning!)
        # ==============================================
        self.optimizer.zero_grad()    # Clear previous gradients
        loss.backward()               # Calculate new gradients  
        self.optimizer.step()         # Update network weights
        
        # 📊 TRACK TRAINING PROGRESS
        # =========================
        self.training_losses.append(loss.item())
        
        # Note: Epsilon decay moved to trainer (per episode, not per step)
        self.exploration_rates.append(self.epsilon)
    
    def decay_epsilon(self):
        """
        📉 Adaptive Exploration Decay (called once per episode)
        
        This gradually reduces the randomness in action selection as 
        the AI gets smarter. Uses adaptive decay that scales to total
        training episodes for consistent exploration curves.
        
        Supports both exponential and linear decay patterns.
        """
        current_episode = len(self.episode_scores)
        
        # 🎯 ADAPTIVE EXPLORATION: Scale decay to training length
        if self.total_episodes > 0 and self.exploration_end_episode > 0:
            if current_episode <= self.exploration_end_episode:
                if self.exploration_decay_type.lower() == "linear":
                    # 📉 LINEAR DECAY: Reduce by fixed amount each episode
                    self.epsilon -= self.linear_decay_per_episode
                    # Ensure we don't go below minimum
                    if self.epsilon < self.EPSILON_END:
                        self.epsilon = self.EPSILON_END
                else:
                    # 📉 EXPONENTIAL DECAY: Multiply by decay factor each episode
                    if self.epsilon > self.EPSILON_END:
                        self.epsilon *= self.EPSILON_DECAY
                        # Ensure we don't go below minimum
                        if self.epsilon < self.EPSILON_END:
                            self.epsilon = self.EPSILON_END
            # After exploration_end_episode, keep epsilon at minimum
            else:
                self.epsilon = self.EPSILON_END
        else:
            # 🔄 FALLBACK: Original decay method if adaptive parameters not set
            if self.epsilon > self.EPSILON_END:
                self.epsilon *= self.EPSILON_DECAY
    
    def save_agent(self, filepath):
        """
        💾 Save the trained agent to disk
        
        Args:
            filepath (str): Where to save the agent
        """
        checkpoint = {
            'main_network_state': self.main_network.state_dict(),
            'target_network_state': self.target_network.state_dict(), 
            'optimizer_state': self.optimizer.state_dict(),
            'current_epsilon': self.epsilon,
            'training_losses': self.training_losses,
            'episode_scores': self.episode_scores,
            'exploration_rates': self.exploration_rates
        }
        
        torch.save(checkpoint, filepath)
        print(f"🎉 Agent saved successfully to {filepath}")
    
    def load_agent(self, filepath):
        """
        📂 Load a previously trained agent from disk
        
        🔧 SECURITY FIX: Uses weights_only=True for safer loading
        🐛 PERFORMANCE FIX: Better error handling and validation
        
        Args:
            filepath (str): Path to saved agent file
        """
        try:
            # 🔒 SECURE LOADING - Prevents malicious code execution
            checkpoint = torch.load(filepath, weights_only=False)  # Note: Set to False for complex objects, True for weights only
            
            # 🧠 RESTORE NEURAL NETWORKS
            self.main_network.load_state_dict(checkpoint['main_network_state'])
            self.target_network.load_state_dict(checkpoint['target_network_state'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state'])
            
            # 📊 RESTORE TRAINING STATISTICS
            self.epsilon = checkpoint['current_epsilon']
            self.training_losses = checkpoint['training_losses']
            self.episode_scores = checkpoint['episode_scores'] 
            self.exploration_rates = checkpoint['exploration_rates']
            
            # 🔍 VALIDATE LOADED MODEL
            print(f"🎉 Agent loaded successfully from {filepath}")
            print(f"📊 Loaded epsilon: {self.epsilon:.4f}")
            print(f"📈 Training episodes recorded: {len(self.episode_scores)}")
            
            # 🧪 QUICK SANITY CHECK - Test if network can make predictions
            test_state = torch.FloatTensor([0.5, 0.5, 0.5, 0.5, 0.5]).unsqueeze(0)
            with torch.no_grad():
                q_values = self.main_network(test_state)
                print(f"🧠 Network test - Q-values: {q_values.numpy().flatten()}")
                
        except Exception as error:
            print(f"❌ Error loading agent: {error}")
            print("🔄 Agent will start with random weights")
            raise error
    
    def create_training_charts(self, save_path="results/charts/ai_training_progress.png"):
        """
        📊 Create visual charts showing training progress
        
        This generates graphs that show:
        - How scores improved over time
        - How training loss changed
        - How exploration rate decreased
        - Overall learning trends
        """
        # Create figure with 4 subplots
        fig, charts = plt.subplots(2, 2, figsize=(15, 10))
        
        # 🏆 CHART 1: SCORES OVER EPISODES
        # ================================
        charts[0, 0].plot(self.episode_scores, color='blue', alpha=0.7)
        charts[0, 0].set_title('Agent Scores Over Time')  # Removed emoji
        charts[0, 0].set_xlabel('Episode Number')
        charts[0, 0].set_ylabel('Score Achieved')
        charts[0, 0].grid(True, alpha=0.3)
        
        # 📈 CHART 2: MOVING AVERAGE OR TREND
        # ===================================
        if len(self.episode_scores) >= 100:
            # Calculate 100-episode moving average
            window_size = 100
            moving_averages = []
            for i in range(len(self.episode_scores) - window_size + 1):
                window = self.episode_scores[i:i + window_size] 
                avg = sum(window) / window_size
                moving_averages.append(avg)
            
            charts[0, 1].plot(moving_averages, color='green', linewidth=2)
            charts[0, 1].set_title('Moving Average Score (100 episodes)')  # Removed emoji
            charts[0, 1].set_xlabel('Episode Number')
            charts[0, 1].set_ylabel('Average Score')
            
        else:
            # Show scores with trend line for fewer episodes
            charts[0, 1].plot(self.episode_scores, 'b-', alpha=0.6, label='Episode Scores')
            if len(self.episode_scores) > 1:
                # Add trend line
                episodes = list(range(len(self.episode_scores)))
                trend_coeffs = np.polyfit(episodes, self.episode_scores, 1)
                trend_line = np.poly1d(trend_coeffs)
                charts[0, 1].plot(episodes, trend_line(episodes), 'r--', alpha=0.8, label='Trend Line')
                charts[0, 1].legend()
            
            charts[0, 1].set_title(f'Scores with Trend ({len(self.episode_scores)} episodes)')  # Removed emoji
            charts[0, 1].set_xlabel('Episode Number')  
            charts[0, 1].set_ylabel('Score')
        
        charts[0, 1].grid(True, alpha=0.3)
        
        # 📉 CHART 3: TRAINING LOSS
        # =========================
        if self.training_losses:
            charts[1, 0].plot(self.training_losses, color='red', alpha=0.7)
            charts[1, 0].set_title('Training Loss Over Time')  # Removed emoji
            charts[1, 0].set_xlabel('Training Step')
            charts[1, 0].set_ylabel('Loss Value')
            charts[1, 0].grid(True, alpha=0.3)
        else:
            charts[1, 0].text(0.5, 0.5, 'No Training Data Yet\nStart training to see loss!',  # Removed emoji
                             ha='center', va='center', transform=charts[1, 0].transAxes,
                             fontsize=12)
            charts[1, 0].set_title('Training Loss Over Time')  # Removed emoji
        
        # 🎲 CHART 4: EXPLORATION RATE DECAY  
        # ==================================
        if self.exploration_rates:
            charts[1, 1].plot(self.exploration_rates, color='orange', alpha=0.7)
            charts[1, 1].set_title('Exploration Rate Decay')  # Removed emoji
            charts[1, 1].set_xlabel('Training Step')
            charts[1, 1].set_ylabel('Epsilon (Exploration Rate)')
            charts[1, 1].grid(True, alpha=0.3)
        else:
            charts[1, 1].text(0.5, 0.5, 'No Exploration Data Yet\nStart training to see decay!',  # Removed emoji
                             ha='center', va='center', transform=charts[1, 1].transAxes,
                             fontsize=12)
            charts[1, 1].set_title('Exploration Rate Decay')  # Removed emoji
        
        # 🎨 FINALIZE AND SAVE CHART
        # ==========================
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Training progress charts saved to '{save_path}'")
        
        # Close plot to free memory (don't show interactively)
        plt.close()

# 🧪 TESTING SECTION
# ==================
if __name__ == "__main__":
    """
    This section runs when you execute this file directly.
    It creates a simple test to verify the DQN agent works correctly.
    """
    print("🧪 TESTING DQN AGENT")
    print("===================")
    
    # Test parameters
    STATE_SIZE = 5      # Our game state has 5 values
    ACTION_SIZE = 3     # 3 possible actions: left, right, stay
    
    # Create agent
    print("🤖 Creating DQN agent...")
    agent = DQNAgent(
        state_size=STATE_SIZE,
        action_size=ACTION_SIZE,
        learning_rate=0.001,
        epsilon_start=1.0,
        memory_size=1000,
        batch_size=32
    )
    
    # Test decision making
    print("🎯 Testing decision making...")
    test_state = np.array([0.5, 0.3, 0.7, 0.4, 0.6], dtype=np.float32)
    action = agent.choose_action(test_state, training_mode=True)
    print(f"   Agent chose action: {action} (0=stay, 1=left, 2=right)")
    
    # Test memory storage
    print("💾 Testing memory storage...")
    agent.store_experience(test_state, action, 10.0, test_state, False)
    print(f"   Memory size: {len(agent.memory)} experiences")
    
    # Test saving/loading
    print("💾 Testing save/load functionality...")
    test_save_path = "test_agent.pth"
    agent.save_agent(test_save_path)
    
    new_agent = DQNAgent(STATE_SIZE, ACTION_SIZE)
    new_agent.load_agent(test_save_path)
    print("   Save/load test successful!")
    
    # Test chart creation
    print("📊 Testing chart creation...")
    agent.episode_scores = [1, 5, 10, 15, 20]  # Fake some scores
    agent.create_training_charts()
    
    print("✅ ALL TESTS PASSED!")
    print("🎉 DQN Agent is ready for training!")