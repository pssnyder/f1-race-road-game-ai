"""
🏋️ F1 Race AI Training System 🤖
=================================

Welcome to the AI Training Center! 🎓✨    # Calculate the decay rate needed to reach epsilon_end at the target episode
    # Using: epsilon_end = epsilon_start * (decay_rate ^ target_episodes)
    # Solving: decay_rate = (epsilon_end / epsilon_start) ^ (1 / target_episodes)
    import math
    
    print(f"🎯 ADAPTIVE EXPLORATION SYSTEM")
    print(f"   📊 Total episodes: {episodes}")
    print(f"   📉 Exploration ends at episode: {exploration_end_episode} ({exploration_completion_ratio*100:.0f}% through training)")
    print(f"   📈 Exploration range: {EXPLORATION_START:.2f} → {EXPLORATION_END:.3f}")
    print(f"   🔄 Decay type: {exploration_decay_type}")
    
    if exploration_end_episode > 0:
        if exploration_decay_type.lower() == "linear":
            # Linear decay: epsilon decreases by fixed amount each episode
            EXPLORATION_DECAY = "linear"  # Special flag for linear decay
            linear_decay_per_episode = (EXPLORATION_START - EXPLORATION_END) / exploration_end_episode
            print(f"   📉 Linear decay per episode: -{linear_decay_per_episode:.6f}")
        else:
            # Exponential decay: epsilon multiplied by decay factor each episode
            EXPLORATION_DECAY = math.pow(EXPLORATION_END / EXPLORATION_START, 1.0 / exploration_end_episode)
            print(f"   📉 Exponential decay rate: {EXPLORATION_DECAY:.6f}")
    else:
        EXPLORATION_DECAY = 0.9995  # Fallback to previous default
        print(f"   ⚠️  Using fallback decay rate: {EXPLORATION_DECAY:.6f}")ere the magic happens - where we transform a clueless AI into a racing expert!
Think of this as the "gym" where our artificial race car driver learns to become a champion.

🎯 WHAT HAPPENS DURING TRAINING?
-------------------------------
Imagine teaching a friend to drive:
- 🚗 They start by crashing into everything (random actions)  
- 💡 Gradually they learn "don't hit red things" (negative rewards)
- 🏆 Eventually they become skilled drivers (positive rewards)
- 📈 Each practice session makes them better (learning from experience)

Our AI goes through the same process, but MUCH faster! 🚀

🧠 THE TRAINING PROCESS:
1. 🎮 CREATE ENVIRONMENT: Set up the racing game
2. 🤖 CREATE AI AGENT: Initialize the "student driver" 
3. 🔄 PRACTICE LOOP: Let AI play thousands of games
4. 📊 TRACK PROGRESS: Monitor how well AI is learning
5. 💾 SAVE RESULTS: Keep the trained "brain" for later use

🎓 TRAINING PHASES:
- 🌟 EXPLORATION: AI tries random actions to discover what works
- 🧠 LEARNING: AI updates its "brain" based on rewards/penalties  
- 🎯 EXPLOITATION: AI uses learned knowledge to get better scores
- 🏆 MASTERY: AI becomes expert driver that rarely crashes!

👥 AUDIENCE NOTES:
- 🔬 Data Scientists: Notice the hyperparameter tuning and performance metrics
- 👩‍🏫 Educators: Great example of trial-and-error learning scaled up
- 👶 Young Coders: Watch a computer learn to play games just like humans!
- 🤓 AI Curious: See reinforcement learning in action with real-time feedback

Author: Pat Snyder 💻
Created for: Learning Labs Portfolio 🌟
"""

import numpy as np
import matplotlib.pyplot as plt
from environment import F1RaceEnvironment
from agent import DQNAgent
import time
import os
import signal
import sys

import signal
import sys

# Global flag for graceful shutdown
graceful_shutdown = False

def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully"""
    global graceful_shutdown
    print("\n\n🛑 GRACEFUL SHUTDOWN REQUESTED")
    print("=" * 50)
    print("🔄 Finishing current episode and saving progress...")
    print("💾 Please wait for clean shutdown...")
    print("   (Press Ctrl+C again to force quit)")
    graceful_shutdown = True
    
    # Set up handler for second Ctrl+C to force quit
    signal.signal(signal.SIGINT, lambda s, f: sys.exit(1))

# Set up the signal handler
signal.signal(signal.SIGINT, signal_handler)

def train_racing_ai(episodes=2000, target_update_frequency=100, save_frequency=500, 
                   show_training=False, resume_checkpoint: str | None = None,
                   framerate_multiplier: int = 100, chart_update_frequency: int = 100,
                   exploration_completion_ratio: float = 0.8, exploration_decay_type: str = "exponential"):
    """
    🏋️ Train the AI to become an expert F1 race car driver!
    
    This function runs the complete training process where the AI plays
    thousands of racing games and learns from each experience.
    
    Args:
        episodes (int): How many games to play (2000 = good for learning)
        target_update_frequency (int): How often to update target network (100 = stable)
        save_frequency (int): How often to save progress (500 = reasonable backup)
        show_training (bool): Whether to show the game while training (False = faster)
        resume_checkpoint (str | None): Path to checkpoint to resume from
        framerate_multiplier (int): Speed multiplier 1-500 (100 = normal speed, 200 = 2x speed)
        chart_update_frequency (int): How often to update training charts (100 = every 100 episodes)
        exploration_completion_ratio (float): At what fraction of training should exploration reach minimum (0.8 = 80%)
        exploration_decay_type (str): Type of decay curve - "exponential" or "linear" (exponential = smooth curve, linear = straight line)
        
    Returns:
        DQNAgent: The trained AI agent
    """
    
    print("🎮 INITIALIZING F1 RACE AI TRAINING SYSTEM")
    print("=" * 50)
    
    # 🎛️ TRAINING CONFIGURATION - Easy to modify!
    # ===========================================
    LEARNING_RATE = 0.001        # 📚 How fast AI learns (0.001 = stable default)
    DISCOUNT_FACTOR = 0.95       # 🔮 How much AI cares about future (0.95 = forward-thinking)
    EXPLORATION_START = 1.0       # 🎲 Initial randomness (100% random at start)
    EXPLORATION_END = 0.01       # 🎲 Final randomness (1% random when expert)  
    MEMORY_SIZE = 15000         # 🧠 How many experiences to remember
    BATCH_SIZE = 64             # 📦 How many experiences to learn from at once
    
    # 📏 DYNAMIC EXPLORATION DECAY CALCULATION
    # ========================================
    # Calculate when exploration should reach minimum (based on total episodes)
    exploration_end_episode = int(episodes * exploration_completion_ratio)
    
    # Calculate the decay rate needed to reach epsilon_end at the target episode
    # Using: epsilon_end = epsilon_start * (decay_rate ^ target_episodes)
    # Solving: decay_rate = (epsilon_end / epsilon_start) ^ (1 / target_episodes)
    import math
    if exploration_end_episode > 0:
        EXPLORATION_DECAY = math.pow(EXPLORATION_END / EXPLORATION_START, 1.0 / exploration_end_episode)
    else:
        EXPLORATION_DECAY = 0.9995  # Fallback to previous default
    
    print(f"🎯 ADAPTIVE EXPLORATION SYSTEM")
    print(f"   � Total episodes: {episodes}")
    print(f"   📉 Exploration ends at episode: {exploration_end_episode} ({exploration_completion_ratio*100:.0f}% through training)")
    print(f"   🎲 Calculated decay rate: {EXPLORATION_DECAY:.6f}")
    print(f"   📈 Exploration range: {EXPLORATION_START:.2f} → {EXPLORATION_END:.3f}")
    print()
    
    # 🏗️ CREATE TRAINING ENVIRONMENT AND AI AGENT
    # ============================================
    print("🏎️  Creating racing environment...")
    # When not showing training, max out framerate for fastest training
    effective_framerate = 500 if not show_training else framerate_multiplier
    env = F1RaceEnvironment(render=show_training, framerate_multiplier=effective_framerate)
    
    print("🤖 Creating AI agent...")
    agent = DQNAgent(
        state_size=env.state_space_size,      # 5 numbers describing game state
        action_size=env.action_space_size,    # 3 possible actions  
        learning_rate=LEARNING_RATE,
        gamma=DISCOUNT_FACTOR,
        epsilon_start=EXPLORATION_START,
        epsilon_end=EXPLORATION_END,
        epsilon_decay=EXPLORATION_DECAY,
        memory_size=MEMORY_SIZE,
        batch_size=BATCH_SIZE
    )
    
    # Set total episodes for adaptive exploration tracking
    agent.total_episodes = episodes
    agent.exploration_end_episode = exploration_end_episode
    agent.exploration_decay_type = exploration_decay_type
    if exploration_decay_type.lower() == "linear" and exploration_end_episode > 0:
        agent.linear_decay_per_episode = (EXPLORATION_START - EXPLORATION_END) / exploration_end_episode
    if resume_checkpoint and os.path.isfile(resume_checkpoint):
        print(f"📂 Resuming from checkpoint: {resume_checkpoint}")
        agent.load_agent(resume_checkpoint)
    
    # 📊 DISPLAY TRAINING CONFIGURATION
    # =================================
    print(f"📏 State space: {env.state_space_size} values")
    print(f"🎯 Action space: {env.action_space_size} actions")
    print(f"📚 Learning rate: {LEARNING_RATE}")  
    print(f"🔮 Discount factor: {DISCOUNT_FACTOR}")
    print(f"🎮 Training episodes: {episodes}")
    print(f"👁️  Show training: {show_training}")
    print(f"⚡ Framerate multiplier: {effective_framerate}% ({'MAX' if effective_framerate == 500 else 'normal' if effective_framerate == 100 else 'custom'})")
    print(f"📊 Chart update frequency: every {chart_update_frequency} episodes")
    print("-" * 50)
    
    # 📊 TRAINING METRICS - Track AI's progress
    # ========================================
    # Initialize scores from checkpoint if resuming
    if resume_checkpoint and agent.episode_scores:
        all_scores = agent.episode_scores.copy()  # Include previous scores for averages
        all_episode_lengths = []  # Episode lengths aren't saved, so start fresh
        best_score_ever = max(agent.episode_scores)
        print(f"📊 Restored {len(all_scores)} previous episodes with best score: {best_score_ever}")
    else:
        all_scores = []           # Score from each episode
        all_episode_lengths = []  # How long each episode lasted
        best_score_ever = 0       # Best score achieved so far
    
    training_start_time = time.time()
    
    # 🏋️ MAIN TRAINING LOOP
    # =====================
    print("🚀 STARTING TRAINING!")
    print("Watch the AI transform from terrible to talented! 🎓")
    print("💡 Press Ctrl+C anytime for graceful shutdown (saves progress)")
    print()
    
    # If resuming, start from the current episode count
    start_episode = len(agent.episode_scores) if resume_checkpoint else 0
    if start_episode > 0:
        print(f"📂 Resuming from episode {start_episode} (continuing where we left off)")
    
    for episode_idx in range(episodes):
        # 🛑 CHECK FOR GRACEFUL SHUTDOWN
        # ==============================
        if graceful_shutdown:
            print(f"\n🛑 Graceful shutdown at episode {start_episode + episode_idx}")
            break
            
        episode_number = start_episode + episode_idx  # Actual episode number for display
        
        # 🔄 START NEW EPISODE
        # ===================
        current_state = env.reset()
        total_reward = 0
        steps_taken = 0
        episode_start_time = time.time()
        
        # 🎮 PLAY ONE COMPLETE GAME
        # ========================
        while True:
            # 🛑 CHECK FOR GRACEFUL SHUTDOWN DURING EPISODE
            # =============================================
            if graceful_shutdown:
                print(f"   🛑 Graceful shutdown during episode {episode_number}")
                break
                
            # 🤔 AI DECIDES WHAT TO DO
            # ========================
            action = agent.choose_action(current_state, training_mode=True)
            
            # 🎬 TAKE ACTION IN GAME  
            # ======================
            next_state, reward, game_over, info = env.step(action)
            
            # 💾 STORE EXPERIENCE FOR LEARNING
            # ================================
            agent.store_experience(current_state, action, reward, next_state, game_over)
            
            # 📊 UPDATE TRACKING VARIABLES
            # ============================
            current_state = next_state
            total_reward = total_reward + reward
            steps_taken = steps_taken + 1
            
            # 🎨 SHOW GAME IF ENABLED
            # =======================
            if show_training:
                env.render()
                time.sleep(0.01)  # Slow down so humans can watch
            
            # 🎓 LEARN FROM EXPERIENCES (if enough stored)
            # ===========================================
            if len(agent.memory) > agent.batch_size:
                agent.train_from_experience()
            
            # 🏁 CHECK IF EPISODE FINISHED
            # ============================
            if game_over:
                break
        
        # 📈 RECORD EPISODE RESULTS
        # ========================
        episode_score = int(info['score'])
        all_scores.append(episode_score)
        all_episode_lengths.append(steps_taken)
        agent.episode_scores.append(int(episode_score))
        
        # 📉 DECAY EXPLORATION (once per episode, not per step!)
        # =====================================================
        agent.decay_epsilon()
        
        # 🎯 UPDATE TARGET NETWORK PERIODICALLY
        # ====================================
        if episode_idx % target_update_frequency == 0:
            agent.copy_to_target_network()
        
        # 📊 DISPLAY PROGRESS UPDATES
        # ===========================
        show_progress = (episode_idx % 100 == 0 or episode_score > best_score_ever)
        
        if show_progress:
            episode_time = time.time() - episode_start_time
            
            # Calculate averages
            if len(all_scores) >= 100:
                avg_score = sum(all_scores[-100:]) / 100
                avg_length = sum(all_episode_lengths[-100:]) / 100
            else:
                avg_score = sum(all_scores) / len(all_scores)
                avg_length = sum(all_episode_lengths) / len(all_episode_lengths)
            
            print(f"Episode {episode_number:4d} | "
                  f"Score: {episode_score:3d} | "
                  f"Steps: {steps_taken:4d} | "
                  f"Avg Score: {avg_score:.2f} | "
                  f"Avg Steps: {avg_length:.1f} | "
                  f"Exploration: {agent.epsilon:.3f} | "
                  f"Time: {episode_time:.2f}s")
            
            # 🏆 CHECK FOR NEW RECORD
            # =======================
            if episode_score > best_score_ever:
                best_score_ever = episode_score
                print(f"   🎉 NEW RECORD! Best score: {best_score_ever}")
        
        # 💾 SAVE PROGRESS PERIODICALLY
        # =============================
        if episode_idx % save_frequency == 0 and episode_idx > 0:
            os.makedirs('models/checkpoints', exist_ok=True)
            checkpoint_filename = os.path.join('models', 'checkpoints', f'ai_driver_checkpoint_episode_{episode_number}.pth')
            agent.save_agent(checkpoint_filename)
            print(f"   💾 Progress saved: {checkpoint_filename}")
        
        # 📊 UPDATE CHARTS PERIODICALLY
        # =============================
        if episode_idx % chart_update_frequency == 0 and episode_idx > 0:
            print(f"   📊 Updating training charts...")
            os.makedirs('results/charts', exist_ok=True)
            chart_path = os.path.join('results', 'charts', 'ai_training_progress.png')
            agent.create_training_charts(save_path=chart_path)
    
    # 🏆 TRAINING COMPLETED OR INTERRUPTED!
    # =====================================
    total_training_time = time.time() - training_start_time
    actual_episodes = len(agent.episode_scores) - (len(agent.episode_scores) if not resume_checkpoint else len(agent.episode_scores) - start_episode)
    
    if graceful_shutdown:
        print("\n" + "=" * 50)
        print("🛑 TRAINING GRACEFULLY INTERRUPTED")
        print("=" * 50)
        print("✅ Current progress has been preserved!")
    else:
        print("\n" + "=" * 50)
        print("🎓 TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 50)
    
    # 📊 FINAL STATISTICS
    # ==================
    if len(all_scores) > 0:
        final_avg_score = sum(all_scores[-100:]) / min(100, len(all_scores))
        print(f"🏆 Best score achieved: {best_score_ever}")
        print(f"📈 Final average score: {final_avg_score:.2f}")
    else:
        print(f"🏆 Best score achieved: {best_score_ever}")
        print("📈 No episodes completed")
        
    print(f"⏱️  Total training time: {total_training_time/60:.1f} minutes")
    print(f"🎮 Episodes completed: {len(agent.episode_scores)}")
    print(f"🧠 Final exploration rate: {agent.epsilon:.4f}")
    
    # 💾 SAVE CURRENT STATE (ALWAYS SAVE ON SHUTDOWN)
    # ===============================================
    os.makedirs('models/final', exist_ok=True)
    if graceful_shutdown:
        # Save with a special graceful shutdown name
        current_episode = len(agent.episode_scores)
        final_model_name = os.path.join('models', 'final', f'f1_race_ai_interrupted_episode_{current_episode}.pth')
        print(f"💾 Saving interrupted training state...")
    else:
        final_model_name = os.path.join('models', 'final', 'f1_race_ai_final_model.pth')
        print(f"💾 Saving final trained model...")
    
    agent.save_agent(final_model_name)
    
    # 📊 CREATE TRAINING CHARTS
    # =========================
    print("📊 Creating training progress charts...")
    chart_path = os.path.join('results', 'charts', 'ai_training_progress.png')
    agent.create_training_charts(save_path=chart_path)
    print(f"📊 Chart saved at: {chart_path}")
    
    # Don't show chart automatically - it blocks the program
    # User can view it separately or via the dashboard
    
    # 🚪 CLEANUP
    # ==========
    env.close()
    
    if graceful_shutdown:
        print(f"🎉 Your AI training was safely interrupted and saved as '{final_model_name}'!")
        print("💡 You can resume training from this point using the 'resume' option.")
    else:
        print(f"🎉 Your AI race car driver is now trained and saved as '{final_model_name}'!")
    
    return agent

def test_trained_ai(model_path, num_test_episodes=5, show_games=True, framerate_multiplier: int = 100):
    """
    🧪 Test how well our trained AI performs!
    
    This function loads a trained AI and lets it play several games
    to see how skilled it has become.
    
    Args:
        model_path (str): Path to the trained AI model file
        num_test_episodes (int): How many test games to play
        show_games (bool): Whether to show the games visually
        framerate_multiplier (int): Speed multiplier 1-500 (100 = normal speed, 200 = 2x speed)
        
    Returns:
        list: Scores achieved in test episodes
    """
    print(f"🧪 TESTING TRAINED AI")
    print("=" * 30)
    print(f"📂 Loading model: {model_path}")
    print(f"🎮 Test episodes: {num_test_episodes}")
    
    # 🏗️ CREATE TESTING ENVIRONMENT AND LOAD TRAINED AI
    # =================================================
    env = F1RaceEnvironment(render=show_games, framerate_multiplier=framerate_multiplier)
    agent = DQNAgent(
        state_size=env.state_space_size,
        action_size=env.action_space_size
    )
    
    # 📂 LOAD THE TRAINED AI "BRAIN"
    # ==============================
    agent.load_agent(model_path)
    
    # 🔧 PERFORMANCE FIX: Explicitly disable exploration
    original_epsilon = agent.epsilon
    agent.epsilon = 0.0  # Completely disable exploration for testing
    
    print("🤖 AI loaded successfully!")
    print(f"🔧 Original epsilon: {original_epsilon:.4f} → Testing epsilon: {agent.epsilon:.4f}")
    print("🎯 Testing mode: No exploration (pure skill)")
    print("-" * 30)
    
    # 📊 TEST RESULTS TRACKING
    # =======================
    test_scores = []
    test_episode_lengths = []
    
    # 🎮 RUN TEST EPISODES
    # ===================
    for episode in range(num_test_episodes):
        current_state = env.reset()
        total_reward = 0
        steps = 0
        
        print(f"\n🎮 Test Episode {episode + 1}")
        print("   Watch the AI show off its skills! 🏆")
        
        # 🏁 PLAY ONE COMPLETE GAME
        # =========================
        action_counts = [0, 0, 0]  # Track action usage [stay, left, right]
        
        while True:
            # 🎯 AI CHOOSES BEST ACTION (no randomness)
            # ========================================
            action = agent.choose_action(current_state, training_mode=False)
            action_counts[action] += 1  # Count this action
            
            # 🎬 TAKE ACTION
            # ==============
            next_state, reward, game_over, info = env.step(action)
            
            # 📊 UPDATE TRACKING
            # ==================
            current_state = next_state
            total_reward = total_reward + reward
            steps = steps + 1
            
            # 🎨 SHOW GAME
            # ============
            if show_games:
                env.render()
                time.sleep(0.03)  # Slower for better viewing
            
            # 🏁 CHECK IF GAME ENDED
            # ======================
            if game_over:
                break
        
        # 📊 RECORD TEST RESULTS
        # =====================
        final_score = info['score']
        test_scores.append(final_score)
        test_episode_lengths.append(steps)
        
        print(f"   ✅ Episode {episode + 1} Results:")
        print(f"      🏆 Score: {final_score}")
        print(f"      📏 Steps survived: {steps}")
        print(f"      🎁 Total reward: {total_reward:.2f}")
        print(f"      🎮 Actions - Stay: {action_counts[0]}, Left: {action_counts[1]}, Right: {action_counts[2]}")
    
    # 📊 CALCULATE FINAL TEST STATISTICS
    # ==================================
    avg_score = sum(test_scores) / len(test_scores)
    best_test_score = max(test_scores)
    avg_length = sum(test_episode_lengths) / len(test_episode_lengths)
    
    print(f"\n🏆 FINAL TEST RESULTS")
    print("=" * 25)
    print(f"📊 Average Score: {avg_score:.2f}")
    print(f"🥇 Best Score: {best_test_score}")
    print(f"📏 Average Episode Length: {avg_length:.1f} steps")
    print(f"🎯 All Scores: {test_scores}")
    
    # 🚪 CLEANUP
    # ==========
    env.close()
    return test_scores

def test_random_driver(num_episodes=10, framerate_multiplier: int = 100):
    """
    🎲 Test a completely random driver as baseline comparison
    
    This shows how badly a random (untrained) driver performs,
    which helps us appreciate how much our AI has learned!
    
    Args:
        num_episodes (int): Number of random games to play
        framerate_multiplier (int): Speed multiplier 1-500 (100 = normal speed, 200 = 2x speed)
        
    Returns:
        list: Scores achieved by random driver
    """
    print("🎲 TESTING RANDOM DRIVER (BASELINE)")
    print("=" * 40)
    print("This shows what happens without AI training! 😅")
    
    # 🎮 CREATE ENVIRONMENT
    # ====================
    env = F1RaceEnvironment(render=True, framerate_multiplier=framerate_multiplier)
    random_scores = []
    random_lengths = []
    
    # 🎮 PLAY RANDOM GAMES
    # ===================
    for episode in range(num_episodes):
        env.reset()
        steps = 0
        
        print(f"\n🎲 Random Episode {episode + 1}")
        print("   Watching chaos unfold... 💥")
        
        while True:
            # 🎲 CHOOSE COMPLETELY RANDOM ACTION
            # =================================
            random_action = np.random.choice(env.action_space_size)
            
            # 🎬 TAKE RANDOM ACTION
            # ====================
            _, _, game_over, info = env.step(random_action)
            steps = steps + 1
            
            # 🎨 SHOW THE CHAOS
            # =================
            env.render()
            time.sleep(0.03)
            
            if game_over:
                break
        
        # 📊 RECORD RANDOM RESULTS
        # =======================
        score = info['score']
        random_scores.append(score)
        random_lengths.append(steps)
        
        print(f"   💥 Episode {episode + 1}: Score = {score}, Steps = {steps}")
    
    # 📊 RANDOM DRIVER STATISTICS
    # ===========================
    avg_random_score = sum(random_scores) / len(random_scores)
    avg_random_length = sum(random_lengths) / len(random_lengths)
    
    print(f"\n📊 RANDOM DRIVER RESULTS:")
    print(f"   📊 Average Score: {avg_random_score:.2f}")
    print(f"   📏 Average Length: {avg_random_length:.1f} steps")
    print(f"   🎯 All Scores: {random_scores}")
    print("   💡 This is why we need AI training! 🤖")
    
    env.close()
    return random_scores

# 🎮 MAIN PROGRAM INTERFACE
# =========================
if __name__ == "__main__":
    print("🏎️  F1 RACE AI TRAINING SYSTEM")
    print("=" * 50)
    print("Welcome to the AI Driver Training Center! 🎓")
    print()
    
    # 📂 CHECK FOR EXISTING TRAINED MODELS
    # ====================================
    # Gather models from root and models directories
    saved_models = []
    root_models = [f for f in os.listdir('.') if f.endswith('.pth')]
    saved_models.extend(root_models)
    for subdir in ['models', os.path.join('models', 'final'), os.path.join('models', 'checkpoints')]:
        if os.path.isdir(subdir):
            for f in os.listdir(subdir):
                if f.endswith('.pth'):
                    saved_models.append(os.path.join(subdir, f))
    
    if saved_models:
        print("📂 Found existing trained models:")
        for i, model in enumerate(saved_models):
            print(f"   {i}: {model}")
        print()
    
    # 🎯 MENU OPTIONS
    # ===============
    print("🎯 What would you like to do?")
    print("   📚 'train'    - Train a new AI driver from scratch (with framerate & chart options)")
    print("   🧪 'test'     - Test an existing trained model (with framerate control)") 
    print("   🔁 'resume'   - Resume training from a checkpoint (with framerate & chart options)")
    print("   🖼️  'chart'    - View the last training chart if available")
    print("   � 'dashboard'- Launch web dashboard for real-time training monitoring")
    print("   �🎲 'baseline' - Watch a random (untrained) driver fail (with framerate control)")
    print()
    print("   💡 NEW: Framerate multiplier (1-500%) controls training/testing speed!")
    print("   💡 NEW: Charts update periodically during training for real-time progress!")
    print("   💡 NEW: Adaptive exploration decay automatically scales to any training length!")
    print("   💡 NEW: Web dashboard for monitoring training without terminal clutter!")
    print()
    
    # 👤 GET USER CHOICE
    # ==================
    user_choice = input("Enter your choice: ").lower().strip()
    
    # 🚀 EXECUTE USER CHOICE
    # ======================
    if user_choice == 'train':
        print("\n🏋️ TRAINING MODE SELECTED")
        print("-" * 30)
        
        # Get training parameters
        show_visual = input("🎨 Show training visually? (y/n, default=n): ").lower().strip() == 'y'
        episodes_input = input("🎮 Number of episodes (default=2000): ").strip()
        episodes = int(episodes_input) if episodes_input else 2000
        
        # Get framerate configuration
        if show_visual:
            framerate_input = input("⚡ Framerate multiplier 1-500% (default=100): ").strip()
            framerate = int(framerate_input) if framerate_input and framerate_input.isdigit() else 100
            framerate = max(1, min(500, framerate))  # Clamp to valid range
        else:
            framerate = 500  # Max speed for headless training
            print("⚡ Using maximum framerate (500%) for headless training")
        
        # Get chart update frequency
        chart_freq_input = input("📊 Chart update frequency in episodes (default=100): ").strip()
        chart_freq = int(chart_freq_input) if chart_freq_input and chart_freq_input.isdigit() else 100
        
        # Get exploration completion ratio
        exploration_input = input("🎯 Exploration completion % (at what % of training should exploration reach minimum, default=80): ").strip()
        exploration_ratio = float(exploration_input) / 100.0 if exploration_input and exploration_input.replace('.', '').isdigit() else 0.8
        exploration_ratio = max(0.1, min(1.0, exploration_ratio))  # Clamp between 10% and 100%
        
        # Get exploration decay type
        decay_type_input = input("📉 Exploration decay type (exponential/linear, default=exponential): ").strip().lower()
        decay_type = "linear" if decay_type_input == "linear" else "exponential"
        
        # Start training
        print(f"\n🚀 Starting training for {episodes} episodes...")
        print(f"   ⚡ Framerate: {framerate}%")
        print(f"   📊 Charts will update every {chart_freq} episodes")
        print(f"   🎯 Exploration will reach minimum at {exploration_ratio*100:.0f}% of training")
        print(f"   📉 Using {decay_type} decay curve")
        trained_agent = train_racing_ai(episodes=episodes, show_training=show_visual, 
                                       framerate_multiplier=framerate, chart_update_frequency=chart_freq,
                                       exploration_completion_ratio=exploration_ratio, exploration_decay_type=decay_type)
        
        # Offer to test the newly trained agent
        test_new = input("\n🧪 Test the newly trained AI? (y/n): ").lower().strip() == 'y'
        if test_new:
            final_model_name = os.path.join('models', 'final', 'f1_race_ai_final_model.pth')
            test_trained_ai(final_model_name, num_test_episodes=5, framerate_multiplier=100)
    
    elif user_choice == 'test' and saved_models:
        print("\n🧪 TESTING MODE SELECTED")  
        print("-" * 25)
        
        # Show available models
        print("📂 Available models:")
        for i, model in enumerate(saved_models):
            print(f"   {i}: {model}")
        
        # Get model selection
        model_choice = int(input("🎯 Select model number: "))
        selected_model = saved_models[model_choice]
        
        # Get test parameters  
        test_episodes_input = input("🎮 Number of test episodes (default=5): ").strip()
        test_episodes = int(test_episodes_input) if test_episodes_input else 5
        
        # Get framerate configuration for testing
        framerate_input = input("⚡ Framerate multiplier 1-500% (default=100): ").strip()
        framerate = int(framerate_input) if framerate_input and framerate_input.isdigit() else 100
        framerate = max(1, min(500, framerate))
        
        # Run test
        test_trained_ai(selected_model, num_test_episodes=test_episodes, framerate_multiplier=framerate)
    
    elif user_choice == 'resume':
        print("\n🔁 RESUME MODE SELECTED")
        print("-" * 30)
        # Filter checkpoints - include regular checkpoints AND interrupted training files
        checkpoints = []
        for m in saved_models:
            filename = os.path.basename(m)
            # Include regular checkpoints, interrupted training files, and any file with "episode" in name
            if ('checkpoint' in filename or 
                'interrupted_episode' in filename or 
                'ai_driver_checkpoint_episode_' in filename):
                checkpoints.append(m)
        
        # Sort checkpoints by modification time (most recent first) 
        checkpoints.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        
        if not checkpoints:
            print("❌ No checkpoints found. Train first to create checkpoints.")
        else:
            print("📂 Available checkpoints (most recent first):")
            for i, model in enumerate(checkpoints):
                # Show modification time for clarity
                mod_time = os.path.getmtime(model)
                mod_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(mod_time))
                print(f"   {i}: {model} ({mod_time_str})")
            idx = int(input("🎯 Select checkpoint number: "))
            resume_path = checkpoints[idx]
            show_visual = input("🎨 Show training visually? (y/n, default=n): ").lower().strip() == 'y'
            episodes_input = input("🎮 Additional episodes to train (default=500): ").strip()
            episodes = int(episodes_input) if episodes_input else 500
            
            # Get framerate configuration  
            if show_visual:
                framerate_input = input("⚡ Framerate multiplier 1-500% (default=100): ").strip()
                framerate = int(framerate_input) if framerate_input and framerate_input.isdigit() else 100
                framerate = max(1, min(500, framerate))
            else:
                framerate = 500  # Max speed for headless training
                print("⚡ Using maximum framerate (500%) for headless training")
            
            # Get chart update frequency
            chart_freq_input = input("📊 Chart update frequency in episodes (default=100): ").strip()
            chart_freq = int(chart_freq_input) if chart_freq_input and chart_freq_input.isdigit() else 100
            
            # Get exploration completion ratio
            exploration_input = input("🎯 Exploration completion % (at what % of training should exploration reach minimum, default=80): ").strip()
            exploration_ratio = float(exploration_input) / 100.0 if exploration_input and exploration_input.replace('.', '').isdigit() else 0.8
            exploration_ratio = max(0.1, min(1.0, exploration_ratio))  # Clamp between 10% and 100%
            
            # Get exploration decay type
            decay_type_input = input("📉 Exploration decay type (exponential/linear, default=exponential): ").strip().lower()
            decay_type = "linear" if decay_type_input == "linear" else "exponential"
            
            print(f"\n🚀 Resuming training for {episodes} episodes from {resume_path}...")
            print(f"   ⚡ Framerate: {framerate}%")  
            print(f"   📊 Charts will update every {chart_freq} episodes")
            print(f"   🎯 Exploration will reach minimum at {exploration_ratio*100:.0f}% of training")
            print(f"   📉 Using {decay_type} decay curve")
            trained_agent = train_racing_ai(episodes=episodes, show_training=show_visual, resume_checkpoint=resume_path,
                           framerate_multiplier=framerate, chart_update_frequency=chart_freq,
                           exploration_completion_ratio=exploration_ratio, exploration_decay_type=decay_type)
            
            # Offer to test the newly trained agent
            test_new = input("\n🧪 Test the newly trained AI? (y/n): ").lower().strip() == 'y'
            if test_new:
                final_model_name = os.path.join('models', 'final', 'f1_race_ai_final_model.pth')
                test_trained_ai(final_model_name, num_test_episodes=5, framerate_multiplier=100)

    elif user_choice == 'chart':
        print("\n🖼️  VIEW CHART MODE")
        print("-" * 20)
        chart_path = os.path.join('results', 'charts', 'ai_training_progress.png')
        if os.path.isfile(chart_path):
            try:
                img = plt.imread(chart_path)
                plt.figure(figsize=(8, 6))
                plt.imshow(img)
                plt.axis('off')
                plt.title('Last Training Progress')
                plt.show()
            except Exception as _err:
                print(f"📊 Chart saved at: {chart_path}")
        else:
            print("❌ No chart found yet. Train first to generate one.")

    elif user_choice == 'dashboard':
        print("\n🌐 DASHBOARD MODE SELECTED")
        print("-" * 30)
        print("🚀 Starting web dashboard for real-time training monitoring...")
        print()
        print("📊 The dashboard provides:")
        print("   • Real-time training charts")
        print("   • Live statistics and metrics")
        print("   • Training status monitoring")
        print("   • Model information")
        print()
        print("🌍 Dashboard will open at: http://localhost:5000")
        print("💡 Run training in another terminal: python train_ai.py")
        print()
        
        try:
            # Use the simple standalone dashboard
            import subprocess
            import sys
            dashboard_script = os.path.join(os.path.dirname(__file__), 'dashboard_simple.py')
            if os.path.exists(dashboard_script):
                print("🎯 Launching standalone dashboard...")
                subprocess.run([sys.executable, dashboard_script])
            else:
                print("❌ Dashboard script not found!")
                print("💡 Try running: python dashboard_simple.py")
                
        except KeyboardInterrupt:
            print("\n🛑 Dashboard stopped")
        except Exception as e:
            print(f"❌ Error starting dashboard: {e}")
            print("💡 Try running: python dashboard_simple.py")

    elif user_choice == 'baseline':
        print("\n🎲 BASELINE MODE SELECTED")
        print("-" * 30)
        
        baseline_episodes_input = input("🎮 Number of random episodes (default=10): ").strip()
        baseline_episodes = int(baseline_episodes_input) if baseline_episodes_input else 10
        
        # Get framerate configuration for baseline
        framerate_input = input("⚡ Framerate multiplier 1-500% (default=100): ").strip()
        framerate = int(framerate_input) if framerate_input and framerate_input.isdigit() else 100
        framerate = max(1, min(500, framerate))
        
        print("\n⚠️  Warning: This will be painful to watch! 😅")
        input("Press Enter to continue...")
        
        test_random_driver(num_episodes=baseline_episodes, framerate_multiplier=framerate)
    
    elif user_choice == 'test' and not saved_models:
        print("\n❌ No trained models found!")
        print("   💡 Please train a model first using 'train' option")
    
    else:
        print("\n❌ Invalid choice or no models available")
        print("   💡 Valid options: 'train', 'test', 'baseline'")
    
    print("\n🎉 Thanks for using the F1 Race AI Training System!")
    print("👋 Happy AI training! 🤖🏎️")