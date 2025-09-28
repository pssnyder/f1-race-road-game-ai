"""
🏋️ F1 Race AI Training System 🤖
=================================

Welcome to the AI Training Center! 🎓✨

This is where the magic happens - where we transform a clueless AI into a racing expert!
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
from f1_race_env import F1RaceEnvironment
from dqn_agent import DQNAgent
import time
import os

def train_racing_ai(episodes=2000, target_update_frequency=100, save_frequency=500, 
                   show_training=False):
    """
    🏋️ Train the AI to become an expert F1 race car driver!
    
    This function runs the complete training process where the AI plays
    thousands of racing games and learns from each experience.
    
    Args:
        episodes (int): How many games to play (2000 = good for learning)
        target_update_frequency (int): How often to update target network (100 = stable)
        save_frequency (int): How often to save progress (500 = reasonable backup)
        show_training (bool): Whether to show the game while training (False = faster)
        
    Returns:
        DQNAgent: The trained AI agent
    """
    
    print("🎮 INITIALIZING F1 RACE AI TRAINING SYSTEM")
    print("=" * 50)
    
    # 🎛️ TRAINING CONFIGURATION - Easy to modify!
    # ===========================================
    LEARNING_RATE = 0.001        # 📚 How fast AI learns (0.001 = stable default)
    DISCOUNT_FACTOR = 0.99       # 🔮 How much AI cares about future (0.99 = forward-thinking)
    EXPLORATION_START = 1.0      # 🎲 Initial randomness (100% random at start)
    EXPLORATION_END = 0.01       # 🎲 Final randomness (1% random when expert)
    EXPLORATION_DECAY = 0.995    # 📉 How fast to reduce randomness (0.995 = gradual)
    MEMORY_SIZE = 10000         # 🧠 How many experiences to remember
    BATCH_SIZE = 32             # 📦 How many experiences to learn from at once
    
    # 🏗️ CREATE TRAINING ENVIRONMENT AND AI AGENT
    # ============================================
    print("🏎️  Creating racing environment...")
    env = F1RaceEnvironment(render=show_training)
    
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
    
    # 📊 DISPLAY TRAINING CONFIGURATION
    # =================================
    print(f"📏 State space: {env.state_space_size} values")
    print(f"🎯 Action space: {env.action_space_size} actions")
    print(f"📚 Learning rate: {LEARNING_RATE}")  
    print(f"🔮 Discount factor: {DISCOUNT_FACTOR}")
    print(f"🎮 Training episodes: {episodes}")
    print(f"👁️  Show training: {show_training}")
    print("-" * 50)
    
    # 📊 TRAINING METRICS - Track AI's progress
    # ========================================
    all_scores = []           # Score from each episode
    all_episode_lengths = []  # How long each episode lasted
    best_score_ever = 0       # Best score achieved so far
    training_start_time = time.time()
    
    # 🏋️ MAIN TRAINING LOOP
    # =====================
    print("🚀 STARTING TRAINING!")
    print("Watch the AI transform from terrible to talented! 🎓\n")
    
    for episode_number in range(episodes):
        # 🔄 START NEW EPISODE
        # ===================
        current_state = env.reset()
        total_reward = 0
        steps_taken = 0
        episode_start_time = time.time()
        
        # 🎮 PLAY ONE COMPLETE GAME
        # ========================
        while True:
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
        episode_score = info['score']
        all_scores.append(episode_score)
        all_episode_lengths.append(steps_taken)
        agent.episode_scores.append(episode_score)
        
        # 🎯 UPDATE TARGET NETWORK PERIODICALLY
        # ====================================
        if episode_number % target_update_frequency == 0:
            agent.copy_to_target_network()
        
        # 📊 DISPLAY PROGRESS UPDATES
        # ===========================
        show_progress = (episode_number % 100 == 0 or episode_score > best_score_ever)
        
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
        if episode_number % save_frequency == 0 and episode_number > 0:
            checkpoint_filename = f'ai_driver_checkpoint_episode_{episode_number}.pth'
            agent.save_agent(checkpoint_filename)
            print(f"   💾 Progress saved: {checkpoint_filename}")
    
    # 🏆 TRAINING COMPLETED!
    # =====================
    total_training_time = time.time() - training_start_time
    
    print("\n" + "=" * 50)
    print("🎓 TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 50)
    
    # 📊 FINAL STATISTICS
    # ==================
    final_avg_score = sum(all_scores[-100:]) / min(100, len(all_scores))
    print(f"🏆 Best score achieved: {best_score_ever}")
    print(f"📈 Final average score: {final_avg_score:.2f}")
    print(f"⏱️  Total training time: {total_training_time/60:.1f} minutes")
    print(f"🎮 Episodes completed: {episodes}")
    print(f"🧠 Final exploration rate: {agent.epsilon:.4f}")
    
    # 💾 SAVE FINAL TRAINED MODEL
    # ===========================
    final_model_name = 'f1_race_ai_final_model.pth'
    agent.save_agent(final_model_name)
    
    # 📊 CREATE TRAINING CHARTS
    # =========================
    print("📊 Creating training progress charts...")
    agent.create_training_charts()
    
    # 🚪 CLEANUP
    # ==========
    env.close()
    
    print(f"🎉 Your AI race car driver is now trained and saved as '{final_model_name}'!")
    return agent

def test_trained_ai(model_path, num_test_episodes=5, show_games=True):
    """
    🧪 Test how well our trained AI performs!
    
    This function loads a trained AI and lets it play several games
    to see how skilled it has become.
    
    Args:
        model_path (str): Path to the trained AI model file
        num_test_episodes (int): How many test games to play
        show_games (bool): Whether to show the games visually
        
    Returns:
        list: Scores achieved in test episodes
    """
    print(f"🧪 TESTING TRAINED AI")
    print("=" * 30)
    print(f"📂 Loading model: {model_path}")
    print(f"🎮 Test episodes: {num_test_episodes}")
    
    # 🏗️ CREATE TESTING ENVIRONMENT AND LOAD TRAINED AI
    # =================================================
    env = F1RaceEnvironment(render=show_games)
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

def test_random_driver(num_episodes=10):
    """
    🎲 Test a completely random driver as baseline comparison
    
    This shows how badly a random (untrained) driver performs,
    which helps us appreciate how much our AI has learned!
    
    Args:
        num_episodes (int): Number of random games to play
        
    Returns:
        list: Scores achieved by random driver
    """
    print("🎲 TESTING RANDOM DRIVER (BASELINE)")
    print("=" * 40)
    print("This shows what happens without AI training! 😅")
    
    # 🎮 CREATE ENVIRONMENT
    # ====================
    env = F1RaceEnvironment(render=True)
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
    saved_models = [f for f in os.listdir('.') if f.endswith('.pth')]
    
    if saved_models:
        print("📂 Found existing trained models:")
        for i, model in enumerate(saved_models):
            print(f"   {i}: {model}")
        print()
    
    # 🎯 MENU OPTIONS
    # ===============
    print("🎯 What would you like to do?")
    print("   📚 'train'    - Train a new AI driver from scratch")
    print("   🧪 'test'     - Test an existing trained model") 
    print("   🎲 'baseline' - Watch a random (untrained) driver fail")
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
        
        # Start training
        print(f"\n🚀 Starting training for {episodes} episodes...")
        trained_agent = train_racing_ai(episodes=episodes, show_training=show_visual)
        
        # Offer to test the newly trained agent
        test_new = input("\n🧪 Test the newly trained AI? (y/n): ").lower().strip() == 'y'
        if test_new:
            test_trained_ai('f1_race_ai_final_model.pth', num_test_episodes=5)
    
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
        
        # Run test
        test_trained_ai(selected_model, num_test_episodes=test_episodes)
    
    elif user_choice == 'baseline':
        print("\n🎲 BASELINE MODE SELECTED")
        print("-" * 30)
        
        baseline_episodes_input = input("🎮 Number of random episodes (default=10): ").strip()
        baseline_episodes = int(baseline_episodes_input) if baseline_episodes_input else 10
        
        print("\n⚠️  Warning: This will be painful to watch! 😅")
        input("Press Enter to continue...")
        
        test_random_driver(num_episodes=baseline_episodes)
    
    elif user_choice == 'test' and not saved_models:
        print("\n❌ No trained models found!")
        print("   💡 Please train a model first using 'train' option")
    
    else:
        print("\n❌ Invalid choice or no models available")
        print("   💡 Valid options: 'train', 'test', 'baseline'")
    
    print("\n🎉 Thanks for using the F1 Race AI Training System!")
    print("👋 Happy AI training! 🤖🏎️")