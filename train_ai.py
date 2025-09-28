import numpy as np
import matplotlib.pyplot as plt
from f1_race_env import F1RaceEnvironment
from dqn_agent import DQNAgent
import time
import os

def train_agent(episodes=2000, target_update_frequency=100, save_frequency=500, render_training=False):
    """Train the DQN agent on F1 Race game"""
    
    print("Initializing F1 Race AI Training...")
    
    # Create environment and agent
    env = F1RaceEnvironment(render=render_training)
    agent = DQNAgent(
        state_size=env.state_space_size,
        action_size=env.action_space_size,
        lr=0.001,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        memory_size=10000,
        batch_size=32
    )
    
    print(f"Environment: State space size = {env.state_space_size}, Action space size = {env.action_space_size}")
    print(f"Agent: Learning rate = {agent.lr}, Gamma = {agent.gamma}")
    print(f"Training for {episodes} episodes...")
    print("-" * 50)
    
    # Training metrics
    episode_scores = []
    episode_lengths = []
    best_score = 0
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        steps = 0
        start_time = time.time()
        
        while True:
            # Choose action
            action = agent.act(state)
            
            # Take action
            next_state, reward, done, info = env.step(action)
            
            # Store experience
            agent.remember(state, action, reward, next_state, done)
            
            # Update state
            state = next_state
            total_reward += reward
            steps += 1
            
            # Render if enabled
            if render_training:
                env.render()
                time.sleep(0.01)  # Slow down for viewing
            
            # Train agent
            if len(agent.memory) > agent.batch_size:
                agent.replay()
            
            if done:
                break
        
        # Record episode metrics
        episode_scores.append(info['score'])
        episode_lengths.append(steps)
        agent.scores.append(info['score'])
        
        # Update target network
        if episode % target_update_frequency == 0:
            agent.update_target_network()
        
        # Print progress
        if episode % 100 == 0 or info['score'] > best_score:
            episode_time = time.time() - start_time
            avg_score = np.mean(episode_scores[-100:]) if len(episode_scores) >= 100 else np.mean(episode_scores)
            avg_length = np.mean(episode_lengths[-100:]) if len(episode_lengths) >= 100 else np.mean(episode_lengths)
            
            print(f"Episode {episode:4d} | Score: {info['score']:3d} | Steps: {steps:4d} | "
                  f"Avg Score: {avg_score:.2f} | Avg Length: {avg_length:.1f} | "
                  f"Epsilon: {agent.epsilon:.3f} | Time: {episode_time:.2f}s")
            
            if info['score'] > best_score:
                best_score = info['score']
                print(f"New best score: {best_score}!")
        
        # Save model periodically
        if episode % save_frequency == 0 and episode > 0:
            agent.save(f'dqn_model_episode_{episode}.pth')
    
    # Save final model
    agent.save('dqn_model_final.pth')
    
    # Plot training results
    print("\nTraining completed!")
    print(f"Best score achieved: {best_score}")
    print(f"Final average score (last 100 episodes): {np.mean(episode_scores[-100:]):.2f}")
    
    agent.plot_training_metrics()
    
    env.close()
    return agent

def test_agent(model_path, num_episodes=5, render=True):
    """Test a trained agent"""
    print(f"Testing trained agent from {model_path}")
    
    # Create environment and agent
    env = F1RaceEnvironment(render=render)
    agent = DQNAgent(
        state_size=env.state_space_size,
        action_size=env.action_space_size
    )
    
    # Load trained model
    agent.load(model_path)
    agent.epsilon = 0.0  # No exploration during testing
    
    test_scores = []
    
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        steps = 0
        
        print(f"\nTest Episode {episode + 1}")
        
        while True:
            # Choose action (no exploration)
            action = agent.act(state, training=False)
            
            # Take action
            next_state, reward, done, info = env.step(action)
            
            state = next_state
            total_reward += reward
            steps += 1
            
            if render:
                env.render()
                time.sleep(0.03)  # Slow down for viewing
            
            if done:
                break
        
        test_scores.append(info['score'])
        print(f"Episode {episode + 1} completed: Score = {info['score']}, Steps = {steps}, Total Reward = {total_reward:.2f}")
    
    print(f"\nTest Results:")
    print(f"Average Score: {np.mean(test_scores):.2f}")
    print(f"Best Score: {max(test_scores)}")
    print(f"All Scores: {test_scores}")
    
    env.close()
    return test_scores

def random_baseline(num_episodes=10):
    """Test random agent as baseline"""
    print("Testing random agent baseline...")
    
    env = F1RaceEnvironment(render=True)
    random_scores = []
    
    for episode in range(num_episodes):
        state = env.reset()
        steps = 0
        
        while True:
            # Random action
            action = np.random.choice(env.action_space_size)
            
            _, _, done, info = env.step(action)
            steps += 1
            
            env.render()
            time.sleep(0.03)
            
            if done:
                break
        
        random_scores.append(info['score'])
        print(f"Random Episode {episode + 1}: Score = {info['score']}, Steps = {steps}")
    
    print(f"Random Agent Average Score: {np.mean(random_scores):.2f}")
    env.close()
    return random_scores

if __name__ == "__main__":
    print("F1 Race AI Training System")
    print("=" * 50)
    
    # Check if we have a saved model to test
    model_files = [f for f in os.listdir('.') if f.endswith('.pth')]
    
    if model_files:
        print(f"Found saved models: {model_files}")
        choice = input("Enter 'train' to train new model, 'test' to test existing model, or 'baseline' for random baseline: ").lower()
    else:
        print("No saved models found.")
        choice = input("Enter 'train' to train new model or 'baseline' for random baseline: ").lower()
    
    if choice == 'train':
        # Train the agent
        render_choice = input("Render training? (y/n): ").lower() == 'y'
        episodes = int(input("Number of episodes (default 2000): ") or "2000")
        
        agent = train_agent(episodes=episodes, render_training=render_choice)
        
        # Test the trained agent
        test_choice = input("Test the trained agent? (y/n): ").lower() == 'y'
        if test_choice:
            test_agent('dqn_model_final.pth', num_episodes=5)
    
    elif choice == 'test' and model_files:
        # Test existing model
        print("Available models:")
        for i, model in enumerate(model_files):
            print(f"{i}: {model}")
        
        model_idx = int(input("Select model index: "))
        model_path = model_files[model_idx]
        
        num_episodes = int(input("Number of test episodes (default 5): ") or "5")
        test_agent(model_path, num_episodes=num_episodes)
    
    elif choice == 'baseline':
        # Test random baseline
        num_episodes = int(input("Number of baseline episodes (default 10): ") or "10")
        random_baseline(num_episodes=num_episodes)
    
    else:
        print("Invalid choice or no models available.")
    
    print("Program finished!")