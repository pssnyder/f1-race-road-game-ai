"""
🔄 Enhanced Trainer with Backward Compatibility
===============================================

This trainer allows you to:
1. Use your existing 20K model with new environment
2. Gradually transition to enhanced features
3. Compare old vs new performance

Author: Pat Snyder 💻
"""

import numpy as np
from environment import F1RaceEnvironment
from agent import DQNAgent
import os

def create_compatible_environment():
    """Create environment that provides backward-compatible state"""
    class CompatibleEnvironment(F1RaceEnvironment):
        def __init__(self, use_enhanced_features=True, **kwargs):
            super().__init__(**kwargs)
            self.use_enhanced_features = use_enhanced_features
            
            # Adjust state space size based on feature set
            if use_enhanced_features:
                self.state_space_size = 7
            else:
                self.state_space_size = 5
        
        def get_current_state(self):
            """Get state in either old (5) or new (7) format"""
            # Always calculate full 7-feature state
            full_state = super().get_current_state()
            
            if self.use_enhanced_features:
                return full_state  # Return all 7 features
            else:
                return full_state[:5]  # Return only first 5 features (backward compatible)
    
    return CompatibleEnvironment

def test_model_compatibility(model_path, episodes=5):
    """
    Test how your existing model performs with and without new features
    """
    print(f"🧪 TESTING MODEL COMPATIBILITY")
    print("=" * 50)
    
    results = {}
    
    # Test 1: Old model with old features (5-state)
    print("🔄 Test 1: Old model with original 5-state environment")
    CompatibleEnv = create_compatible_environment()
    env_old = CompatibleEnv(render=False, use_enhanced_features=False)
    
    agent_old = DQNAgent(state_size=5, action_size=3)
    agent_old.load_agent(model_path)
    agent_old.epsilon = 0.0  # No exploration for testing
    
    old_scores = []
    for ep in range(episodes):
        state = env_old.reset()
        total_reward = 0
        steps = 0
        
        while True:
            action = agent_old.choose_action(state, training_mode=False)
            state, reward, done, info = env_old.step(action)
            total_reward += reward
            steps += 1
            if done:
                break
        
        old_scores.append(info['score'])
        print(f"   Episode {ep+1}: Score={info['score']}, Steps={steps}")
    
    results['old_model_old_env'] = {
        'scores': old_scores,
        'average': np.mean(old_scores),
        'description': "Your 20K model in original environment"
    }
    
    env_old.close()
    
    # Test 2: Adaptation layer - use old model with feature adaptation
    print(f"\n🔧 Test 2: Old model with feature adaptation (simulated new environment)")
    env_new = CompatibleEnv(render=False, use_enhanced_features=True)
    
    adapted_scores = []
    for ep in range(episodes):
        state = env_new.reset()
        total_reward = 0
        steps = 0
        
        while True:
            # Adapt 7-feature state to 5-feature state for old model
            adapted_state = state[:5]  # Use first 5 features only
            action = agent_old.choose_action(adapted_state, training_mode=False)
            state, reward, done, info = env_new.step(action)
            total_reward += reward
            steps += 1
            if done:
                break
        
        adapted_scores.append(info['score'])
        print(f"   Episode {ep+1}: Score={info['score']}, Steps={steps}")
    
    results['old_model_adapted'] = {
        'scores': adapted_scores,
        'average': np.mean(adapted_scores),
        'description': "Your 20K model adapted to new environment"
    }
    
    env_new.close()
    
    # Summary comparison
    print(f"\n📊 COMPATIBILITY TEST RESULTS")
    print("=" * 50)
    for test_name, data in results.items():
        print(f"{data['description']}:")
        print(f"   Average Score: {data['average']:.2f}")
        print(f"   All Scores: {data['scores']}")
        print()
    
    # Performance comparison
    old_avg = results['old_model_old_env']['average']
    adapted_avg = results['old_model_adapted']['average']
    
    if adapted_avg > old_avg:
        improvement = ((adapted_avg - old_avg) / old_avg) * 100
        print(f"🎉 GOOD NEWS: Adaptation improved performance by {improvement:.1f}%!")
        print("✅ Your 20K model benefits from the enhanced environment!")
    elif adapted_avg < old_avg:
        decline = ((old_avg - adapted_avg) / old_avg) * 100
        print(f"⚠️ Performance declined by {decline:.1f}% with adaptation")
        print("💡 Enhanced features help, but new training would be better")
    else:
        print("📊 Performance remained the same")
        print("💡 Enhanced environment is compatible but no immediate benefit")
    
    return results

def train_with_backward_compatibility(old_model_path, episodes=1000):
    """
    Train a new model while leveraging knowledge from old model
    """
    print(f"🎓 TRAINING WITH BACKWARD COMPATIBILITY")
    print("=" * 50)
    
    CompatibleEnv = create_compatible_environment()
    env = CompatibleEnv(render=False, use_enhanced_features=True)
    
    # Create new agent for 7-feature environment
    new_agent = DQNAgent(state_size=7, action_size=3)
    
    # Load old agent for knowledge transfer
    old_agent = DQNAgent(state_size=5, action_size=3)
    old_agent.load_agent(old_model_path)
    old_agent.epsilon = 0.0  # No exploration - just provide guidance
    
    print(f"📚 Training new 7-feature model with guidance from 20K model")
    print(f"🎯 Episodes: {episodes}")
    
    scores = []
    guidance_weight = 0.3  # How much to follow old model's advice
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        steps = 0
        
        while True:
            # Get action from new model
            new_action = new_agent.choose_action(state, training_mode=True)
            
            # Get advice from old model (using adapted state)
            old_state = state[:5]
            old_action = old_agent.choose_action(old_state, training_mode=False)
            
            # Occasionally follow old model's advice (knowledge transfer)
            if np.random.random() < guidance_weight and episode < episodes // 2:
                action = old_action
            else:
                action = new_action
            
            # Execute action
            next_state, reward, done, info = env.step(action)
            
            # Store experience in new model
            new_agent.store_experience(state, action, reward, next_state, done)
            
            # Train new model
            if len(new_agent.memory) > new_agent.batch_size:
                new_agent.train_from_experience()
            
            state = next_state
            total_reward += reward
            steps += 1
            
            if done:
                break
        
        scores.append(info['score'])
        new_agent.episode_scores.append(info['score'])
        new_agent.decay_epsilon()
        
        # Gradually reduce guidance from old model
        if episode < episodes // 2:
            guidance_weight = 0.3 * (1 - episode / (episodes // 2))
        
        # Progress updates
        if episode % 100 == 0:
            avg_score = np.mean(scores[-100:]) if len(scores) >= 100 else np.mean(scores)
            print(f"Episode {episode}: Score={info['score']}, Avg={avg_score:.2f}, "
                  f"Guidance={guidance_weight:.3f}, Epsilon={new_agent.epsilon:.3f}")
    
    # Save the enhanced model
    os.makedirs('models/enhanced', exist_ok=True)
    enhanced_model_path = 'models/enhanced/enhanced_7_feature_model.pth'
    new_agent.save_agent(enhanced_model_path)
    
    env.close()
    
    print(f"\n✅ Enhanced model saved to: {enhanced_model_path}")
    print(f"📊 Final average score: {np.mean(scores[-100:]):.2f}")
    
    return new_agent, enhanced_model_path

# 🧪 EXAMPLE USAGE
if __name__ == "__main__":
    print("🔄 Backward Compatible Training System")
    print("=" * 50)
    
    # Check for existing models
    model_paths = []
    for subdir in ['models/final', 'models']:
        if os.path.isdir(subdir):
            for f in os.listdir(subdir):
                if f.endswith('.pth'):
                    model_paths.append(os.path.join(subdir, f))
    
    if model_paths:
        print("📂 Available models:")
        for i, path in enumerate(model_paths):
            print(f"   {i}: {path}")
        
        # Use the first model found (or let user choose)
        model_path = model_paths[0]
        print(f"\n🎯 Using model: {model_path}")
        
        # Test compatibility
        print(f"\n🧪 Testing compatibility...")
        results = test_model_compatibility(model_path, episodes=3)
        
        # Offer to train enhanced model
        train_choice = input(f"\n🎓 Train enhanced model using knowledge transfer? (y/n): ")
        if train_choice.lower() == 'y':
            episodes = int(input("📊 Number of episodes (default=1000): ") or "1000")
            enhanced_agent, enhanced_path = train_with_backward_compatibility(
                model_path, episodes=episodes
            )
            print(f"\n🎉 Enhanced training complete!")
    else:
        print("❌ No trained models found. Train a model first.")