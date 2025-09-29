#!/usr/bin/env python3
"""
🧪 Quick Test Script for F1 Race AI Enhancements
=================================================

This script verifies that the new enhancements work correctly:
1. Framerate multiplier configuration
2. Periodic chart updates
3. Backwards compatibility

Run this to verify the changes before full training.
"""

import sys
import os
sys.path.append('src')

from src.environment import F1RaceEnvironment
from src.agent import DQNAgent
import numpy as np

def test_framerate_multiplier():
    """Test that framerate multiplier parameter works"""
    print("🧪 Testing framerate multiplier...")
    
    # Test different framerate configurations
    test_cases = [
        (100, True),   # Normal speed with visuals
        (200, True),   # 2x speed with visuals  
        (500, False),  # Max speed headless
        (50, True),    # Half speed with visuals
    ]
    
    for multiplier, render in test_cases:
        print(f"   Testing framerate={multiplier}%, render={render}")
        try:
            env = F1RaceEnvironment(render=render, framerate_multiplier=multiplier)
            # Verify the multiplier was set correctly
            assert env.framerate_multiplier == multiplier
            print(f"   ✅ Multiplier set to {env.framerate_multiplier}%")
            print(f"   ✅ Target FPS: {env.target_fps}")
            env.close()
        except Exception as e:
            print(f"   ❌ Error: {e}")
            return False
    
    print("✅ Framerate multiplier tests passed!")
    return True

def test_environment_compatibility():
    """Test that environment still works with old parameters"""
    print("🧪 Testing backwards compatibility...")
    
    try:
        # Test old-style initialization (should use default framerate)
        env = F1RaceEnvironment(render=False)
        assert env.framerate_multiplier == 100  # Default value
        env.close()
        print("✅ Backwards compatibility maintained!")
        return True
    except Exception as e:
        print(f"❌ Compatibility error: {e}")
        return False

def test_agent_chart_creation():
    """Test that agent can create charts"""
    print("🧪 Testing chart creation...")
    
    try:
        # Create a minimal agent
        agent = DQNAgent(state_size=5, action_size=3)
        
        # Add some dummy data
        agent.episode_scores = [0, 1, 2, 5, 8, 12, 10, 15, 18, 20]
        agent.training_losses = [1.0, 0.8, 0.6, 0.5, 0.4, 0.3, 0.25, 0.2, 0.18, 0.15]
        agent.exploration_rates = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
        
        # Test chart creation
        os.makedirs('test_results', exist_ok=True)
        chart_path = 'test_results/test_chart.png'
        agent.create_training_charts(out_path=chart_path)
        
        # Verify chart was created
        if os.path.exists(chart_path):
            print("✅ Chart creation successful!")
            # Clean up
            os.remove(chart_path)
            os.rmdir('test_results')
            return True
        else:
            print("❌ Chart file not created")
            return False
            
    except Exception as e:
        print(f"❌ Chart creation error: {e}")
        return False

def test_quick_environment_run():
    """Quick test of environment functionality"""
    print("🧪 Testing environment quick run...")
    
    try:
        # Create headless environment with high speed
        env = F1RaceEnvironment(render=False, framerate_multiplier=500)
        
        # Run a few steps
        state = env.reset()
        print(f"   Initial state shape: {state.shape}")
        
        total_reward = 0
        for step in range(10):
            action = np.random.choice(env.action_space_size)
            next_state, reward, done, info = env.step(action)
            total_reward += reward
            
            if done:
                break
        
        print(f"   Completed {step+1} steps, total reward: {total_reward:.2f}")
        env.close()
        print("✅ Environment quick run successful!")
        return True
        
    except Exception as e:
        print(f"❌ Environment run error: {e}")
        return False

def main():
    """Run all tests"""
    print("🏎️  F1 Race AI Enhancement Tests")
    print("=" * 40)
    
    tests = [
        test_framerate_multiplier,
        test_environment_compatibility, 
        test_agent_chart_creation,
        test_quick_environment_run
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            print()  # Add spacing between tests
        except Exception as e:
            print(f"❌ Test failed with exception: {e}\n")
    
    print("=" * 40)
    print(f"🎯 Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All tests passed! Enhancements are working correctly!")
        print("\n💡 Ready to use:")
        print("   python train_ai.py")
        return True
    else:
        print("⚠️  Some tests failed. Please check the implementation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)