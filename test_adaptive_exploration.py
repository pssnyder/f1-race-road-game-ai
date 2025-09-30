#!/usr/bin/env python3
"""
🧪 Test Script for Adaptive Exploration System
==============================================

This script demonstrates how the new adaptive exploration system 
scales to different training lengths and decay types.

Author: Pat Snyder 💻
"""

import math
import matplotlib.pyplot as plt
import numpy as np

def calculate_exploration_curve(episodes, completion_ratio=0.8, decay_type="exponential"):
    """
    Calculate exploration curve for given parameters
    
    Args:
        episodes (int): Total number of episodes
        completion_ratio (float): Fraction of training when exploration reaches minimum
        decay_type (str): "exponential" or "linear"
    
    Returns:
        tuple: (episode_numbers, epsilon_values)
    """
    EXPLORATION_START = 1.0
    EXPLORATION_END = 0.01
    
    exploration_end_episode = int(episodes * completion_ratio)
    episode_numbers = list(range(episodes + 1))
    epsilon_values = []
    
    if decay_type.lower() == "linear":
        # Linear decay
        linear_decay_per_episode = (EXPLORATION_START - EXPLORATION_END) / exploration_end_episode
        for episode in episode_numbers:
            if episode <= exploration_end_episode:
                epsilon = max(EXPLORATION_START - (linear_decay_per_episode * episode), EXPLORATION_END)
            else:
                epsilon = EXPLORATION_END
            epsilon_values.append(epsilon)
    else:
        # Exponential decay
        if exploration_end_episode > 0:
            decay_rate = math.pow(EXPLORATION_END / EXPLORATION_START, 1.0 / exploration_end_episode)
        else:
            decay_rate = 0.9995
            
        epsilon = EXPLORATION_START
        for episode in episode_numbers:
            if episode <= exploration_end_episode and epsilon > EXPLORATION_END:
                epsilon *= decay_rate
                epsilon = max(epsilon, EXPLORATION_END)
            else:
                epsilon = EXPLORATION_END
            epsilon_values.append(epsilon)
    
    return episode_numbers, epsilon_values

def plot_comparison():
    """Generate comparison plots showing adaptive exploration for different scenarios"""
    
    # Test scenarios
    scenarios = [
        (1000, "1K Episodes"),
        (5000, "5K Episodes"), 
        (10000, "10K Episodes"),
        (50000, "50K Episodes")
    ]
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('🎯 Adaptive Exploration System Comparison', fontsize=16, fontweight='bold')
    
    # Plot exponential decay for different episode counts
    ax1.set_title('📉 Exponential Decay (80% Completion)')
    for episodes, label in scenarios:
        episode_nums, epsilon_vals = calculate_exploration_curve(episodes, 0.8, "exponential")
        ax1.plot(np.array(episode_nums) / episodes, epsilon_vals, label=label, linewidth=2)
    ax1.set_xlabel('Training Progress (Fraction)')
    ax1.set_ylabel('Exploration Rate (Epsilon)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.05)
    
    # Plot linear decay for different episode counts
    ax2.set_title('📉 Linear Decay (80% Completion)')
    for episodes, label in scenarios:
        episode_nums, epsilon_vals = calculate_exploration_curve(episodes, 0.8, "linear")
        ax2.plot(np.array(episode_nums) / episodes, epsilon_vals, label=label, linewidth=2)
    ax2.set_xlabel('Training Progress (Fraction)')
    ax2.set_ylabel('Exploration Rate (Epsilon)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.05)
    
    # Compare exponential vs linear for 10K episodes
    ax3.set_title('🔄 Exponential vs Linear (10K Episodes)')
    episodes = 10000
    
    # Different completion ratios
    for ratio, style in [(0.6, '--'), (0.8, '-'), (1.0, ':')]:
        # Exponential
        episode_nums, epsilon_vals = calculate_exploration_curve(episodes, ratio, "exponential")
        ax3.plot(episode_nums, epsilon_vals, label=f'Exp {ratio*100:.0f}%', linestyle=style, color='blue', linewidth=2)
        
        # Linear
        episode_nums, epsilon_vals = calculate_exploration_curve(episodes, ratio, "linear")
        ax3.plot(episode_nums, epsilon_vals, label=f'Lin {ratio*100:.0f}%', linestyle=style, color='red', linewidth=2)
    
    ax3.set_xlabel('Episode Number')
    ax3.set_ylabel('Exploration Rate (Epsilon)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 1.05)
    
    # Show the "old system" problem
    ax4.set_title('❌ Old Fixed Decay vs ✅ New Adaptive')
    
    # Old system with fixed 0.9995 decay
    episodes_list = [1000, 5000, 10000]
    old_decay = 0.9995
    
    for episodes in episodes_list:
        # Old system
        old_epsilon = []
        epsilon = 1.0
        for episode in range(episodes + 1):
            old_epsilon.append(max(epsilon, 0.01))
            epsilon *= old_decay
        ax4.plot(range(episodes + 1), old_epsilon, label=f'Old {episodes}', linestyle='--', alpha=0.7)
        
        # New adaptive system
        episode_nums, epsilon_vals = calculate_exploration_curve(episodes, 0.8, "exponential")
        ax4.plot(episode_nums, epsilon_vals, label=f'New {episodes}', linewidth=2)
    
    ax4.set_xlabel('Episode Number')
    ax4.set_ylabel('Exploration Rate (Epsilon)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(0, 1.05)
    
    plt.tight_layout()
    plt.savefig('adaptive_exploration_comparison.png', dpi=300, bbox_inches='tight')
    print("📊 Comparison chart saved as 'adaptive_exploration_comparison.png'")
    plt.show()

def demonstrate_scaling():
    """Show numerical examples of how exploration scales"""
    
    print("🎯 ADAPTIVE EXPLORATION SCALING DEMONSTRATION")
    print("=" * 60)
    
    scenarios = [
        (500, "Quick Test"),
        (2000, "Standard Training"),
        (10000, "Deep Training"),
        (50000, "Extended Training")
    ]
    
    print(f"{'Scenario':<20} {'Episodes':<10} {'End Episode':<12} {'Decay Rate':<15} {'Final ε':<10}")
    print("-" * 70)
    
    for episodes, name in scenarios:
        exploration_end_episode = int(episodes * 0.8)
        decay_rate = math.pow(0.01 / 1.0, 1.0 / exploration_end_episode)
        
        # Calculate final epsilon after the exploration period
        final_epsilon = 1.0 * (decay_rate ** exploration_end_episode)
        
        print(f"{name:<20} {episodes:<10} {exploration_end_episode:<12} {decay_rate:<15.6f} {final_epsilon:<10.4f}")
    
    print("\n🔍 Key Benefits:")
    print("✅ All scenarios reach ~0.01 exploration at 80% completion")
    print("✅ Decay rate automatically adjusts to training length")
    print("✅ Consistent exploration curves regardless of episode count")
    print("✅ No more guessing optimal decay rates!")

if __name__ == "__main__":
    print("🧪 Testing Adaptive Exploration System")
    print("=" * 50)
    
    # Show numerical scaling
    demonstrate_scaling()
    
    print("\n📊 Generating comparison charts...")
    
    try:
        # Generate visual comparison
        plot_comparison()
        
        print("\n🎉 Test completed successfully!")
        print("💡 The new system automatically scales exploration to any training length!")
        
    except ImportError as e:
        print(f"⚠️  Matplotlib not available: {e}")
        print("📊 Charts cannot be generated, but numerical demonstration completed.")
    except Exception as e:
        print(f"❌ Error generating charts: {e}")
        print("📊 But the numerical demonstration shows the concept works!")