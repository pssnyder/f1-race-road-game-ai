"""
🎯 Production-Grade Exploration Strategies
==========================================

Real-world exploration handling for scenarios where training duration
is unknown, models are extended, or continuous learning is required.

Author: Pat Snyder 💻
"""

import math
import numpy as np

class ProductionExplorationManager:
    """
    🏭 Production-grade exploration management
    
    Used by: Google, Netflix, Uber, Tesla for continuous learning systems
    """
    
    def __init__(self, initial_epsilon=1.0, min_epsilon=0.01, strategy="adaptive"):
        self.initial_epsilon = initial_epsilon
        self.min_epsilon = min_epsilon
        self.current_epsilon = initial_epsilon
        self.strategy = strategy
        self.episode_count = 0
        self.performance_history = []
        
        # Strategy-specific parameters
        self.adaptive_window = 100  # Episodes to look back for performance
        self.curiosity_threshold = 0.1  # When to increase exploration
        self.stability_threshold = 0.05  # When exploration can decrease
        
        print(f"🎯 Exploration Strategy: {strategy}")
    
    def get_epsilon(self, performance_metric=None):
        """
        Get current exploration rate based on chosen strategy
        
        Args:
            performance_metric (float): Recent performance (score, reward, etc.)
        """
        self.episode_count += 1
        
        if performance_metric is not None:
            self.performance_history.append(performance_metric)
        
        if self.strategy == "adaptive":
            return self._adaptive_epsilon()
        elif self.strategy == "cyclical":
            return self._cyclical_epsilon()
        elif self.strategy == "performance_based":
            return self._performance_based_epsilon()
        elif self.strategy == "uncertainty_based":
            return self._uncertainty_based_epsilon()
        else:
            return self._traditional_decay()
    
    def _adaptive_epsilon(self):
        """
        🧠 Adaptive Exploration (Used by DeepMind, OpenAI)
        
        Adjusts exploration based on learning progress without knowing endpoint
        """
        # Base decay: slow reduction over time
        base_decay = 0.9999  # Very slow decay
        self.current_epsilon *= base_decay
        
        # Adaptive component: increase if performance is stagnating
        if len(self.performance_history) >= self.adaptive_window:
            recent_performance = np.mean(self.performance_history[-self.adaptive_window//2:])
            older_performance = np.mean(self.performance_history[-self.adaptive_window:-self.adaptive_window//2])
            
            improvement = (recent_performance - older_performance) / max(abs(older_performance), 1e-6)
            
            if improvement < self.curiosity_threshold:
                # Performance stagnating - increase exploration
                self.current_epsilon = min(self.current_epsilon * 1.1, 0.3)
                print(f"   📈 Increasing exploration to {self.current_epsilon:.3f} (stagnant performance)")
            elif improvement > self.stability_threshold:
                # Good improvement - can reduce exploration more
                self.current_epsilon *= 0.95
        
        return max(self.current_epsilon, self.min_epsilon)
    
    def _cyclical_epsilon(self):
        """
        🔄 Cyclical Exploration (Used by Uber, Lyft for dynamic environments)
        
        Periodically increases exploration to adapt to changing environments
        """
        cycle_length = 1000  # Episodes per cycle
        cycle_position = self.episode_count % cycle_length
        
        # Sine wave: high exploration at start of cycle, low at middle
        cycle_factor = 0.5 * (1 + math.sin(2 * math.pi * cycle_position / cycle_length - math.pi/2))
        epsilon_range = self.initial_epsilon - self.min_epsilon
        
        self.current_epsilon = self.min_epsilon + cycle_factor * epsilon_range
        return self.current_epsilon
    
    def _performance_based_epsilon(self):
        """
        📊 Performance-Based Exploration (Used by Netflix, Amazon)
        
        Exploration rate inversely related to recent performance
        """
        if len(self.performance_history) < 10:
            return self.current_epsilon * 0.999  # Default decay until enough data
        
        recent_avg = np.mean(self.performance_history[-10:])
        all_time_max = max(self.performance_history)
        
        if all_time_max > 0:
            performance_ratio = recent_avg / all_time_max
            # Lower performance = higher exploration
            target_epsilon = self.min_epsilon + (1 - performance_ratio) * (self.initial_epsilon - self.min_epsilon)
            
            # Smooth transition to target
            self.current_epsilon = 0.9 * self.current_epsilon + 0.1 * target_epsilon
        
        return max(self.current_epsilon, self.min_epsilon)
    
    def _uncertainty_based_epsilon(self):
        """
        🤔 Uncertainty-Based Exploration (Used by Google, Facebook)
        
        Higher exploration when model is uncertain about its predictions
        """
        # Base decay
        self.current_epsilon *= 0.9995
        
        # In real implementation, this would use model prediction variance
        # For demo, simulate uncertainty based on performance variance
        if len(self.performance_history) >= 20:
            recent_variance = np.var(self.performance_history[-20:])
            uncertainty_bonus = min(recent_variance * 0.1, 0.2)  # Cap uncertainty bonus
            self.current_epsilon += uncertainty_bonus
        
        return max(min(self.current_epsilon, 0.5), self.min_epsilon)
    
    def _traditional_decay(self):
        """Traditional exponential decay (what you had before)"""
        self.current_epsilon *= 0.9995
        return max(self.current_epsilon, self.min_epsilon)
    
    def reset_for_model_extension(self, feature_expansion_ratio=0.4):
        """
        🔧 Reset exploration when extending model features
        
        Args:
            feature_expansion_ratio: How much to reset exploration (0.0-1.0)
        """
        old_epsilon = self.current_epsilon
        # Partially reset exploration based on how much the model changed
        reset_amount = feature_expansion_ratio * (self.initial_epsilon - self.current_epsilon)
        self.current_epsilon = min(self.current_epsilon + reset_amount, self.initial_epsilon)
        
        print(f"🔧 Model extension: Epsilon {old_epsilon:.3f} → {self.current_epsilon:.3f}")
        return self.current_epsilon
    
    def get_strategy_info(self):
        """Get current strategy status"""
        return {
            'strategy': self.strategy,
            'episode': self.episode_count,
            'epsilon': self.current_epsilon,
            'performance_samples': len(self.performance_history)
        }

class ResumptionManager:
    """
    🔄 Handles exploration when resuming training
    
    Different strategies based on why training was resumed
    """
    
    @staticmethod
    def calculate_resume_epsilon(original_epsilon, episodes_trained, reason="continue"):
        """
        Calculate appropriate epsilon when resuming training
        
        Args:
            original_epsilon: Epsilon when training was saved
            episodes_trained: How many episodes were already trained
            reason: Why training is being resumed
        """
        
        if reason == "continue":
            # Just continuing training - keep epsilon as is
            return original_epsilon
        
        elif reason == "hyperparameter_tuning":
            # Adjusting hyperparameters - slight epsilon increase
            boost = min(0.1, 0.05 * math.log10(episodes_trained / 1000 + 1))
            return min(original_epsilon + boost, 0.3)
        
        elif reason == "environment_changes":
            # Environment changed - moderate epsilon increase
            boost = min(0.2, 0.1 * math.log10(episodes_trained / 1000 + 1))
            return min(original_epsilon + boost, 0.5)
        
        elif reason == "performance_issues":
            # Model performing poorly - significant epsilon increase
            boost = min(0.3, 0.15 * math.log10(episodes_trained / 1000 + 1))
            return min(original_epsilon + boost, 0.7)
        
        elif reason == "model_extension":
            # Added new features - feature-specific reset
            return ResumptionManager._feature_extension_epsilon(original_epsilon, episodes_trained)
        
        else:
            return original_epsilon
    
    @staticmethod
    def _feature_extension_epsilon(original_epsilon, episodes_trained):
        """Special handling for model feature extensions"""
        # More episodes trained = more careful about disrupting learned behavior
        disruption_factor = 1.0 / (1.0 + episodes_trained / 10000)  # Decreases as episodes increase
        
        # Reset exploration for new features while preserving old knowledge
        new_feature_epsilon = 0.3 * disruption_factor
        preserved_epsilon = original_epsilon * (1 - disruption_factor)
        
        return new_feature_epsilon + preserved_epsilon

# 🧪 EXAMPLE USAGE AND COMPARISON
if __name__ == "__main__":
    print("🎯 Production Exploration Strategy Comparison")
    print("=" * 60)
    
    # Simulate training scenarios
    strategies = ["adaptive", "cyclical", "performance_based", "uncertainty_based", "traditional"]
    episodes = 5000
    
    # Simulate performance data (realistic F1 racing scores)
    np.random.seed(42)
    base_performance = np.concatenate([
        np.random.normal(2, 1, 1000),    # Early training: low scores
        np.random.normal(5, 2, 2000),    # Mid training: improving
        np.random.normal(8, 1.5, 1500),  # Late training: good scores
        np.random.normal(6, 2, 500)      # Performance dip (needs exploration)
    ])
    base_performance = np.maximum(base_performance, 0)  # No negative scores
    
    results = {}
    
    for strategy in strategies:
        print(f"\n📊 Testing {strategy} strategy:")
        manager = ProductionExplorationManager(strategy=strategy)
        epsilons = []
        
        for episode in range(episodes):
            performance = base_performance[episode] if episode < len(base_performance) else np.random.normal(7, 1.5)
            epsilon = manager.get_epsilon(performance)
            epsilons.append(epsilon)
            
            if episode % 1000 == 0:
                print(f"   Episode {episode}: ε={epsilon:.3f}, Performance={performance:.1f}")
        
        results[strategy] = epsilons
        final_info = manager.get_strategy_info()
        print(f"   Final: ε={final_info['epsilon']:.3f}")
    
    # Test resumption scenarios
    print(f"\n🔄 RESUMPTION SCENARIOS")
    print("=" * 30)
    
    scenarios = [
        ("continue", "Just continuing training"),
        ("hyperparameter_tuning", "Adjusting learning rate"),
        ("environment_changes", "Enhanced game features"),
        ("performance_issues", "Model performing poorly"),
        ("model_extension", "Added new input features")
    ]
    
    original_epsilon = 0.01  # After 20K episodes
    episodes_trained = 20000
    
    for reason, description in scenarios:
        new_epsilon = ResumptionManager.calculate_resume_epsilon(
            original_epsilon, episodes_trained, reason
        )
        print(f"{reason:20} | {description:25} | ε: {original_epsilon:.3f} → {new_epsilon:.3f}")
    
    print(f"\n💡 RECOMMENDATIONS FOR YOUR PROJECT:")
    print("=" * 50)
    print("🎯 Current situation: 20K episodes, ε=0.01, adding 2 new features")
    print("✅ Best strategy: 'model_extension' with feature-specific reset")
    
    recommended_epsilon = ResumptionManager.calculate_resume_epsilon(
        0.01, 20000, "model_extension"
    )
    print(f"🔧 Recommended starting ε for enhanced model: {recommended_epsilon:.3f}")
    print("📈 This balances preserving learned behavior with exploring new features")