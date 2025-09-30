#!/usr/bin/env python3
"""
📊 Enhanced Training Data Logger
================================

Comprehensive logging system for F1 Race AI training data.
Saves detailed metrics beyond just the visual charts.
"""

import json
import csv
import time
from datetime import datetime
from pathlib import Path


class TrainingLogger:
    """Enhanced training data logger that captures detailed metrics"""
    
    def __init__(self, log_dir="results/logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create timestamped log files
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_id = timestamp
        
        # Initialize log files
        self.csv_file = self.log_dir / f"training_log_{timestamp}.csv"
        self.json_file = self.log_dir / f"training_summary_{timestamp}.json"
        self.loss_file = self.log_dir / f"loss_details_{timestamp}.csv"
        
        self._init_csv_files()
        self.session_data = {
            "start_time": datetime.now().isoformat(),
            "episodes": [],
            "configuration": {},
            "performance_analysis": {}
        }
    
    def _init_csv_files(self):
        """Initialize CSV files with headers"""
        # Episode data CSV
        with open(self.csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'episode', 'score', 'steps', 'episode_time', 'epsilon', 
                'avg_score_100', 'best_score_so_far', 'total_time',
                'car_x', 'car_y', 'obstacle_x', 'obstacle_y', 'distance_to_obstacle',
                'game_speed', 'actions_taken'
            ])
        
        # Loss details CSV
        with open(self.loss_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'episode', 'training_step', 'loss_value', 'q_value_mean', 
                'q_value_std', 'target_mean', 'prediction_mean', 'timestamp'
            ])
    
    def log_episode(self, episode_data):
        """Log detailed episode data"""
        # Add to CSV
        with open(self.csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                episode_data.get('episode', 0),
                episode_data.get('score', 0),
                episode_data.get('steps', 0),
                episode_data.get('episode_time', 0),
                episode_data.get('epsilon', 0),
                episode_data.get('avg_score_100', 0),
                episode_data.get('best_score_so_far', 0),
                episode_data.get('total_time', 0),
                episode_data.get('final_car_x', 0),
                episode_data.get('final_car_y', 0),
                episode_data.get('final_obstacle_x', 0),
                episode_data.get('final_obstacle_y', 0),
                episode_data.get('distance_to_obstacle', 0),
                episode_data.get('game_speed', 0),
                episode_data.get('actions_taken', '[]')
            ])
        
        # Add to session data
        self.session_data["episodes"].append(episode_data)
    
    def log_training_loss(self, episode, training_step, loss_data):
        """Log detailed training loss information"""
        with open(self.loss_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                episode,
                training_step,
                loss_data.get('loss', 0),
                loss_data.get('q_mean', 0),
                loss_data.get('q_std', 0),
                loss_data.get('target_mean', 0),
                loss_data.get('prediction_mean', 0),
                datetime.now().isoformat()
            ])
    
    def log_configuration(self, config):
        """Log training configuration"""
        self.session_data["configuration"] = config
    
    def analyze_performance(self, all_scores, training_losses):
        """Analyze performance patterns and identify issues"""
        if len(all_scores) < 100:
            return
        
        import numpy as np
        
        # Performance analysis
        scores_array = np.array(all_scores)
        recent_scores = scores_array[-1000:] if len(scores_array) >= 1000 else scores_array
        
        analysis = {
            "peak_performance": {
                "max_score": float(np.max(scores_array)),
                "max_score_episode": int(np.argmax(scores_array)),
                "peak_window_start": max(0, int(np.argmax(scores_array)) - 500),
                "peak_window_end": min(len(scores_array), int(np.argmax(scores_array)) + 500)
            },
            "performance_trends": {
                "overall_improvement": float(np.mean(recent_scores) - np.mean(scores_array[:100])),
                "recent_trend": float(np.mean(recent_scores[-100:]) - np.mean(recent_scores[-500:-400])) if len(recent_scores) >= 500 else 0,
                "volatility": float(np.std(recent_scores)),
                "stability_score": float(1.0 / (1.0 + np.std(recent_scores) / max(1.0, np.mean(recent_scores))))
            },
            "loss_analysis": {
                "loss_volatility": float(np.std(training_losses)) if training_losses else 0,
                "average_loss": float(np.mean(training_losses)) if training_losses else 0,
                "loss_spikes": int(np.sum(np.array(training_losses) > (np.mean(training_losses) + 2 * np.std(training_losses)))) if training_losses else 0
            },
            "recommendations": self._generate_recommendations(scores_array, training_losses)
        }
        
        self.session_data["performance_analysis"] = analysis
        return analysis
    
    def _generate_recommendations(self, scores, losses):
        """Generate training recommendations based on analysis"""
        recommendations = []
        
        if len(scores) < 100:
            return recommendations
        
        import numpy as np
        
        # Check for performance degradation
        if len(scores) >= 2000:
            peak_idx = np.argmax(scores)
            if peak_idx < len(scores) * 0.8:  # Peak happened in first 80% of training
                recent_avg = np.mean(scores[-500:])
                peak_window_avg = np.mean(scores[max(0, peak_idx-250):peak_idx+250])
                if recent_avg < peak_window_avg * 0.7:  # Recent performance < 70% of peak
                    recommendations.append({
                        "issue": "performance_degradation",
                        "description": "Performance dropped significantly after peak",
                        "suggestion": f"Consider reverting to checkpoint around episode {peak_idx}",
                        "severity": "high"
                    })
        
        # Check for high loss volatility
        if losses and len(losses) > 100:
            loss_std = np.std(losses)
            loss_mean = np.mean(losses)
            if loss_std > loss_mean:  # High volatility
                recommendations.append({
                    "issue": "training_instability", 
                    "description": "High loss volatility detected",
                    "suggestion": "Consider reducing learning rate or increasing batch size",
                    "severity": "medium"
                })
        
        # Check for plateau
        if len(scores) >= 1000:
            recent_trend = np.mean(scores[-200:]) - np.mean(scores[-400:-200])
            if abs(recent_trend) < 0.1:  # Very little change
                recommendations.append({
                    "issue": "performance_plateau",
                    "description": "Performance has plateaued",
                    "suggestion": "Consider increasing exploration rate or adjusting reward function",
                    "severity": "low"
                })
        
        return recommendations
    
    def save_session_summary(self):
        """Save complete session summary to JSON"""
        self.session_data["end_time"] = datetime.now().isoformat()
        
        with open(self.json_file, 'w') as f:
            json.dump(self.session_data, f, indent=2)
        
        print(f"📊 Training logs saved:")
        print(f"   📈 Episode data: {self.csv_file}")
        print(f"   🎯 Loss details: {self.loss_file}")
        print(f"   📋 Session summary: {self.json_file}")
    
    def get_checkpoint_recommendation(self, all_scores):
        """Get recommendation for best checkpoint to use"""
        if len(all_scores) < 500:
            return None
        
        import numpy as np
        
        # Find the episode with best sustained performance (not just single peak)
        window_size = 100
        best_avg = -1
        best_episode = -1
        
        for i in range(window_size, len(all_scores) - window_size):
            window_avg = np.mean(all_scores[i-window_size:i+window_size])
            if window_avg > best_avg:
                best_avg = window_avg
                best_episode = i
        
        return {
            "recommended_episode": best_episode,
            "average_performance": float(best_avg),
            "confidence": "high" if best_episode < len(all_scores) * 0.9 else "low"
        }


# Global logger instance
logger = None

def get_logger():
    """Get or create global logger instance"""
    global logger
    if logger is None:
        logger = TrainingLogger()
    return logger