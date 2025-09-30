"""
📊 Real-Time Training Status System
==================================

Provides live training updates for dashboard and chart integration
"""

import json
import os
import time
from datetime import datetime
from typing import Dict, List, Optional
import threading

class TrainingStatusManager:
    """Manages real-time training status and metrics"""
    
    def __init__(self, status_file="results/training_status.json"):
        self.status_file = status_file
        self.status_lock = threading.Lock()
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(status_file), exist_ok=True)
        
        # Initialize status
        self.status = {
            "is_training": False,
            "training_mode": None,
            "current_episode": 0,
            "total_episodes": 0,
            "start_time": None,
            "last_update": None,
            "current_score": 0,
            "current_epsilon": 0.0,
            "best_score": 0,
            "average_score": 0.0,
            "recent_scores": [],
            "exploration_rates": [],
            "training_config": {},
            "model_info": {},
            "performance_metrics": {
                "episodes_per_minute": 0.0,
                "time_remaining_estimate": "Unknown",
                "total_training_time": 0.0
            }
        }
        
        # Load existing status if available
        self._load_status()
    
    def _load_status(self):
        """Load existing status from file"""
        try:
            if os.path.exists(self.status_file):
                with open(self.status_file, 'r') as f:
                    saved_status = json.load(f)
                    # Only load if not currently training
                    if not saved_status.get("is_training", False):
                        self.status.update(saved_status)
        except Exception as e:
            print(f"⚠️  Could not load training status: {e}")
    
    def _save_status(self):
        """Save current status to file"""
        try:
            with self.status_lock:
                with open(self.status_file, 'w') as f:
                    json.dump(self.status, f, indent=2)
        except Exception as e:
            print(f"⚠️  Could not save training status: {e}")
    
    def start_training(self, mode: str, total_episodes: int, config: dict, model_info: dict = None):
        """Mark training as started"""
        with self.status_lock:
            self.status.update({
                "is_training": True,
                "training_mode": mode,
                "current_episode": 0,
                "total_episodes": total_episodes,
                "start_time": datetime.now().isoformat(),
                "last_update": datetime.now().isoformat(),
                "training_config": config,
                "model_info": model_info or {},
                "recent_scores": [],
                "exploration_rates": []
            })
        self._save_status()
        print(f"📊 Training status: Started {mode} mode ({total_episodes} episodes)")
    
    def update_episode(self, episode: int, score: float, epsilon: float):
        """Update current episode progress"""
        with self.status_lock:
            self.status["current_episode"] = episode
            self.status["current_score"] = score
            self.status["current_epsilon"] = epsilon
            self.status["last_update"] = datetime.now().isoformat()
            
            # Update recent scores (keep last 100)
            self.status["recent_scores"].append(score)
            if len(self.status["recent_scores"]) > 100:
                self.status["recent_scores"] = self.status["recent_scores"][-100:]
            
            # Update exploration rates (keep last 100)
            self.status["exploration_rates"].append(epsilon)
            if len(self.status["exploration_rates"]) > 100:
                self.status["exploration_rates"] = self.status["exploration_rates"][-100:]
            
            # Update best score
            if score > self.status["best_score"]:
                self.status["best_score"] = score
            
            # Update average score
            if self.status["recent_scores"]:
                self.status["average_score"] = sum(self.status["recent_scores"]) / len(self.status["recent_scores"])
            
            # Calculate performance metrics
            self._update_performance_metrics()
        
        # Save every 10 episodes to avoid too frequent disk writes
        if episode % 10 == 0:
            self._save_status()
    
    def _update_performance_metrics(self):
        """Calculate performance metrics"""
        if not self.status["start_time"]:
            return
        
        try:
            start_time = datetime.fromisoformat(self.status["start_time"])
            current_time = datetime.now()
            elapsed_seconds = (current_time - start_time).total_seconds()
            
            if elapsed_seconds > 0:
                episodes_completed = self.status["current_episode"]
                episodes_per_second = episodes_completed / elapsed_seconds
                episodes_per_minute = episodes_per_second * 60
                
                self.status["performance_metrics"]["episodes_per_minute"] = episodes_per_minute
                self.status["performance_metrics"]["total_training_time"] = elapsed_seconds
                
                # Estimate time remaining
                remaining_episodes = self.status["total_episodes"] - episodes_completed
                if episodes_per_second > 0 and remaining_episodes > 0:
                    remaining_seconds = remaining_episodes / episodes_per_second
                    remaining_minutes = remaining_seconds / 60
                    
                    if remaining_minutes < 60:
                        self.status["performance_metrics"]["time_remaining_estimate"] = f"{remaining_minutes:.1f} minutes"
                    else:
                        remaining_hours = remaining_minutes / 60
                        self.status["performance_metrics"]["time_remaining_estimate"] = f"{remaining_hours:.1f} hours"
                else:
                    self.status["performance_metrics"]["time_remaining_estimate"] = "Calculating..."
        except Exception as e:
            print(f"⚠️  Error calculating metrics: {e}")
    
    def end_training(self, final_score: float = None):
        """Mark training as completed"""
        with self.status_lock:
            self.status["is_training"] = False
            self.status["last_update"] = datetime.now().isoformat()
            
            if final_score is not None:
                self.status["current_score"] = final_score
            
            # Calculate final metrics
            self._update_performance_metrics()
        
        self._save_status()
        print("📊 Training status: Completed")
    
    def get_status(self) -> dict:
        """Get current training status"""
        with self.status_lock:
            return self.status.copy()
    
    def get_dashboard_data(self) -> dict:
        """Get data formatted for dashboard display"""
        status = self.get_status()
        
        # Calculate additional dashboard metrics
        dashboard_data = {
            "training": {
                "is_active": status["is_training"],
                "mode": status["training_mode"],
                "progress": {
                    "current": status["current_episode"],
                    "total": status["total_episodes"],
                    "percentage": (status["current_episode"] / max(status["total_episodes"], 1)) * 100
                }
            },
            "current_metrics": {
                "score": status["current_score"],
                "epsilon": status["current_epsilon"],
                "best_score": status["best_score"],
                "average_score": round(status["average_score"], 2)
            },
            "performance": status["performance_metrics"],
            "charts": {
                "scores": status["recent_scores"][-50:],  # Last 50 for chart
                "exploration": status["exploration_rates"][-50:],
                "episodes": list(range(max(0, status["current_episode"]-49), status["current_episode"]+1))
            },
            "timestamps": {
                "started": status["start_time"],
                "last_update": status["last_update"]
            },
            "model_info": status["model_info"]
        }
        
        return dashboard_data

# Global instance for easy access
_status_manager = None

def get_status_manager() -> TrainingStatusManager:
    """Get global status manager instance"""
    global _status_manager
    if _status_manager is None:
        _status_manager = TrainingStatusManager()
    return _status_manager

def update_training_status(episode: int, score: float, epsilon: float):
    """Quick update function for training loops"""
    get_status_manager().update_episode(episode, score, epsilon)

def start_training_session(mode: str, total_episodes: int, config: dict, model_info: dict = None):
    """Quick start function for training sessions"""
    get_status_manager().start_training(mode, total_episodes, config, model_info)

def end_training_session(final_score: float = None):
    """Quick end function for training sessions"""
    get_status_manager().end_training(final_score)

# Chart update integration
def update_charts_with_status(chart_path: str = "results/charts/ai_training_progress.png"):
    """Update charts and sync with status system"""
    try:
        # This will be called from the training loop
        status = get_status_manager().get_status()
        
        # Create a marker file to indicate chart was updated
        chart_update_file = "results/charts/last_update.json"
        os.makedirs(os.path.dirname(chart_update_file), exist_ok=True)
        
        with open(chart_update_file, 'w') as f:
            json.dump({
                "last_chart_update": datetime.now().isoformat(),
                "episode": status["current_episode"],
                "chart_path": chart_path
            }, f, indent=2)
        
        print(f"📊 Chart update marker saved: episode {status['current_episode']}")
        
    except Exception as e:
        print(f"⚠️  Error updating chart status: {e}")