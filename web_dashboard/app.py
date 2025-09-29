#!/usr/bin/env python3
"""
🌐 F1 Race AI Training Dashboard
===============================

Real-time web dashboard to monitor AI training progress!

This lightweight Flask app provides a beautiful, auto-refreshing
web interface to track your AI's learning journey without
disrupting the training process.

Features:
🏎️ Live training charts (auto-updated)
📊 Real-time statistics and metrics
⚡ Training configuration display  
📈 Episode progress tracking
🎯 Best score highlights
🧠 Current exploration rate

Just run this alongside your training and visit:
http://localhost:5000

Author: F1 Race AI Project
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path

import torch
from flask import Flask, jsonify, render_template

app = Flask(__name__)

# Configuration
PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
CHARTS_DIR = RESULTS_DIR / "charts"
MODELS_DIR = PROJECT_ROOT / "models"
CHART_PATH = CHARTS_DIR / "ai_training_progress.png"


class TrainingMonitor:
    """
    🔍 Monitors training progress by reading files and model checkpoints
    
    This class tracks training without interfering with the training process
    by reading the same files the training system writes to.
    """
    
    def __init__(self):
        self.last_chart_update = 0
        self.last_model_update = 0
        self.training_active = False
        self.start_time = time.time()
        
    def get_chart_info(self):
        """Get information about the training chart"""
        if not CHART_PATH.exists():
            return {
                "exists": False,
                "message": "Training chart not yet generated. Start training to see progress!",
                "last_updated": "Never"
            }
        
        stat = CHART_PATH.stat()
        self.last_chart_update = stat.st_mtime
        
        return {
            "exists": True,
            "path": str(CHART_PATH.relative_to(PROJECT_ROOT)),
            "size": f"{stat.st_size / 1024:.1f} KB",
            "last_updated": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
            "url": "/chart?" + str(int(stat.st_mtime))  # Cache busting
        }
    
    def get_latest_model_info(self):
        """Get information about the most recent model/checkpoint"""
        model_files = []
        
        # Check for final models
        final_dir = MODELS_DIR / "final"
        if final_dir.exists():
            for model_file in final_dir.glob("*.pth"):
                stat = model_file.stat()
                model_files.append({
                    "name": model_file.name,
                    "path": str(model_file.relative_to(PROJECT_ROOT)),
                    "size": f"{stat.st_size / 1024:.1f} KB",
                    "last_modified": stat.st_mtime,
                    "type": "final"
                })
        
        # Check for checkpoints
        checkpoint_dir = MODELS_DIR / "checkpoints"
        if checkpoint_dir.exists():
            for checkpoint_file in checkpoint_dir.glob("*.pth"):
                stat = checkpoint_file.stat()
                model_files.append({
                    "name": checkpoint_file.name,
                    "path": str(checkpoint_file.relative_to(PROJECT_ROOT)),
                    "size": f"{stat.st_size / 1024:.1f} KB",
                    "last_modified": stat.st_mtime,
                    "type": "checkpoint"
                })
        
        # Sort by modification time (newest first)
        model_files.sort(key=lambda x: x["last_modified"], reverse=True)
        
        # Add formatted timestamps
        for model in model_files:
            model["last_updated"] = datetime.fromtimestamp(model["last_modified"]).strftime("%Y-%m-%d %H:%M:%S")
        
        return model_files
    
    def get_training_status(self):
        """Determine if training is currently active"""
        # Check if any model was updated in the last 60 seconds
        latest_models = self.get_latest_model_info()
        if latest_models:
            latest_update = latest_models[0]["last_modified"]
            self.training_active = (time.time() - latest_update) < 60
        
        # Check if chart was updated recently
        chart_info = self.get_chart_info()
        if chart_info["exists"]:
            chart_stat = CHART_PATH.stat()
            chart_recent = (time.time() - chart_stat.st_mtime) < 120  # 2 minutes for chart
            self.training_active = self.training_active or chart_recent
        
        return {
            "active": self.training_active,
            "status": "🟢 Training Active" if self.training_active else "🔴 Training Idle",
            "uptime": self._format_uptime(time.time() - self.start_time)
        }
    
    def _format_uptime(self, seconds):
        """Format uptime in a human-readable way"""
        hours, remainder = divmod(int(seconds), 3600)
        minutes, seconds = divmod(remainder, 60)
        
        if hours > 0:
            return f"{hours}h {minutes}m {seconds}s"
        elif minutes > 0:
            return f"{minutes}m {seconds}s"
        else:
            return f"{seconds}s"
    
    def extract_model_stats(self, model_path):
        """Extract statistics from a model file"""
        try:
            if not Path(model_path).exists():
                return None
                
            # Load the model checkpoint
            checkpoint = torch.load(model_path, weights_only=False, map_location='cpu')
            
            stats = {
                "episode_scores": checkpoint.get("episode_scores", []),
                "training_losses": checkpoint.get("training_losses", []),
                "exploration_rates": checkpoint.get("exploration_rates", []),
                "current_epsilon": checkpoint.get("current_epsilon", "N/A")
            }
            
            # Calculate derived statistics
            if stats["episode_scores"]:
                scores = stats["episode_scores"]
                stats["total_episodes"] = len(scores)
                stats["best_score"] = max(scores)
                stats["average_score"] = sum(scores) / len(scores)
                stats["recent_average"] = sum(scores[-min(100, len(scores)):]) / min(100, len(scores))
            
            return stats
            
        except Exception as e:
            print(f"Error loading model stats: {e}")
            return None


# Global monitor instance
monitor = TrainingMonitor()


@app.route('/')
def dashboard():
    """Main dashboard page"""
    return render_template('dashboard.html')


@app.route('/api/status')
def api_status():
    """API endpoint for current training status"""
    status = monitor.get_training_status()
    chart_info = monitor.get_chart_info()
    models = monitor.get_latest_model_info()
    
    # Get stats from the most recent model
    model_stats = None
    if models:
        latest_model_path = PROJECT_ROOT / models[0]["path"]
        model_stats = monitor.extract_model_stats(latest_model_path)
    
    return jsonify({
        "status": status,
        "chart": chart_info,
        "models": models[:5],  # Limit to 5 most recent
        "model_stats": model_stats,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })


@app.route('/chart')
def serve_chart():
    """Serve the training chart image"""
    from flask import send_file
    
    if CHART_PATH.exists():
        return send_file(CHART_PATH, mimetype='image/png')
    else:
        # Return a placeholder image
        return "Chart not available", 404


@app.route('/api/models')
def api_models():
    """API endpoint for model information"""
    models = monitor.get_latest_model_info()
    return jsonify(models)


if __name__ == '__main__':
    print("🌐 F1 Race AI Training Dashboard")
    print("=" * 40)
    print("🚀 Starting web dashboard server...")
    print("📊 Monitor your AI training progress at:")
    print("   🌍 http://localhost:5000")
    print()
    print("💡 Features:")
    print("   📈 Real-time training charts")
    print("   📊 Live statistics and metrics")
    print("   🏎️ Training status monitoring")
    print("   ⚡ Auto-refreshing interface")
    print()
    print("🎯 Usage:")
    print("   1. Start this dashboard: python web_dashboard/app.py")
    print("   2. Start training: python train_ai.py")
    print("   3. Watch progress in your browser!")
    print()
    print("Press Ctrl+C to stop the dashboard")
    print("-" * 40)
    
    # Create required directories
    RESULTS_DIR.mkdir(exist_ok=True)
    CHARTS_DIR.mkdir(exist_ok=True)
    
    # Run the Flask app
    app.run(host='localhost', port=5000, debug=True, use_reloader=False)