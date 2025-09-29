#!/usr/bin/env python3
"""
🌐 F1 Race AI Web Dashboard - Standalone Version
================================================

A simple web server that monitors F1 Race AI training progress
in real-time without requiring complex dependencies.

This version uses Python's built-in HTTP server with basic
HTML/CSS/JavaScript for maximum compatibility.
"""

import json
import os
import time
import webbrowser
from datetime import datetime
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse, parse_qs
import threading

# Project paths
PROJECT_ROOT = Path(__file__).parent
RESULTS_DIR = PROJECT_ROOT / "results"
CHARTS_DIR = RESULTS_DIR / "charts"
MODELS_DIR = PROJECT_ROOT / "models"
CHART_PATH = CHARTS_DIR / "ai_training_progress.png"


class DashboardHandler(SimpleHTTPRequestHandler):
    """Custom HTTP handler for the dashboard"""
    
    def do_GET(self):
        """Handle GET requests"""
        parsed_path = urlparse(self.path)
        
        if parsed_path.path == '/' or parsed_path.path == '/dashboard':
            self.serve_dashboard()
        elif parsed_path.path == '/api/status':
            self.serve_status_api()
        elif parsed_path.path == '/chart':
            self.serve_chart()
        elif parsed_path.path.startswith('/api/'):
            self.serve_api_404()
        else:
            # Serve static files if they exist
            super().do_GET()
    
    def serve_dashboard(self):
        """Serve the main dashboard HTML"""
        dashboard_html = self.get_dashboard_html()
        
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(dashboard_html.encode('utf-8'))
    
    def serve_status_api(self):
        """Serve the status API endpoint"""
        try:
            status_data = self.get_training_status()
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(status_data).encode('utf-8'))
        except Exception as e:
            self.send_error(500, f"Error getting status: {e}")
    
    def serve_chart(self):
        """Serve the training chart image"""
        if CHART_PATH.exists():
            self.send_response(200)
            self.send_header('Content-type', 'image/png')
            self.send_header('Cache-Control', 'no-cache')
            self.end_headers()
            
            with open(CHART_PATH, 'rb') as f:
                self.wfile.write(f.read())
        else:
            self.send_error(404, "Chart not found")
    
    def serve_api_404(self):
        """Serve 404 for unknown API endpoints"""
        self.send_error(404, "API endpoint not found")
    
    def get_training_status(self):
        """Get current training status and metrics"""
        # Check chart info
        chart_info = self.get_chart_info()
        
        # Get model info
        models = self.get_latest_models()
        
        # Get model statistics
        model_stats = None
        if models:
            try:
                model_stats = self.extract_model_stats(PROJECT_ROOT / models[0]["path"])
            except:
                pass
        
        # Improved training detection logic
        training_active = False
        current_time = time.time()
        
        # Check if any model was updated recently (more sensitive - 30 seconds for checkpoints)
        if models:
            latest_model_time = models[0]["last_modified"]
            model_age = current_time - latest_model_time
            training_active = model_age < 60  # 1 minute for model updates
        
        # Check if chart was updated recently (charts update every 100 episodes)
        if chart_info["exists"]:
            chart_stat = CHART_PATH.stat()
            chart_age = current_time - chart_stat.st_mtime
            # Charts are less frequent, so allow longer window
            chart_recently_updated = chart_age < 300  # 5 minutes for chart updates
            training_active = training_active or chart_recently_updated
        
        # If we have multiple recent checkpoints, that's a strong indicator of active training
        if len(models) >= 2:
            recent_models = [m for m in models if (current_time - m["last_modified"]) < 600]  # 10 minutes
            if len(recent_models) >= 2:
                training_active = True
        
        return {
            "status": {
                "active": training_active,
                "status": "🟢 Training Active" if training_active else "🔴 Training Idle",
                "uptime": self.format_uptime(current_time - start_time)
            },
            "chart": chart_info,
            "models": models[:5],
            "model_stats": model_stats,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    
    def get_chart_info(self):
        """Get chart file information"""
        if not CHART_PATH.exists():
            return {
                "exists": False,
                "message": "Training chart not yet generated. Start training to see progress!",
                "last_updated": "Never"
            }
        
        stat = CHART_PATH.stat()
        return {
            "exists": True,
            "path": str(CHART_PATH.relative_to(PROJECT_ROOT)),
            "size": f"{stat.st_size / 1024:.1f} KB",
            "last_updated": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
            "url": "/chart?" + str(int(stat.st_mtime))
        }
    
    def get_latest_models(self):
        """Get information about recent models"""
        model_files = []
        
        # Check final models
        final_dir = MODELS_DIR / "final"
        if final_dir.exists():
            for model_file in final_dir.glob("*.pth"):
                stat = model_file.stat()
                model_files.append({
                    "name": model_file.name,
                    "path": str(model_file.relative_to(PROJECT_ROOT)),
                    "size": f"{stat.st_size / 1024:.1f} KB",
                    "last_modified": stat.st_mtime,
                    "last_updated": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
                    "type": "final"
                })
        
        # Check checkpoints
        checkpoint_dir = MODELS_DIR / "checkpoints"
        if checkpoint_dir.exists():
            for checkpoint_file in checkpoint_dir.glob("*.pth"):
                stat = checkpoint_file.stat()
                model_files.append({
                    "name": checkpoint_file.name,
                    "path": str(checkpoint_file.relative_to(PROJECT_ROOT)),
                    "size": f"{stat.st_size / 1024:.1f} KB",
                    "last_modified": stat.st_mtime,
                    "last_updated": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
                    "type": "checkpoint"
                })
        
        # Sort by modification time (most recent first) - this is correct for active training
        model_files.sort(key=lambda x: x["last_modified"], reverse=True)
        return model_files
    
    def extract_model_stats(self, model_path):
        """Extract statistics from model file (simplified)"""
        try:
            # Try to import torch only when needed
            import torch
            
            if not Path(model_path).exists():
                return None
            
            checkpoint = torch.load(model_path, weights_only=False, map_location='cpu')
            
            stats = {
                "episode_scores": checkpoint.get("episode_scores", []),
                "training_losses": checkpoint.get("training_losses", []),
                "exploration_rates": checkpoint.get("exploration_rates", []),
                "current_epsilon": checkpoint.get("current_epsilon", "N/A")
            }
            
            if stats["episode_scores"]:
                scores = stats["episode_scores"]
                stats["total_episodes"] = len(scores)
                stats["best_score"] = max(scores)
                stats["average_score"] = sum(scores) / len(scores)
                stats["recent_average"] = sum(scores[-min(100, len(scores)):]) / min(100, len(scores))
            
            return stats
        except Exception as e:
            print(f"Warning: Could not load model stats: {e}")
            return None
    
    def format_uptime(self, seconds):
        """Format uptime in human readable form"""
        hours, remainder = divmod(int(seconds), 3600)
        minutes, seconds = divmod(remainder, 60)
        
        if hours > 0:
            return f"{hours}h {minutes}m {seconds}s"
        elif minutes > 0:
            return f"{minutes}m {seconds}s"
        else:
            return f"{seconds}s"
    
    def get_dashboard_html(self):
        """Generate the dashboard HTML"""
        # Read the template file if it exists, otherwise use embedded HTML
        template_path = PROJECT_ROOT / "web_dashboard" / "templates" / "dashboard.html"
        
        if template_path.exists():
            try:
                with open(template_path, 'r', encoding='utf-8') as f:
                    return f.read()
            except:
                pass
        
        # Fallback to embedded simple HTML
        return self.get_simple_dashboard_html()
    
    def get_simple_dashboard_html(self):
        """Simple embedded dashboard HTML"""
        return """
<!DOCTYPE html>
<html>
<head>
    <title>🏎️ F1 Race AI Training Dashboard</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f0f0f0; }
        .container { max-width: 1200px; margin: 0 auto; }
        .header { text-align: center; background: white; padding: 20px; border-radius: 10px; margin-bottom: 20px; }
        .card { background: white; padding: 20px; border-radius: 10px; margin: 10px 0; }
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }
        .stat { text-align: center; padding: 10px; background: #f8f9fa; border-radius: 5px; margin: 5px; }
        .stat-value { font-size: 1.5em; font-weight: bold; color: #007bff; }
        .status-active { color: #28a745; }
        .status-idle { color: #6c757d; }
        img { max-width: 100%; height: auto; border-radius: 5px; }
        .loading { text-align: center; color: #6c757d; padding: 20px; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🏎️ F1 Race AI Training Dashboard</h1>
            <p>Real-time monitoring of your AI race car driver's learning progress</p>
            <p id="status" class="loading">🔄 Loading...</p>
            <p><small>Last updated: <span id="timestamp">--</span></small></p>
        </div>
        
        <div class="grid">
            <div class="card">
                <h2>📊 Training Statistics</h2>
                <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px;">
                    <div class="stat">
                        <div class="stat-value" id="episodes">--</div>
                        <div>Episodes</div>
                    </div>
                    <div class="stat">
                        <div class="stat-value" id="best-score">--</div>
                        <div>Best Score</div>
                    </div>
                    <div class="stat">
                        <div class="stat-value" id="avg-score">--</div>
                        <div>Avg Score</div>
                    </div>
                </div>
            </div>
            
            <div class="card">
                <h2>💾 Recent Models</h2>
                <div id="models" class="loading">Loading...</div>
            </div>
            
            <div class="card" style="grid-column: 1 / -1;">
                <h2>📈 Training Progress Chart</h2>
                <div id="chart-container" style="text-align: center;">
                    <div class="loading">Chart will appear once training begins</div>
                </div>
            </div>
        </div>
    </div>

    <script>
        function updateDashboard() {
            fetch('/api/status')
                .then(response => response.json())
                .then(data => {
                    // Update status
                    const statusEl = document.getElementById('status');
                    statusEl.textContent = data.status.status;
                    statusEl.className = data.status.active ? 'status-active' : 'status-idle';
                    
                    document.getElementById('timestamp').textContent = data.timestamp;
                    
                    // Update stats
                    if (data.model_stats) {
                        document.getElementById('episodes').textContent = data.model_stats.total_episodes || '--';
                        document.getElementById('best-score').textContent = data.model_stats.best_score || '--';
                        document.getElementById('avg-score').textContent = 
                            data.model_stats.average_score ? data.model_stats.average_score.toFixed(1) : '--';
                    }
                    
                    // Update models
                    const modelsEl = document.getElementById('models');
                    if (data.models && data.models.length > 0) {
                        modelsEl.innerHTML = data.models.map(m => 
                            `<div style="margin: 5px 0; padding: 8px; background: #f8f9fa; border-radius: 3px;">
                                <strong>${m.name}</strong> (${m.type})<br>
                                <small>${m.size} • ${m.last_updated}</small>
                            </div>`
                        ).join('');
                    } else {
                        modelsEl.innerHTML = '<div class="loading">No models found yet</div>';
                    }
                    
                    // Update chart
                    const chartEl = document.getElementById('chart-container');
                    if (data.chart && data.chart.exists) {
                        chartEl.innerHTML = `<img src="${data.chart.url}" alt="Training Chart">`;
                    }
                })
                .catch(error => {
                    console.error('Error:', error);
                    document.getElementById('status').textContent = '❌ Connection Error';
                });
        }

        // Update every 5 seconds
        updateDashboard();
        setInterval(updateDashboard, 5000);
    </script>
</body>
</html>
        """

# Global start time for uptime calculation
start_time = time.time()


def main():
    """Main function to run the dashboard server"""
    print("🌐 F1 Race AI Training Dashboard (Standalone)")
    print("=" * 50)
    
    # Create required directories
    RESULTS_DIR.mkdir(exist_ok=True)
    CHARTS_DIR.mkdir(exist_ok=True)
    
    # Setup server
    port = 5000
    server_address = ('localhost', port)
    httpd = HTTPServer(server_address, DashboardHandler)
    
    print("🚀 Starting dashboard server...")
    print(f"📊 Dashboard available at: http://localhost:{port}")
    print("💡 Monitor your AI training progress in real-time!")
    print()
    print("🎯 Usage:")
    print("   1. Keep this dashboard running")
    print("   2. Start training: python train_ai.py")
    print("   3. Watch progress at: http://localhost:5000")
    print()
    print("Press Ctrl+C to stop the dashboard")
    print("-" * 50)
    
    # Auto-open browser
    def open_browser():
        time.sleep(2)
        try:
            webbrowser.open(f'http://localhost:{port}')
            print("✅ Browser opened automatically!")
        except:
            pass
    
    threading.Thread(target=open_browser, daemon=True).start()
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n🛑 Dashboard stopped by user")
        httpd.server_close()


if __name__ == "__main__":
    main()