#!/usr/bin/env python3
"""
🚀 Launch F1 Race AI Training Dashboard
======================================

Simple launcher for the web-based training monitor.

Usage:
    python launch_dashboard.py

Then visit: http://localhost:5000
"""

import os
import sys
import webbrowser
from pathlib import Path

def main():
    print("🌐 F1 Race AI Training Dashboard Launcher")
    print("=" * 50)
    
    # Get the project root directory
    project_root = Path(__file__).parent
    dashboard_app = project_root / "web_dashboard" / "app.py"
    
    # Check if the dashboard app exists
    if not dashboard_app.exists():
        print("❌ Dashboard app not found!")
        print(f"   Expected: {dashboard_app}")
        sys.exit(1)
    
    print("🚀 Starting F1 Race AI Training Dashboard...")
    print("📊 Monitor your AI training progress at:")
    print("   🌍 http://localhost:5000")
    print()
    print("💡 Tips:")
    print("   • Start training in another terminal: python train_ai.py")
    print("   • Dashboard auto-refreshes every 5 seconds")
    print("   • Charts update as training progresses")
    print("   • Press Ctrl+C to stop the dashboard")
    print()
    
    # Ask if user wants to open browser automatically
    open_browser = input("🌐 Open dashboard in browser automatically? (y/n, default=y): ").lower().strip()
    should_open = open_browser != 'n'
    
    if should_open:
        print("🌍 Opening browser in 3 seconds...")
        import threading
        import time
        
        def open_browser_delayed():
            time.sleep(3)  # Give Flask time to start
            try:
                webbrowser.open('http://localhost:5000')
                print("✅ Browser opened!")
            except Exception as e:
                print(f"⚠️  Could not open browser: {e}")
        
        threading.Thread(target=open_browser_delayed, daemon=True).start()
    
    print("-" * 50)
    
    # Change to the dashboard directory and run the app
    os.chdir(project_root)
    
    try:
        # Import and run the Flask app
        sys.path.insert(0, str(project_root / "web_dashboard"))
        from app import app
        
        # Run the app
        app.run(host='localhost', port=5000, debug=False, use_reloader=False)
        
    except KeyboardInterrupt:
        print("\n🛑 Dashboard stopped by user")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Make sure Flask is installed: pip install flask")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error starting dashboard: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()