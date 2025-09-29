"""
🏎️ F1 Race AI Training System - Main Entry Point
===============================================

This is the main entry point for the F1 Racing AI system.
All functionality is organized in the src/ folder:

📁 src/
  - agent.py       🤖 DQN AI agent with learning algorithms
  - environment.py 🎮 Game environment with physics and rendering  
  - trainer.py     🏋️ Training system with all enhancements
  - dashboard.py   📊 Web dashboard for monitoring

🚀 ENHANCEMENTS INCLUDED:
- ⚡ Framerate multiplier (1-500%) for faster training
- 📊 Periodic chart updates during training
- 🌐 Real-time web dashboard monitoring
- 🛑 Graceful shutdown with Ctrl+C
- 💾 Resume training from any checkpoint
- 🎨 Visual/headless training modes

Author: Pat Snyder
"""

# Import the trainer which handles everything
from src.trainer import main

if __name__ == "__main__":
    main()