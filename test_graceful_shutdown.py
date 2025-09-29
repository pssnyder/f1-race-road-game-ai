#!/usr/bin/env python3
"""
🧪 Test Graceful Shutdown Feature
=================================

Quick test to verify the graceful shutdown feature works correctly.
This will start a short training session that you can interrupt with Ctrl+C.
"""

from train_ai import train_racing_ai
import signal
import time

def test_graceful_shutdown():
    """Test the graceful shutdown feature"""
    print("🧪 TESTING GRACEFUL SHUTDOWN FEATURE")
    print("=" * 50)
    print("This will start a training session.")
    print("Press Ctrl+C after a few seconds to test graceful shutdown.")
    print("The training should save progress and exit cleanly.")
    print()
    
    input("Press Enter to start test training (then Ctrl+C to test shutdown)...")
    
    try:
        # Start a training session with many episodes
        agent = train_racing_ai(
            episodes=5000,  # Long training
            show_training=False,  # Headless for speed
            save_frequency=50,    # Frequent saves for testing
            chart_update_frequency=25  # Frequent chart updates
        )
        
        print("✅ Training completed normally (not interrupted)")
        
    except SystemExit:
        print("✅ Graceful shutdown worked correctly!")
        
if __name__ == "__main__":
    test_graceful_shutdown()