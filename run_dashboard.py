#!/usr/bin/env python3
"""
Simple script to run just the dashboard
Perfect for sharing with others!
"""

import subprocess
import sys
import os

def main():
    print("📊 Starting Federated Learning Dashboard")
    print("=" * 40)
    print("🌐 Dashboard will open at: http://localhost:8501")
    print("📱 Share this URL with others to show the results!")
    print("=" * 40)
    
    try:
        # Start dashboard
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "simple_dashboard.py",
            "--server.port", "8501",
            "--server.address", "0.0.0.0"  # Allow external access
        ])
    except KeyboardInterrupt:
        print("\n🛑 Dashboard stopped!")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Make sure you have streamlit installed:")
        print("   pip install streamlit")

if __name__ == "__main__":
    main() 