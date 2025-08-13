#!/usr/bin/env python3
"""
Simple Federated Learning Coordinator
Easy to understand and run!
"""

import subprocess
import time
import requests
import sys
import os

def start_server():
    """Start the server"""
    print("🚀 Starting server...")
    os.chdir("Server")
    server_process = subprocess.Popen([sys.executable, "server.py"])
    os.chdir("..")
    time.sleep(3)  # Wait for server to start
    return server_process

def start_bank1():
    """Start Bank-1 client"""
    print("🏦 Starting Bank-1...")
    os.chdir("Bank-1")
    bank1_process = subprocess.Popen([sys.executable, "client.py"])
    os.chdir("..")
    return bank1_process

def start_bank2():
    """Start Bank-2 client"""
    print("🏦 Starting Bank-2...")
    os.chdir("Bank-2")
    bank2_process = subprocess.Popen([sys.executable, "client.py"])
    os.chdir("..")
    return bank2_process

def start_dashboard():
    """Start the dashboard"""
    print("📊 Starting dashboard...")
    dashboard_process = subprocess.Popen([
        sys.executable, "-m", "streamlit", "run", "simple_dashboard.py", 
        "--server.port", "8501", "--server.address", "localhost"
    ])
    time.sleep(3)  # Wait for dashboard to start
    return dashboard_process

def check_server():
    """Check if server is running"""
    try:
        response = requests.get("http://localhost:5000/status", timeout=5)
        return response.status_code == 200
    except:
        return False

def trigger_aggregation():
    """Tell server to combine models"""
    try:
        response = requests.post("http://localhost:5000/aggregate", timeout=10)
        if response.status_code == 200:
            result = response.json()
            if 'round' in result:
                print(f"✅ Aggregation done! Round: {result['round']}")
                
                # Display evaluation metrics if available
                if 'metrics' in result:
                    metrics = result['metrics']
                    print(f"📊 Evaluation Metrics:")
                    print(f"   Accuracy: {metrics['accuracy']:.3f}")
                    print(f"   Precision: {metrics['precision']:.3f}")
                    print(f"   Recall: {metrics['recall']:.3f}")
                    print(f"   F1-Score: {metrics['f1_score']:.3f}")
                
                return True
            else:
                print("❌ Aggregation failed - no round info!")
                return False
        else:
            print("❌ Aggregation failed!")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    """Main function - very simple!"""
    print("🎯 Simple Federated Learning System")
    print("=" * 40)
    
    # Start server
    server_process = start_server()
    
    # Wait for server
    print("⏳ Waiting for server...")
    for i in range(10):
        if check_server():
            print("✅ Server is ready!")
            break
        time.sleep(1)
    else:
        print("❌ Server failed to start!")
        return
    
    # Start dashboard
    dashboard_process = start_dashboard()
    print("📊 Dashboard started at: http://localhost:8501")
    
    # Start banks
    bank1_process = start_bank1()
    time.sleep(2)
    bank2_process = start_bank2()
    time.sleep(2)
    
    print("\n🎯 All started! Running 5 rounds...")
    print("📊 View dashboard at: http://localhost:8501")
    print("💾 Models will be saved to both bank directories")
    print("=" * 40)
    
    # Do 5 rounds
    for round_num in range(5):
        print(f"\n🔄 ROUND {round_num + 1}/5")
        print("-" * 20)
        
        # Wait for banks to finish
        print("⏳ Waiting for banks to train...")
        time.sleep(10)  # Give more time for banks to train
        
        # Combine models
        if trigger_aggregation():
            print(f"✅ Round {round_num + 1} completed!")
            print(f"💾 Global model saved to both bank directories")
        else:
            print(f"❌ Round {round_num + 1} failed!")
        
        time.sleep(3)
    
    print("\n🎉 All done! Federated learning completed!")
    print("📊 Dashboard: http://localhost:8501")
    print("💾 Check Bank-1/ and Bank-2/ directories for saved models")
    print("💡 Press Ctrl+C to stop all processes")
    
    # Keep running
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Stopping...")
        server_process.terminate()
        bank1_process.terminate()
        bank2_process.terminate()
        dashboard_process.terminate()
        print("✅ All stopped!")

if __name__ == "__main__":
    main() 