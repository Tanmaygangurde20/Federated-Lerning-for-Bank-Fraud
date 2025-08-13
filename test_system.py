#!/usr/bin/env python3
"""
Simple test script for federated learning system
Easy to understand!
"""

import sys
import os
import pandas as pd
import torch

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_data():
    """Test if data files exist"""
    print("🔍 Testing data files...")
    
    try:
        # Check Bank-1 data
        bank1_data = pd.read_csv("Bank-1/bank1.csv")
        print(f"✅ Bank-1 data: {len(bank1_data)} rows")
        
        # Check Bank-2 data
        bank2_data = pd.read_csv("Bank-2/bank2.csv")
        print(f"✅ Bank-2 data: {len(bank2_data)} rows")
        
        return True
    except Exception as e:
        print(f"❌ Data test failed: {e}")
        return False

def test_model():
    """Test if model works"""
    print("\n🔍 Testing model...")
    
    try:
        from model import SimpleFraudModel
        
        # Create model
        model = SimpleFraudModel()
        print("✅ Model created")
        
        # Test with sample data
        test_input = torch.randn(5, 7)  # 5 samples, 7 features
        output = model(test_input)
        print(f"✅ Model works! Output shape: {output.shape}")
        
        return True
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        return False

def test_data_processing():
    """Test data processing"""
    print("\n🔍 Testing data processing...")
    
    try:
        from model import load_and_prepare_data
        
        # Load sample data
        X, y = load_and_prepare_data("Bank-1/bank1.csv")
        print(f"✅ Data processed! Features: {X.shape}, Labels: {y.shape}")
        
        return True
    except Exception as e:
        print(f"❌ Data processing failed: {e}")
        return False

def test_training():
    """Test training"""
    print("\n🔍 Testing training...")
    
    try:
        from model import SimpleFraudModel, load_and_prepare_data, train_simple_model, evaluate_simple_model
        
        # Load data
        X, y = load_and_prepare_data("Bank-1/bank1.csv")
        
        # Create and train model
        model = SimpleFraudModel()
        trained_model = train_simple_model(model, X[:1000], y[:1000], epochs=2)  # Use small sample
        
        # Test accuracy
        accuracy = evaluate_simple_model(trained_model, X[:100], y[:100])
        print(f"✅ Training works! Accuracy: {accuracy:.3f}")
        
        return True
    except Exception as e:
        print(f"❌ Training test failed: {e}")
        return False

def test_packages():
    """Test if packages are installed"""
    print("\n🔍 Testing packages...")
    
    packages = ['torch', 'pandas', 'numpy', 'sklearn', 'requests', 'flask']
    
    for package in packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
            return False
    
    return True

def main():
    """Run all tests"""
    print("🧪 Simple Federated Learning Tests")
    print("=" * 40)
    
    tests = [
        ("Packages", test_packages),
        ("Data Files", test_data),
        ("Model", test_model),
        ("Data Processing", test_data_processing),
        ("Training", test_training)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}")
        print("-" * 20)
        
        if test_func():
            passed += 1
            print(f"✅ {test_name} PASSED")
        else:
            print(f"❌ {test_name} FAILED")
    
    print("\n" + "=" * 40)
    print(f"📊 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Ready to run!")
        print("\n🚀 To start federated learning:")
        print("   python run_federated_learning.py")
    else:
        print("⚠️  Some tests failed!")
        print("\n💡 Try:")
        print("   pip install -r requirements.txt")

if __name__ == "__main__":
    main() 