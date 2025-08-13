import torch
import requests
from flask import Flask, request, jsonify
import json
import sys
import os
import pickle

# Add parent directory to path so we can import model
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import SimpleFraudModel, load_and_prepare_data, evaluate_simple_model

app = Flask(__name__)

# Simple server class
class SimpleServer:
    def __init__(self):
        self.global_model = SimpleFraudModel()
        self.client_models = {}
        self.round = 0
        self.evaluation_history = []
        
        # Load test data for evaluation
        self.load_test_data()
    
    def load_test_data(self):
        """Load test data for evaluation"""
        try:
            # Use Bank-1 data for testing
            self.X_test, self.y_test = load_and_prepare_data("../Bank-1/bank1.csv")
            print(f"✅ Loaded test data: {len(self.X_test)} samples")
        except Exception as e:
            print(f"❌ Error loading test data: {e}")
            self.X_test, self.y_test = None, None
    
    def evaluate_global_model(self):
        """Evaluate global model and return metrics"""
        if self.X_test is None or self.y_test is None:
            return {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0}
        
        try:
            self.global_model.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(self.X_test)
                outputs = self.global_model(X_tensor)
                predictions = (outputs >= 0.5).float().squeeze()
                y_tensor = torch.FloatTensor(self.y_test)
                
                # Calculate accuracy
                accuracy = (predictions == y_tensor).float().mean().item()
                
                # Calculate precision, recall, f1-score
                tp = ((predictions == 1) & (y_tensor == 1)).sum().item()
                fp = ((predictions == 1) & (y_tensor == 0)).sum().item()
                fn = ((predictions == 0) & (y_tensor == 1)).sum().item()
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                
                return {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1_score
                }
        except Exception as e:
            print(f"❌ Evaluation error: {e}")
            return {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0}
    
    def save_global_model(self):
        """Save global model to both bank directories"""
        try:
            # Save model state dict
            model_state = self.global_model.state_dict()
            
            # Save to Bank-1 directory
            bank1_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Bank-1", "global_model.pth")
            torch.save(model_state, bank1_path)
            print(f"✅ Saved global model to Bank-1: {bank1_path}")
            
            # Save to Bank-2 directory
            bank2_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Bank-2", "global_model.pth")
            torch.save(model_state, bank2_path)
            print(f"✅ Saved global model to Bank-2: {bank2_path}")
            
            # Save evaluation metrics
            metrics = self.evaluate_global_model()
            self.evaluation_history.append({
                'round': self.round,
                'metrics': metrics
            })
            
            # Save metrics to both directories
            metrics_data = {
                'current_round': self.round,
                'current_metrics': metrics,
                'history': self.evaluation_history
            }
            
            # Save to Bank-1
            metrics_path1 = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Bank-1", "evaluation_metrics.pkl")
            with open(metrics_path1, 'wb') as f:
                pickle.dump(metrics_data, f)
            
            # Save to Bank-2
            metrics_path2 = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Bank-2", "evaluation_metrics.pkl")
            with open(metrics_path2, 'wb') as f:
                pickle.dump(metrics_data, f)
            
            print(f"✅ Saved evaluation metrics to both banks")
            
        except Exception as e:
            print(f"❌ Error saving model: {e}")
    
    def aggregate_models(self):
        """Simple FedAvg - just average the models"""
        if not self.client_models:
            print("No models to aggregate!")
            return False
        
        print(f"Round {self.round}: Aggregating {len(self.client_models)} models")
        
        # Get all model states
        model_states = list(self.client_models.values())
        
        # Average the parameters
        averaged_state = {}
        for param_name in self.global_model.state_dict().keys():
            averaged_state[param_name] = torch.zeros_like(self.global_model.state_dict()[param_name])
            for model_state in model_states:
                averaged_state[param_name] += model_state[param_name]
            averaged_state[param_name] /= len(model_states)
        
        # Update global model
        self.global_model.load_state_dict(averaged_state)
        self.round += 1
        
        # Save the global model
        self.save_global_model()
        
        print(f"Global model updated! Round {self.round}")
        return True

# Create server instance
server = SimpleServer()

@app.route('/get_model', methods=['GET'])
def get_model():
    """Send global model to clients"""
    model_state = server.global_model.state_dict()
    
    # Convert to lists for JSON
    serializable_state = {}
    for key, value in model_state.items():
        serializable_state[key] = value.cpu().numpy().tolist()
    
    return jsonify({
        'model_state': serializable_state,
        'round': server.round
    })

@app.route('/submit_model', methods=['POST'])
def submit_model():
    """Receive model from client"""
    data = request.json
    client_id = data['client_id']
    model_state = data['model_state']
    
    # Convert back to tensors
    state_dict = {}
    for key, value in model_state.items():
        state_dict[key] = torch.tensor(value)
    
    # Store client model
    server.client_models[client_id] = state_dict
    print(f"Received model from {client_id}")
    
    return jsonify({'status': 'success'})

@app.route('/aggregate', methods=['POST'])
def aggregate():
    """Trigger aggregation"""
    success = server.aggregate_models()
    if success:
        # Clear client models for next round
        server.client_models = {}
        
        # Get current metrics
        current_metrics = server.evaluate_global_model()
        
        return jsonify({
            'status': 'success', 
            'round': server.round,
            'metrics': current_metrics
        })
    else:
        return jsonify({'error': 'No models to aggregate', 'round': server.round})

@app.route('/status', methods=['GET'])
def status():
    """Get server status"""
    current_metrics = server.evaluate_global_model()
    return jsonify({
        'round': server.round,
        'num_clients': len(server.client_models),
        'metrics': current_metrics
    })

@app.route('/evaluation_history', methods=['GET'])
def get_evaluation_history():
    """Get evaluation history"""
    return jsonify({
        'history': server.evaluation_history,
        'current_round': server.round
    })

if __name__ == '__main__':
    print("🚀 Simple Federated Learning Server Starting...")
    print("Server will listen on http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=True) 