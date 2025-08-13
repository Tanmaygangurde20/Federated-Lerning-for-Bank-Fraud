import torch
import requests
import time
import sys
import os

# Add parent directory to path so we can import model
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import SimpleFraudModel, load_and_prepare_data, train_simple_model, evaluate_simple_model

# Simple client class
class SimpleClient:
    def __init__(self, client_id, data_file):
        self.client_id = client_id
        self.data_file = data_file
        self.model = SimpleFraudModel()
        
        # Load data
        print(f"Loading data for {client_id}...")
        try:
            self.X, self.y = load_and_prepare_data(data_file)
            print(f"✅ Loaded {len(self.X)} samples")
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            raise e
    
    def get_global_model(self):
        """Download global model from server"""
        try:
            response = requests.get("http://localhost:5000/get_model", timeout=10)
            if response.status_code == 200:
                data = response.json()
                model_state = data['model_state']
                
                # Convert to tensors
                state_dict = {}
                for key, value in model_state.items():
                    state_dict[key] = torch.tensor(value)
                
                self.model.load_state_dict(state_dict)
                print(f"✅ Downloaded global model")
                return True
            else:
                print(f"❌ Failed to get model: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Error: {e}")
            return False
    
    def train_local_model(self):
        """Train model on local data"""
        try:
            print(f"Training {self.client_id} model...")
            self.model = train_simple_model(self.model, self.X, self.y, epochs=3)
            
            # Check accuracy
            accuracy = evaluate_simple_model(self.model, self.X, self.y)
            print(f"✅ {self.client_id} accuracy: {accuracy:.3f}")
            return accuracy
        except Exception as e:
            print(f"❌ Training error: {e}")
            return 0.0
    
    def submit_model(self):
        """Send model to server"""
        try:
            # Get model state
            model_state = self.model.state_dict()
            
            # Convert to lists
            serializable_state = {}
            for key, value in model_state.items():
                serializable_state[key] = value.cpu().numpy().tolist()
            
            # Send to server
            data = {
                'client_id': self.client_id,
                'model_state': serializable_state
            }
            
            response = requests.post("http://localhost:5000/submit_model", json=data, timeout=10)
            
            if response.status_code == 200:
                print(f"✅ {self.client_id} model submitted")
                return True
            else:
                print(f"❌ Failed to submit: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Error submitting: {e}")
            return False
    
    def participate_in_round(self):
        """Do one round of federated learning"""
        print(f"\n🔄 {self.client_id} - Starting Round")
        
        # Get global model
        if not self.get_global_model():
            print(f"❌ {self.client_id} failed to get global model")
            return None
        
        # Train locally
        accuracy = self.train_local_model()
        
        # Submit model
        if not self.submit_model():
            print(f"❌ {self.client_id} failed to submit model")
            return None
        
        return accuracy

def main():
    try:
        # Create client
        client = SimpleClient("Bank-2", "bank2.csv")
        
        # Do 5 rounds
        for round_num in range(5):
            print(f"\n{'='*40}")
            print(f"ROUND {round_num + 1}/5")
            print(f"{'='*40}")
            
            accuracy = client.participate_in_round()
            
            if accuracy is not None:
                print(f"✅ Round {round_num + 1} completed! Accuracy: {accuracy:.3f}")
            else:
                print(f"❌ Round {round_num + 1} failed!")
            
            time.sleep(2)  # Wait before next round
        
        print("\n🎉 Federated learning completed!")
        
    except Exception as e:
        print(f"❌ Client error: {e}")

if __name__ == "__main__":
    main() 