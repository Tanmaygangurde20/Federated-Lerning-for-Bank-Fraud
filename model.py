import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Very simple neural network for beginners
class SimpleFraudModel(nn.Module):
    def __init__(self):
        super(SimpleFraudModel, self).__init__()
        # Simple 3-layer network: 7 inputs -> 10 -> 5 -> 1 output
        self.layer1 = nn.Linear(7, 10)
        self.layer2 = nn.Linear(10, 5)
        self.layer3 = nn.Linear(5, 1)
        
    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = torch.sigmoid(self.layer3(x))
        return x

# Simple data processing
def load_and_prepare_data(file_path):
    """Load CSV and prepare simple features"""
    data = pd.read_csv(file_path)
    
    # Simple feature selection - just use numerical columns
    features = ['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 
               'oldbalanceDest', 'newbalanceDest']
    
    # Add transaction type as number
    type_map = {'CASH_IN': 0, 'CASH_OUT': 1, 'TRANSFER': 2, 'PAYMENT': 3}
    data['type_num'] = data['type'].map(type_map)
    features.append('type_num')
    
    # Check if all features exist
    missing_features = [f for f in features if f not in data.columns]
    if missing_features:
        print(f"Warning: Missing features: {missing_features}")
        # Remove missing features
        features = [f for f in features if f in data.columns]
    
    X = data[features].values
    y = data['isFraud'].values
    
    # Handle any NaN values
    X = np.nan_to_num(X, nan=0.0)
    y = np.nan_to_num(y, nan=0.0)
    
    # Simple scaling
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    return X, y

# Simple training function
def train_simple_model(model, X, y, epochs=5):
    """Train model with simple settings"""
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.BCELoss()
    
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.FloatTensor(y).unsqueeze(1)
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)
        loss.backward()
        optimizer.step()
    
    return model

# Simple evaluation
def evaluate_simple_model(model, X, y):
    """Simple accuracy calculation"""
    model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X)
        outputs = model(X_tensor)
        predictions = (outputs >= 0.5).float().squeeze()
        accuracy = (predictions == torch.FloatTensor(y)).float().mean().item()
    return accuracy 