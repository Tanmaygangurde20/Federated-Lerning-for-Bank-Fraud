# 🏦 Federated Learning for Bank Fraud Detection

This project implements a federated learning system using the FedAvg algorithm for bank fraud detection. Two banks (Bank-1 and Bank-2) collaborate to train a global fraud detection model without sharing their raw data.

## 🎯 Project Overview

- **Federated Learning Algorithm**: FedAvg (Federated Averaging)
- **Model**: Neural Network for binary classification (fraud detection)
- **Data**: Bank transaction data with fraud labels
- **Visualization**: Streamlit dashboard for real-time monitoring

## 📁 Project Structure

```
Federated Learning -2/
├── Bank-1/
│   ├── bank1.csv          # Bank-1 transaction data
│   └── client.py          # Bank-1 federated client
├── Bank-2/
│   ├── bank2.csv          # Bank-2 transaction data
│   └── client.py          # Bank-2 federated client
├── Server/
│   └── server.py          # Central aggregation server
├── model.py               # Neural network model and utilities
├── dashboard.py           # Streamlit dashboard
├── run_federated_learning.py  # Coordinator script
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Complete System

```bash
python run_federated_learning.py
```

This will:
- Start the central server
- Launch both bank clients
- Start the Streamlit dashboard
- Run 5 federated learning rounds
- Display real-time metrics

### 3. Access the Dashboard

Open your browser and go to: http://localhost:8501

## 🔧 Manual Setup (Alternative)

If you prefer to run components individually:

### Start the Server
```bash
cd Server
python server.py
```

### Start Bank-1 Client (in a new terminal)
```bash
cd Bank-1
python client.py
```

### Start Bank-2 Client (in a new terminal)
```bash
cd Bank-2
python client.py
```

### Start the Dashboard (in a new terminal)
```bash
streamlit run dashboard.py
```

## 📊 Dashboard Features

The Streamlit dashboard provides:

1. **Real-time Metrics**: Current round, active clients, global accuracy
2. **Performance Visualization**: 
   - Accuracy comparison across rounds
   - F1-score trends
   - Local vs global model performance
3. **Data Distribution**: Transaction types and fraud distribution for both banks
4. **Model Architecture**: Neural network structure visualization
5. **Interactive Controls**: Manual aggregation triggering

## 🧠 Model Architecture

The fraud detection model uses:

- **Input Layer**: 7 features (step, amount, balances, transaction type)
- **Hidden Layers**: 64 → 32 → 16 neurons with ReLU activation
- **Output Layer**: 1 neuron with Sigmoid activation
- **Regularization**: Dropout (0.3) for overfitting prevention
- **Loss Function**: Binary Cross Entropy
- **Optimizer**: Adam with learning rate 0.001

## 🔄 Federated Learning Process

1. **Initialization**: Server initializes global model
2. **Distribution**: Global model sent to all clients
3. **Local Training**: Each client trains on local data (3-5 epochs)
4. **Model Submission**: Clients send updated models to server
5. **Aggregation**: Server aggregates models using FedAvg
6. **Evaluation**: Global model evaluated on test data
7. **Repeat**: Process continues for specified rounds

## 📈 Expected Results

After 5 rounds of federated learning, you should see:

- **Global Model Accuracy**: ~85-90%
- **Local Model Accuracy**: ~80-85% (varies by bank)
- **F1-Score**: ~75-80%
- **Convergence**: Steady improvement across rounds

## 🛠️ API Endpoints

The server provides these REST endpoints:

- `GET /get_global_model` - Download global model
- `POST /submit_model` - Submit local model
- `POST /aggregate` - Trigger model aggregation
- `GET /status` - Get server status

## 📋 Data Format

The bank transaction data includes:

- `step`: Time step of transaction
- `type`: Transaction type (CASH_IN, CASH_OUT, TRANSFER, PAYMENT)
- `amount`: Transaction amount
- `nameOrig`: Origin account
- `oldbalanceOrg`: Origin account balance before
- `newbalanceOrig`: Origin account balance after
- `nameDest`: Destination account
- `oldbalanceDest`: Destination account balance before
- `newbalanceDest`: Destination account balance after
- `isFraud`: Fraud label (0/1)
- `isFlaggedFraud`: Flagged fraud (0/1)

## 🔍 Key Features

- **Privacy-Preserving**: Raw data never leaves local banks
- **Scalable**: Easy to add more banks/clients
- **Real-time Monitoring**: Live dashboard with metrics
- **Robust**: Error handling and graceful shutdown
- **Configurable**: Adjustable parameters for rounds, epochs, etc.

## 🐛 Troubleshooting

### Common Issues:

1. **Port Already in Use**: 
   - Change ports in the scripts or kill existing processes
   - Server: 5000, Dashboard: 8501

2. **Import Errors**:
   - Ensure all dependencies are installed: `pip install -r requirements.txt`
   - Check Python path and working directory

3. **Data Loading Errors**:
   - Verify CSV files exist in Bank-1/ and Bank-2/ directories
   - Check file permissions

4. **Connection Errors**:
   - Ensure server is running before starting clients
   - Check firewall settings

## 📝 Customization

### Modify Training Parameters:
- Edit `epochs` parameter in client scripts
- Change learning rate in `model.py`
- Adjust number of rounds in coordinator

### Add More Banks:
1. Create new bank directory with data
2. Copy and modify client script
3. Update coordinator to include new client

### Change Model Architecture:
- Modify `FraudDetectionModel` class in `model.py`
- Adjust input size based on feature engineering

## 🤝 Contributing

Feel free to contribute by:
- Adding new features
- Improving the model architecture
- Enhancing the dashboard
- Adding more evaluation metrics
- Implementing additional federated learning algorithms

## 📄 License

This project is for educational and research purposes.

---

**Happy Federated Learning! 🎉** 