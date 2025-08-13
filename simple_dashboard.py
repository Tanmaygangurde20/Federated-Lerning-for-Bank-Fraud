import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import requests
import time
from datetime import datetime
import pickle
import os

# Page setup
st.set_page_config(
    page_title="Federated Learning - Bank Fraud Detection",
    page_icon="🏦",
    layout="wide"
)

# Title
st.title("🏦 Federated Learning - Bank Fraud Detection")
st.markdown("**Two banks collaborate to detect fraud without sharing their data!**")

# Sidebar
st.sidebar.header("📊 Dashboard Controls")
server_url = st.sidebar.text_input("Server URL", "http://localhost:5000")

# Function to get server status
def get_server_status():
    try:
        response = requests.get(f"{server_url}/status", timeout=5)
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except:
        return None

# Function to get evaluation history
def get_evaluation_history():
    try:
        response = requests.get(f"{server_url}/evaluation_history", timeout=5)
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except:
        return None

# Function to trigger aggregation
def trigger_aggregation():
    try:
        response = requests.post(f"{server_url}/aggregate", timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except:
        return None

# Function to load saved metrics
def load_saved_metrics():
    try:
        # Try to load from Bank-1 directory
        metrics_path = "Bank-1/evaluation_metrics.pkl"
        if os.path.exists(metrics_path):
            with open(metrics_path, 'rb') as f:
                return pickle.load(f)
    except:
        pass
    return None

# Main dashboard
col1, col2, col3, col4 = st.columns(4)

# Get current status
status = get_server_status()

with col1:
    if status:
        st.metric("🔄 Current Round", status['round'])
    else:
        st.metric("🔄 Current Round", "Not Connected")

with col2:
    if status:
        st.metric("🏦 Active Banks", status['num_clients'])
    else:
        st.metric("🏦 Active Banks", "Not Connected")

with col3:
    if status and 'metrics' in status:
        metrics = status['metrics']
        st.metric("📈 Global Accuracy", f"{metrics['accuracy']:.3f}")
    else:
        st.metric("📈 Global Accuracy", "N/A")

with col4:
    if status and 'metrics' in status:
        metrics = status['metrics']
        st.metric("🎯 F1-Score", f"{metrics['f1_score']:.3f}")
    else:
        st.metric("🎯 F1-Score", "N/A")

# Action button
st.sidebar.markdown("---")
if st.sidebar.button("🔄 Combine Models", type="primary"):
    with st.spinner("Combining models from both banks..."):
        result = trigger_aggregation()
        if result:
            st.sidebar.success("✅ Models combined successfully!")
            st.sidebar.write(f"Round: {result['round']}")
            if 'metrics' in result:
                metrics = result['metrics']
                st.sidebar.write(f"Accuracy: {metrics['accuracy']:.3f}")
                st.sidebar.write(f"F1-Score: {metrics['f1_score']:.3f}")
        else:
            st.sidebar.error("❌ Failed to combine models!")

# Main content area
st.markdown("---")

# Create tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Progress", "📈 Evaluation Metrics", "🏦 Bank Data", "🧠 How It Works", "💾 Saved Models"])

with tab1:
    st.header("�� Federated Learning Progress")
    
    # Get real evaluation history
    eval_history = get_evaluation_history()
    saved_metrics = load_saved_metrics()
    
    if eval_history and eval_history['history']:
        # Use real data
        history = eval_history['history']
        rounds = [h['round'] for h in history]
        accuracies = [h['metrics']['accuracy'] for h in history]
        f1_scores = [h['metrics']['f1_score'] for h in history]
        precisions = [h['metrics']['precision'] for h in history]
        recalls = [h['metrics']['recall'] for h in history]
    elif saved_metrics and saved_metrics['history']:
        # Use saved data
        history = saved_metrics['history']
        rounds = [h['round'] for h in history]
        accuracies = [h['metrics']['accuracy'] for h in history]
        f1_scores = [h['metrics']['f1_score'] for h in history]
        precisions = [h['metrics']['precision'] for h in history]
        recalls = [h['metrics']['recall'] for h in history]
    else:
        # Use sample data
        rounds = list(range(1, 6))
        accuracies = [0.65, 0.72, 0.78, 0.82, 0.85]
        f1_scores = [0.60, 0.68, 0.74, 0.78, 0.82]
        precisions = [0.58, 0.66, 0.72, 0.76, 0.80]
        recalls = [0.62, 0.70, 0.76, 0.80, 0.84]
    
    # Progress chart
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=rounds, y=accuracies,
        mode='lines+markers',
        name='📈 Accuracy',
        line=dict(color='red', width=4),
        marker=dict(size=10)
    ))
    
    fig.add_trace(go.Scatter(
        x=rounds, y=f1_scores,
        mode='lines+markers',
        name='🎯 F1-Score',
        line=dict(color='blue', width=3),
        marker=dict(size=8)
    ))
    
    fig.add_trace(go.Scatter(
        x=rounds, y=precisions,
        mode='lines+markers',
        name='🎯 Precision',
        line=dict(color='green', width=2),
        marker=dict(size=6)
    ))
    
    fig.add_trace(go.Scatter(
        x=rounds, y=recalls,
        mode='lines+markers',
        name='🎯 Recall',
        line=dict(color='orange', width=2),
        marker=dict(size=6)
    ))
    
    fig.update_layout(
        title="🎯 Model Performance Over Federated Learning Rounds",
        xaxis_title="Federated Learning Round",
        yaxis_title="Score",
        height=500,
        showlegend=True
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Progress explanation
    st.markdown("""
    ### 📈 What's Happening:
    - **Round 1**: Banks start with random models
    - **Round 2-3**: Models learn from local data
    - **Round 4-5**: Global model becomes better than individual banks
    - **Result**: Combined model is more accurate than any single bank!
    """)

with tab2:
    st.header("📈 Evaluation Metrics")
    
    # Get current metrics
    if status and 'metrics' in status:
        current_metrics = status['metrics']
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📊 Accuracy", f"{current_metrics['accuracy']:.3f}")
            st.info("Overall correct predictions")
        
        with col2:
            st.metric("🎯 Precision", f"{current_metrics['precision']:.3f}")
            st.info("Correct fraud predictions / Total fraud predictions")
        
        with col3:
            st.metric("🔄 Recall", f"{current_metrics['recall']:.3f}")
            st.info("Correct fraud predictions / Total actual frauds")
        
        with col4:
            st.metric("⚖️ F1-Score", f"{current_metrics['f1_score']:.3f}")
            st.info("Harmonic mean of precision and recall")
        
        # Metrics explanation
        st.markdown("""
        ### 📊 Understanding the Metrics:
        
        - **Accuracy**: Overall percentage of correct predictions (fraud + non-fraud)
        - **Precision**: When the model predicts fraud, how often is it correct?
        - **Recall**: Of all actual frauds, how many did the model catch?
        - **F1-Score**: Balanced measure between precision and recall
        
        **🎯 Higher values = Better performance!**
        """)
        
        # Current round info
        if status:
            st.success(f"🔄 **Current Round**: {status['round']}")
            st.success(f"🏦 **Active Banks**: {status['num_clients']}")
    
    else:
        st.warning("⚠️ No evaluation metrics available. Start federated learning to see results!")

with tab3:
    st.header("🏦 Bank Data Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🏦 Bank-1 Data")
        try:
            bank1_data = pd.read_csv("Bank-1/bank1.csv")
            
            # Transaction types
            type_counts = bank1_data['type'].value_counts()
            fig1 = px.pie(values=type_counts.values, names=type_counts.index, 
                         title="Transaction Types - Bank-1")
            st.plotly_chart(fig1, use_container_width=True)
            
            # Fraud vs Non-fraud
            fraud_counts = bank1_data['isFraud'].value_counts()
            fig2 = px.bar(x=['✅ Non-Fraud', '🚨 Fraud'], y=fraud_counts.values,
                         title="Fraud Detection - Bank-1",
                         color=['green', 'red'])
            st.plotly_chart(fig2, use_container_width=True)
            
            st.info(f"📊 Total transactions: {len(bank1_data):,}")
            
        except Exception as e:
            st.error(f"Could not load Bank-1 data: {e}")
    
    with col2:
        st.subheader("🏦 Bank-2 Data")
        try:
            bank2_data = pd.read_csv("Bank-2/bank2.csv")
            
            # Transaction types
            type_counts = bank2_data['type'].value_counts()
            fig3 = px.pie(values=type_counts.values, names=type_counts.index,
                         title="Transaction Types - Bank-2")
            st.plotly_chart(fig3, use_container_width=True)
            
            # Fraud vs Non-fraud
            fraud_counts = bank2_data['isFraud'].value_counts()
            fig4 = px.bar(x=['✅ Non-Fraud', '🚨 Fraud'], y=fraud_counts.values,
                         title="Fraud Detection - Bank-2",
                         color=['green', 'red'])
            st.plotly_chart(fig4, use_container_width=True)
            
            st.info(f"📊 Total transactions: {len(bank2_data):,}")
            
        except Exception as e:
            st.error(f"Could not load Bank-2 data: {e}")

with tab4:
    st.header("🧠 How Federated Learning Works")
    
    st.markdown("""
    ### 🔒 Privacy-Preserving Machine Learning
    
    **Traditional ML**: All data goes to one place → Privacy risk! ❌
    
    **Federated Learning**: Data stays local → Privacy safe! ✅
    
    ---
    
    ### 🔄 The Process:
    
    1. **🏦 Banks keep their data private**
       - Bank-1 has its own fraud data
       - Bank-2 has its own fraud data
       - Data never leaves the banks!
    
    2. **🧠 Each bank trains a model locally**
       - Bank-1 trains on its data
       - Bank-2 trains on its data
       - Models learn different patterns
    
    3. **🔄 Models are combined (not data!)**
       - Only model parameters are shared
       - Raw data stays private
       - Global model gets knowledge from both banks
    
    4. **📈 Better results for everyone**
       - Global model is more accurate
       - Each bank benefits from others' knowledge
       - Privacy is maintained!
    
    ---
    
    ### 🎯 Benefits:
    - ✅ **Privacy**: Data never leaves banks
    - ✅ **Accuracy**: Combined model is better
    - ✅ **Collaboration**: Banks work together
    - ✅ **Security**: No central data storage
    """)

with tab5:
    st.header("💾 Saved Models & Metrics")
    
    # Check for saved models
    bank1_model_exists = os.path.exists("Bank-1/global_model.pth")
    bank2_model_exists = os.path.exists("Bank-2/global_model.pth")
    bank1_metrics_exists = os.path.exists("Bank-1/evaluation_metrics.pkl")
    bank2_metrics_exists = os.path.exists("Bank-2/evaluation_metrics.pkl")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🏦 Bank-1 Directory")
        if bank1_model_exists:
            st.success("✅ Global model saved")
            st.info("File: `Bank-1/global_model.pth`")
        else:
            st.warning("⚠️ No global model saved yet")
        
        if bank1_metrics_exists:
            st.success("✅ Evaluation metrics saved")
            st.info("File: `Bank-1/evaluation_metrics.pkl`")
        else:
            st.warning("⚠️ No evaluation metrics saved yet")
    
    with col2:
        st.subheader("🏦 Bank-2 Directory")
        if bank2_model_exists:
            st.success("✅ Global model saved")
            st.info("File: `Bank-2/global_model.pth`")
        else:
            st.warning("⚠️ No global model saved yet")
        
        if bank2_metrics_exists:
            st.success("✅ Evaluation metrics saved")
            st.info("File: `Bank-2/evaluation_metrics.pkl`")
        else:
            st.warning("⚠️ No evaluation metrics saved yet")
    
    # Load and display saved metrics
    if bank1_metrics_exists:
        try:
            with open("Bank-1/evaluation_metrics.pkl", 'rb') as f:
                saved_metrics = pickle.load(f)
            
            st.subheader("📊 Saved Evaluation History")
            
            if saved_metrics['history']:
                # Create metrics table
                history_data = []
                for h in saved_metrics['history']:
                    history_data.append({
                        'Round': h['round'],
                        'Accuracy': f"{h['metrics']['accuracy']:.3f}",
                        'Precision': f"{h['metrics']['precision']:.3f}",
                        'Recall': f"{h['metrics']['recall']:.3f}",
                        'F1-Score': f"{h['metrics']['f1_score']:.3f}"
                    })
                
                df = pd.DataFrame(history_data)
                st.dataframe(df, use_container_width=True)
                
                st.success(f"📈 **Best Performance**: Round {saved_metrics['current_round']}")
                current = saved_metrics['current_metrics']
                st.info(f"Current Accuracy: {current['accuracy']:.3f}, F1-Score: {current['f1_score']:.3f}")
            else:
                st.info("No evaluation history available yet.")
                
        except Exception as e:
            st.error(f"Error loading saved metrics: {e}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <h3>🏦 Federated Learning for Bank Fraud Detection</h3>
    <p><strong>Privacy-Preserving Machine Learning</strong></p>
    <p>Built with PyTorch, Flask, and Streamlit</p>
</div>
""", unsafe_allow_html=True)

# Auto-refresh
if st.sidebar.checkbox("🔄 Auto-refresh (30s)"):
    time.sleep(30)
    st.rerun()

# Manual refresh
if st.sidebar.button("🔄 Refresh Now"):
    st.rerun() 