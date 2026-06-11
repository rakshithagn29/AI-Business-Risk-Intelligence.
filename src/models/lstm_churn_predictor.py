# src/models/lstm_churn_predictor.py
# Deep Learning LSTM model for churn prediction

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, roc_auc_score,
                             precision_score, recall_score, f1_score)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
MODEL_PATH = os.path.join(BASE_DIR, "models_saved")

# ============================================================
# LSTM MODEL ARCHITECTURE
# ============================================================
class ChurnLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64,
                 num_layers=2, dropout=0.3):
        super(ChurnLSTM, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Attention mechanism
        self.attention = nn.Linear(hidden_size, 1)

        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 1)

        # Activation functions
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)
        self.batch_norm = nn.BatchNorm1d(32)

    def forward(self, x):
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers,
                         x.size(0),
                         self.hidden_size)
        c0 = torch.zeros(self.num_layers,
                         x.size(0),
                         self.hidden_size)

        # LSTM forward pass
        out, _ = self.lstm(x, (h0, c0))

        # Attention weights
        attention_weights = torch.softmax(
            self.attention(out), dim=1)
        context = torch.sum(
            attention_weights * out, dim=1)

        # Fully connected layers
        out = self.fc1(context)
        out = self.batch_norm(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc3(out)
        out = self.sigmoid(out)

        return out.squeeze()


# ============================================================
# DATA PREPARATION
# ============================================================
def encode_data(df):
    """Encode categorical features"""
    df = df.copy()
    le = LabelEncoder()

    # Drop prediction columns if present
    drop_cols = ['churn_prob_30day', 'churn_prob_60day',
                 'churn_prob_90day', 'churn_risk',
                 'risk_category', 'risk_score']
    for col in drop_cols:
        if col in df.columns:
            df.drop(col, axis=1, inplace=True)

    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        df[col] = le.fit_transform(df[col].astype(str))

    return df


def prepare_lstm_data(df, target_col='Churn', sequence_len=1):
    """Prepare data for LSTM training"""
    print("📦 Preparing data for LSTM...")

    df_encoded = encode_data(df.copy())

    X = df_encoded.drop(target_col, axis=1)
    y = df_encoded[target_col]

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Save scaler
    joblib.dump(scaler,
                os.path.join(MODEL_PATH, "lstm_scaler.pkl"))

    # Reshape for LSTM — (samples, sequence_len, features)
    X_lstm = X_scaled.reshape(
        X_scaled.shape[0], sequence_len, X_scaled.shape[1])

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_lstm, y.values,
        test_size=0.2, random_state=42, stratify=y)

    print(f"✅ Training samples: {len(X_train)}")
    print(f"✅ Testing samples:  {len(X_test)}")
    print(f"✅ Features:         {X_scaled.shape[1]}")

    return X_train, X_test, y_train, y_test, X.columns.tolist()


# ============================================================
# TRAINING
# ============================================================
def train_lstm_model(df, target_col='Churn',
                     epochs=50, batch_size=32):
    """Train LSTM deep learning model"""

    print("\n🧠 TRAINING LSTM DEEP LEARNING MODEL")
    print("="*50)

    # Prepare data
    X_train, X_test, y_train, y_test, feature_names = \
        prepare_lstm_data(df, target_col)

    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train)
    X_test_t  = torch.FloatTensor(X_test)
    y_test_t  = torch.FloatTensor(y_test)

    # Create DataLoader
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    # Initialize model
    input_size = X_train.shape[2]
    model = ChurnLSTM(
        input_size=input_size,
        hidden_size=64,
        num_layers=2,
        dropout=0.3
    )

    # Loss and optimizer
    # Handle class imbalance
    pos_weight = torch.tensor(
        [len(y_train) / (2 * sum(y_train))])
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=10, gamma=0.5)

    # Training loop
    print("\n⚙️ Training in progress...")
    train_losses = []
    best_auc = 0

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0

        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)

        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                test_outputs = model(X_test_t)
                test_pred = (test_outputs > 0.5).float()
                test_proba = test_outputs.numpy()

            auc = roc_auc_score(y_test, test_proba)
            acc = accuracy_score(
                y_test, test_pred.numpy())

            print(f"  Epoch {epoch+1:3d}/{epochs} | "
                  f"Loss: {avg_loss:.4f} | "
                  f"AUC: {auc*100:.2f}% | "
                  f"Acc: {acc*100:.2f}%")

            if auc > best_auc:
                best_auc = auc
                torch.save(model.state_dict(),
                          os.path.join(MODEL_PATH,
                                      "lstm_best.pth"))

    # Load best model
    model.load_state_dict(
        torch.load(os.path.join(MODEL_PATH, "lstm_best.pth")))

    # Final evaluation
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_t)
        test_pred = (test_outputs > 0.5).float().numpy()
        test_proba = test_outputs.numpy()

    results = {
        'accuracy':  accuracy_score(y_test, test_pred),
        'precision': precision_score(y_test, test_pred),
        'recall':    recall_score(y_test, test_pred),
        'f1':        f1_score(y_test, test_pred),
        'auc_roc':   roc_auc_score(y_test, test_proba)
    }

    print(f"\n📊 LSTM FINAL RESULTS:")
    print("="*45)
    print(f"  Accuracy  : {results['accuracy']*100:.2f}%")
    print(f"  Precision : {results['precision']*100:.2f}%")
    print(f"  Recall    : {results['recall']*100:.2f}%")
    print(f"  F1 Score  : {results['f1']*100:.2f}%")
    print(f"  AUC-ROC   : {results['auc_roc']*100:.2f}%")

    # Save model info
    model_info = {
        'input_size':     input_size,
        'hidden_size':    64,
        'num_layers':     2,
        'feature_names':  feature_names,
        'train_losses':   train_losses
    }
    joblib.dump(model_info,
                os.path.join(MODEL_PATH, "lstm_info.pkl"))

    print("\n✅ LSTM model saved successfully!")

    return model, results, X_test, y_test, test_proba


# ============================================================
# ENSEMBLE — COMBINE XGBoost + LSTM
# ============================================================
def ensemble_predict(xgb_model, lstm_model,
                     X_scaled, lstm_scaler,
                     xgb_weight=0.6, lstm_weight=0.4):
    """
    Combine XGBoost + LSTM predictions
    XGBoost is more accurate so gets higher weight
    """
    # XGBoost prediction
    xgb_proba = xgb_model.predict_proba(X_scaled)[:, 1]

    # LSTM prediction
    X_lstm_input = lstm_scaler.transform(
        X_scaled.values if hasattr(X_scaled, 'values')
        else X_scaled)
    X_lstm_tensor = torch.FloatTensor(
        X_lstm_input.reshape(
            X_lstm_input.shape[0], 1,
            X_lstm_input.shape[1]))

    lstm_model.eval()
    with torch.no_grad():
        lstm_proba = lstm_model(X_lstm_tensor).numpy()

    # Weighted ensemble
    ensemble_proba = (xgb_weight * xgb_proba +
                      lstm_weight * lstm_proba)

    return ensemble_proba * 100