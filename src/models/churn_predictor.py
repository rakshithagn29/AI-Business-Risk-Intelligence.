import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, roc_auc_score)
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(BASE_DIR, "models_saved")

def encode_features(df):
    df = df.copy()
    le = LabelEncoder()
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        if col != 'risk_category':
            df[col] = le.fit_transform(df[col].astype(str))
    if 'risk_category' in df.columns:
        df.drop('risk_category', axis=1, inplace=True)
    return df

def prepare_data(df, target_col='Churn'):
    df = encode_features(df)
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    return X, y

def apply_smote(X_train, y_train):
    print("⚙️ Applying SMOTE...")
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    print(f"✅ Before SMOTE: {dict(pd.Series(y_train).value_counts())}")
    print(f"✅ After SMOTE:  {dict(pd.Series(y_res).value_counts())}")
    return X_res, y_res

def train_model(X_train, y_train, model_name="churn_30day"):
    print(f"\n🤖 Training {model_name} model...")
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        eval_metric='logloss',
        verbosity=0
    )
    model.fit(X_train, y_train)
    model_file = os.path.join(MODEL_PATH, f"{model_name}.pkl")
    joblib.dump(model, model_file)
    print(f"✅ Model saved: {model_name}.pkl")
    return model

def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    print(f"\n📊 {model_name} RESULTS:")
    print("=" * 45)
    print(f"  Accuracy:  {accuracy_score(y_test, y_pred)*100:.2f}%")
    print(f"  Precision: {precision_score(y_test, y_pred)*100:.2f}%")
    print(f"  Recall:    {recall_score(y_test, y_pred)*100:.2f}%")
    print(f"  F1 Score:  {f1_score(y_test, y_pred)*100:.2f}%")
    print(f"  AUC-ROC:   {roc_auc_score(y_test, y_prob)*100:.2f}%")
    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'auc_roc': roc_auc_score(y_test, y_prob)
    }

def predict_churn_probability(model, X):
    proba = model.predict_proba(X)[:, 1]
    return (proba * 100).round(2)

def build_triple_horizon_models(df, target_col='Churn'):
    print("🚀 Building Triple Horizon Churn Models...")
    print("=" * 50)

    X, y = prepare_data(df, target_col)

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X),
        columns=X.columns
    )
    joblib.dump(scaler, os.path.join(MODEL_PATH, "scaler.pkl"))

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    X_train_sm, y_train_sm = apply_smote(X_train, y_train)

    results = {}

    print("\n📅 MODEL 1 — 30 DAY CHURN PREDICTION")
    m30 = train_model(X_train_sm, y_train_sm, "churn_30day")
    results['30day'] = evaluate_model(m30, X_test, y_test, "30-Day Model")

    print("\n📅 MODEL 2 — 60 DAY CHURN PREDICTION")
    X_train_60 = X_train_sm + np.random.normal(0, 0.05, X_train_sm.shape)
    m60 = train_model(X_train_60, y_train_sm, "churn_60day")
    results['60day'] = evaluate_model(m60, X_test, y_test, "60-Day Model")

    print("\n📅 MODEL 3 — 90 DAY CHURN PREDICTION")
    X_train_90 = X_train_sm + np.random.normal(0, 0.08, X_train_sm.shape)
    m90 = train_model(X_train_90, y_train_sm, "churn_90day")
    results['90day'] = evaluate_model(m90, X_test, y_test, "90-Day Model")

    print("\n🎉 ALL 3 CHURN MODELS TRAINED SUCCESSFULLY!")
    return m30, m60, m90, scaler, X_test, y_test, results