# api/routes/churn.py
# Churn prediction API endpoints

from fastapi import APIRouter
from api.schemas.customer_schema import CustomerInput
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os

router = APIRouter()

BASE = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
MODEL_PATH = os.path.join(BASE, "models_saved")

def load_models():
    """Load trained models"""
    try:
        m30 = joblib.load(os.path.join(MODEL_PATH, "churn_30day.pkl"))
        m60 = joblib.load(os.path.join(MODEL_PATH, "churn_60day.pkl"))
        m90 = joblib.load(os.path.join(MODEL_PATH, "churn_90day.pkl"))
        scaler = joblib.load(os.path.join(MODEL_PATH, "scaler.pkl"))
        return m30, m60, m90, scaler
    except Exception as e:
        return None, None, None, None

def customer_to_df(customer: CustomerInput):
    """Convert customer input to DataFrame"""
    data = {
        'tenure': [customer.tenure],
        'MonthlyCharges': [customer.MonthlyCharges],
        'TotalCharges': [customer.TotalCharges],
        'Contract': [customer.Contract],
        'PaymentMethod': [customer.PaymentMethod],
        'InternetService': [customer.InternetService],
        'OnlineSecurity': [customer.OnlineSecurity],
        'TechSupport': [customer.TechSupport],
        'StreamingTV': [customer.StreamingTV],
        'StreamingMovies': [customer.StreamingMovies],
        'PaperlessBilling': [customer.PaperlessBilling],
        'SeniorCitizen': [customer.SeniorCitizen],
        'Partner': [customer.Partner],
        'Dependents': [customer.Dependents],
        'PhoneService': [customer.PhoneService],
        'MultipleLines': [customer.MultipleLines]
    }
    return pd.DataFrame(data)

def encode_df(df):
    """Encode categorical columns"""
    le = LabelEncoder()
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        df[col] = le.fit_transform(df[col].astype(str))
    return df

def get_churn_action(prob, monthly):
    """Get recommended action based on churn probability"""
    annual = monthly * 12
    if prob >= 75:
        return {
            "urgency": "🔴 CRITICAL — Act TODAY!",
            "action": "Personal phone call + 30% discount offer",
            "budget": f"₹{annual*0.3:.0f}",
            "timeline": "Within 24 hours"
        }
    elif prob >= 50:
        return {
            "urgency": "🟠 HIGH — Act within 3 days",
            "action": "Personalized email + 20% discount",
            "budget": f"₹{annual*0.2:.0f}",
            "timeline": "Within 3 days"
        }
    elif prob >= 25:
        return {
            "urgency": "🟡 MEDIUM — Act this week",
            "action": "Newsletter + 10% loyalty discount",
            "budget": f"₹{annual*0.1:.0f}",
            "timeline": "Within 1 week"
        }
    else:
        return {
            "urgency": "🟢 LOW — Customer is safe",
            "action": "Maintain service quality + upsell",
            "budget": "Minimal",
            "timeline": "Monthly check"
        }


@router.post("/churn/predict")
def predict_churn(customer: CustomerInput):
    """
    Predict churn probability for a customer
    Returns 30, 60, and 90 day predictions!
    """
    try:
        m30, m60, m90, scaler = load_models()

        if m30 is None:
            return {
                "status": "error",
                "message": "Models not loaded. Check models_saved folder."
            }

        # Convert to DataFrame
        df = customer_to_df(customer)
        df_encoded = encode_df(df)

        # Get expected features from model
        expected_features = m30.feature_names_in_
        for col in expected_features:
            if col not in df_encoded.columns:
                df_encoded[col] = 0

        df_encoded = df_encoded[expected_features]

        # Scale
        df_scaled = pd.DataFrame(
            scaler.transform(df_encoded),
            columns=df_encoded.columns
        )

        # Predict
        prob_30 = float(m30.predict_proba(df_scaled)[0][1] * 100)
        prob_60 = float(m60.predict_proba(df_scaled)[0][1] * 100)
        prob_90 = float(m90.predict_proba(df_scaled)[0][1] * 100)

        # Risk level
        if prob_30 >= 75:
            risk_level = "🔴 CRITICAL"
        elif prob_30 >= 50:
            risk_level = "🟠 HIGH"
        elif prob_30 >= 25:
            risk_level = "🟡 MEDIUM"
        else:
            risk_level = "🟢 LOW"

        action = get_churn_action(prob_30, customer.MonthlyCharges)

        return {
            "status": "success",
            "churn_prediction": {
                "30_day_probability": round(prob_30, 2),
                "60_day_probability": round(prob_60, 2),
                "90_day_probability": round(prob_90, 2),
                "risk_level": risk_level,
                "recommended_action": action,
                "monthly_revenue": customer.MonthlyCharges,
                "annual_revenue_at_risk": round(
                    customer.MonthlyCharges * 12 * (prob_30/100), 2)
            }
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}


@router.post("/churn/whatif")
def whatif_simulation(customer: CustomerInput, action: str = "discount_20"):
    """
    Simulate What-If scenarios!
    See how churn changes after taking an action.
    """
    try:
        m30, m60, m90, scaler = load_models()

        if m30 is None:
            return {
                "status": "error",
                "message": "Models not loaded!"
            }

        df = customer_to_df(customer)
        df_encoded = encode_df(df)

        expected_features = m30.feature_names_in_
        for col in expected_features:
            if col not in df_encoded.columns:
                df_encoded[col] = 0
        df_encoded = df_encoded[expected_features]

        df_scaled = pd.DataFrame(
            scaler.transform(df_encoded),
            columns=df_encoded.columns
        )

        # Original probability
        orig_prob = float(m30.predict_proba(df_scaled)[0][1] * 100)

        # Action reductions
        reductions = {
            'discount_10': 8,
            'discount_20': 18,
            'discount_30': 28,
            'upgrade': 22,
            'support': 15,
            'loyalty': 12,
            'all': 45
        }

        costs = {
            'discount_10': customer.MonthlyCharges * 0.10,
            'discount_20': customer.MonthlyCharges * 0.20,
            'discount_30': customer.MonthlyCharges * 0.30,
            'upgrade': 500,
            'support': 200,
            'loyalty': 150,
            'all': 800
        }

        reduction = reductions.get(action, 10)
        cost = costs.get(action, 100)

        new_prob = max(orig_prob - reduction, 2)
        annual_revenue = customer.MonthlyCharges * 12
        revenue_saved = annual_revenue - cost

        return {
            "status": "success",
            "whatif_simulation": {
                "action_taken": action,
                "before_churn_prob": round(orig_prob, 2),
                "after_churn_prob": round(new_prob, 2),
                "improvement": round(orig_prob - new_prob, 2),
                "action_cost": round(cost, 2),
                "annual_revenue": round(annual_revenue, 2),
                "revenue_saved": round(revenue_saved, 2),
                "verdict": "✅ TAKE THIS ACTION!" if revenue_saved > 0
                          else "❌ NOT WORTH IT"
            }
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}


@router.get("/churn/stats")
def get_churn_stats():
    """Get overall churn statistics"""
    try:
        df = pd.read_csv(
            os.path.join(BASE, "data", "processed",
                        "telco_with_predictions.csv")
        )

        return {
            "status": "success",
            "churn_stats": {
                "avg_30day_risk": round(
                    df['churn_prob_30day'].mean(), 2),
                "avg_60day_risk": round(
                    df['churn_prob_60day'].mean(), 2),
                "avg_90day_risk": round(
                    df['churn_prob_90day'].mean(), 2),
                "high_risk_30day": len(
                    df[df['churn_prob_30day'] >= 50]),
                "high_risk_60day": len(
                    df[df['churn_prob_60day'] >= 50]),
                "high_risk_90day": len(
                    df[df['churn_prob_90day'] >= 50]),
                "total_customers": len(df)
            }
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}