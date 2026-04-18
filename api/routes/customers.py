from fastapi import APIRouter
import pandas as pd
import os

router = APIRouter()

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

@router.get("/customers")
def get_all_customers():
    try:
        df = pd.read_csv(os.path.join(BASE, "data", "processed", "telco_with_predictions.csv"))
        total = len(df)
        critical = len(df[df['churn_prob_30day'] >= 75])
        high = len(df[(df['churn_prob_30day'] >= 50) & (df['churn_prob_30day'] < 75)])
        medium = len(df[(df['churn_prob_30day'] >= 25) & (df['churn_prob_30day'] < 50)])
        low = len(df[df['churn_prob_30day'] < 25])
        return {
            "status": "success",
            "total_customers": total,
            "risk_breakdown": {
                "critical": critical,
                "high": high,
                "medium": medium,
                "low": low
            },
            "avg_churn_risk": round(df['churn_prob_30day'].mean(), 2)
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

@router.get("/customers/{customer_id}")
def get_customer_by_id(customer_id: int):
    try:
        df = pd.read_csv(os.path.join(BASE, "data", "processed", "telco_with_predictions.csv"))
        if customer_id >= len(df) or customer_id < 0:
            return {"status": "error", "message": f"Customer {customer_id} not found"}
        customer = df.iloc[customer_id]
        prob = customer['churn_prob_30day']
        if prob >= 75: risk_level = "🔴 CRITICAL"
        elif prob >= 50: risk_level = "🟠 HIGH"
        elif prob >= 25: risk_level = "🟡 MEDIUM"
        else: risk_level = "🟢 LOW"
        return {
            "status": "success",
            "customer_id": customer_id,
            "churn_probability": {
                "30_day": round(float(customer['churn_prob_30day']), 2),
                "60_day": round(float(customer['churn_prob_60day']), 2),
                "90_day": round(float(customer['churn_prob_90day']), 2)
            },
            "risk_level": risk_level,
            "monthly_charges": round(float(customer.get('MonthlyCharges', 65)), 2),
            "tenure": int(customer.get('tenure', 0))
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

@router.get("/customers/top-risk/{n}")
def get_top_risk_customers(n: int = 10):
    try:
        df = pd.read_csv(os.path.join(BASE, "data", "processed", "telco_with_predictions.csv"))
        top = df.nlargest(n, 'churn_prob_30day')[
            ['churn_prob_30day', 'churn_prob_60day', 'churn_prob_90day', 'churn_risk']
        ].reset_index()
        customers = []
        for _, row in top.iterrows():
            customers.append({
                "customer_id": int(row['index']),
                "churn_30day": round(float(row['churn_prob_30day']), 2),
                "churn_60day": round(float(row['churn_prob_60day']), 2),
                "churn_90day": round(float(row['churn_prob_90day']), 2),
                "risk_level": row['churn_risk']
            })
        return {"status": "success", "top_risk_customers": customers}
    except Exception as e:
        return {"status": "error", "message": str(e)}