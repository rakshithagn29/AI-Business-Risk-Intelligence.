# api/routes/risk.py
# Risk assessment API endpoints

from fastapi import APIRouter
from api.schemas.customer_schema import CustomerInput
import pandas as pd
import numpy as np
import os

router = APIRouter()

BASE = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))

def calculate_risk_score(customer: CustomerInput):
    """Calculate risk score from customer data"""

    score = 0

    # Financial risk — tenure
    if customer.tenure < 6:
        score += 30
    elif customer.tenure < 12:
        score += 20
    elif customer.tenure < 24:
        score += 10

    # Monthly charges risk
    if customer.MonthlyCharges > 80:
        score += 20
    elif customer.MonthlyCharges > 60:
        score += 10

    # Contract risk
    if customer.Contract == "Month-to-month":
        score += 25
    elif customer.Contract == "One year":
        score += 10

    # Service risks
    if customer.OnlineSecurity == "No":
        score += 8
    if customer.TechSupport == "No":
        score += 7

    # Payment method risk
    if customer.PaymentMethod == "Electronic check":
        score += 10

    # Normalize to 0-100
    score = min(score, 100)

    # Risk category
    if score >= 75:
        category = "🔴 CRITICAL"
    elif score >= 50:
        category = "🟠 HIGH"
    elif score >= 25:
        category = "🟡 MEDIUM"
    else:
        category = "🟢 LOW"

    return score, category


def get_top_risk_factors(customer: CustomerInput):
    """Get top risk factors for customer"""
    factors = []

    if customer.Contract == "Month-to-month":
        factors.append("Month-to-month contract — highest churn risk")
    if customer.tenure < 12:
        factors.append(f"Low tenure ({customer.tenure} months) — new customer")
    if customer.MonthlyCharges > 80:
        factors.append(f"High monthly charges — ₹{customer.MonthlyCharges}")
    if customer.OnlineSecurity == "No":
        factors.append("No online security subscription")
    if customer.TechSupport == "No":
        factors.append("No tech support subscription")
    if customer.PaymentMethod == "Electronic check":
        factors.append("Electronic check payment — higher churn rate")
    if customer.InternetService == "Fiber optic":
        factors.append("Fiber optic — high cost service")

    return factors[:5]  # Top 5 factors


def get_recommendation(risk_score: float, monthly: float):
    """Get AI recommendation based on risk"""
    annual = monthly * 12

    if risk_score >= 75:
        return f"🔴 CRITICAL: Call customer TODAY! Offer 30% discount. Budget: ₹{annual*0.3:.0f}"
    elif risk_score >= 50:
        return f"🟠 HIGH: Send email offer within 2 days. Offer 20% discount. Budget: ₹{annual*0.2:.0f}"
    elif risk_score >= 25:
        return f"🟡 MEDIUM: Send newsletter this week. Offer 10% loyalty discount."
    else:
        return f"🟢 LOW: Customer is happy! Consider upsell opportunity."


@router.post("/risk/assess")
def assess_customer_risk(customer: CustomerInput):
    """
    Assess risk for a customer
    Send customer details — get risk score back!
    """
    try:
        risk_score, risk_category = calculate_risk_score(customer)
        top_factors = get_top_risk_factors(customer)
        recommendation = get_recommendation(
            risk_score, customer.MonthlyCharges)

        return {
            "status": "success",
            "risk_assessment": {
                "risk_score": risk_score,
                "risk_category": risk_category,
                "top_risk_factors": top_factors,
                "recommendation": recommendation,
                "monthly_revenue": customer.MonthlyCharges,
                "annual_revenue": customer.MonthlyCharges * 12,
                "revenue_at_risk": round(
                    customer.MonthlyCharges * 12 * (risk_score/100), 2)
            }
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}


@router.get("/risk/summary")
def get_risk_summary():
    """Get overall risk summary of all customers"""
    try:
        df = pd.read_csv(
            os.path.join(BASE, "data", "processed",
                        "telco_with_predictions.csv")
        )

        total = len(df)
        critical = len(df[df['churn_prob_30day'] >= 75])
        high = len(df[(df['churn_prob_30day'] >= 50) &
                      (df['churn_prob_30day'] < 75)])

        if 'MonthlyCharges' in df.columns:
            total_revenue = df['MonthlyCharges'].sum() * 12
            at_risk = df[df['churn_prob_30day'] >= 50][
                'MonthlyCharges'].sum() * 12
        else:
            total_revenue = total * 65 * 12
            at_risk = (critical + high) * 65 * 12

        return {
            "status": "success",
            "summary": {
                "total_customers": total,
                "critical_risk": critical,
                "high_risk": high,
                "total_annual_revenue": round(total_revenue, 2),
                "revenue_at_risk": round(at_risk, 2),
                "safe_revenue": round(total_revenue - at_risk, 2),
                "avg_churn_risk": round(
                    df['churn_prob_30day'].mean(), 2)
            }
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}