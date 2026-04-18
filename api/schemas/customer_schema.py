# api/schemas/customer_schema.py
# Data models for API requests and responses

from pydantic import BaseModel
from typing import Optional

class CustomerInput(BaseModel):
    """Input data for a single customer prediction"""
    tenure: float = 12
    MonthlyCharges: float = 65.0
    TotalCharges: float = 780.0
    Contract: str = "Month-to-month"
    PaymentMethod: str = "Electronic check"
    InternetService: str = "Fiber optic"
    OnlineSecurity: str = "No"
    TechSupport: str = "No"
    StreamingTV: str = "No"
    StreamingMovies: str = "No"
    PaperlessBilling: str = "Yes"
    SeniorCitizen: str = "0"
    Partner: str = "No"
    Dependents: str = "No"
    PhoneService: str = "Yes"
    MultipleLines: str = "No"

    class Config:
        json_schema_extra = {
            "example": {
                "tenure": 12,
                "MonthlyCharges": 65.0,
                "TotalCharges": 780.0,
                "Contract": "Month-to-month",
                "PaymentMethod": "Electronic check",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "No",
                "TechSupport": "No",
                "StreamingTV": "No",
                "StreamingMovies": "No",
                "PaperlessBilling": "Yes",
                "SeniorCitizen": "0",
                "Partner": "No",
                "Dependents": "No",
                "PhoneService": "Yes",
                "MultipleLines": "No"
            }
        }

class RiskResponse(BaseModel):
    """Response model for risk assessment"""
    customer_data: dict
    risk_score: float
    risk_category: str
    churn_prob_30day: float
    churn_prob_60day: float
    churn_prob_90day: float
    top_risk_factors: list
    recommended_action: str
    revenue_at_risk: float

class ChurnResponse(BaseModel):
    """Response model for churn prediction"""
    churn_prob_30day: float
    churn_prob_60day: float
    churn_prob_90day: float
    risk_level: str
    recommended_action: str

class RevenueResponse(BaseModel):
    """Response model for revenue calculation"""
    monthly_revenue: float
    annual_revenue: float
    revenue_at_risk: float
    action_cost: float
    potential_saving: float
    recommendation: str