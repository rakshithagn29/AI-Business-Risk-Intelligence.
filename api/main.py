# api/main.py
# Main FastAPI application entry point

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Import all routes
from api.routes.customers import router as customers_router
from api.routes.risk import router as risk_router
from api.routes.churn import router as churn_router

# Create FastAPI app
app = FastAPI(
    title="🚨 SENTINEL AI — Business Risk Intelligence API",
    description="""
    ## AI Driven Business Risk Intelligence
    ### Customer Risk Assessment & Churn Prediction

    This API provides:
    - **Customer Risk Scoring** — Multi dimensional risk assessment
    - **Churn Prediction** — Triple horizon prediction (30/60/90 days)
    - **Revenue at Risk** — Calculate business impact in ₹
    - **Action Recommendations** — AI powered retention suggestions

    Built with Python, XGBoost, FastAPI
    """,
    version="1.0.0",
    contact={
        "name": "Rakshitha",
        "url": "https://github.com/rakshithagn29/AI-Business-Risk-Intelligence"
    }
)

# Allow dashboard to talk to API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Include all routes
app.include_router(customers_router, prefix="/api/v1", tags=["Customers"])
app.include_router(risk_router, prefix="/api/v1", tags=["Risk Assessment"])
app.include_router(churn_router, prefix="/api/v1", tags=["Churn Prediction"])

# Home route
@app.get("/")
def home():
    return {
        "message": "🚨 Welcome to SENTINEL AI API",
        "project": "AI Driven Business Risk Intelligence",
        "version": "1.0.0",
        "status": "✅ API is running perfectly!",
        "endpoints": {
            "docs": "/docs",
            "customers": "/api/v1/customers",
            "risk": "/api/v1/risk",
            "churn": "/api/v1/churn"
        }
    }

# Health check
@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "message": "✅ All systems operational!"
    }

if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )