import pandas as pd
import numpy as np

def calculate_financial_risk(df):
    """Score based on charges and tenure"""
    score = pd.Series(0.0, index=df.index)
    
    if 'tenure' in df.columns:
        # Low tenure = high risk
        score += (1 - (df['tenure'] / df['tenure'].max())) * 30
    
    if 'MonthlyCharges' in df.columns:
        # High charges = higher risk
        score += (df['MonthlyCharges'] / df['MonthlyCharges'].max()) * 20
        
    return score

def calculate_engagement_risk(df):
    """Score based on product usage"""
    score = pd.Series(0.0, index=df.index)
    
    if 'numAdminTickets' in df.columns:
        score += (df['numAdminTickets'] / df['numAdminTickets'].max()) * 20
    
    if 'Contract' in df.columns:
        # Month-to-month = highest risk
        score += df['Contract'].map({
            'Month-to-month': 25,
            'One year': 10,
            'Two year': 0
        }).fillna(0)
        
    return score

def calculate_satisfaction_risk(df):
    """Score based on support tickets"""
    score = pd.Series(0.0, index=df.index)
    
    if 'numTechTickets' in df.columns:
        score += (df['numTechTickets'] / df['numTechTickets'].max()) * 25
        
    return score

def calculate_risk_score(df):
    """Calculate final 0-100 risk score"""
    print("⚙️ Calculating risk scores...")
    
    df = df.copy()
    
    financial = calculate_financial_risk(df)
    engagement = calculate_engagement_risk(df)
    satisfaction = calculate_satisfaction_risk(df)
    
    # Combine all scores
    df['risk_score'] = financial + engagement + satisfaction
    
    # Normalize to 0-100
    df['risk_score'] = (df['risk_score'] / df['risk_score'].max()) * 100
    df['risk_score'] = df['risk_score'].round(2)
    
    # Add risk category
    df['risk_category'] = pd.cut(
        df['risk_score'],
        bins=[0, 30, 60, 80, 100],
        labels=['Low', 'Medium', 'High', 'Critical']
    )
    
    print(f"✅ Risk scores calculated!")
    print(f"\n📊 Risk Distribution:")
    print(df['risk_category'].value_counts())
    
    return df