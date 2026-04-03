import pandas as pd
import numpy as np

def clean_telco(df):
    print("🧹 Cleaning Telco dataset...")
    
    # Make a copy
    df = df.copy()
    
    # Fix TotalCharges - it has spaces, convert to number
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    
    # Fill missing TotalCharges with 0
    df['TotalCharges'].fillna(0, inplace=True)
    
    # Convert Churn Yes/No to 1/0
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    
    # Drop customerID - not needed
    df.drop('customerID', axis=1, inplace=True)
    
    # Convert SeniorCitizen to object
    df['SeniorCitizen'] = df['SeniorCitizen'].astype(str)
    
    print(f"✅ Telco cleaned: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"   Missing values remaining: {df.isnull().sum().sum()}")
    return df

def clean_bank(df):
    print("🧹 Cleaning Bank dataset...")
    
    df = df.copy()
    
    # Drop unnecessary columns
    df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1, inplace=True)
    
    print(f"✅ Bank cleaned: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"   Missing values remaining: {df.isnull().sum().sum()}")
    return df

def clean_ecommerce(df):
    print("🧹 Cleaning Ecommerce dataset...")
    
    df = df.copy()
    
    # Fill missing numerical values with median
    num_cols = df.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        df[col].fillna(df[col].median(), inplace=True)
    
    # Fill missing categorical values with mode
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        df[col].fillna(df[col].mode()[0], inplace=True)
    
    print(f"✅ Ecommerce cleaned: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"   Missing values remaining: {df.isnull().sum().sum()}")
    return df

def clean_all(telco, bank, ecommerce):
    print("📂 Cleaning all datasets...")
    telco_clean = clean_telco(telco)
    bank_clean = clean_bank(bank)
    ecommerce_clean = clean_ecommerce(ecommerce)
    print("\n🎉 All datasets cleaned!")
    return telco_clean, bank_clean, ecommerce_clean