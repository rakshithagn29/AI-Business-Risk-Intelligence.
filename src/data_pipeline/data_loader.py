import pandas as pd
import os

# This automatically finds the correct path no matter where you run from
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH = os.path.join(BASE_DIR, "data", "raw")

def load_telco_data():
    path = os.path.join(DATA_PATH, "telco_churn.csv")
    df = pd.read_csv(path)
    print(f"✅ Telco loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df

def load_bank_data():
    path = os.path.join(DATA_PATH, "bank_customers.csv")
    df = pd.read_csv(path)
    print(f"✅ Bank loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df

def load_ecommerce_data():
    path = os.path.join(DATA_PATH, "ecommerce_customers.csv")
    df = pd.read_csv(path)
    print(f"✅ Ecommerce loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df

def load_all_data():
    print("📂 Loading all datasets...")
    telco = load_telco_data()
    bank = load_bank_data()
    ecommerce = load_ecommerce_data()
    print("\n🎉 All datasets loaded successfully!")
    return telco, bank, ecommerce