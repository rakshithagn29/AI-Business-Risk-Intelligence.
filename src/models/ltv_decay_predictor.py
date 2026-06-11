import pandas as pd
import numpy as np
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(BASE_DIR, "models_saved")


def calculate_customer_ltv(monthly_charges, tenure,
                            churn_prob_30, churn_prob_60,
                            churn_prob_90):
    current_ltv = monthly_charges * 12
    survival_30 = (100 - churn_prob_30) / 100
    survival_60 = (100 - churn_prob_60) / 100
    survival_90 = (100 - churn_prob_90) / 100

    months = list(range(0, 13))
    decay_values = []

    for month in months:
        if month == 0:
            value = current_ltv
        elif month <= 1:
            value = current_ltv * survival_30
        elif month <= 2:
            blend = month - 1
            value = current_ltv * (
                survival_30 * (1 - blend) +
                survival_60 * blend)
        elif month <= 3:
            blend = month - 2
            value = current_ltv * (
                survival_60 * (1 - blend) +
                survival_90 * blend)
        else:
            decay_rate = (
                current_ltv -
                current_ltv * survival_90) / 3
            value = max(
                current_ltv * survival_90 -
                decay_rate * (month - 3), 0)
        decay_values.append(round(value, 2))

    turning_point = None
    for i, val in enumerate(decay_values):
        if val < current_ltv * 0.5:
            turning_point = i
            break
    if turning_point is None:
        turning_point = 12

    intervention_month = max(1, turning_point - 1)
    value_at_risk = current_ltv - decay_values[-1]

    if current_ltv > 0:
        decay_rate_pct = (
            (current_ltv - decay_values[3]) /
            current_ltv * 100)
    else:
        decay_rate_pct = 0

    if decay_rate_pct < 15:
        decay_pattern = "🟢 STABLE"
        urgency = "Low"
    elif decay_rate_pct < 35:
        decay_pattern = "🟡 DECLINING"
        urgency = "Medium"
    elif decay_rate_pct < 60:
        decay_pattern = "🟠 RAPID DECLINE"
        urgency = "High"
    else:
        decay_pattern = "🔴 CRITICAL DECAY"
        urgency = "Critical"

    return {
        'current_ltv':        current_ltv,
        'decay_curve':        decay_values,
        'months':             months,
        'turning_point':      turning_point,
        'intervention_month': intervention_month,
        'value_at_risk':      round(value_at_risk, 2),
        'decay_rate_pct':     round(decay_rate_pct, 1),
        'decay_pattern':      decay_pattern,
        'urgency':            urgency,
        'value_month_3':      decay_values[3],
        'value_month_6':      decay_values[6],
        'value_month_12':     decay_values[12]
    }


def calculate_intervention_roi(ltv_data, action,
                                monthly_charges):
    action_costs = {
        'discount_10': monthly_charges * 0.10 * 12,
        'discount_20': monthly_charges * 0.20 * 12,
        'discount_30': monthly_charges * 0.30 * 12,
        'upgrade':     500,
        'support':     200,
        'loyalty':     150,
        'all':         800
    }
    action_retention = {
        'discount_10': 0.25,
        'discount_20': 0.45,
        'discount_30': 0.60,
        'upgrade':     0.50,
        'support':     0.35,
        'loyalty':     0.30,
        'all':         0.75
    }

    cost = action_costs.get(action, 200)
    retention = action_retention.get(action, 0.35)
    value_saved = ltv_data['value_at_risk'] * retention
    net_roi = value_saved - cost
    roi_pct = (net_roi / cost * 100) if cost > 0 else 0

    return {
        'action':            action,
        'action_cost':       round(cost, 2),
        'value_at_risk':     round(
                               ltv_data['value_at_risk'], 2),
        'value_saved':       round(value_saved, 2),
        'net_roi':           round(net_roi, 2),
        'roi_percentage':    round(roi_pct, 1),
        'verdict':           "✅ ACT NOW!"
                             if net_roi > 0
                             else "❌ NOT WORTH IT",
        'best_month_to_act': ltv_data['intervention_month']
    }


def analyze_portfolio_decay(df):
    print("📊 Analyzing portfolio LTV decay...")
    results = []
    monthly_col = ('MonthlyCharges'
                   if 'MonthlyCharges' in df.columns
                   else None)

    for idx, row in df.iterrows():
        monthly = float(
            row[monthly_col]) if monthly_col else 65.0
        tenure  = float(row.get('tenure', 12))
        c30     = float(row.get('churn_prob_30day', 50))
        c60     = float(row.get('churn_prob_60day', 55))
        c90     = float(row.get('churn_prob_90day', 60))

        ltv = calculate_customer_ltv(
            monthly, tenure, c30, c60, c90)

        results.append({
            'customer_id':        idx,
            'current_ltv':        ltv['current_ltv'],
            'ltv_month_3':        ltv['value_month_3'],
            'ltv_month_6':        ltv['value_month_6'],
            'ltv_month_12':       ltv['value_month_12'],
            'value_at_risk':      ltv['value_at_risk'],
            'decay_rate_pct':     ltv['decay_rate_pct'],
            'decay_pattern':      ltv['decay_pattern'],
            'urgency':            ltv['urgency'],
            'intervention_month': ltv['intervention_month'],
            'turning_point':      ltv['turning_point']
        })

    portfolio_df = pd.DataFrame(results)
    total_ltv     = portfolio_df['current_ltv'].sum()
    total_at_risk = portfolio_df['value_at_risk'].sum()
    critical      = len(portfolio_df[
        portfolio_df['urgency'] == 'Critical'])
    avg_decay     = portfolio_df['decay_rate_pct'].mean()

    print(f"✅ Total Portfolio LTV : ₹{total_ltv:,.0f}")
    print(f"✅ Total Value at Risk : ₹{total_at_risk:,.0f}")
    print(f"✅ Critical Decay      : {critical} customers")

    return portfolio_df, {
        'total_ltv':          total_ltv,
        'total_at_risk':      total_at_risk,
        'critical_customers': critical,
        'avg_decay_rate':     avg_decay
    }