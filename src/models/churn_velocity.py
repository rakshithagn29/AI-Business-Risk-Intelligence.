import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


def calculate_churn_velocity(churn_30, churn_60, churn_90):
    velocity_1 = churn_60 - churn_30
    velocity_2 = churn_90 - churn_60
    acceleration = velocity_2 - velocity_1
    avg_velocity = (velocity_1 + velocity_2) / 2

    cvi = max(0, min(10, (avg_velocity / 10) * 10))

    if acceleration > 5:
        cvi = min(10, cvi + 1.5)
    if acceleration > 10:
        cvi = min(10, cvi + 2.5)
    if churn_30 >= 80:
        cvi = min(10, cvi + 2)
    elif churn_30 >= 60:
        cvi = min(10, cvi + 1)

    cvi = round(cvi, 1)

    if cvi >= 8:
        category = "🔴 CRITICAL VELOCITY"
        message  = "Accelerating toward churn FAST!"
        urgency  = "Act within 24 hours"
    elif cvi >= 6:
        category = "🟠 HIGH VELOCITY"
        message  = "Risk increasing rapidly each month"
        urgency  = "Act within 3 days"
    elif cvi >= 4:
        category = "🟡 MODERATE VELOCITY"
        message  = "Risk slowly increasing"
        urgency  = "Act within 1 week"
    elif cvi >= 2:
        category = "🔵 LOW VELOCITY"
        message  = "Risk stable"
        urgency  = "Monthly review"
    else:
        category = "🟢 STABLE"
        message  = "Risk stable or decreasing!"
        urgency  = "No immediate action"

    return {
        'cvi':              cvi,
        'category':         category,
        'message':          message,
        'urgency':          urgency,
        'velocity_month1':  round(velocity_1, 2),
        'velocity_month2':  round(velocity_2, 2),
        'acceleration':     round(acceleration, 2),
        'is_accelerating':  acceleration > 0
    }


def add_cvi_to_dataframe(df):
    print("⚡ Calculating CVI for all customers...")
    results = []

    for _, row in df.iterrows():
        c30 = float(row.get('churn_prob_30day', 50))
        c60 = float(row.get('churn_prob_60day', 55))
        c90 = float(row.get('churn_prob_90day', 60))

        cvi = calculate_churn_velocity(c30, c60, c90)
        results.append({
            'cvi':             cvi['cvi'],
            'cvi_category':    cvi['category'],
            'cvi_urgency':     cvi['urgency'],
            'is_accelerating': cvi['is_accelerating'],
            'velocity_m1':     cvi['velocity_month1'],
            'velocity_m2':     cvi['velocity_month2'],
            'acceleration':    cvi['acceleration']
        })

    cvi_df = pd.DataFrame(results)
    result_df = pd.concat(
        [df.reset_index(drop=True),
         cvi_df.reset_index(drop=True)],
        axis=1)

    total    = len(result_df)
    critical = len(result_df[result_df['cvi'] >= 8])
    high     = len(result_df[
        (result_df['cvi'] >= 6) &
        (result_df['cvi'] < 8)])

    print(f"✅ Critical Velocity : {critical} "
          f"({critical/total*100:.1f}%)")
    print(f"✅ High Velocity     : {high} "
          f"({high/total*100:.1f}%)")
    print(f"✅ Avg CVI Score     : "
          f"{result_df['cvi'].mean():.2f}/10")

    return result_df


def find_dangerous_combinations(df):
    if 'cvi' not in df.columns:
        df = add_cvi_to_dataframe(df)

    dangerous = df[
        (df['churn_prob_30day'] >= 60) &
        (df['cvi'] >= 6)]

    hidden = df[
        (df['churn_prob_30day'] < 60) &
        (df['cvi'] >= 7)]

    print(f"🚨 Dangerous (High Prob + High Vel): "
          f"{len(dangerous)}")
    print(f"🔍 Hidden Danger (Low Prob + High Vel): "
          f"{len(hidden)}")

    return dangerous, hidden