import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="AI Business Risk Intelligence",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE, "models_saved")

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #FF4B4B;
        text-align: center;
        padding: 20px 0;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #aaa;
        text-align: center;
        margin-bottom: 30px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    path = os.path.join(BASE, "data", "processed", "telco_with_predictions.csv")
    return pd.read_csv(path)

@st.cache_resource
def load_models():
    m30 = joblib.load(os.path.join(MODEL_PATH, "churn_30day.pkl"))
    m60 = joblib.load(os.path.join(MODEL_PATH, "churn_60day.pkl"))
    m90 = joblib.load(os.path.join(MODEL_PATH, "churn_90day.pkl"))
    scaler = joblib.load(os.path.join(MODEL_PATH, "scaler.pkl"))
    return m30, m60, m90, scaler

# Sidebar
st.sidebar.image("https://img.icons8.com/color/96/artificial-intelligence.png", width=80)
st.sidebar.title("🚨 SENTINEL AI")
st.sidebar.markdown("**Business Risk Intelligence Platform**")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    ["🏠 Home Dashboard",
     "⚠️ Risk Assessment",
     "🔮 Churn Prediction",
     "💬 Sentiment Analysis",
     "🎮 What-If Simulator",
     "💰 Revenue Impact"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Project:** AI Driven Business Risk Intelligence")
st.sidebar.markdown("**Version:** 1.0.0")

df = load_data()
m30, m60, m90, scaler = load_models()

# ============================================================
# PAGE 1 — HOME DASHBOARD
# ============================================================
if page == "🏠 Home Dashboard":

    st.markdown('<p class="main-header">🚨 AI Driven Business Risk Intelligence</p>',
                unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Customer Risk Assessment & Churn Prediction Platform</p>',
                unsafe_allow_html=True)
    st.markdown("---")

    total = len(df)
    critical = len(df[df['churn_prob_30day'] >= 75])
    high = len(df[(df['churn_prob_30day'] >= 50) & (df['churn_prob_30day'] < 75)])
    avg_churn = df['churn_prob_30day'].mean()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("👥 Total Customers", f"{total:,}")
    col2.metric("🔴 Critical Risk", f"{critical:,}", f"{critical/total*100:.1f}% of total")
    col3.metric("🟠 High Risk", f"{high:,}", f"{high/total*100:.1f}% of total")
    col4.metric("📊 Avg Churn Risk", f"{avg_churn:.1f}%", "30-day average")

    st.markdown("---")

    if 'MonthlyCharges' in df.columns:
        total_revenue = df['MonthlyCharges'].sum() * 12
        at_risk_revenue = df[df['churn_prob_30day'] >= 50]['MonthlyCharges'].sum() * 12
        safe_revenue = total_revenue - at_risk_revenue
    else:
        total_revenue = total * 65 * 12
        at_risk_revenue = (critical + high) * 65 * 12
        safe_revenue = total_revenue - at_risk_revenue

    col5, col6, col7 = st.columns(3)
    col5.metric("💵 Total Annual Revenue", f"₹{total_revenue:,.0f}")
    col6.metric("⚠️ Revenue at Risk", f"₹{at_risk_revenue:,.0f}")
    col7.metric("✅ Safe Revenue", f"₹{safe_revenue:,.0f}")

    st.markdown("---")

    col8, col9 = st.columns(2)

    with col8:
        st.subheader("📊 Customer Risk Distribution")
        risk_counts = {
            'Critical 🔴': len(df[df['churn_prob_30day'] >= 75]),
            'High 🟠': len(df[(df['churn_prob_30day'] >= 50) & (df['churn_prob_30day'] < 75)]),
            'Medium 🟡': len(df[(df['churn_prob_30day'] >= 25) & (df['churn_prob_30day'] < 50)]),
            'Low 🟢': len(df[df['churn_prob_30day'] < 25])
        }
        fig_pie = px.pie(
            values=list(risk_counts.values()),
            names=list(risk_counts.keys()),
            color_discrete_map={
                'Critical 🔴': '#FF0000',
                'High 🟠': '#FF6600',
                'Medium 🟡': '#FFB300',
                'Low 🟢': '#00CC00'
            },
            hole=0.4
        )
        st.plotly_chart(fig_pie, use_container_width=True, key="home_pie")

    with col9:
        st.subheader("📈 Churn Probability Distribution")
        fig_hist = px.histogram(
            df, x='churn_prob_30day',
            nbins=30,
            color_discrete_sequence=['#FF4B4B'],
            labels={'churn_prob_30day': 'Churn Probability (%)'}
        )
        fig_hist.add_vline(x=50, line_dash="dash",
                           line_color="orange",
                           annotation_text="High Risk Threshold")
        fig_hist.add_vline(x=75, line_dash="dash",
                           line_color="red",
                           annotation_text="Critical Threshold")
        st.plotly_chart(fig_hist, use_container_width=True, key="home_hist")

    st.markdown("---")
    st.subheader("🔴 Top 10 Highest Risk Customers — Immediate Action Required!")
    top10 = df.nlargest(10, 'churn_prob_30day')[
        ['churn_prob_30day', 'churn_prob_60day',
         'churn_prob_90day', 'churn_risk']
    ].reset_index()
    top10.columns = ['Customer ID', '30-Day Risk%',
                     '60-Day Risk%', '90-Day Risk%', 'Risk Level']
    st.dataframe(top10, use_container_width=True)

# ============================================================
# PAGE 2 — RISK ASSESSMENT
# ============================================================
elif page == "⚠️ Risk Assessment":

    st.title("⚠️ Customer Risk Assessment")
    st.markdown("Analyze and filter customers by their risk level")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        risk_filter = st.selectbox(
            "Filter by Risk Level",
            ["All", "🔴 Critical (75%+)",
             "🟠 High (50-75%)",
             "🟡 Medium (25-50%)",
             "🟢 Low (<25%)"]
        )
    with col2:
        sort_by = st.selectbox(
            "Sort by",
            ["churn_prob_30day", "churn_prob_60day", "churn_prob_90day"]
        )
    with col3:
        show_rows = st.slider("Show rows", 10, 100, 20)

    filtered_df = df.copy()
    if risk_filter == "🔴 Critical (75%+)":
        filtered_df = df[df['churn_prob_30day'] >= 75]
    elif risk_filter == "🟠 High (50-75%)":
        filtered_df = df[(df['churn_prob_30day'] >= 50) & (df['churn_prob_30day'] < 75)]
    elif risk_filter == "🟡 Medium (25-50%)":
        filtered_df = df[(df['churn_prob_30day'] >= 25) & (df['churn_prob_30day'] < 50)]
    elif risk_filter == "🟢 Low (<25%)":
        filtered_df = df[df['churn_prob_30day'] < 25]

    filtered_df = filtered_df.sort_values(sort_by, ascending=False).head(show_rows)

    col4, col5, col6 = st.columns(3)
    col4.metric("Customers Shown", len(filtered_df))
    col5.metric("Avg Risk Score", f"{filtered_df['churn_prob_30day'].mean():.1f}%")
    if 'MonthlyCharges' in filtered_df.columns:
        col6.metric("Revenue at Risk", f"₹{filtered_df['MonthlyCharges'].sum()*12:,.0f}")

    st.markdown("---")

    if 'MonthlyCharges' in df.columns:
        st.subheader("📊 Risk Score vs Monthly Charges")
        fig_scatter = px.scatter(
            filtered_df,
            x='MonthlyCharges',
            y='churn_prob_30day',
            color='churn_risk',
            size='churn_prob_30day',
            color_discrete_map={
                '🔴 CRITICAL': 'red',
                '🟠 HIGH': 'orange',
                '🟡 MEDIUM': 'gold',
                '🟢 LOW': 'green'
            },
            labels={
                'MonthlyCharges': 'Monthly Revenue (₹)',
                'churn_prob_30day': 'Churn Risk (%)'
            }
        )
        st.plotly_chart(fig_scatter, use_container_width=True, key="risk_scatter")

    st.markdown("---")
    st.subheader("📋 Customer Risk Table")
    display_cols = ['churn_prob_30day', 'churn_prob_60day',
                    'churn_prob_90day', 'churn_risk']
    if 'MonthlyCharges' in df.columns:
        display_cols = ['MonthlyCharges'] + display_cols
    st.dataframe(
        filtered_df[display_cols].reset_index(),
        use_container_width=True
    )

# ============================================================
# PAGE 3 — CHURN PREDICTION
# ============================================================
elif page == "🔮 Churn Prediction":

    st.title("🔮 Triple Horizon Churn Prediction")
    st.markdown("Predict customer churn across **30, 60, and 90 day** horizons")
    st.markdown("---")

    # All 3 charts side by side
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("📊 30-Day")
        fig1 = px.histogram(
            df, x='churn_prob_30day',
            nbins=25,
            color_discrete_sequence=['#FF4B4B'],
            title="30-Day Churn Probability"
        )
        st.plotly_chart(fig1, use_container_width=True, key="churn_30_hist")

    with col2:
        st.subheader("📊 60-Day")
        fig2 = px.histogram(
            df, x='churn_prob_60day',
            nbins=25,
            color_discrete_sequence=['#FF8C00'],
            title="60-Day Churn Probability"
        )
        st.plotly_chart(fig2, use_container_width=True, key="churn_60_hist")

    with col3:
        st.subheader("📊 90-Day")
        fig3 = px.histogram(
            df, x='churn_prob_90day',
            nbins=25,
            color_discrete_sequence=['#00CC00'],
            title="90-Day Churn Probability"
        )
        st.plotly_chart(fig3, use_container_width=True, key="churn_90_hist")

    st.markdown("---")

    # Triple horizon comparison bar chart
    st.subheader("📈 Triple Horizon Comparison")
    horizon_data = pd.DataFrame({
        'Horizon': ['30 Days', '60 Days', '90 Days'],
        'Avg Churn Risk': [
            df['churn_prob_30day'].mean(),
            df['churn_prob_60day'].mean(),
            df['churn_prob_90day'].mean()
        ],
        'High Risk Count': [
            len(df[df['churn_prob_30day'] >= 50]),
            len(df[df['churn_prob_60day'] >= 50]),
            len(df[df['churn_prob_90day'] >= 50])
        ]
    })

    fig_horizon = px.bar(
        horizon_data,
        x='Horizon',
        y='High Risk Count',
        color='Horizon',
        color_discrete_sequence=['red', 'orange', 'green'],
        text='High Risk Count',
        title="High Risk Customers Across Time Horizons"
    )
    st.plotly_chart(fig_horizon, use_container_width=True, key="horizon_bar")

    st.markdown("---")

    # Individual customer lookup
    st.subheader("🔍 Look Up Individual Customer")
    customer_id = st.number_input(
        "Enter Customer Index",
        min_value=0,
        max_value=len(df)-1,
        value=0
    )

    if st.button("🔮 Predict Churn Risk"):
        customer = df.iloc[customer_id]

        col4, col5, col6 = st.columns(3)
        col4.metric("30-Day Risk", f"{customer['churn_prob_30day']:.1f}%")
        col5.metric("60-Day Risk", f"{customer['churn_prob_60day']:.1f}%")
        col6.metric("90-Day Risk", f"{customer['churn_prob_90day']:.1f}%")

        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=customer['churn_prob_30day'],
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': f"Customer {customer_id} — 30-Day Churn Risk"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkred"},
                'steps': [
                    {'range': [0, 25], 'color': "green"},
                    {'range': [25, 50], 'color': "yellow"},
                    {'range': [50, 75], 'color': "orange"},
                    {'range': [75, 100], 'color': "red"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 75
                }
            }
        ))
        st.plotly_chart(fig_gauge, use_container_width=True, key="customer_gauge")

# ============================================================
# PAGE 4 — WHAT-IF SIMULATOR
# ============================================================
elif page == "🎮 What-If Simulator":

    st.title("🎮 What-If Action Simulator")
    st.markdown("Simulate business actions and see predicted impact on churn")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        customer_id = st.number_input(
            "Select Customer",
            min_value=0,
            max_value=len(df)-1,
            value=int(df['churn_prob_30day'].idxmax())
        )
    with col2:
        action = st.selectbox(
            "Select Action to Simulate",
            ["discount_10", "discount_20", "discount_30",
             "upgrade", "support", "loyalty", "all"]
        )

    action_names = {
        'discount_10': '10% Discount',
        'discount_20': '20% Discount',
        'discount_30': '30% Discount',
        'upgrade': 'Upgrade to Annual Contract',
        'support': 'Assign Dedicated Support',
        'loyalty': 'Give Loyalty Rewards',
        'all': 'Complete Retention Package'
    }

    reductions = {
        'discount_10': 8, 'discount_20': 18,
        'discount_30': 28, 'upgrade': 22,
        'support': 15, 'loyalty': 12, 'all': 45
    }

    costs = {
        'discount_10': 50, 'discount_20': 100,
        'discount_30': 150, 'upgrade': 500,
        'support': 200, 'loyalty': 150, 'all': 800
    }

    if st.button("🚀 Run Simulation"):
        customer = df.iloc[customer_id]
        orig_prob = customer['churn_prob_30day']
        new_prob = max(orig_prob - reductions[action], 2)
        monthly = customer.get('MonthlyCharges', 65)
        annual = monthly * 12
        cost = costs[action]
        saved = annual - cost

        st.markdown("---")
        st.subheader("📊 Simulation Results")

        col3, col4, col5 = st.columns(3)
        col3.metric("Before Action", f"{orig_prob:.1f}%", "Churn Risk")
        col4.metric("After Action", f"{new_prob:.1f}%",
                    f"-{orig_prob-new_prob:.1f}% improvement")
        col5.metric("Revenue Saved", f"₹{saved:.0f}")

        st.markdown("---")

        col6, col7 = st.columns(2)

        with col6:
            fig_before = go.Figure(go.Indicator(
                mode="gauge+number",
                value=orig_prob,
                title={'text': "BEFORE Action"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "red"},
                    'steps': [
                        {'range': [0, 25], 'color': "green"},
                        {'range': [25, 50], 'color': "yellow"},
                        {'range': [50, 75], 'color': "orange"},
                        {'range': [75, 100], 'color': "red"}
                    ]
                }
            ))
            st.plotly_chart(fig_before, use_container_width=True, key="gauge_before")

        with col7:
            fig_after = go.Figure(go.Indicator(
                mode="gauge+number",
                value=new_prob,
                title={'text': "AFTER Action"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "green"},
                    'steps': [
                        {'range': [0, 25], 'color': "green"},
                        {'range': [25, 50], 'color': "yellow"},
                        {'range': [50, 75], 'color': "orange"},
                        {'range': [75, 100], 'color': "red"}
                    ]
                }
            ))
            st.plotly_chart(fig_after, use_container_width=True, key="gauge_after")

        st.markdown("---")
        if saved > 0:
            st.success(f"✅ RECOMMENDED: Take this action! Saves ₹{saved:.0f} annually!")
        else:
            st.error(f"❌ NOT RECOMMENDED: Action costs more than revenue!")
            # ============================================================
# PAGE 4 — SENTIMENT ANALYSIS
# ============================================================
elif page == "💬 Sentiment Analysis":

    st.title("💬 Customer Sentiment Analysis")
    st.markdown("Analyze customer feedback and detect sentiment risk")
    st.markdown("---")

    # Load sentiment data
    sentiment_path = os.path.join(
        BASE, "data", "processed", "sentiment_results.csv"
    )

    if os.path.exists(sentiment_path):
        sent_df = pd.read_csv(sentiment_path)
    else:
        st.warning("⚠️ Run notebook 07 first to generate sentiment data!")
        st.stop()

    # KPI Cards
    total = len(sent_df)
    positive = len(sent_df[sent_df['sentiment'] == 'Positive'])
    negative = len(sent_df[sent_df['sentiment'] == 'Negative'])
    neutral = len(sent_df[sent_df['sentiment'] == 'Neutral'])

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📝 Total Feedback", total)
    col2.metric("😊 Positive", f"{positive} ({positive/total*100:.0f}%)")
    col3.metric("😐 Neutral", f"{neutral} ({neutral/total*100:.0f}%)")
    col4.metric("😠 Negative", f"{negative} ({negative/total*100:.0f}%)")

    st.markdown("---")

    col5, col6 = st.columns(2)

    with col5:
        st.subheader("📊 Sentiment Distribution")
        fig_sent_pie = px.pie(
            sent_df,
            names='sentiment',
            color='sentiment',
            color_discrete_map={
                'Positive': '#00CC00',
                'Neutral': '#FFB300',
                'Negative': '#FF0000'
            },
            hole=0.4
        )
        st.plotly_chart(fig_sent_pie,
                        use_container_width=True,
                        key="sent_pie")

    with col6:
        st.subheader("📈 Sentiment Score Distribution")
        fig_sent_hist = px.histogram(
            sent_df,
            x='sentiment_score',
            nbins=20,
            color='sentiment',
            color_discrete_map={
                'Positive': '#00CC00',
                'Neutral': '#FFB300',
                'Negative': '#FF0000'
            },
            title="Sentiment Scores (-1 Negative to +1 Positive)"
        )
        fig_sent_hist.add_vline(
            x=0, line_dash="dash",
            line_color="white",
            annotation_text="Neutral"
        )
        st.plotly_chart(fig_sent_hist,
                        use_container_width=True,
                        key="sent_hist")

    st.markdown("---")

    # Sentiment Risk Chart
    st.subheader("⚠️ Sentiment Risk by Customer")
    fig_risk = px.bar(
        sent_df.sort_values('sentiment_risk', ascending=False),
        x='customer_id',
        y='sentiment_risk',
        color='sentiment',
        color_discrete_map={
            'Positive': '#00CC00',
            'Neutral': '#FFB300',
            'Negative': '#FF0000'
        },
        title="Customer Sentiment Risk Score",
        labels={
            'customer_id': 'Customer ID',
            'sentiment_risk': 'Risk Score (%)'
        }
    )
    st.plotly_chart(fig_risk, use_container_width=True, key="sent_risk")

    st.markdown("---")

    # Feedback filter
    st.subheader("🔍 Filter Feedback by Sentiment")
    filter_sent = st.selectbox(
        "Show feedback",
        ["All", "Positive 😊", "Neutral 😐", "Negative 😠"]
    )

    filtered_sent = sent_df.copy()
    if filter_sent == "Positive 😊":
        filtered_sent = sent_df[sent_df['sentiment'] == 'Positive']
    elif filter_sent == "Neutral 😐":
        filtered_sent = sent_df[sent_df['sentiment'] == 'Neutral']
    elif filter_sent == "Negative 😠":
        filtered_sent = sent_df[sent_df['sentiment'] == 'Negative']

    st.dataframe(
        filtered_sent[['customer_id', 'feedback',
                       'sentiment', 'sentiment_score',
                       'sentiment_risk']],
        use_container_width=True
    )

    st.markdown("---")

    # High risk alert
    high_risk_sent = sent_df[sent_df['sentiment_risk'] >= 50]
    if len(high_risk_sent) > 0:
        st.error(f"🔴 {len(high_risk_sent)} customers have HIGH sentiment risk — immediate attention needed!")
        st.dataframe(
            high_risk_sent[['customer_id', 'feedback', 'sentiment_risk']],
            use_container_width=True
        )

# ============================================================
# PAGE 5 — REVENUE IMPACT
# ============================================================
elif page == "💰 Revenue Impact":

    st.title("💰 Revenue at Risk Calculator")
    st.markdown("Calculate potential revenue loss and savings")
    st.markdown("---")

    critical = df[df['churn_prob_30day'] >= 75]
    high = df[(df['churn_prob_30day'] >= 50) & (df['churn_prob_30day'] < 75)]
    medium = df[(df['churn_prob_30day'] >= 25) & (df['churn_prob_30day'] < 50)]
    low = df[df['churn_prob_30day'] < 25]

    if 'MonthlyCharges' in df.columns:
        crit_rev = critical['MonthlyCharges'].sum() * 12
        high_rev = high['MonthlyCharges'].sum() * 12
        med_rev = medium['MonthlyCharges'].sum() * 12
        total_risk = crit_rev + high_rev
    else:
        avg = 65
        crit_rev = len(critical) * avg * 12
        high_rev = len(high) * avg * 12
        med_rev = len(medium) * avg * 12
        total_risk = crit_rev + high_rev

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🔴 Critical Revenue Risk", f"₹{crit_rev:,.0f}")
    col2.metric("🟠 High Revenue Risk", f"₹{high_rev:,.0f}")
    col3.metric("🟡 Medium Revenue Risk", f"₹{med_rev:,.0f}")
    col4.metric("⚠️ Total at Risk", f"₹{total_risk:,.0f}")

    st.markdown("---")

    st.subheader("🎯 Retention Scenario Calculator")
    retention_rate = st.slider(
        "If we retain this % of high risk customers:",
        min_value=10, max_value=100, value=50, step=10
    )

    saved = total_risk * (retention_rate / 100)

    col5, col6 = st.columns(2)
    col5.metric(f"Revenue Saved ({retention_rate}% retention)", f"₹{saved:,.0f}")
    col6.metric("Still at Risk", f"₹{total_risk - saved:,.0f}")

    st.markdown("---")

    col7, col8 = st.columns(2)

    with col7:
        st.subheader("Revenue at Risk by Category")
        rev_data = pd.DataFrame({
            'Category': ['🔴 Critical', '🟠 High', '🟡 Medium'],
            'Revenue': [crit_rev, high_rev, med_rev]
        })
        fig_rev = px.bar(
            rev_data, x='Category', y='Revenue',
            color='Category',
            color_discrete_sequence=['red', 'orange', 'gold'],
            text='Revenue'
        )
        fig_rev.update_traces(
            texttemplate='₹%{text:,.0f}',
            textposition='outside'
        )
        st.plotly_chart(fig_rev, use_container_width=True, key="revenue_bar")

    with col8:
        st.subheader("Retention Impact Scenarios")
        scenarios = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        savings = [total_risk * (s/100) for s in scenarios]
        fig_scen = px.line(
            x=scenarios, y=savings,
            labels={'x': 'Retention Rate (%)', 'y': 'Revenue Saved (₹)'},
            markers=True,
            color_discrete_sequence=['green']
        )
        fig_scen.add_hline(
            y=saved, line_dash="dash",
            line_color="red",
            annotation_text=f"Current: {retention_rate}%"
        )
        st.plotly_chart(fig_scen, use_container_width=True, key="revenue_line")

    st.markdown("---")
    st.subheader("📋 Action Plan for Maximum Revenue Recovery")

    action_plan = pd.DataFrame({
        'Priority': ['1st', '2nd', '3rd'],
        'Target Segment': ['Critical Risk', 'High Risk', 'Medium Risk'],
        'Customers': [len(critical), len(high), len(medium)],
        'Revenue at Risk': [
            f'₹{crit_rev:,.0f}',
            f'₹{high_rev:,.0f}',
            f'₹{med_rev:,.0f}'
        ],
        'Recommended Action': [
            'Personal call + 30% discount',
            'Email offer + 20% discount',
            'Newsletter + 10% discount'
        ],
        'Timeline': ['Today', 'Within 3 days', 'Within 1 week']
    })

    st.dataframe(action_plan, use_container_width=True)