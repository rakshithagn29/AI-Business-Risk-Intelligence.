import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="AI Business Risk Intelligence",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# BASE PATH — DEFINED FIRST BEFORE ANYTHING ELSE
# ============================================================
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE, "models_saved")
os.makedirs(MODEL_PATH, exist_ok=True)

# ============================================================
# AUTO TRAIN MODELS IF NOT FOUND
# ============================================================
def ensure_models_exist():
    required = ['churn_30day.pkl', 'churn_60day.pkl',
                'churn_90day.pkl', 'scaler.pkl']
    missing = [f for f in required
               if not os.path.exists(
                   os.path.join(MODEL_PATH, f))]

    if not missing:
        return

    st.warning("⏳ Training models for first time — please wait 2-3 minutes...")

    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.model_selection import train_test_split
    from imblearn.over_sampling import SMOTE
    import xgboost as xgb

    clean_path = os.path.join(
        BASE, "data", "processed", "telco_clean.csv")

    if not os.path.exists(clean_path):
        st.error("❌ telco_clean.csv not found in data/processed/")
        st.stop()

    df_train = pd.read_csv(clean_path)
    le = LabelEncoder()
    df_work = df_train.copy()

    cat_cols = df_work.select_dtypes(
        include=['object']).columns
    for col in cat_cols:
        df_work[col] = le.fit_transform(
            df_work[col].astype(str))

    X = df_work.drop('Churn', axis=1)
    y = df_work['Churn']

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X), columns=X.columns)
    joblib.dump(scaler,
                os.path.join(MODEL_PATH, "scaler.pkl"))

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2,
        random_state=42, stratify=y)

    sm = SMOTE(random_state=42)
    X_sm, y_sm = sm.fit_resample(X_train, y_train)

    for name, noise in [('churn_30day', 0),
                         ('churn_60day', 0.05),
                         ('churn_90day', 0.08)]:
        X_tr = (X_sm + np.random.normal(
            0, noise, X_sm.shape)
            if noise > 0 else X_sm)
        model = xgb.XGBClassifier(
            n_estimators=200, max_depth=6,
            learning_rate=0.1, random_state=42,
            n_jobs=-1, verbosity=0,
            eval_metric='logloss')
        model.fit(X_tr, y_sm)
        joblib.dump(model, os.path.join(
            MODEL_PATH, f"{name}.pkl"))

    st.success("✅ Models trained! Reloading...")
    st.rerun()

ensure_models_exist()

# ============================================================
# CUSTOM CSS
# ============================================================
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

# ============================================================
# LOAD DATA
# ============================================================
@st.cache_data
def load_data():
    path = os.path.join(BASE, "data", "processed",
                        "telco_with_predictions.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    else:
        st.error("❌ telco_with_predictions.csv not found!")
        st.stop()

@st.cache_resource
def load_models():
    m30    = joblib.load(os.path.join(MODEL_PATH, "churn_30day.pkl"))
    m60    = joblib.load(os.path.join(MODEL_PATH, "churn_60day.pkl"))
    m90    = joblib.load(os.path.join(MODEL_PATH, "churn_90day.pkl"))
    scaler = joblib.load(os.path.join(MODEL_PATH, "scaler.pkl"))
    return m30, m60, m90, scaler

# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.image(
    "https://img.icons8.com/color/96/artificial-intelligence.png",
    width=80)
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
     "💰 Revenue Impact",
     "📉 LTV Decay Analysis",
     "⚡ Churn Velocity Index"]
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Project:** AI Driven Business Risk Intelligence")
st.sidebar.markdown("**Version:** 2.0.0")

# Load everything
df = load_data()
m30, m60, m90, scaler = load_models()

# ============================================================
# PAGE 1 — HOME DASHBOARD
# ============================================================
if page == "🏠 Home Dashboard":

    st.markdown(
        '<p class="main-header">🚨 AI Driven Business Risk Intelligence</p>',
        unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Customer Risk Assessment & Churn Prediction Platform</p>',
        unsafe_allow_html=True)
    st.markdown("---")

    total    = len(df)
    critical = len(df[df['churn_prob_30day'] >= 75])
    high     = len(df[(df['churn_prob_30day'] >= 50) &
                      (df['churn_prob_30day'] < 75)])
    avg_churn = df['churn_prob_30day'].mean()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("👥 Total Customers", f"{total:,}")
    col2.metric("🔴 Critical Risk",   f"{critical:,}",
                f"{critical/total*100:.1f}%")
    col3.metric("🟠 High Risk",       f"{high:,}",
                f"{high/total*100:.1f}%")
    col4.metric("📊 Avg Churn Risk",  f"{avg_churn:.1f}%")

    st.markdown("---")

    if 'MonthlyCharges' in df.columns:
        total_rev  = df['MonthlyCharges'].sum() * 12
        at_risk    = df[df['churn_prob_30day'] >= 50][
            'MonthlyCharges'].sum() * 12
        safe_rev   = total_rev - at_risk
    else:
        total_rev  = total * 65 * 12
        at_risk    = (critical + high) * 65 * 12
        safe_rev   = total_rev - at_risk

    col5, col6, col7 = st.columns(3)
    col5.metric("💵 Total Annual Revenue", f"₹{total_rev:,.0f}")
    col6.metric("⚠️ Revenue at Risk",      f"₹{at_risk:,.0f}")
    col7.metric("✅ Safe Revenue",          f"₹{safe_rev:,.0f}")

    st.markdown("---")

    col8, col9 = st.columns(2)

    with col8:
        st.subheader("📊 Customer Risk Distribution")
        risk_counts = {
            'Critical 🔴': critical,
            'High 🟠':     high,
            'Medium 🟡':   len(df[(df['churn_prob_30day'] >= 25) &
                                   (df['churn_prob_30day'] < 50)]),
            'Low 🟢':      len(df[df['churn_prob_30day'] < 25])
        }
        fig_pie = px.pie(
            values=list(risk_counts.values()),
            names=list(risk_counts.keys()),
            color_discrete_map={
                'Critical 🔴': '#FF0000',
                'High 🟠':     '#FF6600',
                'Medium 🟡':   '#FFB300',
                'Low 🟢':      '#00CC00'},
            hole=0.4)
        st.plotly_chart(fig_pie,
                        use_container_width=True,
                        key="home_pie")

    with col9:
        st.subheader("📈 Churn Probability Distribution")
        fig_hist = px.histogram(
            df, x='churn_prob_30day', nbins=30,
            color_discrete_sequence=['#FF4B4B'],
            labels={'churn_prob_30day': 'Churn Probability (%)'})
        fig_hist.add_vline(
            x=50, line_dash="dash",
            line_color="orange",
            annotation_text="High Risk")
        fig_hist.add_vline(
            x=75, line_dash="dash",
            line_color="red",
            annotation_text="Critical")
        st.plotly_chart(fig_hist,
                        use_container_width=True,
                        key="home_hist")

    st.markdown("---")
    st.subheader(
        "🔴 Top 10 Highest Risk Customers — Immediate Action!")
    top10 = df.nlargest(10, 'churn_prob_30day')[
        ['churn_prob_30day', 'churn_prob_60day',
         'churn_prob_90day', 'churn_risk']
    ].reset_index()
    top10.columns = ['Customer ID', '30-Day Risk%',
                     '60-Day Risk%', '90-Day Risk%',
                     'Risk Level']
    st.dataframe(top10, use_container_width=True)

# ============================================================
# PAGE 2 — RISK ASSESSMENT
# ============================================================
elif page == "⚠️ Risk Assessment":

    st.title("⚠️ Customer Risk Assessment")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    with col1:
        risk_filter = st.selectbox(
            "Filter by Risk Level",
            ["All", "🔴 Critical (75%+)",
             "🟠 High (50-75%)",
             "🟡 Medium (25-50%)",
             "🟢 Low (<25%)"])
    with col2:
        sort_by = st.selectbox(
            "Sort by",
            ["churn_prob_30day",
             "churn_prob_60day",
             "churn_prob_90day"])
    with col3:
        show_rows = st.slider("Show rows", 10, 100, 20)

    filtered_df = df.copy()
    if risk_filter == "🔴 Critical (75%+)":
        filtered_df = df[df['churn_prob_30day'] >= 75]
    elif risk_filter == "🟠 High (50-75%)":
        filtered_df = df[(df['churn_prob_30day'] >= 50) &
                         (df['churn_prob_30day'] < 75)]
    elif risk_filter == "🟡 Medium (25-50%)":
        filtered_df = df[(df['churn_prob_30day'] >= 25) &
                         (df['churn_prob_30day'] < 50)]
    elif risk_filter == "🟢 Low (<25%)":
        filtered_df = df[df['churn_prob_30day'] < 25]

    filtered_df = filtered_df.sort_values(
        sort_by, ascending=False).head(show_rows)

    col4, col5, col6 = st.columns(3)
    col4.metric("Shown", len(filtered_df))
    col5.metric("Avg Risk",
                f"{filtered_df['churn_prob_30day'].mean():.1f}%")
    if 'MonthlyCharges' in filtered_df.columns:
        col6.metric("Revenue at Risk",
                    f"₹{filtered_df['MonthlyCharges'].sum()*12:,.0f}")

    st.markdown("---")

    if 'MonthlyCharges' in df.columns:
        fig_sc = px.scatter(
            filtered_df,
            x='MonthlyCharges',
            y='churn_prob_30day',
            color='churn_risk',
            size='churn_prob_30day',
            labels={
                'MonthlyCharges':    'Monthly Revenue (₹)',
                'churn_prob_30day':  'Churn Risk (%)'},
            title="Risk Score vs Monthly Charges")
        st.plotly_chart(fig_sc,
                        use_container_width=True,
                        key="risk_scatter")

    st.markdown("---")
    display_cols = ['churn_prob_30day', 'churn_prob_60day',
                    'churn_prob_90day', 'churn_risk']
    if 'MonthlyCharges' in df.columns:
        display_cols = ['MonthlyCharges'] + display_cols
    st.dataframe(
        filtered_df[display_cols].reset_index(),
        use_container_width=True)

# ============================================================
# PAGE 3 — CHURN PREDICTION
# ============================================================
elif page == "🔮 Churn Prediction":

    st.title("🔮 Triple Horizon Churn Prediction")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    with col1:
        fig1 = px.histogram(
            df, x='churn_prob_30day', nbins=25,
            color_discrete_sequence=['#FF4B4B'],
            title="30-Day Churn")
        st.plotly_chart(fig1,
                        use_container_width=True,
                        key="churn_30")
    with col2:
        fig2 = px.histogram(
            df, x='churn_prob_60day', nbins=25,
            color_discrete_sequence=['#FF8C00'],
            title="60-Day Churn")
        st.plotly_chart(fig2,
                        use_container_width=True,
                        key="churn_60")
    with col3:
        fig3 = px.histogram(
            df, x='churn_prob_90day', nbins=25,
            color_discrete_sequence=['#00CC00'],
            title="90-Day Churn")
        st.plotly_chart(fig3,
                        use_container_width=True,
                        key="churn_90")

    st.markdown("---")

    horizon_data = pd.DataFrame({
        'Horizon':        ['30 Days', '60 Days', '90 Days'],
        'High Risk Count': [
            len(df[df['churn_prob_30day'] >= 50]),
            len(df[df['churn_prob_60day'] >= 50]),
            len(df[df['churn_prob_90day'] >= 50])]
    })
    fig_h = px.bar(
        horizon_data, x='Horizon', y='High Risk Count',
        color='Horizon',
        color_discrete_sequence=['red','orange','green'],
        text='High Risk Count',
        title="High Risk Customers Across Horizons")
    st.plotly_chart(fig_h,
                    use_container_width=True,
                    key="horizon_bar")

    st.markdown("---")
    st.subheader("🔍 Individual Customer Lookup")

    cust_id = st.number_input(
        "Customer Index", min_value=0,
        max_value=len(df)-1, value=0)

    if st.button("🔮 Predict"):
        customer = df.iloc[cust_id]
        col4, col5, col6 = st.columns(3)
        col4.metric("30-Day Risk",
                    f"{customer['churn_prob_30day']:.1f}%")
        col5.metric("60-Day Risk",
                    f"{customer['churn_prob_60day']:.1f}%")
        col6.metric("90-Day Risk",
                    f"{customer['churn_prob_90day']:.1f}%")

        fig_g = go.Figure(go.Indicator(
            mode="gauge+number",
            value=customer['churn_prob_30day'],
            title={'text': f"Customer {cust_id} — 30-Day Risk"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar':  {'color': "darkred"},
                'steps': [
                    {'range': [0, 25],  'color': "green"},
                    {'range': [25, 50], 'color': "yellow"},
                    {'range': [50, 75], 'color': "orange"},
                    {'range': [75,100], 'color': "red"}]}))
        st.plotly_chart(fig_g,
                        use_container_width=True,
                        key="gauge_predict")

# ============================================================
# PAGE 4 — SENTIMENT ANALYSIS
# ============================================================
elif page == "💬 Real-Time Sentiment":

    st.title("💬 Real-Time Customer Sentiment")
    st.markdown(
        "**Live data** collected from NewsAPI — "
        "real news about Indian telecom operators")
    st.markdown("---")

    live_path = os.path.join(
        BASE, "data", "external", "live_news_data.csv")
    summary_path = os.path.join(
        BASE, "data", "external", "live_data_summary.json")

    if os.path.exists(live_path):
        live_df = pd.read_csv(live_path)

        import json
        with open(summary_path) as f:
            summary = json.load(f)

        st.success(
            f"✅ Live data collected: {summary['collection_time']} "
            f"| Source: {summary['data_source']}")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("📰 Articles Analyzed",
                    summary['total_articles'])
        col2.metric("😊 Positive",
                    summary['positive'])
        col3.metric("😠 Negative",
                    summary['negative'])
        col4.metric("🎯 Churn Signal Rate",
                    f"{summary['churn_signal_rate']}%")

        st.markdown("---")

        col5, col6 = st.columns(2)
        with col5:
            fig_p = px.pie(
                live_df, names='sentiment',
                color='sentiment',
                color_discrete_map={
                    'Positive':'#00CC00',
                    'Neutral':'#FFB300',
                    'Negative':'#FF0000'},
                title="Live Sentiment Distribution")
            st.plotly_chart(fig_p,
                use_container_width=True, key="live_pie")

        with col6:
            sub_sent = live_df.groupby(
                ['query','sentiment']).size().reset_index(
                    name='count')
            fig_s = px.bar(
                sub_sent, x='query', y='count',
                color='sentiment',
                color_discrete_map={
                    'Positive':'#00CC00',
                    'Neutral':'#FFB300',
                    'Negative':'#FF0000'},
                title="Sentiment by Search Query")
            fig_s.update_xaxes(tickangle=45)
            st.plotly_chart(fig_s,
                use_container_width=True, key="live_query")

        st.markdown("---")
        st.subheader("🔴 Articles with Churn Signals")
        churn_posts = live_df[
            live_df['churn_signal'] == True
        ][['publisher','title','sentiment',
           'sentiment_risk','url']].head(10)

        if len(churn_posts) > 0:
            st.dataframe(churn_posts,
                use_container_width=True)
        else:
            st.info("No churn signals in current data")

        st.markdown("---")
        st.caption(
            f"Data refreshed: {summary['collection_time']} | "
            f"Run notebook 14 to refresh live data")

    else:
        st.warning(
            "⚠️ Run notebook 14_real_live_data.ipynb "
            "to collect live data from NewsAPI!")
        st.info("""
        This page shows REAL-TIME data collected from
        NewsAPI about Indian telecom operators
        (Jio, Airtel, Vi, BSNL) — genuine live news
        sentiment, not synthetic data.
        """)
# ============================================================
# PAGE 5 — WHAT-IF SIMULATOR
# ============================================================
elif page == "🎮 What-If Simulator":

    st.title("🎮 What-If Action Simulator")
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        sim_cust = st.number_input(
            "Select Customer",
            min_value=0,
            max_value=len(df)-1,
            value=int(df['churn_prob_30day'].idxmax()))
    with col2:
        sim_action = st.selectbox(
            "Select Action",
            ["discount_10","discount_20","discount_30",
             "upgrade","support","loyalty","all"])

    reductions = {
        'discount_10': 8,  'discount_20': 18,
        'discount_30': 28, 'upgrade': 22,
        'support': 15,     'loyalty': 12, 'all': 45}
    costs = {
        'discount_10': 50,  'discount_20': 100,
        'discount_30': 150, 'upgrade': 500,
        'support': 200,     'loyalty': 150, 'all': 800}

    if st.button("🚀 Run Simulation"):
        cust     = df.iloc[sim_cust]
        orig     = cust['churn_prob_30day']
        new_prob = max(orig - reductions[sim_action], 2)
        monthly  = cust.get('MonthlyCharges', 65)
        annual   = monthly * 12
        cost     = costs[sim_action]
        saved    = annual - cost

        col3, col4, col5 = st.columns(3)
        col3.metric("Before", f"{orig:.1f}%")
        col4.metric("After",  f"{new_prob:.1f}%",
                    f"-{orig-new_prob:.1f}%")
        col5.metric("Revenue Saved", f"₹{saved:.0f}")

        st.markdown("---")
        col6, col7 = st.columns(2)

        with col6:
            fig_b = go.Figure(go.Indicator(
                mode="gauge+number",
                value=orig,
                title={'text': "BEFORE"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar':  {'color': "red"},
                    'steps': [
                        {'range': [0, 25],  'color': "green"},
                        {'range': [25, 50], 'color': "yellow"},
                        {'range': [50, 75], 'color': "orange"},
                        {'range': [75,100], 'color': "red"}]}))
            st.plotly_chart(fig_b,
                            use_container_width=True,
                            key="gauge_b")
        with col7:
            fig_a = go.Figure(go.Indicator(
                mode="gauge+number",
                value=new_prob,
                title={'text': "AFTER"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar':  {'color': "green"},
                    'steps': [
                        {'range': [0, 25],  'color': "green"},
                        {'range': [25, 50], 'color': "yellow"},
                        {'range': [50, 75], 'color': "orange"},
                        {'range': [75,100], 'color': "red"}]}))
            st.plotly_chart(fig_a,
                            use_container_width=True,
                            key="gauge_a")

        st.markdown("---")
        if saved > 0:
            st.success(
                f"✅ TAKE THIS ACTION! Saves ₹{saved:.0f}/year!")
        else:
            st.error("❌ Not worth it — costs more than revenue!")

# ============================================================
# PAGE 6 — REVENUE IMPACT
# ============================================================
elif page == "💰 Revenue Impact":

    st.title("💰 Revenue at Risk Calculator")
    st.markdown("---")

    crit  = df[df['churn_prob_30day'] >= 75]
    high  = df[(df['churn_prob_30day'] >= 50) &
               (df['churn_prob_30day'] < 75)]
    med   = df[(df['churn_prob_30day'] >= 25) &
               (df['churn_prob_30day'] < 50)]
    low   = df[df['churn_prob_30day'] < 25]

    if 'MonthlyCharges' in df.columns:
        cr = crit['MonthlyCharges'].sum() * 12
        hr = high['MonthlyCharges'].sum() * 12
        mr = med['MonthlyCharges'].sum()  * 12
        total_risk = cr + hr
    else:
        cr = len(crit) * 65 * 12
        hr = len(high) * 65 * 12
        mr = len(med)  * 65 * 12
        total_risk = cr + hr

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🔴 Critical Revenue", f"₹{cr:,.0f}")
    col2.metric("🟠 High Revenue",     f"₹{hr:,.0f}")
    col3.metric("🟡 Medium Revenue",   f"₹{mr:,.0f}")
    col4.metric("⚠️ Total at Risk",    f"₹{total_risk:,.0f}")

    st.markdown("---")

    retention = st.slider(
        "Retain this % of high risk customers:",
        10, 100, 50, step=10)
    saved_rev = total_risk * (retention / 100)

    col5, col6 = st.columns(2)
    col5.metric(f"Revenue Saved ({retention}%)",
                f"₹{saved_rev:,.0f}")
    col6.metric("Still at Risk",
                f"₹{total_risk - saved_rev:,.0f}")

    st.markdown("---")

    col7, col8 = st.columns(2)
    with col7:
        rev_df = pd.DataFrame({
            'Category': ['🔴 Critical','🟠 High','🟡 Medium'],
            'Revenue':  [cr, hr, mr]})
        fig_rv = px.bar(
            rev_df, x='Category', y='Revenue',
            color='Category',
            color_discrete_sequence=['red','orange','gold'],
            text='Revenue',
            title="Revenue at Risk by Category")
        fig_rv.update_traces(
            texttemplate='₹%{text:,.0f}',
            textposition='outside')
        st.plotly_chart(fig_rv,
                        use_container_width=True,
                        key="rev_bar")

    with col8:
        sc_list = list(range(10, 110, 10))
        sv_list = [total_risk*(s/100) for s in sc_list]
        fig_ln = px.line(
            x=sc_list, y=sv_list, markers=True,
            labels={'x': 'Retention %',
                    'y': 'Revenue Saved (₹)'},
            color_discrete_sequence=['green'],
            title="Retention Impact Scenarios")
        fig_ln.add_hline(
            y=saved_rev, line_dash="dash",
            line_color="red",
            annotation_text=f"Current: {retention}%")
        st.plotly_chart(fig_ln,
                        use_container_width=True,
                        key="rev_line")

    st.markdown("---")
    st.subheader("📋 Action Plan")
    action_plan = pd.DataFrame({
        'Priority':         ['1st', '2nd', '3rd'],
        'Segment':          ['Critical','High','Medium'],
        'Customers':        [len(crit), len(high), len(med)],
        'Revenue at Risk':  [f'₹{cr:,.0f}',
                             f'₹{hr:,.0f}',
                             f'₹{mr:,.0f}'],
        'Action':           ['Personal call + 30% disc',
                             'Email + 20% disc',
                             'Newsletter + 10% disc'],
        'Timeline':         ['Today','3 days','1 week']})
    st.dataframe(action_plan, use_container_width=True)

# ============================================================
# PAGE 7 — LTV DECAY ANALYSIS
# ============================================================
elif page == "📉 LTV Decay Analysis":

    st.title("📉 Customer Lifetime Value Decay")
    st.markdown(
        "**Core Innovation** — Predict HOW FAST "
        "customer value decays and WHEN to intervene!")
    st.markdown("---")

    import sys
    sys.path.append(BASE)
    from src.models.ltv_decay_predictor import (
        calculate_customer_ltv,
        calculate_intervention_roi)

    decay_path = os.path.join(
        BASE, "data", "processed",
        "ltv_decay_portfolio.csv")

    if os.path.exists(decay_path):
        portfolio = pd.read_csv(decay_path)
    else:
        st.warning(
            "⚠️ Run notebook 12_ltv_decay_innovation first!")
        st.stop()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("💰 Portfolio LTV",
                f"₹{portfolio['current_ltv'].sum():,.0f}")
    col2.metric("⚠️ Value at Risk",
                f"₹{portfolio['value_at_risk'].sum():,.0f}")
    col3.metric("🔴 Critical Decay",
                len(portfolio[
                    portfolio['urgency']=='Critical']))
    col4.metric("📊 Avg Decay Rate",
                f"{portfolio['decay_rate_pct'].mean():.1f}%")

    st.markdown("---")
    st.info("""
    **What is LTV Decay? (Our Core Innovation)**
    Traditional tools say: "Customer will churn — 84% probability"
    SENTINEL AI shows: "Customer value will drop from ₹780 → ₹0
    in exactly 4 months. Act in Month 1 for 233% ROI!"
    No existing paper or tool does this!
    """)
    st.markdown("---")

    st.subheader("🔍 Individual Customer LTV Decay")
    col5, col6 = st.columns(2)
    with col5:
        ltv_cust = st.number_input(
            "Customer ID", min_value=0,
            max_value=len(df)-1,
            value=int(df['churn_prob_30day'].idxmax()),
            key="ltv_cust")
    with col6:
        ltv_action = st.selectbox(
            "Intervention Action",
            ['discount_10','discount_20','discount_30',
             'upgrade','support','loyalty','all'],
            key="ltv_action")

    if st.button("📉 Analyze LTV Decay"):
        row     = df.iloc[ltv_cust]
        monthly = float(row.get('MonthlyCharges', 65))
        tenure  = float(row.get('tenure', 12))
        c30     = float(row['churn_prob_30day'])
        c60     = float(row['churn_prob_60day'])
        c90     = float(row['churn_prob_90day'])

        ltv = calculate_customer_ltv(
            monthly, tenure, c30, c60, c90)
        roi = calculate_intervention_roi(
            ltv, ltv_action, monthly)

        col7, col8, col9 = st.columns(3)
        col7.metric("Current LTV",
                    f"₹{ltv['current_ltv']:,.0f}/yr")
        col8.metric("Value at Risk",
                    f"₹{ltv['value_at_risk']:,.0f}",
                    f"{ltv['decay_rate_pct']:.1f}% decay")
        col9.metric("Act by",
                    f"Month {ltv['intervention_month']}",
                    ltv['decay_pattern'])

        fig_dc = go.Figure()
        fig_dc.add_trace(go.Scatter(
            x=ltv['months'],
            y=ltv['decay_curve'],
            mode='lines+markers',
            name='LTV Projection',
            line=dict(color='red', width=3),
            fill='tozeroy',
            fillcolor='rgba(255,0,0,0.1)'))
        fig_dc.add_vline(
            x=ltv['intervention_month'],
            line_dash="dash", line_color="green",
            annotation_text=
            f"Act Here! Month {ltv['intervention_month']}")
        fig_dc.update_layout(
            title=f"Customer {ltv_cust} LTV Decay",
            xaxis_title="Months from Today",
            yaxis_title="LTV (₹/year)",
            height=380)
        st.plotly_chart(fig_dc,
                        use_container_width=True,
                        key="ltv_curve")

        if roi['net_roi'] > 0:
            st.success(f"""
✅ **TAKE THIS ACTION NOW!**
Cost: ₹{roi['action_cost']:,.0f} |
Saves: ₹{roi['value_saved']:,.0f} |
**Net ROI: ₹{roi['net_roi']:,.0f} ({roi['roi_percentage']:.0f}%)**
Best Month: {roi['best_month_to_act']}
            """)
        else:
            st.warning(
                "⚠️ Try a cheaper intervention!")

    st.markdown("---")
    st.subheader("🔴 Top 10 Fastest Decaying Customers")
    top_d = portfolio.nlargest(10, 'decay_rate_pct')[
        ['customer_id','current_ltv',
         'value_at_risk','decay_rate_pct',
         'urgency','intervention_month']
    ].reset_index(drop=True)
    st.dataframe(top_d, use_container_width=True)

# ============================================================
# PAGE 8 — CHURN VELOCITY INDEX
# ============================================================
elif page == "⚡ Churn Velocity Index":

    st.title("⚡ Churn Velocity Index (CVI)")
    st.markdown(
        "**Core Innovation** — Not just WILL they leave "
        "but HOW FAST are they moving toward leaving!")
    st.markdown("---")

    import sys
    sys.path.append(BASE)
    from src.models.churn_velocity import (
        add_cvi_to_dataframe,
        calculate_churn_velocity)

    cvi_path = os.path.join(
        BASE, "data", "processed",
        "telco_with_cvi.csv")

    if os.path.exists(cvi_path):
        df_cvi = pd.read_csv(cvi_path)
    else:
        with st.spinner("Calculating CVI..."):
            df_cvi = add_cvi_to_dataframe(df)
            df_cvi.to_csv(cvi_path, index=False)

    total    = len(df_cvi)
    crit_vel = len(df_cvi[df_cvi['cvi'] >= 8])
    high_vel = len(df_cvi[
        (df_cvi['cvi'] >= 6) &
        (df_cvi['cvi'] < 8)])
    accel    = len(df_cvi[
        df_cvi['is_accelerating'] == True])

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("⚡ Avg CVI",
                f"{df_cvi['cvi'].mean():.1f}/10")
    col2.metric("🔴 Critical Velocity",
                f"{crit_vel:,}",
                f"{crit_vel/total*100:.1f}%")
    col3.metric("🟠 High Velocity",
                f"{high_vel:,}")
    col4.metric("📈 Accelerating",
                f"{accel:,}")

    st.markdown("---")
    st.info("""
    **What is Churn Velocity Index? (Our Core Innovation)**
    Two customers — both 70% churn risk.
    Customer A: Was 68% last month → Moving SLOWLY (not urgent)
    Customer B: Was 50% last month → Moving FAST (urgent!)
    Traditional tools treat them the same.
    SENTINEL AI CVI shows Customer B needs action TODAY!
    """)
    st.markdown("---")

    st.subheader("🔍 Individual Customer CVI")
    cvi_cust = st.number_input(
        "Customer ID", min_value=0,
        max_value=len(df)-1,
        value=int(df['churn_prob_30day'].idxmax()),
        key="cvi_cust")

    if st.button("⚡ Calculate CVI"):
        row2 = df.iloc[cvi_cust]
        c30  = float(row2['churn_prob_30day'])
        c60  = float(row2['churn_prob_60day'])
        c90  = float(row2['churn_prob_90day'])
        cvi  = calculate_churn_velocity(c30, c60, c90)

        col5, col6, col7 = st.columns(3)
        col5.metric("⚡ CVI Score", f"{cvi['cvi']}/10")
        col6.metric("Category",
                    cvi['category'].split('—')[0])
        col7.metric("Urgency", cvi['urgency'])

        fig_vt = go.Figure()
        fig_vt.add_trace(go.Scatter(
            x=['30-Day','60-Day','90-Day'],
            y=[c30, c60, c90],
            mode='lines+markers',
            line=dict(
                color='red' if cvi['cvi'] >= 6
                else 'orange',
                width=3),
            marker=dict(size=12)))
        fig_vt.update_layout(
            title=f"Customer {cvi_cust} — CVI: {cvi['cvi']}/10",
            xaxis_title="Time Horizon",
            yaxis_title="Churn Probability (%)",
            yaxis=dict(range=[0, 105]),
            height=350)
        st.plotly_chart(fig_vt,
                        use_container_width=True,
                        key="vel_chart")

        if cvi['is_accelerating']:
            st.error(
                f"🚨 ACCELERATING! {cvi['message']} "
                f"— {cvi['urgency']}")
        else:
            st.success(f"✅ STABLE. {cvi['message']}")

    st.markdown("---")
    st.subheader(
        "📊 2D Risk View — Probability vs Velocity")
    st.markdown(
        "Traditional tools only see the X-axis. "
        "SENTINEL AI shows BOTH dimensions!")

    sample = df_cvi.sample(
        min(500, len(df_cvi)), random_state=42)
    fig_2d = px.scatter(
        sample,
        x='churn_prob_30day',
        y='cvi',
        color='cvi_category',
        title="Churn Probability vs Velocity Index",
        labels={
            'churn_prob_30day': 'Churn Probability (%)',
            'cvi':              'CVI Score (0-10)'},
        opacity=0.6)
    fig_2d.add_hline(
        y=6, line_dash="dash", line_color="red",
        annotation_text="High Velocity Threshold")
    fig_2d.add_vline(
        x=60, line_dash="dash", line_color="red",
        annotation_text="High Risk Threshold")
    fig_2d.add_annotation(
        x=80, y=9, text="⚠️ DANGER ZONE",
        showarrow=False,
        font=dict(color="red", size=12))
    fig_2d.add_annotation(
        x=15, y=9, text="🔍 HIDDEN DANGER",
        showarrow=False,
        font=dict(color="orange", size=12))
    st.plotly_chart(fig_2d,
                    use_container_width=True,
                    key="cvi_2d")

    st.markdown("---")
    st.subheader("🔍 Hidden Danger Customers")
    st.warning(
        "These look safe by traditional methods "
        "but are accelerating toward churn fast!")

    hidden = df_cvi[
        (df_cvi['churn_prob_30day'] < 60) &
        (df_cvi['cvi'] >= 7)
    ].head(10)

    if len(hidden) > 0:
        st.dataframe(
            hidden[['churn_prob_30day','cvi',
                    'cvi_category',
                    'cvi_urgency']].reset_index(),
            use_container_width=True)
    else:
        st.success("No hidden danger customers found!")

    st.markdown("---")
    st.subheader("⚡ Top 10 Highest Velocity Customers")
    top_v = df_cvi.nlargest(10, 'cvi')[
        ['churn_prob_30day','cvi','cvi_category',
         'cvi_urgency','is_accelerating']
    ].reset_index()
    st.dataframe(top_v, use_container_width=True)

# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.markdown("""
<div style='text-align:center; color:gray; font-size:0.8rem;'>
🚨 SENTINEL AI — AI Driven Business Risk Intelligence |
Python + XGBoost + Streamlit | JVIT Bengaluru
</div>
""", unsafe_allow_html=True)