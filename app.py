import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os

# ----------------------------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="RailGuard | Track Safety Predictor",
    page_icon="🚆",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------------
# CUSTOM CSS  (theme base colors also set in .streamlit/config.toml)
# ----------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
    }

    /* Light background with a subtle railway-track pattern (rails + sleepers) */
    [data-testid="stAppViewContainer"], [data-testid="stMain"], .stApp {
        background-color: #f4f4f5 !important;
    }
    [data-testid="stHeader"] {
        background: transparent !important;
    }
    .block-container {
        padding-top: 2rem;
        max-width: 1250px;
    }

    /* Header */
    .rg-header {
        background: linear-gradient(120deg, #14335c 0%, #1f5187 55%, #2f6fae 100%);
        padding: 2.4rem 2.6rem;
        border-radius: 20px;
        margin-bottom: 1.8rem;
        border: 1px solid rgba(255,255,255,0.12);
        box-shadow: 0 12px 30px rgba(20,51,92,0.25);
        position: relative;
        overflow: hidden;
    }
    .rg-header::after {
        content: "";
        position: absolute;
        left: 0; right: 0; bottom: 0;
        height: 34px;
        background-image:
            linear-gradient(rgba(255,255,255,0.12) 4px, transparent 4px),
            repeating-linear-gradient(90deg, rgba(255,255,255,0.16) 0px, rgba(255,255,255,0.16) 5px, transparent 5px, transparent 26px);
        background-position: 0 0, 0 8px;
        opacity: 0.7;
    }
    .rg-header h1 {
        color: #ffffff;
        font-weight: 800;
        font-size: 2.15rem;
        margin-bottom: 0.3rem;
        letter-spacing: -0.5px;
    }
    .rg-header p {
        color: #dbe7f5;
        font-size: 1.0rem;
        margin: 0;
        max-width: 780px;
    }
    .rg-badge {
        display:inline-block;
        background: rgba(255,255,255,0.16);
        color:#ffd98a;
        padding: 5px 16px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 1.2px;
        margin-bottom: 1rem;
        border: 1px solid rgba(255,217,138,0.4);
    }

    /* Section cards */
    .rg-card {
        background: #ffffff;
        border: 1px solid rgba(28,39,51,0.08);
        border-radius: 18px;
        padding: 1.5rem 1.7rem;
        margin-bottom: 1.3rem;
        box-shadow: 0 4px 18px rgba(28,39,51,0.06);
    }
    .rg-card h3 {
        color: #14335c;
        font-size: 1.05rem;
        font-weight: 700;
        margin-bottom: 1rem;
        display:flex;
        align-items:center;
        gap: 8px;
        border-bottom: 1px solid rgba(28,39,51,0.08);
        padding-bottom: 0.7rem;
    }

    /* Result cards */
    .result-safe {
        background: linear-gradient(135deg, #e4f8ef 0%, #c9f0dd 100%);
        border: 1px solid #34b378;
        border-radius: 20px;
        padding: 2.2rem;
        text-align: center;
        box-shadow: 0 10px 26px rgba(52,179,120,0.18);
        height: 100%;
    }
    .result-risk {
        background: linear-gradient(135deg, #fdeaea 0%, #fbd4d4 100%);
        border: 1px solid #e0524f;
        border-radius: 20px;
        padding: 2.2rem;
        text-align: center;
        box-shadow: 0 10px 26px rgba(224,82,79,0.18);
        height: 100%;
    }
    .result-title {
        font-size: 1.5rem;
        font-weight: 800;
        color: #17324f;
        margin-bottom: 0.3rem;
        letter-spacing: 0.5px;
    }
    .result-sub {
        color: #3d5064;
        font-size: 0.9rem;
    }

    /* Gauge ring */
    .gauge-wrap {
        display:flex;
        flex-direction:column;
        align-items:center;
        justify-content:center;
        padding: 0.5rem 0 0.2rem 0;
    }
    .gauge-ring {
        width: 150px;
        height: 150px;
        border-radius: 50%;
        display:flex;
        align-items:center;
        justify-content:center;
        position: relative;
        margin-bottom: 0.6rem;
    }
    .gauge-inner {
        width: 118px;
        height: 118px;
        border-radius: 50%;
        background: #ffffff;
        display:flex;
        flex-direction:column;
        align-items:center;
        justify-content:center;
        box-shadow: inset 0 0 10px rgba(0,0,0,0.08);
    }
    .gauge-val {
        font-size: 1.55rem;
        font-weight: 800;
        color: #17324f;
    }
    .gauge-lbl {
        font-size: 0.68rem;
        color: #66798f;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #ffffff;
        border-right: 1px solid rgba(28,39,51,0.08);
    }
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #14335c;
    }
    section[data-testid="stSidebar"] hr {
        border-color: rgba(28,39,51,0.1);
    }

    /* Buttons */
    .stButton>button {
        background: linear-gradient(120deg, #e0631f, #f4923e);
        color: #ffffff;
        font-weight: 700;
        border-radius: 10px;
        border: none;
        padding: 0.75rem 1.2rem;
        font-size: 1rem;
        width: 100%;
        transition: all 0.2s ease;
        box-shadow: 0 6px 16px rgba(224,99,31,0.28);
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 22px rgba(224,99,31,0.45);
    }

    .footer-note {
        text-align:center;
        color:#7c8ba0;
        font-size:0.8rem;
        margin-top: 2.2rem;
        padding-top: 1.2rem;
        border-top: 1px solid rgba(28,39,51,0.08);
    }

    .factor-chip {
        display:inline-block;
        background: #f2f5f9;
        color:#2c3e50;
        padding: 6px 14px;
        border-radius: 8px;
        font-size: 0.85rem;
        margin: 3px 6px 3px 0;
        border-left: 3px solid #e0631f;
    }

    [data-testid="stDataFrame"] {
        border-radius: 10px;
        overflow: hidden;
    }

    .stat-box {
        background: #f6f8fb;
        border: 1px solid rgba(28,39,51,0.08);
        border-radius: 12px;
        padding: 0.9rem 1rem;
        text-align:center;
    }
    .stat-box .num { font-size:1.3rem; font-weight:800; color:#17324f; }
    .stat-box .lbl { font-size:0.72rem; color:#66798f; text-transform:uppercase; letter-spacing:0.5px; }
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# LOAD MODEL
# ----------------------------------------------------------------------------
MODEL_FILENAME = "logistic_regression_railwaytrack.safety_model.pkl"

def find_model_path():
    candidates = [
        MODEL_FILENAME,
        os.path.join(os.path.dirname(os.path.abspath(__file__)), MODEL_FILENAME),
        os.path.join(os.getcwd(), MODEL_FILENAME),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        f"Could not find '{MODEL_FILENAME}'. Make sure this file is in the SAME folder as app.py. "
        f"Checked: {candidates}"
    )

@st.cache_resource
def load_model():
    return joblib.load(find_model_path())

model = load_model()

FEATURE_ORDER = [
    'Track_Age_Years', 'Previous_Incidents', 'Track_Crack_Yes',
    'Signal_Failure_Yes', 'Track_Obstruction_Yes', 'Maintenance_Overdue_Yes',
    'Weather_Rain', 'Weather_Sunny', 'Visibility_Moderate', 'Visibility_Poor',
    'Train_Speed_Low', 'Train_Speed_Medium', 'Sensor_Status_Working'
]

# ----------------------------------------------------------------------------
# HEADER
# ----------------------------------------------------------------------------
st.markdown("""
<div class="rg-header">
    <div class="rg-badge">AI POWERED &nbsp;•&nbsp; PREDICTIVE MAINTENANCE</div>
    <h1>🚆 RailGuard — Railway Track Safety Predictor</h1>
    <p>Logistic Regression based real-time risk assessment for railway track sections.
    Set the track's current condition on the left, then click Predict to get an instant safety verdict.</p>
</div>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# SIDEBAR - INPUTS
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## ⚙️ Input Parameters")
    st.caption("Step 1: Fill in the current condition of the track section.")

    st.markdown("### 🛤️ Track Condition")
    track_age = st.slider("Track Age (Years)", 0, 60, 10)
    prev_incidents = st.slider("Previous Incidents (count)", 0, 20, 0)
    track_crack = st.radio("Track Crack Detected?", ["No", "Yes"], horizontal=True)
    maintenance_overdue = st.radio("Maintenance Overdue?", ["No", "Yes"], horizontal=True)

    st.markdown("### 🚦 Signal & Obstruction")
    signal_failure = st.radio("Signal Failure?", ["No", "Yes"], horizontal=True)
    track_obstruction = st.radio("Track Obstruction?", ["No", "Yes"], horizontal=True)
    sensor_status = st.radio("Sensor Status", ["Working", "Not Working"], horizontal=True)

    st.markdown("### 🌦️ Environmental Conditions")
    weather = st.selectbox("Weather Condition", ["Sunny", "Rain", "Cloudy / Other"])
    visibility = st.selectbox("Visibility", ["Good", "Moderate", "Poor"])

    st.markdown("### 🚄 Operational Details")
    train_speed = st.selectbox("Train Speed Category", ["Low", "Medium", "High"])

    st.markdown("---")
    st.caption("Step 2: Click below to run the model.")
    predict_clicked = st.button("🔍 Predict Track Safety")

# ----------------------------------------------------------------------------
# BUILD FEATURE VECTOR
# ----------------------------------------------------------------------------
def build_features():
    row = {
        'Track_Age_Years': track_age,
        'Previous_Incidents': prev_incidents,
        'Track_Crack_Yes': 1 if track_crack == "Yes" else 0,
        'Signal_Failure_Yes': 1 if signal_failure == "Yes" else 0,
        'Track_Obstruction_Yes': 1 if track_obstruction == "Yes" else 0,
        'Maintenance_Overdue_Yes': 1 if maintenance_overdue == "Yes" else 0,
        'Weather_Rain': 1 if weather == "Rain" else 0,
        'Weather_Sunny': 1 if weather == "Sunny" else 0,
        'Visibility_Moderate': 1 if visibility == "Moderate" else 0,
        'Visibility_Poor': 1 if visibility == "Poor" else 0,
        'Train_Speed_Low': 1 if train_speed == "Low" else 0,
        'Train_Speed_Medium': 1 if train_speed == "Medium" else 0,
        'Sensor_Status_Working': 1 if sensor_status == "Working" else 0,
    }
    return pd.DataFrame([row])[FEATURE_ORDER]

# ----------------------------------------------------------------------------
# MAIN AREA — CONFIG SUMMARY
# ----------------------------------------------------------------------------
col1, col2 = st.columns([1.3, 1])

with col1:
    st.markdown('<div class="rg-card"><h3>📋 Current Configuration Summary</h3>', unsafe_allow_html=True)
    summary_df = pd.DataFrame({
        "Parameter": ["Track Age", "Previous Incidents", "Track Crack", "Maintenance Overdue",
                      "Signal Failure", "Track Obstruction", "Sensor Status",
                      "Weather", "Visibility", "Train Speed"],
        "Value": [f"{track_age} yrs", prev_incidents, track_crack, maintenance_overdue,
                  signal_failure, track_obstruction, sensor_status,
                  weather, visibility, train_speed]
    })
    st.dataframe(summary_df, hide_index=True, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="rg-card">
        <h3>ℹ️ How It Works</h3>
        <p style="color:#3d5064; font-size:0.9rem; line-height:1.7; margin-bottom:1rem;">
        This tool uses a trained <b>Logistic Regression</b> model on historical railway
        track data. It looks at track age, structural faults, signaling issues,
        weather, visibility and train speed to estimate whether a
        track section is <b>safe</b> or <b>needs attention</b>.
        </p>
        <ol style="color:#66798f; font-size:0.85rem; line-height:1.9; padding-left:1.1rem; margin:0;">
            <li>Set conditions in the sidebar</li>
            <li>Click <b>Predict Track Safety</b></li>
            <li>Read the verdict + probability below</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# GAUGE HELPER
# ----------------------------------------------------------------------------
def gauge_html(percent, color, label):
    percent = max(0, min(100, percent))
    deg = percent * 3.6
    return f"""<div class="gauge-wrap">
<div class="gauge-ring" style="background: conic-gradient({color} {deg}deg, rgba(28,39,51,0.1) {deg}deg);">
<div class="gauge-inner">
<div class="gauge-val">{percent:.1f}%</div>
<div class="gauge-lbl">{label}</div>
</div>
</div>
</div>"""

# ----------------------------------------------------------------------------
# PREDICTION
# ----------------------------------------------------------------------------
if predict_clicked:
    X = build_features()
    proba = model.predict_proba(X)[0]
    pred = model.predict(X)[0]
    risk_prob = proba[1] * 100
    safe_prob = proba[0] * 100

    st.markdown("## 🧠 Prediction Result")

    r1, r2 = st.columns([1, 1])

    with r1:
        if pred == 1:
            st.markdown(
f"""<div class="result-risk">
<div class="result-title">⚠️ UNSAFE — HIGH RISK</div>
{gauge_html(risk_prob, "#e0524f", "RISK LEVEL")}
<div class="result-sub">Estimated probability of an unsafe track condition</div>
</div>""", unsafe_allow_html=True)
        else:
            st.markdown(
f"""<div class="result-safe">
<div class="result-title">✅ SAFE TRACK CONDITION</div>
{gauge_html(safe_prob, "#34b378", "SAFETY LEVEL")}
<div class="result-sub">Estimated probability that the track is safe</div>
</div>""", unsafe_allow_html=True)

    with r2:
        st.markdown("<div class='rg-card'><h3>📊 Probability Breakdown</h3>", unsafe_allow_html=True)
        st.progress(int(risk_prob), text=f"Risk Probability: {risk_prob:.1f}%")
        st.progress(int(safe_prob), text=f"Safe Probability: {safe_prob:.1f}%")

        s1, s2 = st.columns(2)
        with s1:
            st.markdown(f"""<div class="stat-box"><div class="num">{prev_incidents}</div><div class="lbl">Past Incidents</div></div>""", unsafe_allow_html=True)
        with s2:
            st.markdown(f"""<div class="stat-box"><div class="num">{track_age}y</div><div class="lbl">Track Age</div></div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Key contributing factors
    factors = []
    if track_crack == "Yes": factors.append("Track Crack Detected")
    if signal_failure == "Yes": factors.append("Signal Failure")
    if track_obstruction == "Yes": factors.append("Track Obstruction")
    if maintenance_overdue == "Yes": factors.append("Maintenance Overdue")
    if sensor_status == "Not Working": factors.append("Sensor Not Working")
    if visibility == "Poor": factors.append("Poor Visibility")
    if prev_incidents >= 3: factors.append("High Previous Incident Count")
    if track_age >= 30: factors.append("Aging Track (30+ yrs)")

    st.markdown("<div class='rg-card'><h3>🔎 Key Contributing Factors</h3>", unsafe_allow_html=True)
    if factors:
        chips = "".join([f"<span class='factor-chip'>{f}</span>" for f in factors])
        st.markdown(chips, unsafe_allow_html=True)
    else:
        st.markdown("<span style='color:#3d5064;'>No major risk factors detected in current inputs.</span>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    if pred == 1:
        st.warning("🚧 Recommendation: Flag this section for immediate inspection and maintenance before the next scheduled run.")
    else:
        st.success("✅ Recommendation: Track section is within safe operating parameters. Continue routine monitoring.")

else:
    st.info("👈 Set the parameters in the sidebar and click **Predict Track Safety** to see results.")

st.markdown("<div class='footer-note'>RailGuard © 2026 · Powered by Logistic Regression · For decision-support purposes only</div>", unsafe_allow_html=True)
