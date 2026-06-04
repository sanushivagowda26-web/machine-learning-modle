import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pickle
import os

# -----------------------------------------------
# Page Layout & Styling
# -----------------------------------------------
st.set_page_config(
    page_title="Industrial Predictive Maintenance Hub",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling Override
st.markdown("""
    <style>
    .main .block-container { padding-top: 2rem; }
    .stMetric { background-color: #f8f9fa; padding: 15px; border-radius: 10px; border: 1px solid #e9ecef; }
    </style>
""", unsafe_html=True)

# -----------------------------------------------
# Data & Model Loaders (Cached)
# -----------------------------------------------
@st.cache_data
def load_analytics_data():
    if os.path.exists("data/predictive_maintenance.csv"):
        return pd.read_csv("data/predictive_maintenance.csv")
    else:
        st.error("Data file missing! Please place 'predictive_maintenance.csv' in the 'data/' directory.")
        return None

@st.cache_resource
def load_predefined_model():
    if os.path.exists("models/maintenance_model.pkl"):
        with open("models/maintenance_model.pkl", "rb") as f:
            return pickle.load(f)
    else:
        st.error("Model artifact missing! Please place 'maintenance_model.pkl' in the 'models/' directory.")
        return None

df = load_analytics_data()
model = load_predefined_model()

# -----------------------------------------------
# Header Section
# -----------------------------------------------
st.title("⚙️ Industrial Predictive Maintenance Analytics")
st.markdown("Monitor real-time structural asset logs, run exploratory analytics, and run predefined ML classification inference outputs below.")
st.write("---")

if df is not None:
    # -----------------------------------------------
    # Sidebar Configuration & Filters
    # -----------------------------------------------
    st.sidebar.header("Global Dashboard Controls")
    product_type = st.sidebar.multiselect("Filter by Product Variant Type:", 
                                          options=df["Type"].unique(), 
                                          default=df["Type"].unique())
    
    # Filter dataset mapping dynamically
    filtered_df = df[df["Type"].isin(product_type)]

    # -----------------------------------------------
    # Row 1: High-Level Analytics Metrics
    # -----------------------------------------------
    total_records = len(filtered_df)
    total_failures = int(filtered_df["Target"].sum())
    failure_rate = (total_failures / total_records) * 100 if total_records > 0 else 0
    avg_tool_wear = filtered_df["Tool wear [min]"].mean()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric(label="📊 Tracked Assets Logged", value=f"{total_records:,}")
    m2.metric(label="🚨 Registered Failures", value=f"{total_failures:,}", delta=f"{failure_rate:.2f}% Rate", delta_color="inverse")
    m3.metric(label="⏱️ Avg Tool Wear Time", value=f"{avg_tool_wear:.1f} min")
    m4.metric(label="🌡️ Operating System Temp", value=f"{filtered_df['Process temperature [K]'].mean():.1f} K")

    st.write("##")

    # -----------------------------------------------
    # Main Interactive Core Tabs
    # -----------------------------------------------
    tab1, tab2, tab3 = st.tabs(["📊 Exploratory Deep Analytics", "🔮 ML Inference Workspace", "📋 Raw Telemetry Data"])

    # --- TAB 1: EXPLORATORY DEEP ANALYTICS ---
    with tab1:
        st.subheader("Asset Behavior & Failure Correlations")
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Rotational Speed vs. Torque Dynamics**")
            fig_scatter = px.scatter(
                filtered_df, 
                x="Rotational speed [rpm]", 
                y="Torque [Nm]", 
                color="Target",
                color_continuous_scale=["#1f77b4", "#d62728"],
                labels={"Target": "Failure Status"},
                opacity=0.6
            )
            fig_scatter.update_layout(margin=dict(l=20, r=20, t=20, b=20))
            st.plotly_chart(fig_scatter, use_container_width=True)
            
        with c2:
            st.markdown("**Tool Wear Breakdown Distributions**")
            fig_box = px.box(
                filtered_df, 
                x="Type", 
                y="Tool wear [min]", 
                color="Target",
                color_discrete_map={0: "#1f77b4", 1: "#d62728"}
            )
            fig_box.update_layout(margin=dict(l=20, r=20, t=20, b=20))
            st.plotly_chart(fig_box, use_container_width=True)

        st.write("---")
        st.markdown("### 💡 Automatically Generated Insights")
        
        # Static conditional analytical summaries based on the active dataset slice
        high_wear_failure = filtered_df[(filtered_df["Tool wear [min]"] > 180) & (filtered_df["Target"] == 1)]
        st.info(f"**Wear Notice:** There are currently **{len(high_wear_failure)}** documented operational failures linked directly to a critically high tool wear time slice boundary exceeding 180 minutes.")
        
        if total_failures > 0:
            avg_fail_torque = filtered_df[filtered_df["Target"] == 1]["Torque [Nm]"].mean()
            avg_safe_torque = filtered_df[filtered_df["Target"] == 0]["Torque [Nm]"].mean()
            st.warning(f"**Stress Disparity:** Failed machinery units operated at an average torque output limit of **{avg_fail_torque:.2f} Nm** vs safe benchmark baseline levels (**{avg_safe_torque:.2f} Nm**).")

    # --- TAB 2: PRE-TRAINED ML INFERENCE WORKSPACE ---
    with tab2:
        st.subheader("Predefined Model Predictive Classification Sandbox")
        
        if model is not None:
            st.markdown("Adjust input parameter telemetry values below to send vectors to the predefined pipeline pickle file.")
            
            # Form UI layout setup for simulation inputs
            with st.form("inference_form"):
                sc1, sc2, sc3 = st.columns(3)
                with sc1:
                    air_temp = st.number_input("Air temperature [K]", min_value=280.0, max_value=320.0, value=300.0, step=0.1)
                    process_temp = st.number_input("Process temperature [K]", min_value=290.0, max_value=330.0, value=310.0, step=0.1)
                with sc2:
                    rot_speed = st.number_input("Rotational speed [rpm]", min_value=1000, max_value=3000, value=1500, step=10)
                    torque = st.number_input("Torque [Nm]", min_value=0.0, max_value=100.0, value=40.0, step=0.5)
                with sc3:
                    tool_wear = st.slider("Current Tool Wear [min]", min_value=0, max_value=300, value=50)
                
                submit_btn = st.form_submit_button("Compute Diagnostic Classification Risk", use_container_width=True)
                
            if submit_btn:
                # Compile feature array matches model profile expectance standard
                input_vector = np.array([[air_temp, process_temp, rot_speed, torque, tool_wear]])
                
                # Perform forward pass computation over pre-saved pickle pipeline state
                prediction = model.predict(input_vector)[0]
                probabilities = model.predict_proba(input_vector)[0]
                
                st.write("---")
                st.subheader("Classification Outcome")
                
                col_res1, col_res2 = st.columns([1, 2])
                with col_res1:
                    if prediction == 1:
                        st.error("🚨 CRITICAL: FAILURE RISK DETECTED")
                    else:
                        st.success("✅ OPTIMAL: NOMINAL CONDITIONS")
                
                with col_res2:
                    # Render progress metric breakdown mapping breakdown rates
                    st.markdown(f"**Confidence Matrix Probability Breakdown:**")
                    st.progress(float(probabilities[1]), text=f"Failure Likelihood Probability Profile: {probabilities[1]*100:.1f}%")
        else:
            st.warning("Prediction sandbox disabled because the pre-defined model model/maintenance_model.pkl file was not recovered.")

    # --- TAB 3: RAW TELEMETRY DATA ---
    with tab3:
        st.subheader("Raw Fleet Asset Logs")
        st.markdown("Below is the filtered interactive telemetry ledger.")
        st.dataframe(filtered_df, use_container_width=True, hide_index=True)
