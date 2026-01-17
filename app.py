import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from live_predictor import get_live_prediction
import datetime
import os
import numpy as np

# --- Page Config ---
st.set_page_config(
    page_title="Avengers AI Trader",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Premium Look ---
st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
    }
    .metric-card {
        background-color: #1e2530;
        border-radius: 10px;
        padding: 20px;
        color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    .big-signal {
        font-size: 48px;
        font-weight: 800;
        text-align: center;
        margin-bottom: 0px;
    }
    .sub-signal {
        font-size: 18px;
        color: #a0a0a0;
        text-align: center;
        margin-top: -10px;
    }
</style>
""", unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.title("🛡️ Avengers AI")
    st.markdown("---")
    st.markdown("**Target Asset**")
    st.info("Samsung Electronics (005930.KS)")
    
    st.markdown("**Model Specs**")
    st.caption("Type: XGBoost Classifier")
    st.caption("Features: S&P500, RSI, Momentum, Vol, US10Y")
    st.caption("Threshold: 61.0%")
    
    st.markdown("---")
    st.markdown("**Data History**")
    st.caption("Recording since 2024-06-01")
    
    st.markdown("---")
    st.markdown("Created by **Antigravity**")

# --- Tabs ---
tab1, tab2 = st.tabs(["🚀 Daily Signal", "📜 History & Records"])

# === Tab 1: Live Prediction ===
with tab1:
    st.title("Daily Trading Intelligence")
    
    # Run Logic
    if 'prediction' not in st.session_state:
        st.session_state['prediction'] = None

    if st.button("🔄 Analyze New Data", use_container_width=True):
        with st.spinner("Connecting to Global Markets (KR, US)..."):
            st.session_state['prediction'] = get_live_prediction()

    result = st.session_state['prediction']

    if result:
        if "error" in result:
            st.error(f"System Error: {result['error']}")
        else:
            # Extract Data
            prob = result['prob_up']
            signal = result['signal']
            date_used = result['date_used']
            features = result['features']
            
            st.markdown(f"#### 📅 Data Date: {date_used} (Market Close)")
            
            # --- Top Dashboard (2 Columns) ---
            col_left, col_right = st.columns([1, 2])
            
            with col_left:
                # Signal Card
                color = "#00cc66" if signal == "BUY" else "#ff4b4b" if prob < 40 else "#7a7a7a"
                card_bg = "rgba(0, 204, 102, 0.1)" if signal == "BUY" else "rgba(122, 122, 122, 0.1)"
                
                st.markdown(f"""
                <div class="metric-card" style="border: 2px solid {color}; background-color: {card_bg};">
                    <div class="sub-signal">ACTION SIGNAL</div>
                    <div class="big-signal" style="color: {color};">{signal}</div>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("###") # Spacer
                st.metric(label="Win Probability", value=f"{prob:.2f}%", delta=f"{prob-61:.1f}% vs Threshold")

            with col_right:
                # Gauge Chart
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = prob,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "<b>Bullish Probability</b>", 'font': {'size': 20}},
                    gauge = {
                        'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "white"},
                        'bar': {'color': color, 'thickness': 0.75},
                        'bgcolor': "#1e2530",
                        'borderwidth': 2,
                        'bordercolor': "#333",
                        'steps': [
                            {'range': [0, 61], 'color': '#333'},
                            {'range': [61, 100], 'color': '#262626'}],
                        'threshold': {
                            'line': {'color': "white", 'width': 4},
                            'thickness': 0.8,
                            'value': 61}}))
                
                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font={'color': "white", 'family': "Arial"})
                st.plotly_chart(fig, use_container_width=True)

            # --- Feature Analysis Section ---
            st.divider()
            st.subheader("📊 Market factors Analysis")
            
            key_features = {
                'RSI_Golden_Cross_Lag1': 'RSI Golden Cross',
                'Risk_Adj_Mom_Lag1': 'Risk Adj Momentum',
                'LogReturn_SP500_Lag1': 'S&P 500 Return',
                'LogReturn_US10Y_Lag1': 'US 10Y Yield Change',
                'Vol_Change_Lag1': 'Volume Change',
                'Momentum_5D': '5D Momentum'
            }
            
            plot_data = []
            for key, label in key_features.items():
                val = features.get(key, 0)
                if 'Golden' in label:
                    val_fmt = "Active" if val == 1 else "Inactive"
                else:
                    val_fmt = f"{val:.4f}"
                plot_data.append({"Factor": label, "Value": val, "Display": val_fmt})
            
            # 6 Columns for Key Metrics
            cols = st.columns(6)
            for i, row in enumerate(plot_data):
                with cols[i]:
                    st.metric(label=row['Factor'], value=row['Display'])
            
            # Save History Automatically
            if 'history' in result:
                hist_df = result['history']
                hist_df.to_csv('prediction_history.csv')
    else:
        st.info("👋 Welcome! Click 'Analyze New Data' to generate today's trading signal.")

# === Tab 2: History ===
with tab2:
    st.header("📜 Prediction History (Since 2024-06-01)")
    
    # Auto-load or Generate
    if not os.path.exists('prediction_history.csv'):
        with st.spinner("Generating history from live data..."):
            res = get_live_prediction()
            if 'history' in res:
                res['history'].to_csv('prediction_history.csv')
                st.success("History generated successfully!")
                st.rerun()
            else:
                st.error("Could not generate history.")
    
    # Force Refresh Button
    if st.button("🔄 Reset History (Fetch 2 Years)"):
        with st.spinner("Regenerating history..."):
            res = get_live_prediction()
            if 'history' in res:
                res['history'].to_csv('prediction_history.csv')
                st.success("History updated!")
                st.rerun()

    if os.path.exists('prediction_history.csv'):
        hist_df = pd.read_csv('prediction_history.csv', index_col=0)
        hist_df.index = pd.to_datetime(hist_df.index)
        hist_df = hist_df.sort_index(ascending=False) # Newest first
        
        # --- Month Filter ---
        # Get unique months from index
        months = hist_df.index.strftime('%Y-%m').unique().tolist()
        months.insert(0, 'All Time')
        
        selected_month = st.selectbox("Select Month", months)
        
        if selected_month != 'All Time':
            # Filter by month
            filtered_df = hist_df[hist_df.index.strftime('%Y-%m') == selected_month]
        else:
            filtered_df = hist_df
            
        # Stats (Recalculate for filtered view)
        total_days = len(filtered_df)
        if total_days > 0:
            win_rate = (filtered_df['Strategy_Return'] > 0).mean() * 100
            # Cumulative return for this period
            cum_ret = (np.exp(filtered_df['Strategy_Return'].cumsum()) - 1) * 100
            period_return = cum_ret.iloc[-1]
        else:
            win_rate = 0
            period_return = 0
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Trading Days", f"{total_days} Days")
        col2.metric("Period Return", f"{period_return:.2f}%")
        col3.metric("Win Rate", f"{win_rate:.1f}%")
        
        # Chart
        st.subheader("Equity Curve (Selected Period)")
        if not filtered_df.empty:
            # Re-sort for Chart (Chronological)
            chart_df = filtered_df.sort_index(ascending=True)
            equity = (np.exp(chart_df['Strategy_Return'].cumsum()) - 1) * 100
            st.line_chart(equity)
        else:
            st.info("No data for selected period.")
        
        # Table
        st.subheader("Daily Logs")
        
        # formatting (Convert Log Return -> Simple Return for Display)
        display_df = filtered_df.copy()
        display_df['Log_Return'] = display_df['Log_Return'].apply(lambda x: f"{(np.exp(x)-1)*100:.2f}%")
        display_df['Strategy_Return'] = display_df['Strategy_Return'].apply(lambda x: f"{(np.exp(x)-1)*100:.2f}%")
        display_df['Prob_Up'] = display_df['Prob_Up'].apply(lambda x: f"{x:.2f}%")
        display_df['Price'] = display_df['Price'].apply(lambda x: f"{x:,.0f}")
        
        # Rename columns for clarity
        display_df = display_df.rename(columns={
            'Price': 'Close Price',
            'Log_Return': 'Daily Return',
            'Strategy_Return': 'Strat Return'
        })
        
        st.dataframe(display_df, use_container_width=True, height=500)
        
        st.download_button("Download CSV", hist_df.to_csv(), "prediction_history.csv", "text/csv")

# --- Footer ---
st.markdown("---")
st.caption("Automated Trading System v1.1 | Powered by XGBoost")
