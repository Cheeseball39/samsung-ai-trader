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
    .hero-container {
        border-radius: 20px;
        padding: 30px;
        color: white;
        margin-bottom: 20px;
        box-shadow: 0 10px 20px rgba(0,0,0,0.3);
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .hero-left {
        flex: 1;
        border-right: 1px solid rgba(255,255,255,0.2);
        padding-right: 20px;
    }
    .hero-right {
        flex: 1.5;
        padding-left: 30px;
    }
    .signal-label {
        font-size: 1rem;
        text-transform: uppercase;
        letter-spacing: 2px;
        opacity: 0.8;
        margin-bottom: 5px;
    }
    .signal-value {
        font-size: 4rem;
        font-weight: 800;
        line-height: 1;
        text-shadow: 0 2px 4px rgba(0,0,0,0.5);
    }
    .conf-label {
        font-size: 1.2rem;
        margin-bottom: 10px;
        display: flex;
        justify-content: space-between;
    }
    .progress-bg {
        background-color: rgba(255,255,255,0.1);
        border-radius: 10px;
        height: 15px;
        width: 100%;
        overflow: hidden;
        position: relative;
    }
    .progress-fill {
        height: 100%;
        border-radius: 10px;
        transition: width 1s ease-in-out;
    }
    .threshold-marker {
        position: absolute;
        top: 0;
        bottom: 0;
        width: 2px;
        background-color: white;
        z-index: 10;
        box-shadow: 0 0 5px rgba(0,0,0,0.5);
    }
    
    @media (max-width: 768px) {
        .hero-container {
            flex-direction: column;
            text-align: center;
        }
        .hero-left {
            border-right: none;
            border-bottom: 1px solid rgba(255,255,255,0.2);
            padding-right: 0;
            padding-bottom: 20px;
            margin-bottom: 20px;
            width: 100%;
        }
        .hero-right {
            padding-left: 0;
            width: 100%;
        }
        .conf-label {
            justify-content: center;
            gap: 10px;
        }
    }
</style>
""", unsafe_allow_html=True)



# --- Tabs ---
tab1, tab2 = st.tabs(["🚀 Daily Signal", "📜 History & Records"])

# === Tab 1: Live Prediction ===
with tab1:
    st.title("Daily Trading Intelligence")
    
    # Run Logic
    # Run Logic
    if 'prediction' not in st.session_state:
        with st.spinner("Initializing AI Dashboard..."):
            st.session_state['prediction'] = get_live_prediction()

    if st.button("🔄 Refresh Data", use_container_width=True):
        with st.spinner("Fetching Latest Market Data..."):
            st.session_state['prediction'] = get_live_prediction()
            st.rerun()
            
    st.caption("💡 **Best Timing: 15:20 ~ 15:30** (장 마감 직전 확인 후 진입 추천)")

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
            
            st.markdown(f"#### 📅 Target Date: {date_used}")            
            
            # --- Hero Dashboard (Unified Design) ---
            # Dynamic Colors
            if signal == "BUY":
                grad_bg = "linear-gradient(135deg, #004d26 0%, #00cc66 100%)"
                res_color = "#ffffff"
            elif prob < 40:
                grad_bg = "linear-gradient(135deg, #4d0000 0%, #ff4b4b 100%)"
                res_color = "#ffffff"
            else:
                grad_bg = "linear-gradient(135deg, #2c2c2c 0%, #7a7a7a 100%)"
                res_color = "#e0e0e0"
            
            # Progress Bar Logic
            threshold_pct = 61
            
            st.markdown(f"""
            <div class="hero-container" style="background: {grad_bg};">
                <div class="hero-left">
                    <div class="signal-label">AI Action Signal</div>
                    <div class="signal-value" style="color: {res_color};">{signal}</div>
                </div>
                <div class="hero-right">
                    <div class="conf-label">
                        <span>Bullish Probability</span>
                        <span style="font-weight:bold; font-size:1.5rem;">{prob:.1f}%</span>
                    </div>
                    <div class="progress-bg">
                        <div class="progress-fill" style="width: {prob}%; background-color: white;"></div>
                        <div class="threshold-marker" style="left: {threshold_pct}%;" title="Threshold {threshold_pct}%"></div>
                    </div>
                    <div style="font-size: 0.8rem; margin-top: 5px; opacity: 0.8; text-align: right;">
                        Target Threshold: {threshold_pct}%
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

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

# === Tab 2: History ===
with tab2:
    st.header("📜 Prediction History (Since 2024-06-01)")
    
    # Logic to Auto-load or Force Regenerate if history is too short
    should_regenerate = False
    
    if not os.path.exists('prediction_history.csv'):
        should_regenerate = True
    else:
        # Check if existing file is the "old version" (short history starting 2026)
        try:
            temp_df = pd.read_csv('prediction_history.csv', index_col=0)
            if not temp_df.empty:
                temp_df.index = pd.to_datetime(temp_df.index)
                min_date = temp_df.index.min()
                # If data starts after 2025-01-01, it's the old short version. We want 2024-06.
                if min_date > pd.Timestamp("2025-01-01"):
                    should_regenerate = True
            else:
                should_regenerate = True
        except:
            should_regenerate = True

    if should_regenerate:
        with st.spinner("Upgrading history data to 2-year range (Auto)..."):
            res = get_live_prediction()
            if 'history' in res:
                res['history'].to_csv('prediction_history.csv')
                st.success("History upgraded to long-term data!")
                st.rerun()
            else:
                st.error("Could not generate history.")
    
    if should_regenerate:
        with st.spinner("Upgrading history data to 2-year range (Auto)..."):
            res = get_live_prediction()
            if 'history' in res:
                res['history'].to_csv('prediction_history.csv')
                st.success("History upgraded to long-term data!")
                st.rerun()
            else:
                st.error("Could not generate history.")
    
    # (Reset Button Removed as requested)

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
            # Strategy Return
            cum_ret = (np.exp(filtered_df['Strategy_Return'].cumsum()) - 1) * 100
            period_return = cum_ret.iloc[-1]
            
            # Buy & Hold Return (End Price / Start Price - 1)
            # filtered_df is DOscending (Newest first)
            start_price = filtered_df['Price'].iloc[-1]
            end_price = filtered_df['Price'].iloc[0]
            bh_return = (end_price / start_price - 1) * 100
        else:
            period_return = 0
            bh_return = 0
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Trading Days", f"{total_days} Days")
        col2.metric("Model Return", f"{period_return:.2f}%", delta=f"{period_return - bh_return:.2f}% vs B&H")
        col3.metric("Buy & Hold Return", f"{bh_return:.2f}%")
        
        # Chart
        st.subheader("Equity Curve (Selected Period)")
        if not filtered_df.empty:
            # Re-sort for Chart (Chronological)
            chart_df = filtered_df.sort_index(ascending=True)
            equity = (np.exp(chart_df['Strategy_Return'].cumsum()) - 1) * 100
            
            # Add B&H Equity for comparison
            # Normalized B&H: (Price / StartPrice - 1) * 100
            start_p = chart_df['Price'].iloc[0]
            bh_equity = (chart_df['Price'] / start_p - 1) * 100
            
            comp_chart = pd.DataFrame({
                'Model Strat': equity,
                'Buy & Hold': bh_equity
            })
            st.line_chart(comp_chart)
        else:
            st.info("No data for selected period.")
        
        # Table
        st.subheader("Daily Logs")
        
        # formatting (Convert Log Return -> Simple Return for Display)
        display_df = filtered_df.copy()
        
        # Add Hit/Miss Logic
        # Note: Log_Return and Strategy_Return are still floats here
        def evaluate_result(row):
            sig = row['Signal']
            ret = row['Log_Return']
            if sig == 'BUY':
                return "✅ Win" if ret > 0 else "❌ Loss"
            else: # HOLD
                return "🛡️ Saved" if ret < 0 else "⚠️ Missed"

        display_df['Match_Result'] = display_df.apply(evaluate_result, axis=1)
        
        display_df['Log_Return'] = display_df['Log_Return'].apply(lambda x: f"{(np.exp(x)-1)*100:.2f}%")
        display_df['Strategy_Return'] = display_df['Strategy_Return'].apply(lambda x: f"{(np.exp(x)-1)*100:.2f}%")
        display_df['Prob_Up'] = display_df['Prob_Up'].apply(lambda x: f"{x:.2f}%")
        display_df['Price'] = display_df['Price'].apply(lambda x: f"{x:,.0f}")
        
        # Rename columns for clarity
        display_df = display_df.rename(columns={
            'Price': 'Close Price',
            'Log_Return': 'Daily Return',
            'Strategy_Return': 'Strat Return',
            'Match_Result': 'Hit/Miss'
        })
        
        # Reset index to make Date a column and format it
        display_df = display_df.reset_index()
        display_df['Date'] = display_df['Date'].dt.date
        
        # Select and Reorder columns
        cols_to_show = ['Date', 'Hit/Miss', 'Signal', 'Prob_Up', 'Close Price', 'Daily Return', 'Strat Return']
        
        st.dataframe(display_df[cols_to_show], use_container_width=True, height=500, hide_index=True)
        
        st.download_button("Download CSV", hist_df.to_csv(), "prediction_history.csv", "text/csv")

# --- Footer ---
st.markdown("---")
st.caption("Automated Trading System v1.1 | Powered by XGBoost")
