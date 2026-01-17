import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import xgboost as xgb
import numpy as np
import datetime
import os

# --- Page Config ---
st.set_page_config(
    page_title="Samsung AI Predictor - Backward Elimination",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Model Configuration ---
OPTIMAL_FEATURES = [
    'Day_of_Week', 'Dist_to_MA5', 'Intraday_Range',
    'Log_Return_Lag1', 'Log_Return_Lag2', 'Log_Return_Lag3',
    'RSI_14', 'RSI_Death_Cross', 'RSI_Golden_Cross',
    'Simple_Return_Lag1', 'Simple_Return_Lag2', 'Simple_Return_Lag3',
    'Stochastic_D', 'Stochastic_K'
]

OPTIMAL_THRESHOLD = 0.47

# Get the directory where app.py is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'optimized_model.json')
DATA_PATH = os.path.join(BASE_DIR, 'full_features_811.csv')

# --- Custom CSS ---
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 100%);
    }
    .hero-card {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
        border-radius: 20px;
        padding: 40px;
        margin-bottom: 30px;
        box-shadow: 0 20px 60px rgba(0,0,0,0.4);
    }
    .signal-box {
        background: rgba(255,255,255,0.05);
        border: 2px solid rgba(255,255,255,0.1);
        border-radius: 15px;
        padding: 30px;
        text-align: center;
        backdrop-filter: blur(10px);
    }
    .metric-card {
        background: rgba(30,37,48,0.9);
        border-radius: 12px;
        padding: 20px;
        border-left: 4px solid #3b82f6;
    }
    .stat-number {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #60a5fa 0%, #3b82f6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load optimized XGBoost model"""
    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ Model file not found: {MODEL_PATH}")
        return None
    model = xgb.XGBClassifier()
    model.load_model(MODEL_PATH)
    return model

@st.cache_data(ttl=3600)
def load_latest_data():
    """Load latest market data"""
    try:
        data = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
        return data
    except Exception as e:
        st.error(f"❌ Data load error: {e}")
        return None

def get_prediction(model, latest_data):
    """Get prediction from model"""
    try:
        # Get latest row with required features
        latest = latest_data.iloc[-1]
        X = latest[OPTIMAL_FEATURES].values.reshape(1, -1)
        
        # Predict
        prob = model.predict_proba(X)[0, 1]
        signal = "📈 BUY" if prob > OPTIMAL_THRESHOLD else "📉 HOLD"
        confidence = prob * 100
        
        return {
            'signal': signal,
            'probability': prob,
            'confidence': confidence,
            'date': latest_data.index[-1],
            'price': latest.get('Price', 0)
        }
    except Exception as e:
        st.error(f"❌ Prediction error: {e}")
        return None

# --- Main App ---
st.title("🎯 Samsung Electronics AI Predictor")
st.caption("Powered by Backward Elimination (14 Features, Sharpe 8.39)")

# Load model and data
model = load_model()
data = load_latest_data()

if model and data is not None:
    # Get prediction
    pred = get_prediction(model, data)
    
    if pred:
        # Hero Section
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("""
            <div class="signal-box">
                <h1 style="font-size: 4rem; margin: 0;">{}</h1>
                <p style="font-size: 1.2rem; opacity: 0.8; margin-top: 10px;">Current Signal</p>
                <p style="font-size: 0.9rem; opacity: 0.6;">as of {}</p>
            </div>
            """.format(pred['signal'], pred['date'].strftime('%Y-%m-%d')), unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="signal-box">
                <h2>Confidence</h2>
                <div class="stat-number">{pred['confidence']:.1f}%</div>
                <p style="opacity: 0.7; margin-top: 10px;">Probability: {pred['probability']:.3f}</p>
                <p style="opacity: 0.6;">Threshold: {OPTIMAL_THRESHOLD}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Progress Bar
        st.markdown("### Signal Strength")
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=pred['probability'],
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Probability", 'font': {'size': 24, 'color': 'white'}},
            delta={'reference': OPTIMAL_THRESHOLD, 'increasing': {'color': "green"}},
            gauge={
                'axis': {'range': [None, 1], 'tickwidth': 1, 'tickcolor': "white"},
                'bar': {'color': "#3b82f6"},
                'bgcolor': "rgba(255,255,255,0.1)",
                'borderwidth': 2,
                'bordercolor': "white",
                'steps': [
                    {'range': [0, OPTIMAL_THRESHOLD], 'color': 'rgba(255,99,132,0.3)'},
                    {'range': [OPTIMAL_THRESHOLD, 1], 'color': 'rgba(75,192,192,0.3)'}
                ],
                'threshold': {
                    'line': {'color': "white", 'width': 4},
                    'thickness': 0.75,
                    'value': OPTIMAL_THRESHOLD
                }
            }
        ))
        fig_gauge.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font={'color': "white", 'family': "Arial"},
            height=300
        )
        st.plotly_chart(fig_gauge, use_container_width=True)
        
        # Model Performance Stats
        st.markdown("---")
        st.markdown("### 📊 Model Performance (Cross-Validation)")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div class="metric-card">
                <h4>Average Sharpe</h4>
                <div class="stat-number">8.39</div>
                <p style="opacity: 0.7; font-size: 0.9rem;">Across 4 periods</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-card">
                <h4>Average Win Rate</h4>
                <div class="stat-number">91.3%</div>
                <p style="opacity: 0.7; font-size: 0.9rem;">2018-2025</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="metric-card">
                <h4>Avg Return</h4>
                <div class="stat-number">3,341%</div>
                <p style="opacity: 0.7; font-size: 0.9rem;">Per period</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown("""
            <div class="metric-card">
                <h4>Features</h4>
                <div class="stat-number">14</div>
                <p style="opacity: 0.7; font-size: 0.9rem;">From 50 candidates</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Cross-Validation Table
        st.markdown("### 🔬 Cross-Validation Results")
        cv_results = pd.DataFrame({
            'Period': ['2018-2020', '2020-2022', '2022-2024', '2024-2025', 'AVERAGE'],
            'Market': ['Recovery', 'Pandemic', 'Rate Hikes', 'AI Boom', '-'],
            'Sharpe': [8.23, 7.30, 8.12, 9.89, 8.39],
            'Win Rate': ['89.61%', '89.47%', '93.13%', '93.14%', '91.34%'],
            'Return': ['6,643%', '3,094%', '1,635%', '1,992%', '3,341%'],
            'Alpha': ['6,585%p', '3,094%p', '1,631%p', '1,956%p', '3,317%p']
        })
        
        # Style the last row
        def highlight_avg(row):
            if row ['Period'] == 'AVERAGE':
                return ['background-color: #1e3a8a; font-weight: bold'] * len(row)
            return [''] * len(row)
        
        st.dataframe(
            cv_results.style.apply(highlight_avg, axis=1),
            use_container_width=True,
            hide_index=True
        )

# Sidebar
with st.sidebar:
    st.markdown("## ⚙️ Model Info")
    st.markdown(f"""
    **Method**: Backward Elimination  
    **Features**: 14 (from 50)  
    **Threshold**: {OPTIMAL_THRESHOLD}  
    **Training**: 2011-2024  
    """)
    
    st.markdown("---")
    st.markdown("### 🎯 Selected Features")
    
    feature_categories = {
        "📈 Price Lags (6)": [
            "Log_Return_Lag1/2/3",
            "Simple_Return_Lag1/2/3"
        ],
        "📊 RSI (3)": [
            "RSI_14",
            "RSI_Golden/Death_Cross"
        ],
        "📉 Stochastic (2)": [
            "Stochastic_K",
            "Stochastic_D"
        ],
        "🔧 Others (3)": [
            "Dist_to_MA5",
            "Intraday_Range",
            "Day_of_Week"
        ]
    }
    
    for category, features in feature_categories.items():
        with st.expander(category):
            for feat in features:
                st.markdown(f"• {feat}")
    
    st.markdown("---")
    st.markdown("### ❌ Rejected Features")
    st.markdown("""
    **Eliminated as noise:**
    - All macro indicators (S&P500, SOX, US10Y, USD/KRW)
    - MACD, Bollinger Bands
    - Volume indicators
    - Golden/Dead Cross signals
    """)
    
    st.markdown("---")
    st.caption("Built with Streamlit • XGBoost")
