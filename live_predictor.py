import yfinance as yf
import pandas as pd
import numpy as np
import xgboost as xgb
import os
import datetime
import warnings
warnings.filterwarnings('ignore')

base_dir = os.path.dirname(os.path.abspath(__file__))

# Helper Functions
def calculate_rsi_series(prices, period=14):
    delta = prices.diff()
    gain = delta.where(delta > 0, 0)
    loss = (-delta).where(delta < 0, 0)
    avg_gain = gain.ewm(span=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def calculate_atr(high, low, close, period=14):
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = abs(high - prev_close)
    tr3 = abs(low - prev_close)
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return true_range.ewm(span=period, adjust=False).mean()

def get_live_prediction():
    print("Fetching live data from yfinance...")
    
    # 200 days history to ensure rolling windows are populated
    start_date = datetime.datetime.now() - datetime.timedelta(days=300)
    
    tickers = {
        'Samsung': '005930.KS',
        'SP500': '^GSPC',
        'US10Y': '^TNX'
    }
    
    dfs = {}
    
    for name, ticker in tickers.items():
        try:
            # period='2y' to cover requested history from 2024.06
            df = yf.download(ticker, period='2y', progress=False)
            
            # Handle MultiIndex columns (yfinance v0.2+)
            if isinstance(df.columns, pd.MultiIndex):
                # Columns are likely (Price Type, Ticker) e.g. ('Close', '005930.KS')
                # We want to drop the Ticker level to get ['Close', 'High', 'Low'...]
                df.columns = df.columns.droplevel(1)
            
            # Map columns
            rename_map = {
                'Close': 'Price',
                'Volume': 'Vol.',
                'Adj Close': 'Adj Close'
            }
            df = df.rename(columns=rename_map)
            
            # Verify required columns exist
            required = ['Price', 'High', 'Low', 'Vol.']
            missing = [c for c in required if c not in df.columns]
            if missing:
                 # Fallback: maybe columns are already correct or different case?
                 # Try capitalized
                 df.columns = [c.capitalize() if c != 'Vol.' else 'Vol.' for c in df.columns]
                 # Re-check
                 missing = [c for c in required if c not in df.columns]
                 if missing: 
                     return {"error": f"Missing columns for {name}: {missing}. Found: {list(df.columns)}"}
            
            # Ensure index is datetime and normalized (no time)
            df.index = pd.to_datetime(df.index).normalize()
            dfs[name] = df
            
        except Exception as e:
            return {"error": f"Failed to fetch {name}: {str(e)}"}

    df_sam = dfs['Samsung']
    df_sp500 = dfs['SP500']
    df_us10y = dfs['US10Y']
    
    # === Feature Engineering (Exact Replica) ===
    
    # 1. Samsung Features
    df_sam['Log_Return'] = np.log(df_sam['Price'] / df_sam['Price'].shift(1))
    df_sam['Direction'] = (df_sam['Log_Return'] > 0).astype(int)
    df_sam['RSI'] = calculate_rsi_series(df_sam['Price'], period=14)
    df_sam['RSI_Golden_Cross'] = ((df_sam['RSI'].shift(1) < 30) & (df_sam['RSI'] >= 30)).astype(int)
    
    # ATR & Momentum
    df_sam['ATR'] = calculate_atr(df_sam['High'], df_sam['Low'], df_sam['Price'], period=14)
    df_sam['ATR_Pct'] = df_sam['ATR'] / df_sam['Price']
    df_sam['Momentum_5D'] = df_sam['Log_Return'].rolling(window=5).sum()
    df_sam['Risk_Adj_Mom'] = df_sam['Momentum_5D'] / df_sam['ATR_Pct'].replace(0, np.nan)
    df_sam['Risk_Adj_Mom'] = df_sam['Risk_Adj_Mom'].fillna(0).replace([np.inf, -np.inf], 0)
    
    # Volume
    df_sam['Vol_Change'] = np.log(df_sam['Vol.'] / df_sam['Vol.'].shift(1)).fillna(0) # Vol. is already float in yfinance
    
    # 2. External Features
    df_sp500['LogReturn_SP500'] = np.log(df_sp500['Price'] / df_sp500['Price'].shift(1))
    df_us10y['LogReturn_US10Y'] = np.log(df_us10y['Price'] / df_us10y['Price'].shift(1))
    
    # 3. Merge (Inner Join on Date)
    data = df_sam[['Price', 'Momentum_5D', 'Log_Return', 'Direction', 'Vol_Change', 'Risk_Adj_Mom', 'RSI_Golden_Cross']].join(df_sp500[['LogReturn_SP500']], how='inner').join(df_us10y[['LogReturn_US10Y']], how='inner')
    
    # 4. create Lags
    for i in range(1, 6): data[f'Log_Return_Lag{i}'] = data['Log_Return'].shift(i)
    data['Vol_Change_Lag1'] = data['Vol_Change'].shift(1)
    data['LogReturn_US10Y_Lag1'] = data['LogReturn_US10Y'].shift(1)
    data['LogReturn_SP500_Lag1'] = data['LogReturn_SP500'].shift(1)
    data['RSI_Golden_Cross_Lag1'] = data['RSI_Golden_Cross'].shift(1)
    data['Risk_Adj_Mom_Lag1'] = data['Risk_Adj_Mom'].shift(1)
    
    data_clean = data.dropna()
    
    if data_clean.empty:
        return {"error": "Not enough overlapping data after processing."}
        
    last_row = data_clean.iloc[-1:]
    last_date = last_row.index[0]
    
    # 5. Load Model
    features = ['Log_Return_Lag1', 'Log_Return_Lag2', 'Log_Return_Lag3', 'Log_Return_Lag4', 'Log_Return_Lag5', 
                'Vol_Change_Lag1', 'LogReturn_US10Y_Lag1', 'LogReturn_SP500_Lag1', 
                'RSI_Golden_Cross_Lag1', 'Risk_Adj_Mom_Lag1']
    
    model_path = os.path.join(base_dir, 'final_model.json')
    if not os.path.exists(model_path):
        return {"error": "Model file not found. Please run save_final_model.py first."}
        
    model = xgb.XGBClassifier()
    model.load_model(model_path)
    
    # --- Live Prediction (Future Forecast) ---
    # To predict for T+1 (Tomorrow), we need features based on Data(T) (Today).
    # Current 'last_row' is Day T. Its columns (Price, Vol_Change, etc.) are today's values.
    # Its 'Lag1' columns are yesterday's values.
    # We must construct a new input row where 'Lag1' features take Today's values.
    
    future_input = pd.DataFrame(index=[last_row.index[0] + datetime.timedelta(days=1)])
    
    # 1. Shift Standard Lag1 Features
    # Feature_Lag1 (Input) <--- Column (Today)
    future_input['Log_Return_Lag1'] = last_row['Log_Return'].values
    future_input['Vol_Change_Lag1'] = last_row['Vol_Change'].values
    future_input['LogReturn_US10Y_Lag1'] = last_row['LogReturn_US10Y'].values
    future_input['LogReturn_SP500_Lag1'] = last_row['LogReturn_SP500'].values
    future_input['RSI_Golden_Cross_Lag1'] = last_row['RSI_Golden_Cross'].values
    future_input['Risk_Adj_Mom_Lag1'] = last_row['Risk_Adj_Mom'].values
    
    # 2. Shift Deep Lags
    # Log_Return_Lag{k} (Input) <--- Log_Return_Lag{k-1} (Today)
    future_input['Log_Return_Lag2'] = last_row['Log_Return_Lag1'].values
    future_input['Log_Return_Lag3'] = last_row['Log_Return_Lag2'].values
    future_input['Log_Return_Lag4'] = last_row['Log_Return_Lag3'].values
    future_input['Log_Return_Lag5'] = last_row['Log_Return_Lag4'].values
    
    # Predict using the constructed Future Input
    prob_up = model.predict_proba(future_input[features])[0, 1]
    
    threshold = 0.61
    signal = "BUY" if prob_up > threshold else "HOLD"
    
    # For display, we show the features that contributed to THIS prediction.
    # i.e., the values from 'future_input'
    display_features = future_input[features].to_dict('records')[0]
    display_features['Momentum_5D'] = last_row['Momentum_5D'].item() # Keep purely informational
    
    live_result = {
        "status": "success",
        "date_used": f"{last_date.strftime('%Y-%m-%d')} (Data) ▶ Forecast",
        "prob_up": round(prob_up * 100, 2),
        "signal": signal,
        "features": display_features
    }
    
    # --- Historical Batch Prediction (Optional) ---
    # To check history from 2024-06-01
    hist_start = '2024-06-01'
    # Filter data from this date
    # data_clean index is Date.
    hist_data = data_clean[data_clean.index >= hist_start]
    
    if not hist_data.empty:
        probs = model.predict_proba(hist_data[features])[:, 1]
        hist_df = hist_data.copy()
        hist_df['Prob_Up'] = probs * 100
        hist_df['Signal'] = hist_df['Prob_Up'].apply(lambda x: "BUY" if x > 61 else "HOLD")
        
        # Calculate fictional return if we followed the signal
        # Signal at T (calculated using Lag data from T-1) -> Trade at T Open/Close -> Return at T
        # Actually our model uses Lag1, so at Day T, we use Data(T-1) to predict Return(T).
        # y_train was Direction(T). features were Lag1(T-1).
        # So predict_proba(X[T]) predicts Direction(T).
        # If 'Signal' is BUY, we get 'Log_Return'(T).
        
        hist_df['Strategy_Return'] = np.where(hist_df['Signal'] == 'BUY', hist_df['Log_Return'], 0)
        hist_df['Cum_Return'] = hist_df['Strategy_Return'].cumsum()
        
        live_result['history'] = hist_df[['Prob_Up', 'Signal', 'Price', 'Log_Return', 'Strategy_Return']]
    
    return live_result

if __name__ == "__main__":
    result = get_live_prediction()
    if 'history' in result:
        print(f"History rows: {len(result['history'])}")
    print(result['date_used'], result['signal'])
