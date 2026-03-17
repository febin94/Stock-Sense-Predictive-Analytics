"""
Nifty 50 Model Training Script
Trains models for:
  - Daily close price prediction  -> saved to ./model/best/best_model.pkl
  - 1-minute close price prediction -> saved to ./1min-model/best/best_model.pkl
"""

import os
import numpy as np
import pandas as pd
import yfinance as yf
import joblib
from datetime import datetime

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# ============================================================
# Feature engineering (must match app.py exactly)
# ============================================================

def engineer_features_normal(data):
    df = data.copy()
    df['Prev Close'] = df['Close'].shift(1)
    df['Prev Close 2'] = df['Close'].shift(2)
    df['Prev Close 3'] = df['Close'].shift(3)
    df['SMA_5'] = df['Close'].rolling(window=5).mean()
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['EMA_5'] = df['Close'].ewm(span=5, adjust=False).mean()
    df['EMA_10'] = df['Close'].ewm(span=10, adjust=False).mean()
    df['20 Day MA'] = df['Close'].rolling(window=20).mean()
    df['20 Day STD'] = df['Close'].rolling(window=20).std()
    df['Upper Band'] = df['20 Day MA'] + (df['20 Day STD'] * 2)
    df['Lower Band'] = df['20 Day MA'] - (df['20 Day STD'] * 2)
    df['Daily Return'] = (df['Close'] - df['Open']) / df['Open']
    df['Volatility_5'] = df['Daily Return'].rolling(window=5).std()

    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df['RSI'] = 100 - (100 / (1 + rs))

    df['Day of Week'] = df.index.dayofweek
    df['Month'] = df.index.month
    df.dropna(inplace=True)

    features = ['Prev Close', 'Prev Close 2', 'Prev Close 3', 'SMA_5', 'SMA_10',
                'EMA_5', 'EMA_10', 'Upper Band', 'Lower Band', 'Daily Return',
                'Volatility_5', 'RSI', 'Day of Week', 'Month']
    return df, features


def engineer_features_minute(data):
    df = data.copy()
    df['Prev Close'] = df['Close'].shift(1)
    df['SMA_5'] = df['Close'].rolling(window=5).mean()
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['Volatility_5'] = df['Close'].diff().rolling(window=5).std()

    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df['RSI'] = 100 - (100 / (1 + rs))

    df['Minute'] = df.index.minute
    df.dropna(inplace=True)

    features = ['Prev Close', 'SMA_5', 'SMA_10', 'Volatility_5', 'RSI', 'Minute']
    return df, features


# ============================================================
# Model training helper
# ============================================================

def train_best_model(X, y, label="model"):
    """
    Trains Ridge, RandomForest, and GradientBoosting.
    Returns the Pipeline with the lowest validation RMSE.
    """
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)

    candidates = {
        "Ridge": Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0))
        ]),
        "RandomForest": Pipeline([
            ("scaler", StandardScaler()),
            ("model", RandomForestRegressor(n_estimators=200, max_depth=8,
                                            min_samples_leaf=5, random_state=42, n_jobs=-1))
        ]),
        "GradientBoosting": Pipeline([
            ("scaler", StandardScaler()),
            ("model", GradientBoostingRegressor(n_estimators=200, max_depth=4,
                                                learning_rate=0.05, subsample=0.8,
                                                random_state=42))
        ]),
    }

    best_name = None
    best_rmse = float("inf")
    best_pipeline = None

    for name, pipeline in candidates.items():
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, preds))
        print(f"  [{label}] {name:20s}  val RMSE = {rmse:,.2f}")
        if rmse < best_rmse:
            best_rmse = rmse
            best_name = name
            best_pipeline = pipeline

    print(f"  [{label}] Best model: {best_name}  (RMSE = {best_rmse:,.2f})\n")
    return best_pipeline


# ============================================================
# 1. Daily model
# ============================================================
def flatten_columns(df):
    """Flatten MultiIndex columns that newer yfinance versions return."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
    return df

print("=" * 55)
print("Downloading daily Nifty 50 data (2 years)...")
end = pd.Timestamp.now().normalize()
start = end - pd.Timedelta(days=730)
daily_raw = yf.download("^NSEI", start=start, end=end, progress=False)
daily_raw = flatten_columns(daily_raw)
daily_raw.index = pd.to_datetime(daily_raw.index)
print(f"  Rows fetched: {len(daily_raw)}")

daily_df, daily_features = engineer_features_normal(daily_raw)

# Target: next day's closing price
daily_df['Target'] = daily_df['Close'].shift(-1)
daily_df.dropna(inplace=True)

X_daily = daily_df[daily_features].values
y_daily = daily_df['Target'].values.ravel()

print(f"  Training samples: {len(X_daily)}")
print("Training daily models...")
daily_model = train_best_model(X_daily, y_daily, label="daily")

os.makedirs("model/best", exist_ok=True)
joblib.dump(daily_model, "model/best/best_model.pkl")
print("  Saved -> model/best/best_model.pkl")

# ============================================================
# 2. One-minute model
# ============================================================
print("=" * 55)
print("Downloading 1-minute Nifty 50 data (last 7 days)...")
try:
    minute_raw = yf.download("^NSEI",
                             start=end - pd.Timedelta(days=7),
                             end=end, interval='1m', progress=False)
    minute_raw = flatten_columns(minute_raw)
    minute_raw.index = pd.to_datetime(minute_raw.index)
    if len(minute_raw) < 50:
        raise ValueError("Not enough minute data rows.")
    print(f"  Rows fetched: {len(minute_raw)}")
except Exception as e:
    print(f"  ⚠️ Minute data unavailable ({e}). Falling back to daily data for 1-min model.")
    minute_raw = daily_raw.copy()

minute_df, minute_features = engineer_features_minute(minute_raw)

# Target: next bar's closing price
minute_df['Target'] = minute_df['Close'].shift(-1)
minute_df.dropna(inplace=True)

X_minute = minute_df[minute_features].values
y_minute = minute_df['Target'].values.ravel()

print(f"  Training samples: {len(X_minute)}")
print("Training 1-minute models...")
minute_model = train_best_model(X_minute, y_minute, label="1min")

os.makedirs("1min-model/best", exist_ok=True)
joblib.dump(minute_model, "1min-model/best/best_model.pkl")
print("  Saved -> 1min-model/best/best_model.pkl")

print("=" * 55)
print("✅ Training complete! Both models are ready.")
print(f"   Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
