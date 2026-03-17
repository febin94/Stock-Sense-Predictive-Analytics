# Nifty 50 Analysis Dashboard

An interactive financial analytics dashboard built with Python Dash that fetches live Nifty 50 data, applies technical indicators, trains ML models, and visualises predictions — all in one place.

## Features

- **Day Prediction** — Predicts the next day's Nifty 50 closing price using a trained regression model
- **1-Minute Prediction** — Predicts the next minute's closing price using a model trained on intraday data
- **EDA** — Exploratory data analysis: price history, return distribution, box plot, rolling volatility
- **Correlation Heatmap** — Visualises relationships between all engineered features

## Technical Indicators Used

| Indicator | Description |
|---|---|
| SMA 5 / SMA 10 | Simple moving averages |
| EMA 5 / EMA 10 | Exponential moving averages |
| Bollinger Bands | 20-day MA ± 2 standard deviations |
| RSI (14) | Relative Strength Index |
| Daily Return | (Close − Open) / Open |
| Volatility (5-day) | Rolling standard deviation of returns |

## ML Models

Three models are benchmarked on a holdout validation set and the best is saved automatically:

- Ridge Regression
- Random Forest Regressor
- Gradient Boosting Regressor

Evaluation metric: **RMSE** on the last 20% of data (time-ordered split).

## Project Structure

```
├── app.py              # Dash dashboard (run this to launch the app)
├── train_models.py     # Model training script
├── requirements.txt    # Python dependencies
├── model/
│   └── best/
│       └── best_model.pkl      # Trained daily model
└── 1min-model/
    └── best/
        └── best_model.pkl      # Trained 1-minute model
```

## Getting Started

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the models

```bash
python train_models.py
```

This downloads 2 years of daily data and the last 7 days of 1-minute data from Yahoo Finance, trains all three model types, and saves the best performer for each task.

### 3. Run the dashboard

```bash
python app.py
```

Open your browser at `http://localhost:5000`

## Data Source

Live market data is fetched from **Yahoo Finance** via the `yfinance` library. Ticker used: `^NSEI` (Nifty 50 index).

## Requirements

- Python 3.9+
- See `requirements.txt` for full dependency list

## License

MIT
