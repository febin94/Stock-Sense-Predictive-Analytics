import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
import pandas as pd
import yfinance as yf
import plotly.express as px
import numpy as np
import joblib
import plotly.graph_objs as go
from datetime import timedelta
import os

# ============================================================
# 1. Load Pre-trained Models (with fallback if missing)
# ============================================================
def load_model_or_dummy(path):
    if os.path.exists(path):
        return joblib.load(path)
    else:
        print(f"⚠️ Model not found at {path}. Using dummy predictor.")
        return lambda x: x.iloc[-1]['Prev Close'] if 'Prev Close' in x.columns else 0

normal_model_path = './model/best/best_model.pkl'
one_minute_model_path = './1min-model/best/best_model.pkl'

normal_model = load_model_or_dummy(normal_model_path)
one_minute_model = load_model_or_dummy(one_minute_model_path)

# ============================================================
# 2. Fetch and Prepare Data
# ============================================================
ticker = "^NSEI"

def flatten_columns(df):
    """Flatten MultiIndex columns from yfinance into simple column names."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
    return df

end_date = pd.Timestamp.now().normalize()
start_date = end_date - pd.Timedelta(days=365)
nifty_data_normal = yf.download(ticker, start=start_date, end=end_date, progress=False)
nifty_data_normal = flatten_columns(nifty_data_normal)
nifty_data_normal.index = pd.to_datetime(nifty_data_normal.index)

try:
    nifty_data_minute = yf.download(ticker, start=end_date - pd.Timedelta(days=7), end=end_date, interval='1m', progress=False)
    nifty_data_minute = flatten_columns(nifty_data_minute)
    nifty_data_minute.index = pd.to_datetime(nifty_data_minute.index)
except Exception as e:
    print(f"⚠️ Could not fetch minute data: {e}. Using daily data for demo.")
    nifty_data_minute = nifty_data_normal.iloc[-100:].copy()

# ============================================================
# 3. Feature Engineering Functions
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

nifty_data_normal, features_normal = engineer_features_normal(nifty_data_normal)
nifty_data_minute, features_minute = engineer_features_minute(nifty_data_minute)

# ============================================================
# 4. Dash App Initialization
# ============================================================
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Nifty 50 Analysis Dashboard"

# ============================================================
# 5. Page Layout Functions
# ============================================================
def prediction_layout(model, data, feature_columns):
    latest = data.iloc[-1:]
    X_input = latest[feature_columns]
    prediction = float(model.predict(X_input)[0]) if hasattr(model, 'predict') else float(model(X_input))

    last_price = float(data["Close"].iloc[-1])
    recent = data[-100:].copy()

    recent['SMA_5'] = recent['Close'].rolling(5).mean()
    recent['SMA_10'] = recent['Close'].rolling(10).mean()
    recent['EMA_5'] = recent['Close'].ewm(span=5, adjust=False).mean()
    recent['EMA_10'] = recent['Close'].ewm(span=10, adjust=False).mean()
    recent['20 Day MA'] = recent['Close'].rolling(20).mean()
    recent['20 Day STD'] = recent['Close'].rolling(20).std()
    recent['Upper Band'] = recent['20 Day MA'] + 2 * recent['20 Day STD']
    recent['Lower Band'] = recent['20 Day MA'] - 2 * recent['20 Day STD']

    delta = recent['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    recent['RSI'] = 100 - 100 / (1 + rs)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=recent.index, y=recent['Close'], mode='lines', name='Close Price', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=recent.index, y=recent['SMA_5'], mode='lines', name='SMA 5', line=dict(color='orange', dash='dash')))
    fig.add_trace(go.Scatter(x=recent.index, y=recent['SMA_10'], mode='lines', name='SMA 10', line=dict(color='green', dash='dash')))
    fig.add_trace(go.Scatter(x=recent.index, y=recent['EMA_5'], mode='lines', name='EMA 5', line=dict(color='purple', dash='dot')))
    fig.add_trace(go.Scatter(x=recent.index, y=recent['EMA_10'], mode='lines', name='EMA 10', line=dict(color='red', dash='dot')))
    fig.add_trace(go.Scatter(x=recent.index, y=recent['Upper Band'], mode='lines', name='Upper Band', line=dict(color='grey', dash='dot')))
    fig.add_trace(go.Scatter(x=recent.index, y=recent['Lower Band'], mode='lines', name='Lower Band', line=dict(color='grey', dash='dot')))

    future_time = recent.index[-1] + timedelta(minutes=1)
    fig.add_trace(go.Scatter(x=[future_time], y=[prediction], mode='markers', name='Predicted Price',
                             marker=dict(color='red', size=10, symbol='star')))

    fig.update_layout(title='Nifty 50 Price with Indicators and Prediction', xaxis_title='Date', yaxis_title='Price')

    fig_rsi = go.Figure()
    fig_rsi.add_trace(go.Scatter(x=recent.index, y=recent['RSI'], mode='lines', name='RSI', line=dict(color='magenta')))
    fig_rsi.update_layout(title='RSI Indicator', yaxis=dict(title='RSI'))

    return dbc.Container([
        html.H3("Prediction", className="text-center text-primary"),
        dbc.Row([
            dbc.Col(dbc.Card([
                dbc.CardBody([
                    html.H4("Current Price", className="card-title text-white"),
                    html.P(f"{last_price:.2f}", className="card-text text-white"),
                ])
            ], color="primary", inverse=True), width=6),
            dbc.Col(dbc.Card([
                dbc.CardBody([
                    html.H4("Predicted Price", className="card-title text-white"),
                    html.P(f"{prediction:.2f}", className="card-text text-white"),
                ])
            ], color="success", inverse=True), width=6),
        ], className="mb-4"),
        dcc.Graph(id="price_chart", figure=fig),
        dcc.Graph(id="rsi_chart", figure=fig_rsi),
    ], fluid=True)

def eda_layout(data):
    line_fig = px.line(data, x=data.index, y="Close", title="Nifty 50 Closing Prices")
    hist_fig = px.histogram(data, x="Daily Return", nbins=50, title="Histogram of Daily Returns")
    box_fig = px.box(data, y="Close", title="Box Plot of Closing Prices")
    vol_fig = px.line(data, x=data.index, y="Volatility_5", title="5-Day Rolling Volatility")

    return dbc.Container([
        html.H3("Exploratory Data Analysis", className="text-center text-primary"),
        dbc.Row([
            dbc.Col(dcc.Graph(figure=line_fig), md=6),
            dbc.Col(dcc.Graph(figure=hist_fig), md=6),
        ]),
        dbc.Row([
            dbc.Col(dcc.Graph(figure=box_fig), md=6),
            dbc.Col(dcc.Graph(figure=vol_fig), md=6),
        ]),
    ], fluid=True)

def heatmap_layout(data):
    corr_matrix = data[features_normal + ["Close"]].corr()
    fig = px.imshow(corr_matrix, text_auto=True, title="Correlation Heatmap",
                    color_continuous_scale="RdBu", zmin=-1, zmax=1)
    fig.update_layout(autosize=True, width=900, height=700)
    return dbc.Container([
        html.H3("Correlation Heatmap", className="text-center text-primary"),
        dcc.Graph(figure=fig),
    ], fluid=True)

# ============================================================
# 6. Routing Callback
# ============================================================
app.layout = dbc.Container([
    dbc.NavbarSimple(
        brand="Nifty 50 Dashboard",
        color="black",
        dark=True,
        children=[
            dbc.NavItem(dbc.NavLink("Day Prediction", href="/")),
            dbc.NavItem(dbc.NavLink("One-Minute Prediction", href="/one-min-prediction")),
            dbc.NavItem(dbc.NavLink("EDA", href="/eda")),
            dbc.NavItem(dbc.NavLink("Heatmap", href="/heatmap")),
        ]
    ),
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content', style={'backgroundColor': 'white', 'padding': '20px'})
], fluid=True)

@app.callback(
    dash.dependencies.Output('page-content', 'children'),
    [dash.dependencies.Input('url', 'pathname')]
)
def display_page(pathname):
    if pathname == '/':
        return prediction_layout(normal_model, nifty_data_normal, features_normal)
    elif pathname == '/one-min-prediction':
        return prediction_layout(one_minute_model, nifty_data_minute, features_minute)
    elif pathname == '/eda':
        return eda_layout(nifty_data_normal)
    elif pathname == '/heatmap':
        return heatmap_layout(nifty_data_normal)
    else:
        return dbc.Container([html.H3("404: Page not found", className="text-center text-danger")], fluid=True)

# ============================================================
# 7. Run the App
# ============================================================
if __name__ == "__main__":
    port = int(os.environ.get("DASH_PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
