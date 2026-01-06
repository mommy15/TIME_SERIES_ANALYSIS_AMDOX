import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings



from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

warnings.filterwarnings("ignore")
# -------------------- PAGE CONFIG --------------------
st.set_page_config(page_title="Crypto Time Series Analysis", layout="wide")
st.title("Cryptocurrency Time Series Analysis & Forecasting")

# -------------------- DATA COLLECTION --------------------
st.header("1. Cryptocurrency Data Collection")

st.markdown("""
This section allows users to select a cryptocurrency and a custom historical 
date range for analysis. The system dynamically fetches historical price data 
from Yahoo Finance based on the selected inputs.
""")

# -------------------- USER INPUTS --------------------
crypto_map = {
    "Bitcoin (BTC)": "BTC-USD",
    "Ethereum (ETH)": "ETH-USD",
    "Solana (SOL)": "SOL-USD",
    "Cardano (ADA)": "ADA-USD",
    "Ripple (XRP)": "XRP-USD"
}

crypto_name = st.selectbox(
    "Select Cryptocurrency",
    list(crypto_map.keys())
)

symbol = crypto_map[crypto_name]

from datetime import date

start_date = st.date_input(
    "Start Date",
    date(2020, 1, 1),
    min_value=date(2015, 1, 1),
    max_value=date.today()
)

end_date = st.date_input(
    "End Date",
    date.today(),
    min_value=start_date,
    max_value=date.today()
)

# -------------------- DATA LOADING --------------------
@st.cache_data(show_spinner=False)
def load_data(symbol, start_date, end_date):
    data = yf.download(symbol, start=start_date, end=end_date)
    data.reset_index(inplace=True)
    return data

data = load_data(symbol, start_date, end_date)

# -------------------- FIX #1: FLATTEN MULTIINDEX COLUMNS --------------------
# (CRITICAL for Prophet & pandas numeric operations)
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)

# -------------------- VALIDATION --------------------
if data.empty:
    st.error(
        "No data available for the selected date range. "
        "Please choose a valid historical period."
    )
    st.stop()

st.success(
    f"Loaded {len(data)} daily records for {crypto_name} "
    f"from {start_date} to {end_date}."
)

# -------------------- PREVIEW --------------------
with st.expander("View Sample Data"):
    st.dataframe(data.head(100), use_container_width=True)



# -------------------- PREPROCESSING & EDA --------------------
st.header("2. Data Preprocessing & Exploratory Data Analysis")

st.markdown("""
This section explores historical cryptocurrency price behavior through 
**feature engineering and trend visualization**, enabling a clearer understanding 
of short-term momentum and long-term market direction.
""")

# Feature engineering
data['Return'] = data['Close'].pct_change()
data['Volatility'] = data['Return'].rolling(7).std()
data['MA7'] = data['Close'].rolling(7).mean()
data['MA30'] = data['Close'].rolling(30).mean()

# Plot
fig, ax = plt.subplots()
ax.plot(data['Date'], data['Close'], label="Closing Price", linewidth=1)
ax.plot(data['Date'], data['MA7'], label="7-Day Moving Average (MA7)", linestyle="--")
ax.plot(data['Date'], data['MA30'], label="30-Day Moving Average (MA30)", linestyle="-.")
ax.set_xlabel("Date")
ax.set_ylabel("Price")
ax.legend()
st.pyplot(fig)

st.markdown("""
###  Interpretation of Moving Averages

- **MA7 (7-Day Moving Average)**  
  Represents short-term price momentum.  
  It reacts quickly to recent price changes and highlights short-lived trends.

- **MA30 (30-Day Moving Average)**  
  Captures medium-to-long-term market direction.  
  It smooths short-term noise and reflects overall trend stability.

###  Market Insights

- When **MA7 crosses above MA30**, it may indicate **short-term bullish momentum**.
- When **MA7 crosses below MA30**, it may signal **short-term bearish pressure**.
- A widening gap between MA7 and MA30 suggests increasing trend strength or volatility.

###  Why This Matters
Moving averages help reduce price noise and make trends more interpretable, 
serving as foundational indicators for exploratory analysis before applying 
advanced forecasting models.
""")


# -------------------- VOLATILITY ANALYSIS --------------------
st.header("3. Volatility Analysis")

st.markdown("""
This section examines **price volatility**, which measures the degree of variation 
in cryptocurrency prices over time. Volatility is a key indicator of **market risk and uncertainty**.
""")

fig2, ax2 = plt.subplots()
ax2.plot(data['Date'], data['Volatility'], color="red", linewidth=1.2)
ax2.set_title("7-Day Rolling Volatility of Closing Prices")
ax2.set_xlabel("Date")
ax2.set_ylabel("Volatility (Standard Deviation)")
st.pyplot(fig2)

st.markdown("""
###  Interpretation of Results

- **Higher volatility spikes** indicate periods of increased market uncertainty, often caused by:
  - Regulatory announcements  
  - Macroeconomic events  
  - Market speculation or panic selling  

- **Lower volatility regions** suggest relatively stable market conditions and reduced short-term risk.

- Cryptocurrency markets typically exhibit **higher volatility** compared to traditional financial assets, 
  making volatility analysis essential for risk-aware decision-making.

###  Practical Implications
- Traders can use volatility to adjust **position sizing and stop-loss levels**.
- Investors can identify **high-risk periods** and avoid entering the market during extreme fluctuations.
- Volatility insights complement forecasting models by explaining **confidence and risk behind predictions**.
""")


# -------------------- SENTIMENT ANALYSIS --------------------
st.header("4. Sentiment Analysis (NLP)")

st.markdown("""
This section analyzes **market sentiment** from cryptocurrency-related news headlines using 
**VADER (Valence Aware Dictionary for Sentiment Reasoning)**, a lexicon-based NLP model 
commonly used for financial and social media text.
""")

news_samples = [
    "Bitcoin price surges as institutional adoption increases",
    "Cryptocurrency market crashes amid regulatory fears",
    "Investors remain optimistic about blockchain technology"
]

analyzer = SentimentIntensityAnalyzer()
sentiments = [analyzer.polarity_scores(text)['compound'] for text in news_samples]

sent_df = pd.DataFrame({
    "News Headline": news_samples,
    "Sentiment Score (–1 to +1)": sentiments
})

st.dataframe(sent_df, use_container_width=True)

# Plot
fig, ax = plt.subplots()
bars = ax.bar(range(len(sentiments)), sentiments)
ax.axhline(0, color="black", linewidth=0.8)
ax.set_xticks(range(len(sentiments)))
ax.set_xticklabels(["Headline 1", "Headline 2", "Headline 3"])
ax.set_ylabel("Sentiment Polarity")
ax.set_title("Sentiment Polarity of Crypto News Headlines")

st.pyplot(fig)

# Interpretation
st.markdown("""
###  Interpretation of Results

- **Positive scores (> 0)** indicate optimistic or bullish market sentiment.
- **Negative scores (< 0)** reflect fear, uncertainty, or bearish sentiment.
- Headlines related to **market growth or adoption** tend to show positive sentiment.
- Headlines involving **crashes or regulation** typically produce negative sentiment.

###  Why This Matters
Market sentiment often influences **short-term price volatility**.  
By combining sentiment analysis with price-based models, the system provides a **more holistic view** of cryptocurrency market behavior.
""")


st.header("5. Time Series Forecasting Models")
st.info("Click a button to run each forecasting model")

# -------------------- ARIMA FORECAST --------------------
st.subheader("ARIMA Forecast")

if "arima_fig" not in st.session_state:
    st.session_state.arima_fig = None

if st.button("Run ARIMA Forecast"):
    with st.spinner("Running ARIMA model..."):

        close_series = data.set_index("Date")["Close"].dropna()

        arima_model = ARIMA(close_series, order=(5, 1, 0))
        arima_fit = arima_model.fit()

        forecast_steps = 30
        arima_forecast = arima_fit.forecast(steps=forecast_steps)

        forecast_dates = pd.date_range(
            start=close_series.index[-1],
            periods=forecast_steps + 1,
            freq="D"
        )[1:]

        fig, ax = plt.subplots()
        ax.plot(close_series[-200:], label="Historical")
        ax.plot(forecast_dates, arima_forecast, label="ARIMA Forecast")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.legend()

        st.session_state.arima_fig = fig
        st.success("ARIMA forecasting completed!")

if st.session_state.arima_fig is not None:
    st.pyplot(st.session_state.arima_fig)

# ---------- LSTM SAFETY CHECK ----------
if len(data) < 100:
    st.warning("Not enough data for LSTM. Please select a longer date range.")
else:
    st.subheader("LSTM Forecast")

    # Initialize session state
    if "lstm_fig" not in st.session_state:
        st.session_state.lstm_fig = None

    if st.button("Run LSTM Forecast"):
        with st.spinner("Training LSTM model (this may take a moment)..."):

            # Prepare data
            close_series = data.set_index("Date")["Close"].dropna()
            close_prices = close_series.values.reshape(-1, 1)

            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(close_prices)

            lookback = 60
            X, y = [], []
            for i in range(lookback, len(scaled_data)):
                X.append(scaled_data[i - lookback:i])
                y.append(scaled_data[i])

            X, y = np.array(X), np.array(y)

            # Build LSTM
            model = Sequential([
                LSTM(50, return_sequences=True, input_shape=(lookback, 1)),
                LSTM(50),
                Dense(1)
            ])

            model.compile(optimizer="adam", loss="mean_squared_error")

            # Train (lightweight)
            model.fit(
                X, y,
                epochs=2,
                batch_size=64,
                verbose=0
            )

            # Recursive 30-day forecast
            last_sequence = scaled_data[-lookback:]
            predictions = []

            for _ in range(30):
                pred = model.predict(
                    last_sequence.reshape(1, lookback, 1),
                    verbose=0
                )
                predictions.append(pred[0][0])
                last_sequence = np.vstack([last_sequence[1:], pred])

            lstm_pred = scaler.inverse_transform(
                np.array(predictions).reshape(-1, 1)
            )

            # Future dates
            forecast_dates = pd.date_range(
                start=close_series.index[-1],
                periods=31,
                freq="D"
            )[1:]

            # Plot
            fig, ax = plt.subplots()
            ax.plot(close_series[-200:], label="Historical")
            ax.plot(forecast_dates, lstm_pred, label="LSTM Forecast")
            ax.set_xlabel("Date")
            ax.set_ylabel("Price")
            ax.legend()

            # Store plot in session state
            st.session_state.lstm_fig = fig

            st.success("LSTM forecasting completed!")

    # Display persisted plot
    if st.session_state.lstm_fig is not None:
        st.pyplot(st.session_state.lstm_fig)


# -------------------- PROPHET FORECAST --------------------
st.subheader("Prophet Forecast")

# Initialize session state
if "prophet_fig" not in st.session_state:
    st.session_state.prophet_fig = None

if st.button("Run Prophet Forecast"):
    with st.spinner("Running Prophet model..."):

        # Prepare data (now guaranteed 1D)
        prophet_df = data[['Date', 'Close']].copy()
        prophet_df.columns = ['ds', 'y']

        # Enforce correct types
        prophet_df['ds'] = pd.to_datetime(prophet_df['ds'])
        prophet_df['y'] = prophet_df['y'].astype(float)

        # Drop missing values
        prophet_df = prophet_df.dropna()

        if len(prophet_df) < 2:
            st.error("Not enough valid data for Prophet forecasting.")
            st.stop()

        # Build & fit Prophet
        model = Prophet(daily_seasonality=True)
        model.fit(prophet_df)

        # Forecast
        future = model.make_future_dataframe(periods=30)
        forecast = model.predict(future)

        # Plot
        fig = model.plot(forecast)
        st.session_state.prophet_fig = fig

        st.success("Prophet forecasting completed!")

# Persist plot
if st.session_state.prophet_fig is not None:
    st.pyplot(st.session_state.prophet_fig)


# -------------------- REAL-WORLD APPLICATION --------------------
st.header("8. Real-World Applications & Impact")

st.markdown("""
###  Practical Use Cases

This system demonstrates how **time-series analytics and machine learning** can be applied to real-world cryptocurrency markets to support **data-driven decision making**.

**Key applications include:**

- **Market Trend Identification:**  
  Moving averages and historical price analysis help identify bullish and bearish trends, enabling traders to time entry and exit points more effectively.

- **Volatility & Risk Assessment:**  
  Rolling volatility analysis highlights periods of heightened market uncertainty, allowing investors to adjust position sizes and risk exposure.

- **Sentiment-Aware Market Insights:**  
  NLP-based sentiment analysis captures market psychology from crypto-related news, complementing numerical price data with qualitative signals.

- **Multi-Model Forecasting for Decision Support:**  
  Comparing ARIMA (statistical), LSTM (deep learning), and Prophet (trend-based) forecasts provides diverse perspectives on potential future price movements.

- **Educational & Research Use:**  
  The system serves as a practical learning tool for students and researchers exploring financial time series modeling, forecasting techniques, and ML deployment using Streamlit.

---

### ⚠️ Disclaimer
This application is designed for **educational and analytical purposes only**.  
It does **not constitute financial advice**, and predictions should not be used for real trading without professional risk evaluation.
""")


# -------------------- FOOTER --------------------
st.markdown("---")
st.markdown("**Crypto Time Series Analysis Project — Streamlit Dashboard**")
