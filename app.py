import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
import plotly.graph_objects as go
import plotly.express as px
import os
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import google.generativeai as genai
from yahoo_fin import news

# Download necessary NLTK data
nltk.download("vader_lexicon")

# Initialize Google Gemini API
GEMINI_API_KEY = "AIzaSyCg6b34yh04SOvhSnxc5m5ComJKaPTc0oI"  # Replace with your API key
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("models/gemini-1.5-pro")

# Streamlit UI setup
st.set_page_config(page_title="TradeVision - Real-time Stock Analyzer", layout="wide")
st.title("📈 TradeVision - Real-time Stock Analyzer")

# Sidebar user inputs
ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., AAPL, TSLA, GOOG):", "AAPL").upper()
future_days = st.sidebar.slider("Days to Predict:", 1, 30, 7)

# Load sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Cache stock data fetching
@st.cache_data
def fetch_stock_data(ticker):
    return yf.download(ticker, period="2y")

stock_data = fetch_stock_data(ticker)
if stock_data.empty:
    st.sidebar.error("Invalid Ticker! Enter a valid stock symbol.")
    st.stop()

# Sidebar confirmation
st.sidebar.success(f"Data for {ticker} loaded!")

# Data preprocessing
scaler = MinMaxScaler()
stock_data['Scaled_Close'] = scaler.fit_transform(stock_data[['Close']])

def create_sequences(data, time_step=50):
    X, Y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), 0])
        Y.append(data[i + time_step, 0])
    return np.array(X), np.array(Y)

time_step = 50
dataset = stock_data['Scaled_Close'].values.reshape(-1, 1)
X, Y = create_sequences(dataset, time_step)
train_size = int(len(X) * 0.8)
X_train, Y_train = X[:train_size], Y[:train_size]
X_test, Y_test = X[train_size:], Y[train_size:]
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Model path
model_path = f"{ticker}_lstm_model.h5"

def build_lstm_model():
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(time_step, 1)),
        LSTM(50, return_sequences=False),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Load or train model
if os.path.exists(model_path):
    model = load_model(model_path)
else:
    model = build_lstm_model()
    with st.spinner("Training LSTM Model... ⏳"):
        model.fit(X_train, Y_train, epochs=10, batch_size=16, verbose=0)
        model.save(model_path)

# Predictions
Y_pred = model.predict(X_test)
Y_pred = scaler.inverse_transform(Y_pred.reshape(-1, 1))
Y_test = scaler.inverse_transform(Y_test.reshape(-1, 1))

# Model performance metrics
st.subheader("📊 Model Accuracy Metrics")
st.write(f"🔹 Mean Squared Error (MSE): {mean_squared_error(Y_test, Y_pred):.4f}")
st.write(f"🔹 Mean Absolute Error (MAE): {mean_absolute_error(Y_test, Y_pred):.4f}")

# Plot actual vs predicted values
st.subheader("📈 Actual vs Predicted Stock Prices")
fig_actual_predicted = go.Figure()
fig_actual_predicted.add_trace(go.Scatter(x=stock_data.index[-len(Y_test):], y=Y_test.flatten(),
                                          mode='lines', name='Actual Price', line=dict(color='blue')))
fig_actual_predicted.add_trace(go.Scatter(x=stock_data.index[-len(Y_test):], y=Y_pred.flatten(),
                                          mode='lines', name='Predicted Price', line=dict(color='red', dash='dot')))
fig_actual_predicted.update_layout(title=f"Actual vs Predicted Prices ({ticker})",
                                   xaxis_title="Date", yaxis_title="Stock Price (USD)")
st.plotly_chart(fig_actual_predicted)

# Predict future prices
def predict_future_prices(model, last_50_days, future_days):
    future_predictions = []
    current_input = last_50_days.reshape(1, -1, 1)
    for _ in range(future_days):
        next_prediction = model.predict(current_input)[0][0]
        future_predictions.append(next_prediction)
        current_input = np.append(current_input[:, 1:, :], [[[next_prediction]]], axis=1)
    return scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

last_50_days = dataset[-time_step:]
future_prices = predict_future_prices(model, last_50_days, future_days)
future_dates = pd.date_range(stock_data.index[-1] + pd.Timedelta(days=1), periods=future_days)
future_df = pd.DataFrame({'Date': future_dates, 'Predicted_Close': future_prices.flatten()})
st.subheader(f"📅 Predicted Stock Prices for Next {future_days} Days")
st.dataframe(future_df)

# Fetch and analyze stock news
def fetch_stock_news(stock_name):
    try:
        news_articles = news.get_yf_rss(stock_name)
        return "\n".join([article['title'] for article in news_articles[:5]])
    except Exception:
        return "No news found."

stock_news = fetch_stock_news(ticker)
st.subheader(f"📰 Latest News for {ticker}")
for article in news.get_yf_rss(ticker)[:5]:
    st.write(f"📰 {article['title']}\n🔗 {article['link']}\n")

# Sentiment Analysis
def get_gemini_insights(news_text):
    prompt = f"Summarize the following stock news and provide key takeaways:\n\n{news_text}"
    try:
        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error fetching insights from Gemini AI: {e}"

insights = get_gemini_insights(stock_news)
sentiment = sia.polarity_scores(stock_news)
st.subheader(f"📊 Sentiment Analysis & AI Insights for {ticker}")
st.write(f"Gemini AI Insights:\n{insights}")

# Pie chart for sentiment distribution
sentiment_labels = ["Positive", "Neutral", "Negative"]
sentiment_values = [sentiment["pos"], sentiment["neu"], sentiment["neg"]]
fig_pie = px.pie(values=sentiment_values, names=sentiment_labels, title="Sentiment Distribution")
st.plotly_chart(fig_pie)

st.sidebar.markdown("### About TradeVision")
st.sidebar.info("TradeVision provides real-time stock analysis using AI-powered prediction models and sentiment analysis.")
