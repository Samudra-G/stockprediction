import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from workflow import save_database
from datetime import datetime
from utils import (
    calculate_rsi,
    calculate_dividend_yield,
    calculate_volatility,
    plot_macd,
    calculate_macd,
    preprocess_data,
    prepare_test_data,
    make_predictions,
    load_LSTM
)
from financial_functions import py_moving_averages
import matplotlib.pyplot as plt

# Paths
base_dir= 'C:/Coding/Python/stock_prediction/'
model_path = base_dir + '/model/lstm_model.keras'
database_path = base_dir + 'data/stock_data.db'

@st.cache_data
def fetch_stock_data(ticker):
    stock = yf.Ticker(ticker)
    df = stock.history(period='max')
    company_name = stock.info['longName']
    df.reset_index(inplace=True)
    df['Ticker'] = ticker
    df['Name'] = company_name
    return df

def main():
    st.title("Stock Analysis and Prediction App")

    # Input the ticker
    ticker = st.text_input("Enter the stock ticker (e.g., AAPL):", "AAPL").upper()

    if ticker:
        # Fetch and display the stock data
        df = fetch_stock_data(ticker)
        save_database(df, database_path, table_name='stocks')
        st.write(df.head())

        # Descriptive statistics
        st.subheader('Descriptive Statistics')
        st.write(df.describe())

        # Financial calculations
        window_size = st.slider('Moving Average Window Size', 1, 100, 14)
        rsi_period = st.slider('RSI Period', 1, 100, 14)
        short_window = st.slider('MACD Short Window', 1, 50, 12)
        long_window = st.slider('MACD Long Window', 1, 50, 26)
        signal_window = st.slider('MACD Signal Window', 1, 50, 9)

        # Moving Averages
        if len(df) >= 200:
            ma_100 = py_moving_averages(df['Close'].tolist(), 100)
            ma_200 = py_moving_averages(df['Close'].tolist(), 200)

            df['MA100'] = [np.nan] * (100 - 1) + ma_100
            df['MA200'] = [np.nan] * (200 - 1) + ma_200

            st.subheader('Moving Averages')
            st.line_chart(df.set_index('Date')[['Close', 'MA100', 'MA200']])
        else:
            st.warning("Not enough data to calculate 200-day moving average (need at least 200 data points).")

        # Volatility Calculation
        st.subheader('Volatility')
        volatility = calculate_volatility(df)
        st.write(f"Volatility: {volatility:.3f}")

        # Dividend Yield
        st.subheader('Dividend Yield')
        dividend_yield = calculate_dividend_yield(df)
        st.write(f"Dividend Yield: {dividend_yield:.2f}%")

        # RSI Calculation
        st.subheader('RSI')
        rsi_values = calculate_rsi(df, rsi_period)
        padding_length = len(df) - len(rsi_values)
        df['RSI'] = np.concatenate([np.full(padding_length, np.nan), rsi_values])
        st.line_chart(df.set_index('Date')[['RSI']])

        # MACD Calculation
        st.subheader("MACD")
        df_macd = calculate_macd(df, short_window, long_window, signal_window, limit=100)
        macd_fig = plot_macd(df_macd)
        st.pyplot(macd_fig)

        # LSTM Stock Price Prediction
        st.subheader("LSTM Stock Price Prediction")

        # Preprocess data
        X_train, y_train, scaler, train_data_len, dataset = preprocess_data(df)

        # Prepare test data
        X_test, y_test = prepare_test_data(df, dataset, scaler, train_data_len)

        # Load the model
        model = load_LSTM(model_path)

        # Make predictions
        predictions, y_test_actual = make_predictions(model, X_test, scaler, df, train_data_len)

        # Plot results
        train = df[:train_data_len]
        valid = df[train_data_len:].copy()
        valid['Predictions'] = predictions

        st.write("Prediction vs Actual Prices")
        fig, ax = plt.subplots(figsize=(16, 8))
        ax.plot(train.index, train['Close'], label='Training Data')
        ax.plot(valid.index, valid['Close'], label='Actual Prices')
        ax.plot(valid.index, valid['Predictions'], label='Predicted Prices', linestyle='--')
        ax.legend(loc='upper left')
        st.pyplot(fig)

        # Automatically save results to a CSV
        results_df = pd.DataFrame({
            'Actual Prices': valid['Close'].values,
            'Predicted Prices': valid['Predictions'].values
        })
        results_csv_path = base_dir + '/data/prediction_results.csv'
        results_df.to_csv(results_csv_path, index=False)

# Run the app
if __name__ == "__main__":
    main()
