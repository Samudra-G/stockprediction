#Use this to call our functions
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import streamlit as st
import matplotlib.pyplot as plt
import math
from app.financial_functions import py_moving_averages, py_calculate_rsi, py_calculate_volatility, py_calculate_dividend_yield
def save_database(df,db_name='stock_data.db'):
    import sqlite3

    conn = sqlite3.connect(db_name)
    df.to_sql('stocks', conn, if_exists='replace',index=False)

    conn.close()
    print(f"Data saved to {db_name}")

def load_LSTM(file_path):
    
    from tensorflow.keras.models import load_model
    model = load_model(file_path)
    return model

def preprocess_data(df):

    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date',inplace = True)
     # Filter and convert to numpy array
    data = df.filter(['Close'])
    dataset = data.values
    
    # Determine the length of the training data
    train_data_len = math.ceil(len(dataset) * .8)
    
    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)
    
    # Prepare training data
    train_data = scaled_data[0:train_data_len, :]
    X_train = []
    y_train = []

    for i in range(60, len(train_data)):
        X_train.append(train_data[i-60:i, 0])
        y_train.append(train_data[i, 0])
    
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    
    return X_train, y_train, scaler, train_data_len, dataset

def prepare_test_data(df,dataset,scaler,train_data_len,timesteps=60):
    
    data = df.filter(['Close']).values
    scaled_data = scaler.transform(data)
    
    # Create test data sequences
    X_test = []
    y_test = dataset[train_data_len: , :]
    
    for i in range(train_data_len, len(scaled_data)):
        X_test.append(scaled_data[i-timesteps:i, 0])
    
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    
    return X_test, y_test

def make_predictions(model,X_test,scaler,df,train_data_len):
    
    # Make predictions
    predictions = model.predict(X_test)
    
    # Reverse the scaling for predictions
    predictions = scaler.inverse_transform(predictions)
    
    # Get the true prices from the dataset
    y_test = df['Close'].values[train_data_len:]
    
    # Reverse the scaling for true values
    y_test = scaler.inverse_transform(y_test.reshape(-1, 1))
    
    return predictions, y_test

#Streamlit analytics metrics
@st.cache_data
def calculate_rsi(df, period):
    return py_calculate_rsi(df['Close'].tolist(), period)

@st.cache_data
def calculate_volatility(df):
    return py_calculate_volatility(df['High'].tolist(), df['Low'].tolist())

@st.cache_data
def calculate_dividend_yield(df):
    if 'Dividends' in df.columns and df['Dividends'].sum() > 0:
        total_dividend = df['Dividends'].iloc[-365:].sum()
        latest_close_price = df['Close'].iloc[-1]
        return py_calculate_dividend_yield([total_dividend], [latest_close_price])
    return 0

@st.cache_data
def calculate_macd(df, short_window=12, long_window=26, signal_window=9, limit=100):

    if len(df) < limit:
        limit = len(df)  # Use the entire dataset if it's smaller than the limit

    df_limited = df.tail(limit).copy()

    short_ema = df_limited['Close'].ewm(span=short_window, adjust=False).mean()
    long_ema = df_limited['Close'].ewm(span=long_window, adjust=False).mean()

    macd_line = short_ema - long_ema
    signal_line = macd_line.ewm(span=signal_window, adjust=False).mean()
    histogram = macd_line - signal_line

    df_limited['MACD_Line'] = macd_line
    df_limited['Signal_Line'] = signal_line
    df_limited['Histogram'] = histogram

    return df_limited


def plot_macd(df_limited):

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot MACD line and signal line
    ax.plot(df_limited['Date'], df_limited['MACD_Line'], label="MACD Line", color="blue")
    ax.plot(df_limited['Date'], df_limited['Signal_Line'], label="Signal Line", color="orange")
    
    # Plot Histogram
    ax.bar(
        df_limited['Date'], 
        df_limited['Histogram'], 
        color=['green' if val > 0 else 'red' for val in df_limited['Histogram']], 
        label="Histogram",
        width=0.8,
    )

    # Add title and legend
    ax.set_title("MACD (Moving Average Convergence Divergence)")
    ax.axhline(0, color='black', linewidth=0.5, linestyle="--")
    ax.legend(loc="upper left")
    
    # Formatting x-axis labels
    fig.autofmt_xdate()
    
    return fig
