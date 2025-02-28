import yfinance as yf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns


def download_historical_data(tickers, start_date, end_date):
    """
    Download historical financial data for the given tickers and date range.

    Parameters:
    - tickers: List of ticker symbols (e.g., ['TSLA', 'BND', 'SPY'])
    - start_date: Start date for the data (e.g., '2020-01-01')
    - end_date: End date for the data (e.g., '2023-12-31')

    Returns:
    - Pandas DataFrame containing the historical data
    """
    return yf.download(tickers, start=start_date, end=end_date)

def separate_data(historical_data, tickers):
    """
    Separate the historical data into individual DataFrames for each ticker.

    Parameters:
    - historical_data: DataFrame containing the historical data
    - tickers: List of ticker symbols

    Returns:
    - Dictionary of DataFrames, one for each ticker
    """
    data_frames = {}
    for ticker in tickers:
        data = historical_data['Close'][ticker].reset_index()
        data.columns = ['Date', 'Close']
        data['Open'] = historical_data['Open'][ticker].values
        data['High'] = historical_data['High'][ticker].values
        data['Low'] = historical_data['Low'][ticker].values
        data['Volume'] = historical_data['Volume'][ticker].values
        data_frames[ticker] = data
    return data_frames

def check_basic_statistics(data_frames):
    """
    Print basic statistics for each ticker's data.

    Parameters:
    - data_frames: Dictionary of DataFrames, one for each ticker
    """
    for ticker, data in data_frames.items():
        print(f"Basic Statistics for {ticker}:")
        print(data.describe(), "\n")

def ensure_data_types(data_frames):
    """
    Ensure all columns have appropriate data types.

    Parameters:
    - data_frames: Dictionary of DataFrames, one for each ticker
    """
    for ticker, data in data_frames.items():
        print(f"Data Types for {ticker}:")
        print(data.dtypes, "\n")

def standardize_data_types(data_frames):
    """
    Standardize the data types of numerical columns to float64.

    Parameters:
    - data_frames: Dictionary of DataFrames, one for each ticker

    Returns:
    - Dictionary of DataFrames with standardized data types
    """
    for ticker, data in data_frames.items():
        # Convert numerical columns to float64
        data_frames[ticker] = data.astype({
            'Close': 'float64',
            'Open': 'float64',
            'High': 'float64',
            'Low': 'float64',
            'Volume': 'float64'
        })
    return data_frames
def check_missing_values(data_frames):
    """
    Check for missing values in each ticker's data.

    Parameters:
    - data_frames: Dictionary of DataFrames, one for each ticker
    """
    for ticker, data in data_frames.items():
        print(f"Missing Values for {ticker}:")
        print(data.isnull().sum(), "\n")

def handle_missing_values(data_frames, method='ffill'):
    """
    Handle missing values by filling, interpolating, or removing them.

    Parameters:
    - data_frames: Dictionary of DataFrames, one for each ticker
    - method: Method to handle missing values ('ffill', 'bfill', 'interpolate', 'drop')
    """
    for ticker, data in data_frames.items():
        if method == 'drop':
            data_frames[ticker] = data.dropna()
        else:
            data_frames[ticker] = data.fillna(method=method)
    return data_frames

def normalize_data(data_frames):
    """
    Normalize or scale the data using MinMaxScaler.

    Parameters:
    - data_frames: Dictionary of DataFrames, one for each ticker
    """
    scaler = MinMaxScaler()
    for ticker, data in data_frames.items():
        data_frames[ticker][['Close', 'Open', 'High', 'Low', 'Volume']] = scaler.fit_transform(
            data[['Close', 'Open', 'High', 'Low', 'Volume']]
        )
    return data_frames

def display_cleaned_data(data_frames):
    """
    Display the first few rows of each DataFrame after cleaning and scaling.

    Parameters:
    - data_frames: Dictionary of DataFrames, one for each ticker
    """
    for ticker, data in data_frames.items():
        print(f"Cleaned and Scaled Data for {ticker}:")
        print(data.head(), "\n")

def plot_closing_prices(data_frames):
    """
    Visualize the closing price over time for each ticker.

    Parameters:
    - data_frames: Dictionary of DataFrames, one for each ticker
    """
    for ticker, data in data_frames.items():
        plt.figure(figsize=(10, 5))
        plt.plot(data['Date'], data['Close'], label='Close Price')
        plt.title(f'{ticker} Closing Price Over Time')
        plt.xlabel('Date')
        plt.ylabel('Close Price')
        plt.legend()
        plt.grid()
        plt.show()

def calculate_daily_returns(data_frames):
    """
    Calculate and plot the daily percentage change (returns) for each ticker.

    Parameters:
    - data_frames: Dictionary of DataFrames, one for each ticker
    """
    for ticker, data in data_frames.items():
        data['Daily Return'] = data['Close'].pct_change() * 100
        plt.figure(figsize=(10, 5))
        plt.plot(data['Date'], data['Daily Return'], label='Daily Return', color='orange')
        plt.title(f'{ticker} Daily Percentage Change (Returns)')
        plt.xlabel('Date')
        plt.ylabel('Daily Return (%)')
        plt.legend()
        plt.grid()
        plt.show()

def analyze_volatility(data_frames, window=30):
    """
    Analyze volatility by calculating rolling means and standard deviations.

    Parameters:
    - data_frames: Dictionary of DataFrames, one for each ticker
    - window: Rolling window size (default: 30 days)
    """
    for ticker, data in data_frames.items():
        data['Rolling Mean'] = data['Close'].rolling(window=window).mean()
        data['Rolling Std'] = data['Close'].rolling(window=window).std()

        plt.figure(figsize=(10, 5))
        plt.plot(data['Date'], data['Close'], label='Close Price', color='blue')
        plt.plot(data['Date'], data['Rolling Mean'], label=f'{window}-Day Rolling Mean', color='green')
        plt.plot(data['Date'], data['Rolling Std'], label=f'{window}-Day Rolling Std', color='red')
        plt.title(f'{ticker} Volatility Analysis (Rolling Mean & Std)')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid()
        plt.show()

def detect_outliers(data_frames, threshold=3):
    """
    Detect outliers using the Z-score method.

    Parameters:
    - data_frames: Dictionary of DataFrames, one for each ticker
    - threshold: Z-score threshold for outlier detection (default: 3)
    """
    from scipy.stats import zscore

    for ticker, data in data_frames.items():
        data['Z-Score'] = zscore(data['Close'])
        outliers = data[abs(data['Z-Score']) > threshold]

        print(f"Outliers for {ticker}:")
        print(outliers[['Date', 'Close', 'Z-Score']], "\n")

def analyze_unusual_returns(data_frames, threshold=2):
    """
    Analyze days with unusually high or low returns.

    Parameters:
    - data_frames: Dictionary of DataFrames, one for each ticker
    - threshold: Threshold for unusual returns (default: 2%)
    """
    for ticker, data in data_frames.items():
        if 'Daily Return' not in data.columns:
            data['Daily Return'] = data['Close'].pct_change() * 100

        unusual_returns = data[(data['Daily Return'] > threshold) | (data['Daily Return'] < -threshold)]

        print(f"Unusual Returns for {ticker}:")
        print(unusual_returns[['Date', 'Close', 'Daily Return']], "\n")