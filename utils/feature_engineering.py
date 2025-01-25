import pandas as pd
import numpy as np

def add_technical_indicators(df):
    """
    Add common technical indicators to the DataFrame.

    Parameters:
        df (pd.DataFrame): Original DataFrame with 'Close', 'High', and 'Low' columns.

    Returns:
        pd.DataFrame: DataFrame with additional technical indicators.
    """
    # Moving Averages
    df['MA7'] = df['Close'].rolling(window=7).mean()
    df['MA21'] = df['Close'].rolling(window=21).mean()

    # RSI
    delta = df['Close'].diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # MACD
    ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema_12 - ema_26

    # ATR (Average True Range) as a replacement for Bollinger Bands
    df['ATR'] = calculate_atr(df)

    # Drop rows with NaN values introduced by rolling calculations
    df = df.dropna()

    return df

def calculate_atr(df, window=14):
    """
    Calculate the Average True Range (ATR).
    
    Parameters:
        df (pd.DataFrame): DataFrame with 'High', 'Low', and 'Close' columns.
        window (int): Lookback period for ATR calculation.

    Returns:
        pd.Series: The ATR values.
    """
    high_low = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift()).abs()
    low_close = (df['Low'] - df['Close'].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.rolling(window=window).mean()
    return atr


def add_lagged_features(df, columns=['Close'], lags=5):
    """
    Add lagged features to the DataFrame.

    Parameters:
        df (pd.DataFrame): Original DataFrame.
        columns (list): Columns for lagging.
        lags (int): Number of lags.

    Returns:
        pd.DataFrame: DataFrame with lagged features.
    """
    for col in columns:
        for lag in range(1, lags + 1):
            df[f'{col}_lag{lag}'] = df[col].shift(lag)
    df = df.dropna()
    return df