# src/analysis/preprocessing.py
import pandas as pd
import numpy as np
from datetime import datetime

def load_and_preprocess_data(filepath):
    """
    Load and preprocess Brent oil price data.
    
    Args:
        filepath (str): Path to the CSV file containing price data
        
    Returns:
        pd.DataFrame: Processed DataFrame with datetime index and log returns
    """
    df = pd.read_csv(filepath)
    
    # Convert date to datetime
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    df.set_index('Date', inplace=True)
    
    # Ensure prices are numeric
    df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
    
    # Remove any missing values
    df.dropna(inplace=True)
    
    # Calculate log returns
    df['Log_Return'] = np.log(df['Price']) - np.log(df['Price'].shift(1))
    df.dropna(inplace=True)
    
    return df

def load_key_events(filepath):
    """
    Load key events data with dates and descriptions.
    
    Args:
        filepath (str): Path to the CSV file containing event data
        
    Returns:
        pd.DataFrame: Processed DataFrame with datetime index
    """
    events = pd.read_csv(filepath)
    events['Date'] = pd.to_datetime(events['Date'])
    events.set_index('Date', inplace=True)
    return events