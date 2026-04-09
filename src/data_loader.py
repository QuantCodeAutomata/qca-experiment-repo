"""
Data Loader Module

This module handles data retrieval from the Massive API and preprocessing
for portfolio optimization experiments.
"""

import os
from typing import List, Optional, Dict
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from massive import RESTClient


class DataLoader:
    """
    Data loader for retrieving and preprocessing financial market data.
    
    Uses the Massive API to retrieve historical price data for multiple assets.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the DataLoader.
        
        Parameters
        ----------
        api_key : str, optional
            Massive API key. If None, reads from MASSIVE_API_KEY environment variable.
        """
        self.api_key = api_key or os.getenv("MASSIVE_API_KEY") or os.getenv("MASSIVE_TOKEN")
        if not self.api_key:
            raise ValueError("API key must be provided or set in MASSIVE_API_KEY/MASSIVE_TOKEN environment variable")
        
        self.client = RESTClient(api_key=self.api_key)
    
    def fetch_price_data(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str,
        timespan: str = "day",
        multiplier: int = 1
    ) -> pd.DataFrame:
        """
        Fetch historical price data for multiple tickers.
        
        Parameters
        ----------
        tickers : List[str]
            List of ticker symbols to retrieve
        start_date : str
            Start date in format 'YYYY-MM-DD'
        end_date : str
            End date in format 'YYYY-MM-DD'
        timespan : str, default='day'
            Timespan for aggregates ('minute', 'hour', 'day', 'week', 'month')
        multiplier : int, default=1
            Multiplier for timespan
            
        Returns
        -------
        pd.DataFrame
            DataFrame with datetime index and columns for each ticker's close price
        """
        all_data = {}
        
        for ticker in tickers:
            try:
                print(f"Fetching data for {ticker}...")
                aggs = []
                
                # Fetch aggregates with pagination
                for agg in self.client.list_aggs(
                    ticker=ticker,
                    multiplier=multiplier,
                    timespan=timespan,
                    from_=start_date,
                    to=end_date,
                    limit=50000
                ):
                    aggs.append(agg)
                
                if not aggs:
                    print(f"Warning: No data retrieved for {ticker}")
                    continue
                
                # Convert to DataFrame
                df = pd.DataFrame([{
                    'timestamp': pd.to_datetime(a['t'], unit='ms'),
                    'close': a['c'],
                    'volume': a['v'],
                    'open': a['o'],
                    'high': a['h'],
                    'low': a['l']
                } for a in aggs])
                
                df.set_index('timestamp', inplace=True)
                df.sort_index(inplace=True)
                
                all_data[ticker] = df['close']
                
            except Exception as e:
                print(f"Error fetching data for {ticker}: {e}")
                continue
        
        if not all_data:
            raise ValueError("No data was successfully retrieved for any ticker")
        
        # Combine all tickers into single DataFrame
        prices_df = pd.DataFrame(all_data)
        
        # Handle missing data by forward filling then backward filling
        prices_df = prices_df.fillna(method='ffill').fillna(method='bfill')
        
        return prices_df
    
    def calculate_returns(
        self,
        prices: pd.DataFrame,
        method: str = 'simple'
    ) -> pd.DataFrame:
        """
        Calculate returns from price data.
        
        Parameters
        ----------
        prices : pd.DataFrame
            DataFrame with price data
        method : str, default='simple'
            Return calculation method ('simple' or 'log')
            
        Returns
        -------
        pd.DataFrame
            DataFrame with returns
        """
        if method == 'simple':
            returns = prices.pct_change()
        elif method == 'log':
            returns = np.log(prices / prices.shift(1))
        else:
            raise ValueError(f"Unknown method: {method}. Use 'simple' or 'log'")
        
        # Drop first row with NaN
        returns = returns.dropna()
        
        return returns
    
    def align_data(
        self,
        prices: pd.DataFrame,
        min_observations: int = 252
    ) -> pd.DataFrame:
        """
        Align data across assets and ensure minimum observations.
        
        Parameters
        ----------
        prices : pd.DataFrame
            DataFrame with price data
        min_observations : int, default=252
            Minimum number of observations required
            
        Returns
        -------
        pd.DataFrame
            Aligned price data
        """
        # Remove columns with too many missing values
        missing_pct = prices.isnull().sum() / len(prices)
        valid_cols = missing_pct[missing_pct < 0.1].index
        prices = prices[valid_cols]
        
        # Drop rows with any missing values
        prices = prices.dropna()
        
        if len(prices) < min_observations:
            raise ValueError(
                f"Insufficient data: {len(prices)} observations, "
                f"minimum required: {min_observations}"
            )
        
        return prices
    
    def get_sample_tickers(self) -> List[str]:
        """
        Get a sample list of tickers for testing.
        
        Returns
        -------
        List[str]
            List of ticker symbols
        """
        # Sample of liquid, diverse assets
        return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'JPM', 'XOM', 'JNJ', 'PG', 'GLD', 'TLT']


def load_data_for_experiment(
    tickers: Optional[List[str]] = None,
    start_date: str = "2020-01-01",
    end_date: str = "2024-01-01",
    api_key: Optional[str] = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convenience function to load and preprocess data for experiments.
    
    Parameters
    ----------
    tickers : List[str], optional
        List of tickers. If None, uses sample tickers.
    start_date : str, default='2020-01-01'
        Start date for data
    end_date : str, default='2024-01-01'
        End date for data
    api_key : str, optional
        Massive API key
        
    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        Tuple of (prices, returns) DataFrames
    """
    loader = DataLoader(api_key=api_key)
    
    if tickers is None:
        tickers = loader.get_sample_tickers()
    
    prices = loader.fetch_price_data(tickers, start_date, end_date)
    prices = loader.align_data(prices)
    returns = loader.calculate_returns(prices)
    
    return prices, returns
