"""
Backtester Module

This module implements a backtesting framework for portfolio strategies.
"""

from typing import Optional, Dict, List, Callable
import numpy as np
import pandas as pd
from datetime import datetime


class Backtester:
    """
    Backtesting framework for portfolio strategies.
    
    Supports rolling window optimization and rebalancing.
    """
    
    def __init__(
        self,
        returns: pd.DataFrame,
        prices: pd.DataFrame,
        initial_capital: float = 100000.0
    ):
        """
        Initialize the backtester.
        
        Parameters
        ----------
        returns : pd.DataFrame
            DataFrame of asset returns
        prices : pd.DataFrame
            DataFrame of asset prices
        initial_capital : float, default=100000.0
            Initial portfolio capital
        """
        self.returns = returns
        self.prices = prices
        self.initial_capital = initial_capital
        self.results = None
    
    def run_backtest(
        self,
        optimization_func: Callable,
        train_window: int = 252,
        rebalance_frequency: int = 21,
        min_history: int = 252,
        **opt_kwargs
    ) -> pd.DataFrame:
        """
        Run backtest with rolling window optimization.
        
        Parameters
        ----------
        optimization_func : Callable
            Function that takes returns DataFrame and returns weights array
        train_window : int, default=252
            Number of days to use for training (lookback window)
        rebalance_frequency : int, default=21
            Number of days between rebalancing (21 ≈ monthly)
        min_history : int, default=252
            Minimum history required before starting backtest
        **opt_kwargs
            Additional keyword arguments passed to optimization_func
            
        Returns
        -------
        pd.DataFrame
            DataFrame with backtest results including portfolio value and returns
        """
        # Ensure we have enough data
        if len(self.returns) < min_history:
            raise ValueError(
                f"Insufficient data: {len(self.returns)} observations, "
                f"minimum required: {min_history}"
            )
        
        # Initialize results storage
        portfolio_values = []
        portfolio_returns = []
        weights_history = []
        dates = []
        
        # Start after minimum history
        start_idx = min_history
        current_value = self.initial_capital
        
        # Track current weights
        current_weights = None
        last_rebalance_idx = start_idx
        
        for i in range(start_idx, len(self.returns)):
            current_date = self.returns.index[i]
            
            # Check if we need to rebalance
            if current_weights is None or (i - last_rebalance_idx) >= rebalance_frequency:
                # Get training data
                train_start = max(0, i - train_window)
                train_returns = self.returns.iloc[train_start:i]
                
                # Optimize portfolio
                try:
                    current_weights = optimization_func(train_returns, **opt_kwargs)
                    last_rebalance_idx = i
                except Exception as e:
                    print(f"Optimization failed at {current_date}: {e}")
                    # Keep previous weights or use equal weights
                    if current_weights is None:
                        current_weights = np.ones(self.returns.shape[1]) / self.returns.shape[1]
            
            # Calculate portfolio return for this period
            period_returns = self.returns.iloc[i].values
            portfolio_return = np.dot(current_weights, period_returns)
            
            # Update portfolio value
            current_value = current_value * (1 + portfolio_return)
            
            # Store results
            portfolio_values.append(current_value)
            portfolio_returns.append(portfolio_return)
            weights_history.append(current_weights.copy())
            dates.append(current_date)
        
        # Create results DataFrame
        self.results = pd.DataFrame({
            'date': dates,
            'portfolio_value': portfolio_values,
            'portfolio_return': portfolio_returns
        })
        self.results.set_index('date', inplace=True)
        
        # Store weights history
        self.weights_history = pd.DataFrame(
            weights_history,
            index=dates,
            columns=self.returns.columns
        )
        
        return self.results
    
    def calculate_performance_metrics(
        self,
        risk_free_rate: float = 0.02
    ) -> Dict[str, float]:
        """
        Calculate comprehensive performance metrics.
        
        Parameters
        ----------
        risk_free_rate : float, default=0.02
            Annual risk-free rate
            
        Returns
        -------
        Dict[str, float]
            Dictionary of performance metrics
        """
        if self.results is None:
            raise ValueError("No backtest results available. Run backtest first.")
        
        returns = self.results['portfolio_return']
        values = self.results['portfolio_value']
        
        # Annualization factor
        annual_factor = 252
        
        # Total return
        total_return = (values.iloc[-1] - self.initial_capital) / self.initial_capital
        
        # Annualized return
        n_years = len(returns) / annual_factor
        annualized_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0.0
        
        # Volatility
        volatility = returns.std() * np.sqrt(annual_factor)
        
        # Sharpe ratio
        excess_return = annualized_return - risk_free_rate
        sharpe_ratio = excess_return / volatility if volatility > 0 else 0.0
        
        # Sortino ratio
        downside_returns = returns[returns < 0]
        downside_vol = downside_returns.std() * np.sqrt(annual_factor) if len(downside_returns) > 0 else 0.0
        sortino_ratio = excess_return / downside_vol if downside_vol > 0 else 0.0
        
        # Maximum drawdown
        cumulative = values / self.initial_capital
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Calmar ratio
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0.0
        
        # Win rate
        win_rate = (returns > 0).sum() / len(returns)
        
        # Average win/loss
        wins = returns[returns > 0]
        losses = returns[returns < 0]
        avg_win = wins.mean() if len(wins) > 0 else 0.0
        avg_loss = losses.mean() if len(losses) > 0 else 0.0
        
        # Profit factor
        total_wins = wins.sum() if len(wins) > 0 else 0.0
        total_losses = abs(losses.sum()) if len(losses) > 0 else 0.0
        profit_factor = total_wins / total_losses if total_losses > 0 else 0.0
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'n_trades': len(returns)
        }
    
    def get_drawdown_series(self) -> pd.Series:
        """
        Get drawdown series over time.
        
        Returns
        -------
        pd.Series
            Series of drawdown values
        """
        if self.results is None:
            raise ValueError("No backtest results available. Run backtest first.")
        
        values = self.results['portfolio_value']
        cumulative = values / self.initial_capital
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        
        return drawdown
    
    def get_rolling_metrics(
        self,
        window: int = 252,
        risk_free_rate: float = 0.02
    ) -> pd.DataFrame:
        """
        Calculate rolling performance metrics.
        
        Parameters
        ----------
        window : int, default=252
            Rolling window size in days
        risk_free_rate : float, default=0.02
            Annual risk-free rate
            
        Returns
        -------
        pd.DataFrame
            DataFrame with rolling metrics
        """
        if self.results is None:
            raise ValueError("No backtest results available. Run backtest first.")
        
        returns = self.results['portfolio_return']
        
        # Rolling return
        rolling_return = returns.rolling(window).mean() * 252
        
        # Rolling volatility
        rolling_vol = returns.rolling(window).std() * np.sqrt(252)
        
        # Rolling Sharpe
        rolling_sharpe = (rolling_return - risk_free_rate) / rolling_vol
        
        return pd.DataFrame({
            'rolling_return': rolling_return,
            'rolling_volatility': rolling_vol,
            'rolling_sharpe': rolling_sharpe
        })


def run_simple_backtest(
    returns: pd.DataFrame,
    prices: pd.DataFrame,
    optimization_func: Callable,
    initial_capital: float = 100000.0,
    **backtest_kwargs
) -> tuple[pd.DataFrame, Dict[str, float]]:
    """
    Convenience function to run backtest and return results and metrics.
    
    Parameters
    ----------
    returns : pd.DataFrame
        DataFrame of asset returns
    prices : pd.DataFrame
        DataFrame of asset prices
    optimization_func : Callable
        Optimization function
    initial_capital : float, default=100000.0
        Initial capital
    **backtest_kwargs
        Additional arguments for backtesting
        
    Returns
    -------
    tuple[pd.DataFrame, Dict[str, float]]
        Tuple of (results DataFrame, performance metrics)
    """
    backtester = Backtester(returns, prices, initial_capital)
    results = backtester.run_backtest(optimization_func, **backtest_kwargs)
    metrics = backtester.calculate_performance_metrics()
    
    return results, metrics
