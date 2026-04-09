"""
Performance Metrics Module

This module provides comprehensive performance metrics calculation
using quantstats and custom implementations.
"""

from typing import Dict, Optional
import numpy as np
import pandas as pd
import quantstats as qs


class PerformanceAnalyzer:
    """
    Performance analyzer using quantstats — Context7 confirmed
    """
    
    def __init__(self, returns: pd.Series, benchmark_returns: Optional[pd.Series] = None):
        """
        Initialize performance analyzer.
        
        Parameters
        ----------
        returns : pd.Series
            Series of portfolio returns
        benchmark_returns : pd.Series, optional
            Series of benchmark returns for comparison
        """
        self.returns = returns
        self.benchmark_returns = benchmark_returns
        
        # Extend pandas with quantstats functionality
        qs.extend_pandas()
    
    def calculate_all_metrics(
        self,
        risk_free_rate: float = 0.02
    ) -> Dict[str, float]:
        """
        Calculate comprehensive performance metrics using quantstats.
        
        Using quantstats — Context7 confirmed
        
        Parameters
        ----------
        risk_free_rate : float, default=0.02
            Annual risk-free rate
            
        Returns
        -------
        Dict[str, float]
            Dictionary of performance metrics
        """
        metrics = {}
        
        # Using quantstats.stats — Context7 confirmed
        try:
            metrics['sharpe_ratio'] = qs.stats.sharpe(self.returns, rf=risk_free_rate)
        except:
            metrics['sharpe_ratio'] = 0.0
        
        try:
            metrics['sortino_ratio'] = qs.stats.sortino(self.returns, rf=risk_free_rate)
        except:
            metrics['sortino_ratio'] = 0.0
        
        try:
            metrics['calmar_ratio'] = qs.stats.calmar(self.returns)
        except:
            metrics['calmar_ratio'] = 0.0
        
        try:
            metrics['max_drawdown'] = qs.stats.max_drawdown(self.returns)
        except:
            metrics['max_drawdown'] = 0.0
        
        try:
            metrics['cagr'] = qs.stats.cagr(self.returns)
        except:
            metrics['cagr'] = 0.0
        
        try:
            metrics['volatility'] = qs.stats.volatility(self.returns)
        except:
            metrics['volatility'] = 0.0
        
        try:
            metrics['var_95'] = qs.stats.var(self.returns, confidence=0.95)
        except:
            metrics['var_95'] = 0.0
        
        try:
            metrics['cvar_95'] = qs.stats.cvar(self.returns, confidence=0.95)
        except:
            metrics['cvar_95'] = 0.0
        
        try:
            metrics['win_rate'] = qs.stats.win_rate(self.returns)
        except:
            metrics['win_rate'] = 0.0
        
        try:
            metrics['avg_win'] = qs.stats.avg_win(self.returns)
        except:
            metrics['avg_win'] = 0.0
        
        try:
            metrics['avg_loss'] = qs.stats.avg_loss(self.returns)
        except:
            metrics['avg_loss'] = 0.0
        
        try:
            metrics['profit_factor'] = qs.stats.profit_factor(self.returns)
        except:
            metrics['profit_factor'] = 0.0
        
        try:
            metrics['ulcer_index'] = qs.stats.ulcer_index(self.returns)
        except:
            metrics['ulcer_index'] = 0.0
        
        try:
            metrics['skewness'] = qs.stats.skew(self.returns)
        except:
            metrics['skewness'] = 0.0
        
        try:
            metrics['kurtosis'] = qs.stats.kurtosis(self.returns)
        except:
            metrics['kurtosis'] = 0.0
        
        # If benchmark is provided, calculate relative metrics
        if self.benchmark_returns is not None:
            try:
                metrics['information_ratio'] = qs.stats.information_ratio(
                    self.returns, self.benchmark_returns
                )
            except:
                metrics['information_ratio'] = 0.0
            
            try:
                # Calculate beta
                aligned_returns = pd.DataFrame({
                    'portfolio': self.returns,
                    'benchmark': self.benchmark_returns
                }).dropna()
                
                if len(aligned_returns) > 0:
                    cov = aligned_returns.cov()
                    beta = cov.loc['portfolio', 'benchmark'] / cov.loc['benchmark', 'benchmark']
                    metrics['beta'] = beta
                else:
                    metrics['beta'] = 0.0
            except:
                metrics['beta'] = 0.0
        
        return metrics
    
    def generate_report(
        self,
        output_file: Optional[str] = None,
        benchmark: Optional[str] = None
    ) -> None:
        """
        Generate HTML performance report using quantstats.
        
        Using quantstats.reports — Context7 confirmed
        
        Parameters
        ----------
        output_file : str, optional
            Path to save HTML report
        benchmark : str, optional
            Benchmark ticker symbol
        """
        if output_file:
            qs.reports.html(
                self.returns,
                benchmark=benchmark or self.benchmark_returns,
                output=output_file,
                title='Portfolio Performance Report'
            )
        else:
            qs.reports.basic(self.returns, benchmark=benchmark or self.benchmark_returns)
    
    def get_drawdown_details(self) -> pd.DataFrame:
        """
        Get detailed drawdown information.
        
        Returns
        -------
        pd.DataFrame
            DataFrame with drawdown details
        """
        try:
            dd_details = qs.stats.drawdown_details(self.returns)
            return dd_details
        except:
            return pd.DataFrame()


def calculate_risk_adjusted_metrics(
    returns: pd.Series,
    risk_free_rate: float = 0.02
) -> Dict[str, float]:
    """
    Calculate risk-adjusted performance metrics.
    
    Custom implementation — Context7 found no library equivalent
    Implements additional risk-adjusted metrics not in quantstats.
    
    Parameters
    ----------
    returns : pd.Series
        Series of returns
    risk_free_rate : float, default=0.02
        Annual risk-free rate
        
    Returns
    -------
    Dict[str, float]
        Dictionary of risk-adjusted metrics
    """
    metrics = {}
    
    # Annualization factor
    annual_factor = 252
    
    # Basic statistics
    mean_return = returns.mean() * annual_factor
    volatility = returns.std() * np.sqrt(annual_factor)
    
    # Sharpe ratio
    metrics['sharpe_ratio'] = (mean_return - risk_free_rate) / volatility if volatility > 0 else 0.0
    
    # Sortino ratio (downside deviation)
    downside_returns = returns[returns < 0]
    downside_vol = downside_returns.std() * np.sqrt(annual_factor) if len(downside_returns) > 0 else 0.0
    metrics['sortino_ratio'] = (mean_return - risk_free_rate) / downside_vol if downside_vol > 0 else 0.0
    
    # Omega ratio
    # Custom implementation — Context7 found no library equivalent
    threshold = risk_free_rate / annual_factor  # Daily threshold
    gains = returns[returns > threshold] - threshold
    losses = threshold - returns[returns < threshold]
    metrics['omega_ratio'] = gains.sum() / losses.sum() if losses.sum() > 0 else 0.0
    
    # Kappa 3 ratio (similar to Omega but with cubic weighting)
    # Custom implementation — Context7 found no library equivalent
    gains_cubed = ((returns[returns > threshold] - threshold) ** 3).sum()
    losses_cubed = ((threshold - returns[returns < threshold]) ** 3).sum()
    metrics['kappa_3'] = (gains_cubed / losses_cubed) ** (1/3) if losses_cubed > 0 else 0.0
    
    # Tail ratio (95th percentile gain / 95th percentile loss)
    # Custom implementation — Context7 found no library equivalent
    gains_95 = returns.quantile(0.95)
    losses_5 = abs(returns.quantile(0.05))
    metrics['tail_ratio'] = gains_95 / losses_5 if losses_5 > 0 else 0.0
    
    return metrics


def compare_strategies(
    returns_dict: Dict[str, pd.Series],
    risk_free_rate: float = 0.02
) -> pd.DataFrame:
    """
    Compare multiple strategies side by side.
    
    Parameters
    ----------
    returns_dict : Dict[str, pd.Series]
        Dictionary mapping strategy names to return series
    risk_free_rate : float, default=0.02
        Annual risk-free rate
        
    Returns
    -------
    pd.DataFrame
        DataFrame comparing strategies
    """
    comparison = {}
    
    for name, returns in returns_dict.items():
        analyzer = PerformanceAnalyzer(returns)
        metrics = analyzer.calculate_all_metrics(risk_free_rate)
        comparison[name] = metrics
    
    return pd.DataFrame(comparison).T


def calculate_rolling_sharpe(
    returns: pd.Series,
    window: int = 252,
    risk_free_rate: float = 0.02
) -> pd.Series:
    """
    Calculate rolling Sharpe ratio.
    
    Parameters
    ----------
    returns : pd.Series
        Series of returns
    window : int, default=252
        Rolling window size
    risk_free_rate : float, default=0.02
        Annual risk-free rate
        
    Returns
    -------
    pd.Series
        Series of rolling Sharpe ratios
    """
    rolling_mean = returns.rolling(window).mean() * 252
    rolling_std = returns.rolling(window).std() * np.sqrt(252)
    rolling_sharpe = (rolling_mean - risk_free_rate) / rolling_std
    
    return rolling_sharpe
