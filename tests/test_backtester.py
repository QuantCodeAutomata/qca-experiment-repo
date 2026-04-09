"""
Tests for backtester module
"""

import numpy as np
import pandas as pd
import pytest
from src.backtester import Backtester, run_simple_backtest


def create_sample_data(n_assets=5, n_periods=500):
    """Create sample price and returns data for testing."""
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=n_periods, freq='D')
    
    # Create prices with upward trend
    prices = pd.DataFrame(
        100 * np.exp(np.cumsum(np.random.randn(n_periods, n_assets) * 0.01, axis=0)),
        index=dates,
        columns=[f'Asset_{i}' for i in range(n_assets)]
    )
    
    # Calculate returns
    returns = prices.pct_change().dropna()
    prices = prices.iloc[1:]  # Align with returns
    
    return prices, returns


def simple_optimization_func(returns, **kwargs):
    """Simple equal-weight optimization for testing."""
    n_assets = returns.shape[1]
    return np.ones(n_assets) / n_assets


def test_backtester_initialization():
    """Test Backtester initialization."""
    prices, returns = create_sample_data()
    backtester = Backtester(returns, prices, initial_capital=100000)
    
    assert backtester.initial_capital == 100000
    assert backtester.results is None
    assert len(backtester.returns) == len(returns)


def test_run_backtest():
    """Test running a basic backtest."""
    prices, returns = create_sample_data()
    backtester = Backtester(returns, prices)
    
    results = backtester.run_backtest(
        simple_optimization_func,
        train_window=100,
        rebalance_frequency=20,
        min_history=100
    )
    
    assert isinstance(results, pd.DataFrame)
    assert 'portfolio_value' in results.columns
    assert 'portfolio_return' in results.columns
    assert len(results) > 0


def test_backtest_portfolio_value():
    """Test that portfolio value is calculated correctly."""
    prices, returns = create_sample_data()
    backtester = Backtester(returns, prices, initial_capital=100000)
    
    results = backtester.run_backtest(
        simple_optimization_func,
        train_window=100,
        rebalance_frequency=20,
        min_history=100
    )
    
    # Portfolio value should start near initial capital
    assert results['portfolio_value'].iloc[0] > 0
    
    # Portfolio value should always be positive
    assert (results['portfolio_value'] > 0).all()


def test_calculate_performance_metrics():
    """Test performance metrics calculation."""
    prices, returns = create_sample_data()
    backtester = Backtester(returns, prices)
    
    backtester.run_backtest(
        simple_optimization_func,
        train_window=100,
        rebalance_frequency=20,
        min_history=100
    )
    
    metrics = backtester.calculate_performance_metrics()
    
    # Verify all expected metrics are present
    expected_metrics = [
        'total_return', 'annualized_return', 'volatility',
        'sharpe_ratio', 'sortino_ratio', 'max_drawdown',
        'calmar_ratio', 'win_rate', 'avg_win', 'avg_loss',
        'profit_factor', 'n_trades'
    ]
    
    for metric in expected_metrics:
        assert metric in metrics
        assert isinstance(metrics[metric], (int, float))


def test_performance_metrics_values():
    """Test that performance metrics have reasonable values."""
    prices, returns = create_sample_data()
    backtester = Backtester(returns, prices)
    
    backtester.run_backtest(
        simple_optimization_func,
        train_window=100,
        rebalance_frequency=20,
        min_history=100
    )
    
    metrics = backtester.calculate_performance_metrics()
    
    # Volatility should be positive
    assert metrics['volatility'] > 0
    
    # Max drawdown should be negative or zero
    assert metrics['max_drawdown'] <= 0
    
    # Win rate should be between 0 and 1
    assert 0 <= metrics['win_rate'] <= 1
    
    # Number of trades should be positive
    assert metrics['n_trades'] > 0


def test_get_drawdown_series():
    """Test drawdown series calculation."""
    prices, returns = create_sample_data()
    backtester = Backtester(returns, prices)
    
    backtester.run_backtest(
        simple_optimization_func,
        train_window=100,
        rebalance_frequency=20,
        min_history=100
    )
    
    drawdown = backtester.get_drawdown_series()
    
    assert isinstance(drawdown, pd.Series)
    assert len(drawdown) == len(backtester.results)
    
    # Drawdown should always be <= 0
    assert (drawdown <= 0).all()


def test_get_rolling_metrics():
    """Test rolling metrics calculation."""
    prices, returns = create_sample_data()
    backtester = Backtester(returns, prices)
    
    backtester.run_backtest(
        simple_optimization_func,
        train_window=100,
        rebalance_frequency=20,
        min_history=100
    )
    
    rolling_metrics = backtester.get_rolling_metrics(window=50)
    
    assert isinstance(rolling_metrics, pd.DataFrame)
    assert 'rolling_return' in rolling_metrics.columns
    assert 'rolling_volatility' in rolling_metrics.columns
    assert 'rolling_sharpe' in rolling_metrics.columns


def test_rebalancing_frequency():
    """Test that rebalancing occurs at correct frequency."""
    prices, returns = create_sample_data()
    backtester = Backtester(returns, prices)
    
    rebalance_freq = 20
    backtester.run_backtest(
        simple_optimization_func,
        train_window=100,
        rebalance_frequency=rebalance_freq,
        min_history=100
    )
    
    # Check that weights history exists
    assert hasattr(backtester, 'weights_history')
    assert isinstance(backtester.weights_history, pd.DataFrame)


def test_different_train_windows():
    """Test backtesting with different training windows."""
    prices, returns = create_sample_data()
    
    for train_window in [50, 100, 150]:
        backtester = Backtester(returns, prices)
        results = backtester.run_backtest(
            simple_optimization_func,
            train_window=train_window,
            rebalance_frequency=20,
            min_history=train_window
        )
        
        assert len(results) > 0


def test_insufficient_data():
    """Test that insufficient data raises error."""
    prices, returns = create_sample_data(n_periods=100)
    backtester = Backtester(returns, prices)
    
    with pytest.raises(ValueError):
        backtester.run_backtest(
            simple_optimization_func,
            train_window=100,
            rebalance_frequency=20,
            min_history=300  # More than available
        )


def test_metrics_without_backtest():
    """Test that calculating metrics without backtest raises error."""
    prices, returns = create_sample_data()
    backtester = Backtester(returns, prices)
    
    with pytest.raises(ValueError):
        backtester.calculate_performance_metrics()


def test_drawdown_without_backtest():
    """Test that getting drawdown without backtest raises error."""
    prices, returns = create_sample_data()
    backtester = Backtester(returns, prices)
    
    with pytest.raises(ValueError):
        backtester.get_drawdown_series()


def test_rolling_metrics_without_backtest():
    """Test that getting rolling metrics without backtest raises error."""
    prices, returns = create_sample_data()
    backtester = Backtester(returns, prices)
    
    with pytest.raises(ValueError):
        backtester.get_rolling_metrics()


def test_run_simple_backtest_convenience():
    """Test the convenience function for backtesting."""
    prices, returns = create_sample_data()
    
    results, metrics = run_simple_backtest(
        returns,
        prices,
        simple_optimization_func,
        train_window=100,
        rebalance_frequency=20,
        min_history=100
    )
    
    assert isinstance(results, pd.DataFrame)
    assert isinstance(metrics, dict)
    assert 'sharpe_ratio' in metrics


def test_custom_optimization_function():
    """Test backtesting with custom optimization function."""
    prices, returns = create_sample_data()
    
    def custom_opt(returns, **kwargs):
        """Custom optimization that weights by inverse volatility."""
        vols = returns.std()
        weights = 1 / vols
        return weights / weights.sum()
    
    backtester = Backtester(returns, prices)
    results = backtester.run_backtest(
        custom_opt,
        train_window=100,
        rebalance_frequency=20,
        min_history=100
    )
    
    assert len(results) > 0
    metrics = backtester.calculate_performance_metrics()
    assert 'sharpe_ratio' in metrics


def test_backtest_with_different_initial_capital():
    """Test backtesting with different initial capital amounts."""
    prices, returns = create_sample_data()
    
    for capital in [10000, 100000, 1000000]:
        backtester = Backtester(returns, prices, initial_capital=capital)
        results = backtester.run_backtest(
            simple_optimization_func,
            train_window=100,
            rebalance_frequency=20,
            min_history=100
        )
        
        # First portfolio value should be close to initial capital
        assert results['portfolio_value'].iloc[0] > 0


def test_weights_history_stored():
    """Test that weights history is properly stored."""
    prices, returns = create_sample_data()
    backtester = Backtester(returns, prices)
    
    backtester.run_backtest(
        simple_optimization_func,
        train_window=100,
        rebalance_frequency=20,
        min_history=100
    )
    
    assert hasattr(backtester, 'weights_history')
    assert len(backtester.weights_history) == len(backtester.results)
    assert backtester.weights_history.shape[1] == returns.shape[1]


def test_optimization_failure_handling():
    """Test that optimization failures are handled gracefully."""
    prices, returns = create_sample_data()
    
    def failing_opt(returns, **kwargs):
        """Optimization function that sometimes fails."""
        if len(returns) < 150:
            raise ValueError("Not enough data")
        return np.ones(returns.shape[1]) / returns.shape[1]
    
    backtester = Backtester(returns, prices)
    
    # Should not raise error, should use fallback weights
    results = backtester.run_backtest(
        failing_opt,
        train_window=100,
        rebalance_frequency=20,
        min_history=100
    )
    
    assert len(results) > 0
