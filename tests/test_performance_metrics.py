"""
Tests for performance metrics module
"""

import numpy as np
import pandas as pd
import pytest
from src.performance_metrics import (
    PerformanceAnalyzer,
    calculate_risk_adjusted_metrics,
    compare_strategies,
    calculate_rolling_sharpe
)


def create_sample_returns(n_periods=252, mean=0.0005, std=0.01):
    """Create sample returns for testing."""
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=n_periods, freq='D')
    returns = pd.Series(
        np.random.randn(n_periods) * std + mean,
        index=dates,
        name='returns'
    )
    return returns


def test_performance_analyzer_initialization():
    """Test PerformanceAnalyzer initialization."""
    returns = create_sample_returns()
    analyzer = PerformanceAnalyzer(returns)
    
    assert len(analyzer.returns) == len(returns)
    assert analyzer.benchmark_returns is None


def test_performance_analyzer_with_benchmark():
    """Test PerformanceAnalyzer with benchmark."""
    returns = create_sample_returns()
    benchmark = create_sample_returns(mean=0.0003)
    
    analyzer = PerformanceAnalyzer(returns, benchmark)
    
    assert analyzer.benchmark_returns is not None
    assert len(analyzer.benchmark_returns) == len(benchmark)


def test_calculate_all_metrics():
    """Test calculation of all performance metrics."""
    returns = create_sample_returns()
    analyzer = PerformanceAnalyzer(returns)
    
    metrics = analyzer.calculate_all_metrics()
    
    # Verify key metrics are present
    expected_metrics = [
        'sharpe_ratio', 'sortino_ratio', 'calmar_ratio',
        'max_drawdown', 'cagr', 'volatility'
    ]
    
    for metric in expected_metrics:
        assert metric in metrics
        assert isinstance(metrics[metric], (int, float))


def test_metrics_with_benchmark():
    """Test metrics calculation with benchmark."""
    returns = create_sample_returns()
    benchmark = create_sample_returns(mean=0.0003)
    
    analyzer = PerformanceAnalyzer(returns, benchmark)
    metrics = analyzer.calculate_all_metrics()
    
    # Should include benchmark-relative metrics
    assert 'information_ratio' in metrics or 'beta' in metrics


def test_calculate_risk_adjusted_metrics():
    """Test custom risk-adjusted metrics calculation."""
    returns = create_sample_returns()
    
    metrics = calculate_risk_adjusted_metrics(returns)
    
    expected_metrics = [
        'sharpe_ratio', 'sortino_ratio', 'omega_ratio',
        'kappa_3', 'tail_ratio'
    ]
    
    for metric in expected_metrics:
        assert metric in metrics
        assert isinstance(metrics[metric], (int, float))


def test_sharpe_ratio_calculation():
    """Test Sharpe ratio calculation."""
    # Create returns with known properties
    returns = create_sample_returns(mean=0.001, std=0.01)
    
    metrics = calculate_risk_adjusted_metrics(returns, risk_free_rate=0.02)
    
    # Sharpe ratio should be finite
    assert np.isfinite(metrics['sharpe_ratio'])


def test_sortino_ratio_calculation():
    """Test Sortino ratio calculation."""
    returns = create_sample_returns()
    
    metrics = calculate_risk_adjusted_metrics(returns)
    
    # Sortino ratio should be finite
    assert np.isfinite(metrics['sortino_ratio'])


def test_omega_ratio_calculation():
    """Test Omega ratio calculation."""
    returns = create_sample_returns()
    
    metrics = calculate_risk_adjusted_metrics(returns)
    
    # Omega ratio should be positive
    assert metrics['omega_ratio'] >= 0


def test_tail_ratio_calculation():
    """Test tail ratio calculation."""
    returns = create_sample_returns()
    
    metrics = calculate_risk_adjusted_metrics(returns)
    
    # Tail ratio should be positive
    assert metrics['tail_ratio'] >= 0


def test_compare_strategies():
    """Test strategy comparison function."""
    returns1 = create_sample_returns(mean=0.0005)
    returns2 = create_sample_returns(mean=0.0007)
    returns3 = create_sample_returns(mean=0.0003)
    
    returns_dict = {
        'Strategy1': returns1,
        'Strategy2': returns2,
        'Strategy3': returns3
    }
    
    comparison = compare_strategies(returns_dict)
    
    assert isinstance(comparison, pd.DataFrame)
    assert len(comparison) == 3
    assert 'Strategy1' in comparison.index
    assert 'sharpe_ratio' in comparison.columns


def test_calculate_rolling_sharpe():
    """Test rolling Sharpe ratio calculation."""
    returns = create_sample_returns(n_periods=500)
    
    rolling_sharpe = calculate_rolling_sharpe(returns, window=100)
    
    assert isinstance(rolling_sharpe, pd.Series)
    assert len(rolling_sharpe) == len(returns)
    
    # First values should be NaN due to window
    assert rolling_sharpe.iloc[:99].isna().all()
    
    # Later values should be finite
    assert rolling_sharpe.iloc[100:].notna().any()


def test_rolling_sharpe_different_windows():
    """Test rolling Sharpe with different window sizes."""
    returns = create_sample_returns(n_periods=500)
    
    for window in [50, 100, 252]:
        rolling_sharpe = calculate_rolling_sharpe(returns, window=window)
        
        # First (window-1) values should be NaN
        assert rolling_sharpe.iloc[:window-1].isna().all()


def test_metrics_with_positive_returns():
    """Test metrics with consistently positive returns."""
    returns = create_sample_returns(mean=0.001, std=0.005)
    
    metrics = calculate_risk_adjusted_metrics(returns)
    
    # Sharpe should be positive with positive mean returns
    assert metrics['sharpe_ratio'] > 0


def test_metrics_with_negative_returns():
    """Test metrics with consistently negative returns."""
    returns = create_sample_returns(mean=-0.001, std=0.005)
    
    metrics = calculate_risk_adjusted_metrics(returns)
    
    # Sharpe should be negative with negative mean returns
    assert metrics['sharpe_ratio'] < 0


def test_metrics_with_zero_volatility():
    """Test metrics with zero volatility (edge case)."""
    # Create constant returns
    dates = pd.date_range('2020-01-01', periods=252, freq='D')
    returns = pd.Series(0.001, index=dates)
    
    # Should handle gracefully without division by zero
    metrics = calculate_risk_adjusted_metrics(returns)
    
    # Metrics should be calculated (may be inf or very large)
    assert 'sharpe_ratio' in metrics


def test_metrics_with_single_observation():
    """Test metrics with single observation."""
    dates = pd.date_range('2020-01-01', periods=1, freq='D')
    returns = pd.Series([0.01], index=dates)
    
    # Should handle gracefully
    metrics = calculate_risk_adjusted_metrics(returns)
    
    assert isinstance(metrics, dict)


def test_metrics_with_extreme_values():
    """Test metrics with extreme return values."""
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=252, freq='D')
    returns = pd.Series(np.random.randn(252) * 0.1, index=dates)  # High volatility
    
    metrics = calculate_risk_adjusted_metrics(returns)
    
    # Should handle extreme values
    assert np.isfinite(metrics['sharpe_ratio'])
    assert np.isfinite(metrics['sortino_ratio'])


def test_compare_strategies_empty():
    """Test strategy comparison with empty dict."""
    comparison = compare_strategies({})
    
    assert isinstance(comparison, pd.DataFrame)
    assert len(comparison) == 0


def test_compare_strategies_single():
    """Test strategy comparison with single strategy."""
    returns = create_sample_returns()
    
    comparison = compare_strategies({'Strategy1': returns})
    
    assert len(comparison) == 1
    assert 'Strategy1' in comparison.index


def test_performance_analyzer_edge_cases():
    """Test PerformanceAnalyzer with edge cases."""
    # Very short returns series
    dates = pd.date_range('2020-01-01', periods=10, freq='D')
    returns = pd.Series(np.random.randn(10) * 0.01, index=dates)
    
    analyzer = PerformanceAnalyzer(returns)
    metrics = analyzer.calculate_all_metrics()
    
    # Should return metrics even with short series
    assert isinstance(metrics, dict)
    assert len(metrics) > 0


def test_different_risk_free_rates():
    """Test metrics with different risk-free rates."""
    returns = create_sample_returns()
    
    for rf in [0.0, 0.02, 0.05]:
        metrics = calculate_risk_adjusted_metrics(returns, risk_free_rate=rf)
        
        assert 'sharpe_ratio' in metrics
        assert np.isfinite(metrics['sharpe_ratio'])


def test_kappa_3_calculation():
    """Test Kappa 3 ratio calculation."""
    returns = create_sample_returns()
    
    metrics = calculate_risk_adjusted_metrics(returns)
    
    # Kappa 3 should be calculated
    assert 'kappa_3' in metrics
    assert isinstance(metrics['kappa_3'], (int, float))


def test_metrics_consistency():
    """Test that metrics are consistent across multiple calls."""
    returns = create_sample_returns()
    
    metrics1 = calculate_risk_adjusted_metrics(returns)
    metrics2 = calculate_risk_adjusted_metrics(returns)
    
    # Should produce identical results
    for key in metrics1:
        assert metrics1[key] == metrics2[key]


def test_rolling_sharpe_with_risk_free_rate():
    """Test rolling Sharpe with different risk-free rates."""
    returns = create_sample_returns(n_periods=500)
    
    for rf in [0.0, 0.02, 0.05]:
        rolling_sharpe = calculate_rolling_sharpe(returns, window=100, risk_free_rate=rf)
        
        assert isinstance(rolling_sharpe, pd.Series)
        assert len(rolling_sharpe) == len(returns)
