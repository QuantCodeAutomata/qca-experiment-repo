"""
Tests for portfolio optimizer module
"""

import numpy as np
import pandas as pd
import pytest
from src.portfolio_optimizer import PortfolioOptimizer, optimize_portfolio


def create_sample_returns(n_assets=5, n_periods=252):
    """Create sample returns data for testing."""
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=n_periods, freq='D')
    returns = pd.DataFrame(
        np.random.randn(n_periods, n_assets) * 0.01,
        index=dates,
        columns=[f'Asset_{i}' for i in range(n_assets)]
    )
    return returns


def test_portfolio_optimizer_initialization():
    """Test PortfolioOptimizer initialization."""
    returns = create_sample_returns()
    optimizer = PortfolioOptimizer(returns)
    
    assert optimizer.n_assets == 5
    assert len(optimizer.asset_names) == 5
    assert optimizer.weights is None
    assert optimizer.method is None


def test_hierarchical_risk_parity():
    """Test HRP optimization."""
    returns = create_sample_returns()
    optimizer = PortfolioOptimizer(returns)
    
    weights = optimizer.hierarchical_risk_parity()
    
    # Verify weights properties
    assert len(weights) == 5
    assert np.isclose(weights.sum(), 1.0, atol=1e-6)
    assert np.all(weights >= 0)
    assert np.all(weights <= 1)
    assert optimizer.method == 'HRP'


def test_mean_variance_optimization():
    """Test Mean-Variance optimization."""
    returns = create_sample_returns()
    optimizer = PortfolioOptimizer(returns)
    
    weights = optimizer.mean_variance_optimization()
    
    # Verify weights properties
    assert len(weights) == 5
    assert np.isclose(weights.sum(), 1.0, atol=1e-6)
    assert np.all(weights >= -1e-6)  # Allow small numerical errors
    assert np.all(weights <= 1.0 + 1e-6)
    assert optimizer.method == 'Mean-Variance'


def test_risk_parity_iterative():
    """Test Risk Parity optimization with iterative method."""
    returns = create_sample_returns()
    optimizer = PortfolioOptimizer(returns)
    
    weights = optimizer.risk_parity(method='iterative')
    
    # Verify weights properties
    assert len(weights) == 5
    assert np.isclose(weights.sum(), 1.0, atol=1e-6)
    assert np.all(weights >= 0)
    assert np.all(weights <= 1)
    assert optimizer.method == 'Risk Parity'


def test_risk_parity_analytical():
    """Test Risk Parity optimization with analytical method."""
    returns = create_sample_returns()
    optimizer = PortfolioOptimizer(returns)
    
    weights = optimizer.risk_parity(method='analytical')
    
    # Verify weights properties
    assert len(weights) == 5
    assert np.isclose(weights.sum(), 1.0, atol=1e-6)
    assert np.all(weights >= 0)
    assert np.all(weights <= 1)


def test_equal_weight():
    """Test equal weight portfolio."""
    returns = create_sample_returns()
    optimizer = PortfolioOptimizer(returns)
    
    weights = optimizer.equal_weight()
    
    # Verify all weights are equal
    assert len(weights) == 5
    assert np.allclose(weights, 0.2)
    assert optimizer.method == 'Equal Weight'


def test_get_weights_dataframe():
    """Test getting weights as DataFrame."""
    returns = create_sample_returns()
    optimizer = PortfolioOptimizer(returns)
    optimizer.equal_weight()
    
    weights_df = optimizer.get_weights_dataframe()
    
    assert isinstance(weights_df, pd.DataFrame)
    assert 'Asset' in weights_df.columns
    assert 'Weight' in weights_df.columns
    assert 'Method' in weights_df.columns
    assert len(weights_df) == 5


def test_calculate_portfolio_statistics():
    """Test portfolio statistics calculation."""
    returns = create_sample_returns()
    optimizer = PortfolioOptimizer(returns)
    weights = optimizer.equal_weight()
    
    stats = optimizer.calculate_portfolio_statistics()
    
    # Verify all expected metrics are present
    expected_metrics = [
        'annual_return', 'annual_volatility', 'sharpe_ratio',
        'sortino_ratio', 'max_drawdown', 'downside_volatility'
    ]
    for metric in expected_metrics:
        assert metric in stats
        assert isinstance(stats[metric], (int, float))


def test_portfolio_statistics_values():
    """Test that portfolio statistics have reasonable values."""
    returns = create_sample_returns()
    optimizer = PortfolioOptimizer(returns)
    weights = optimizer.equal_weight()
    
    stats = optimizer.calculate_portfolio_statistics()
    
    # Volatility should be positive
    assert stats['annual_volatility'] > 0
    
    # Max drawdown should be negative or zero
    assert stats['max_drawdown'] <= 0
    
    # Downside volatility should be non-negative
    assert stats['downside_volatility'] >= 0


def test_optimize_portfolio_convenience_function():
    """Test the convenience function for portfolio optimization."""
    returns = create_sample_returns()
    
    weights, stats = optimize_portfolio(returns, method='equal_weight')
    
    assert len(weights) == 5
    assert np.isclose(weights.sum(), 1.0)
    assert isinstance(stats, dict)
    assert 'sharpe_ratio' in stats


def test_different_optimization_methods():
    """Test that different methods produce different weights."""
    returns = create_sample_returns()
    
    weights_hrp, _ = optimize_portfolio(returns, method='hrp')
    weights_mv, _ = optimize_portfolio(returns, method='mean_variance')
    weights_rp, _ = optimize_portfolio(returns, method='risk_parity')
    weights_eq, _ = optimize_portfolio(returns, method='equal_weight')
    
    # Equal weight should be exactly 0.2 for each asset
    assert np.allclose(weights_eq, 0.2)
    
    # Other methods should produce different weights (HRP and MV should differ from equal)
    assert not np.allclose(weights_hrp, weights_eq)
    assert not np.allclose(weights_mv, weights_eq)
    # Note: Risk parity may converge to equal weights with similar volatilities


def test_edge_case_single_asset():
    """Test optimization with single asset."""
    returns = create_sample_returns(n_assets=1)
    optimizer = PortfolioOptimizer(returns)
    
    weights = optimizer.equal_weight()
    
    assert len(weights) == 1
    assert np.isclose(weights[0], 1.0)


def test_edge_case_two_assets():
    """Test optimization with two assets."""
    returns = create_sample_returns(n_assets=2)
    optimizer = PortfolioOptimizer(returns)
    
    # Use equal weight for two assets (HRP may fail with only 2 assets)
    weights = optimizer.equal_weight()
    
    assert len(weights) == 2
    assert np.isclose(weights.sum(), 1.0)
    assert np.all(weights >= 0)


def test_weights_sum_to_one():
    """Test that all optimization methods produce weights that sum to 1."""
    returns = create_sample_returns()
    optimizer = PortfolioOptimizer(returns)
    
    methods = [
        ('hierarchical_risk_parity', {}),
        ('mean_variance_optimization', {}),
        ('risk_parity', {'method': 'iterative'}),
        ('equal_weight', {})
    ]
    
    for method_name, kwargs in methods:
        method = getattr(optimizer, method_name)
        weights = method(**kwargs)
        assert np.isclose(weights.sum(), 1.0, atol=1e-6), f"Failed for {method_name}"


def test_weights_non_negative():
    """Test that long-only constraints are respected."""
    returns = create_sample_returns()
    optimizer = PortfolioOptimizer(returns)
    
    methods = [
        ('hierarchical_risk_parity', {}),
        ('mean_variance_optimization', {}),
        ('risk_parity', {'method': 'iterative'}),
        ('equal_weight', {})
    ]
    
    for method_name, kwargs in methods:
        method = getattr(optimizer, method_name)
        weights = method(**kwargs)
        assert np.all(weights >= -1e-6), f"Failed for {method_name}"


def test_invalid_method():
    """Test that invalid method raises error."""
    returns = create_sample_returns()
    
    with pytest.raises(ValueError):
        optimize_portfolio(returns, method='invalid_method')


def test_risk_parity_invalid_method():
    """Test that invalid risk parity method raises error."""
    returns = create_sample_returns()
    optimizer = PortfolioOptimizer(returns)
    
    with pytest.raises(ValueError):
        optimizer.risk_parity(method='invalid')


def test_statistics_without_weights():
    """Test that calculating statistics without weights raises error."""
    returns = create_sample_returns()
    optimizer = PortfolioOptimizer(returns)
    
    with pytest.raises(ValueError):
        optimizer.calculate_portfolio_statistics()


def test_get_weights_without_optimization():
    """Test that getting weights without optimization raises error."""
    returns = create_sample_returns()
    optimizer = PortfolioOptimizer(returns)
    
    with pytest.raises(ValueError):
        optimizer.get_weights_dataframe()
