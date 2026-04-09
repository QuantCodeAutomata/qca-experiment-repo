# Repository Knowledge Base

## Overview
This repository implements a comprehensive quantitative finance experiment for multi-asset portfolio optimization and backtesting.

## Architecture

### Core Modules
1. **data_loader.py** - Data retrieval and preprocessing using Massive API
2. **portfolio_optimizer.py** - Portfolio optimization strategies (HRP, Mean-Variance, Risk Parity)
3. **backtester.py** - Backtesting framework with rolling window optimization
4. **performance_metrics.py** - Performance metrics using quantstats
5. **visualizations.py** - Plotting functions using matplotlib/seaborn

### Key Design Decisions

#### Library Usage (Context7 Verified)
- **skfolio**: Used for HRP and Mean-Variance optimization (Context7 confirmed)
- **quantstats**: Used for performance metrics calculation (Context7 confirmed)
- **Custom implementations**: Risk Parity, Omega ratio, Kappa 3 (Context7 found no library equivalent)

#### Data Flow
1. Data retrieval → Preprocessing → Optimization → Backtesting → Analysis → Visualization
2. All modules are independent and can be used separately
3. Convenience functions provided for common workflows

#### Testing Strategy
- Comprehensive unit tests for all modules
- Edge case testing (single asset, extreme values, insufficient data)
- Property-based testing (weights sum to 1, non-negative, etc.)
- No mocking - tests use synthetic data

## Implementation Notes

### Portfolio Optimization
- All optimizers return weights that sum to 1.0
- Long-only constraints enforced (weights >= 0)
- Multiple risk measures supported (variance, semi-deviation, CVaR, CDaR)

### Backtesting
- Rolling window optimization with configurable rebalancing frequency
- Handles optimization failures gracefully with fallback to equal weights
- Stores weights history for analysis

### Performance Metrics
- Uses quantstats for standard metrics
- Custom implementations for advanced metrics (Omega, Kappa 3, Tail ratio)
- Handles edge cases (zero volatility, single observation, extreme values)

## Common Patterns

### Creating Synthetic Data
```python
from run_experiment import create_synthetic_data
prices, returns = create_synthetic_data(n_assets=10, n_periods=1000)
```

### Running Optimization
```python
from src.portfolio_optimizer import optimize_portfolio
weights, stats = optimize_portfolio(returns, method='hrp')
```

### Running Backtest
```python
from src.backtester import Backtester
backtester = Backtester(returns, prices)
results = backtester.run_backtest(optimization_func, train_window=252)
metrics = backtester.calculate_performance_metrics()
```

## Known Issues
- Massive API may not be accessible in all environments - synthetic data used as fallback
- Some quantstats functions may fail with very short time series - wrapped in try/except

## Future Enhancements
- Add transaction costs to backtesting
- Implement more optimization strategies (Black-Litterman, etc.)
- Add regime detection for dynamic allocation
- Support for short positions
