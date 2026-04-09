# Experiment Summary

## Overview
This repository implements a comprehensive multi-asset portfolio optimization and backtesting framework using modern quantitative finance libraries.

## Experiment Design

### Objective
Compare the performance of different portfolio optimization strategies through rigorous backtesting on synthetic multi-asset data.

### Strategies Tested
1. **Equal Weight** - Baseline strategy with equal allocation across all assets
2. **Hierarchical Risk Parity (HRP)** - Clustering-based diversification approach
3. **Mean-Variance Optimization** - Modern Portfolio Theory (Markowitz)
4. **Risk Parity** - Equal risk contribution across assets

### Methodology

#### Data Generation
- 10 synthetic assets with realistic correlation structure
- 1000 daily observations (approximately 4 years)
- Varying volatilities and expected returns

#### Backtesting Framework
- Walk-forward optimization with rolling windows
- Training window: 252 days (1 year)
- Rebalancing frequency: 21 days (monthly)
- Initial capital: $100,000
- Transaction costs: Not included (can be added)

#### Performance Metrics
- Sharpe Ratio
- Sortino Ratio
- Calmar Ratio
- Maximum Drawdown
- Win Rate
- Profit Factor
- Value at Risk (VaR 95%)
- Conditional VaR (CVaR 95%)
- Omega Ratio
- Kappa 3
- Tail Ratio

## Key Results

### Optimization Phase (Full Sample)
| Strategy      | Annual Return | Volatility | Sharpe Ratio | Max Drawdown |
|---------------|---------------|------------|--------------|--------------|
| Equal Weight  | 13.30%        | 12.90%     | 0.876        | -18.82%      |
| HRP           | 13.15%        | 12.08%     | 0.923        | -17.12%      |
| Mean-Variance | 21.22%        | 13.36%     | 1.439        | -13.70%      |
| Risk Parity   | 15.91%        | 25.45%     | 0.546        | -36.64%      |

### Backtesting Phase (Out-of-Sample)
| Strategy      | Total Return | Annual Return | Sharpe Ratio | Max Drawdown |
|---------------|--------------|---------------|--------------|--------------|
| Equal Weight  | 24.69%       | 7.72%         | 0.443        | -18.82%      |
| **HRP**       | **29.69%**   | **9.15%**     | **0.591**    | **-16.06%**  |
| Mean-Variance | 19.41%       | 6.16%         | 0.269        | -26.59%      |
| Risk Parity   | 23.54%       | 7.38%         | 0.192        | -34.47%      |

## Key Findings

1. **HRP Outperforms**: Hierarchical Risk Parity achieved the best risk-adjusted returns in backtesting
   - Highest Sharpe Ratio (0.591)
   - Best annual return (9.15%)
   - Lowest maximum drawdown (-16.06%)

2. **Mean-Variance Overfitting**: Despite excellent in-sample performance (Sharpe 1.439), Mean-Variance optimization showed significant degradation out-of-sample (Sharpe 0.269)
   - Classic sign of overfitting to historical data
   - Higher drawdown in backtesting (-26.59%)

3. **Risk Parity Challenges**: Risk Parity showed high volatility (28.10%) and poor risk-adjusted returns
   - May benefit from additional constraints or risk budgeting

4. **Equal Weight Baseline**: Simple equal weighting provided competitive performance
   - Sharpe ratio of 0.443
   - Demonstrates the difficulty of consistently beating naive diversification

## Technical Implementation

### Libraries Used (Context7 Verified)
- **skfolio** - Portfolio optimization (HRP, Mean-Variance)
- **quantstats** - Performance analytics and metrics
- **numpy/pandas** - Data manipulation
- **matplotlib/seaborn** - Visualizations

### Custom Implementations
- Risk Parity optimization (iterative and analytical methods)
- Backtesting engine with rolling window optimization
- Advanced performance metrics (Kappa, Omega, Tail Ratio)
- Weight evolution tracking

### Code Quality
- 61 comprehensive tests (100% passing)
- Type hints throughout
- Detailed docstrings
- Modular architecture

## Repository Structure
```
qca-experiment-repo/
├── src/
│   ├── data_loader.py          # Data fetching and preprocessing
│   ├── portfolio_optimizer.py  # Optimization strategies
│   ├── backtester.py           # Backtesting engine
│   ├── performance_metrics.py  # Performance analytics
│   └── visualizations.py       # Plotting utilities
├── tests/
│   ├── test_portfolio_optimizer.py
│   ├── test_backtester.py
│   └── test_performance_metrics.py
├── results/
│   ├── RESULTS.md              # Detailed results
│   └── *.png                   # Visualizations
├── run_experiment.py           # Main experiment runner
├── requirements.txt            # Dependencies
└── README.md                   # Documentation
```

## Visualizations Generated
- Correlation matrix
- Portfolio weights comparison
- Cumulative returns
- Drawdown analysis
- Returns distribution
- Weight evolution over time
- Performance comparison charts

## How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Run tests
pytest tests/ -v

# Run experiment
python run_experiment.py
```

## Future Enhancements
1. Add transaction costs and slippage
2. Implement more optimization strategies (Black-Litterman, CVaR optimization)
3. Test on real market data using massive API
4. Add regime detection and adaptive strategies
5. Implement portfolio constraints (sector limits, turnover constraints)
6. Add Monte Carlo simulation for robustness testing
7. Implement walk-forward optimization parameter tuning

## Conclusion
This experiment demonstrates the importance of out-of-sample testing in portfolio optimization. While sophisticated optimization methods can show impressive in-sample results, simpler diversification approaches like HRP often provide more robust out-of-sample performance. The framework provides a solid foundation for further quantitative research and strategy development.
