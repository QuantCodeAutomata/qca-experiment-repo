# Multi-Asset Portfolio Optimization and Backtesting Experiment

## Overview

This repository implements a comprehensive quantitative finance experiment focusing on multi-asset portfolio optimization using various methodologies including:

- **Hierarchical Risk Parity (HRP)** - Using skfolio library
- **Mean-Variance Optimization** - Using skfolio library
- **Risk Parity** - Custom implementation
- **Backtesting Framework** - Performance evaluation and metrics

## Methodology

The experiment follows a systematic approach:

1. **Data Acquisition**: Retrieve historical price data for multiple assets using the Massive API
2. **Data Preprocessing**: Calculate returns, handle missing data, and align time series
3. **Portfolio Optimization**: Apply multiple optimization strategies
4. **Backtesting**: Evaluate portfolio performance out-of-sample
5. **Performance Analysis**: Calculate comprehensive metrics using quantstats
6. **Visualization**: Generate plots for returns, weights, and risk metrics

## Repository Structure

```
.
├── src/
│   ├── data_loader.py          # Data retrieval using Massive API
│   ├── portfolio_optimizer.py  # Portfolio optimization strategies
│   ├── backtester.py           # Backtesting framework
│   ├── performance_metrics.py  # Performance calculation
│   └── visualizations.py       # Plotting functions
├── tests/
│   ├── test_data_loader.py
│   ├── test_portfolio_optimizer.py
│   ├── test_backtester.py
│   └── test_performance_metrics.py
├── results/
│   └── RESULTS.md              # Experiment results and metrics
├── run_experiment.py           # Main experiment runner
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Run the complete experiment:

```bash
python run_experiment.py
```

## Results

All results, including performance metrics and visualizations, are saved in the `results/` directory.

## Libraries Used

- **skfolio**: Portfolio optimization (HRP, Mean-Variance)
- **quantstats**: Performance metrics and analytics
- **massive**: Financial data retrieval
- **pandas/numpy**: Data manipulation
- **matplotlib/seaborn**: Visualization
- **scipy/statsmodels**: Statistical analysis

## License

MIT License
