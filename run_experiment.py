"""
Main Experiment Runner

This script runs the complete portfolio optimization and backtesting experiment.
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.portfolio_optimizer import PortfolioOptimizer, optimize_portfolio
from src.backtester import Backtester
from src.performance_metrics import (
    PerformanceAnalyzer,
    calculate_risk_adjusted_metrics,
    compare_strategies
)
from src.visualizations import (
    plot_portfolio_performance,
    plot_returns_distribution,
    plot_drawdown,
    plot_portfolio_weights,
    plot_weights_evolution,
    plot_rolling_metrics,
    plot_correlation_matrix,
    plot_strategy_comparison,
    plot_cumulative_returns
)


def create_synthetic_data(n_assets=10, n_periods=1000, seed=42):
    """
    Create synthetic market data for the experiment.
    
    Since we may not have access to real market data via API,
    we create realistic synthetic data with known properties.
    """
    np.random.seed(seed)
    
    dates = pd.date_range('2020-01-01', periods=n_periods, freq='D')
    
    # Create correlation matrix
    base_corr = 0.3
    corr_matrix = np.full((n_assets, n_assets), base_corr)
    np.fill_diagonal(corr_matrix, 1.0)
    
    # Generate correlated returns
    mean_returns = np.random.uniform(0.0003, 0.0008, n_assets)
    volatilities = np.random.uniform(0.01, 0.02, n_assets)
    
    # Cholesky decomposition for correlated returns
    L = np.linalg.cholesky(corr_matrix)
    uncorrelated = np.random.randn(n_periods, n_assets)
    correlated = uncorrelated @ L.T
    
    # Scale by volatility and add mean
    returns = correlated * volatilities + mean_returns
    
    returns_df = pd.DataFrame(
        returns,
        index=dates,
        columns=[f'Asset_{i+1}' for i in range(n_assets)]
    )
    
    # Calculate prices from returns
    prices_df = (1 + returns_df).cumprod() * 100
    
    return prices_df, returns_df


def run_optimization_comparison(returns, results_dir):
    """
    Compare different optimization strategies.
    """
    print("\n" + "="*80)
    print("PORTFOLIO OPTIMIZATION COMPARISON")
    print("="*80)
    
    strategies = {
        'Equal Weight': ('equal_weight', {}),
        'HRP': ('hrp', {'risk_measure': 'variance'}),
        'Mean-Variance': ('mean_variance', {'objective': 'max_sharpe'}),
        'Risk Parity': ('risk_parity', {})
    }
    
    results = {}
    
    for name, (method, kwargs) in strategies.items():
        print(f"\nOptimizing with {name}...")
        
        weights, stats = optimize_portfolio(returns, method=method, **kwargs)
        
        results[name] = {
            'weights': weights,
            'stats': stats
        }
        
        print(f"  Annual Return: {stats['annual_return']:.2%}")
        print(f"  Annual Volatility: {stats['annual_volatility']:.2%}")
        print(f"  Sharpe Ratio: {stats['sharpe_ratio']:.3f}")
        print(f"  Max Drawdown: {stats['max_drawdown']:.2%}")
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame({
        name: data['stats'] for name, data in results.items()
    }).T
    
    print("\n" + "-"*80)
    print("OPTIMIZATION COMPARISON SUMMARY")
    print("-"*80)
    print(comparison_df.to_string())
    
    # Plot comparison
    plot_strategy_comparison(
        comparison_df,
        title="Portfolio Optimization Strategy Comparison",
        save_path=os.path.join(results_dir, 'optimization_comparison.png')
    )
    
    # Plot weights for each strategy
    for name, data in results.items():
        optimizer = PortfolioOptimizer(returns)
        optimizer.weights = data['weights']
        optimizer.method = name
        weights_df = optimizer.get_weights_dataframe()
        
        plot_portfolio_weights(
            weights_df,
            title=f"{name} Portfolio Weights",
            save_path=os.path.join(results_dir, f'weights_{name.lower().replace(" ", "_")}.png')
        )
    
    return results, comparison_df


def run_backtesting(prices, returns, results_dir):
    """
    Run backtesting for different strategies.
    """
    print("\n" + "="*80)
    print("BACKTESTING")
    print("="*80)
    
    strategies = {
        'Equal Weight': lambda r: np.ones(r.shape[1]) / r.shape[1],
        'HRP': lambda r: optimize_portfolio(r, method='hrp')[0],
        'Mean-Variance': lambda r: optimize_portfolio(r, method='mean_variance')[0],
        'Risk Parity': lambda r: optimize_portfolio(r, method='risk_parity')[0]
    }
    
    backtest_results = {}
    
    for name, opt_func in strategies.items():
        print(f"\nBacktesting {name}...")
        
        backtester = Backtester(returns, prices, initial_capital=100000)
        
        try:
            results = backtester.run_backtest(
                opt_func,
                train_window=252,
                rebalance_frequency=21,  # Monthly rebalancing
                min_history=252
            )
            
            metrics = backtester.calculate_performance_metrics()
            
            backtest_results[name] = {
                'results': results,
                'metrics': metrics,
                'backtester': backtester
            }
            
            print(f"  Total Return: {metrics['total_return']:.2%}")
            print(f"  Annual Return: {metrics['annualized_return']:.2%}")
            print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
            print(f"  Max Drawdown: {metrics['max_drawdown']:.2%}")
            print(f"  Win Rate: {metrics['win_rate']:.2%}")
            
        except Exception as e:
            print(f"  Error: {e}")
            continue
    
    # Create comparison DataFrame
    metrics_df = pd.DataFrame({
        name: data['metrics'] for name, data in backtest_results.items()
    }).T
    
    print("\n" + "-"*80)
    print("BACKTESTING COMPARISON SUMMARY")
    print("-"*80)
    print(metrics_df.to_string())
    
    # Plot cumulative returns
    returns_dict = {
        name: data['results']['portfolio_return']
        for name, data in backtest_results.items()
    }
    
    plot_cumulative_returns(
        returns_dict,
        title="Cumulative Returns Comparison",
        save_path=os.path.join(results_dir, 'cumulative_returns.png')
    )
    
    # Plot individual strategy results
    for name, data in backtest_results.items():
        # Portfolio performance
        plot_portfolio_performance(
            data['results'],
            title=f"{name} Portfolio Performance",
            save_path=os.path.join(results_dir, f'performance_{name.lower().replace(" ", "_")}.png')
        )
        
        # Drawdown
        drawdown = data['backtester'].get_drawdown_series()
        plot_drawdown(
            drawdown,
            title=f"{name} Drawdown",
            save_path=os.path.join(results_dir, f'drawdown_{name.lower().replace(" ", "_")}.png')
        )
        
        # Returns distribution
        plot_returns_distribution(
            data['results']['portfolio_return'],
            title=f"{name} Returns Distribution",
            save_path=os.path.join(results_dir, f'returns_dist_{name.lower().replace(" ", "_")}.png')
        )
        
        # Weights evolution
        if hasattr(data['backtester'], 'weights_history'):
            plot_weights_evolution(
                data['backtester'].weights_history,
                title=f"{name} Weights Evolution",
                save_path=os.path.join(results_dir, f'weights_evolution_{name.lower().replace(" ", "_")}.png')
            )
    
    # Plot strategy comparison
    plot_strategy_comparison(
        metrics_df,
        metrics_to_plot=['sharpe_ratio', 'sortino_ratio', 'calmar_ratio', 
                        'max_drawdown', 'annualized_return', 'volatility'],
        title="Backtesting Strategy Comparison",
        save_path=os.path.join(results_dir, 'backtest_comparison.png')
    )
    
    return backtest_results, metrics_df


def run_performance_analysis(backtest_results, results_dir):
    """
    Run detailed performance analysis.
    """
    print("\n" + "="*80)
    print("PERFORMANCE ANALYSIS")
    print("="*80)
    
    analysis_results = {}
    
    for name, data in backtest_results.items():
        print(f"\nAnalyzing {name}...")
        
        returns = data['results']['portfolio_return']
        analyzer = PerformanceAnalyzer(returns)
        
        # Calculate comprehensive metrics
        metrics = analyzer.calculate_all_metrics()
        
        # Calculate custom risk-adjusted metrics
        custom_metrics = calculate_risk_adjusted_metrics(returns)
        
        # Combine metrics
        all_metrics = {**metrics, **custom_metrics}
        
        analysis_results[name] = all_metrics
        
        print(f"  Sharpe Ratio: {all_metrics.get('sharpe_ratio', 0):.3f}")
        print(f"  Sortino Ratio: {all_metrics.get('sortino_ratio', 0):.3f}")
        print(f"  Omega Ratio: {all_metrics.get('omega_ratio', 0):.3f}")
        print(f"  Calmar Ratio: {all_metrics.get('calmar_ratio', 0):.3f}")
    
    # Create comprehensive comparison
    analysis_df = pd.DataFrame(analysis_results).T
    
    print("\n" + "-"*80)
    print("COMPREHENSIVE PERFORMANCE ANALYSIS")
    print("-"*80)
    print(analysis_df.to_string())
    
    return analysis_df


def save_results(optimization_results, backtest_results, analysis_df, results_dir):
    """
    Save all results to files.
    """
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)
    
    # Save optimization results
    opt_df = pd.DataFrame({
        name: data['stats'] for name, data in optimization_results.items()
    }).T
    opt_df.to_csv(os.path.join(results_dir, 'optimization_results.csv'))
    print(f"Saved optimization results to {results_dir}/optimization_results.csv")
    
    # Save backtest results
    backtest_df = pd.DataFrame({
        name: data['metrics'] for name, data in backtest_results.items()
    }).T
    backtest_df.to_csv(os.path.join(results_dir, 'backtest_results.csv'))
    print(f"Saved backtest results to {results_dir}/backtest_results.csv")
    
    # Save analysis results
    analysis_df.to_csv(os.path.join(results_dir, 'performance_analysis.csv'))
    print(f"Saved performance analysis to {results_dir}/performance_analysis.csv")
    
    # Create RESULTS.md
    with open(os.path.join(results_dir, 'RESULTS.md'), 'w') as f:
        f.write("# Experiment Results\n\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Optimization Results\n\n")
        f.write(opt_df.to_markdown())
        f.write("\n\n")
        
        f.write("## Backtesting Results\n\n")
        f.write(backtest_df.to_markdown())
        f.write("\n\n")
        
        f.write("## Performance Analysis\n\n")
        f.write(analysis_df.to_markdown())
        f.write("\n\n")
        
        f.write("## Key Findings\n\n")
        
        # Find best strategy by Sharpe ratio
        best_sharpe = backtest_df['sharpe_ratio'].idxmax()
        f.write(f"- **Best Sharpe Ratio:** {best_sharpe} ({backtest_df.loc[best_sharpe, 'sharpe_ratio']:.3f})\n")
        
        # Find best strategy by return
        best_return = backtest_df['annualized_return'].idxmax()
        f.write(f"- **Best Annual Return:** {best_return} ({backtest_df.loc[best_return, 'annualized_return']:.2%})\n")
        
        # Find lowest drawdown
        best_dd = backtest_df['max_drawdown'].idxmax()  # Closest to 0
        f.write(f"- **Lowest Max Drawdown:** {best_dd} ({backtest_df.loc[best_dd, 'max_drawdown']:.2%})\n")
        
        f.write("\n## Visualizations\n\n")
        f.write("All visualizations are saved in the results directory:\n\n")
        f.write("- `optimization_comparison.png` - Comparison of optimization strategies\n")
        f.write("- `cumulative_returns.png` - Cumulative returns for all strategies\n")
        f.write("- `backtest_comparison.png` - Backtesting metrics comparison\n")
        f.write("- Individual strategy plots (performance, drawdown, returns distribution, weights)\n")
    
    print(f"Saved comprehensive results to {results_dir}/RESULTS.md")


def main():
    """
    Main experiment runner.
    """
    print("="*80)
    print("MULTI-ASSET PORTFOLIO OPTIMIZATION AND BACKTESTING EXPERIMENT")
    print("="*80)
    
    # Create results directory
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)
    
    # Generate synthetic data
    print("\nGenerating synthetic market data...")
    prices, returns = create_synthetic_data(n_assets=10, n_periods=1000)
    
    print(f"Data shape: {returns.shape}")
    print(f"Date range: {returns.index[0]} to {returns.index[-1]}")
    
    # Plot correlation matrix
    plot_correlation_matrix(
        returns,
        title="Asset Correlation Matrix",
        save_path=os.path.join(results_dir, 'correlation_matrix.png')
    )
    
    # Run optimization comparison
    optimization_results, opt_comparison = run_optimization_comparison(returns, results_dir)
    
    # Run backtesting
    backtest_results, backtest_comparison = run_backtesting(prices, returns, results_dir)
    
    # Run performance analysis
    analysis_df = run_performance_analysis(backtest_results, results_dir)
    
    # Save all results
    save_results(optimization_results, backtest_results, analysis_df, results_dir)
    
    print("\n" + "="*80)
    print("EXPERIMENT COMPLETED SUCCESSFULLY")
    print("="*80)
    print(f"\nAll results saved to: {results_dir}/")
    print(f"Summary report: {results_dir}/RESULTS.md")


if __name__ == '__main__':
    main()
