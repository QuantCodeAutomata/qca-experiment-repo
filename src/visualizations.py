"""
Visualizations Module

This module provides visualization functions for portfolio analysis.
Uses matplotlib and seaborn for plotting.
"""

from typing import Optional, List, Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


def plot_portfolio_performance(
    results: pd.DataFrame,
    benchmark: Optional[pd.Series] = None,
    title: str = "Portfolio Performance",
    save_path: Optional[str] = None
) -> None:
    """
    Plot portfolio value over time.
    
    Parameters
    ----------
    results : pd.DataFrame
        DataFrame with 'portfolio_value' column
    benchmark : pd.Series, optional
        Benchmark values for comparison
    title : str, default='Portfolio Performance'
        Plot title
    save_path : str, optional
        Path to save the figure
    """
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Plot portfolio value
    ax.plot(results.index, results['portfolio_value'], label='Portfolio', linewidth=2)
    
    # Plot benchmark if provided
    if benchmark is not None:
        ax.plot(benchmark.index, benchmark.values, label='Benchmark', linewidth=2, alpha=0.7)
    
    ax.set_xlabel('Date')
    ax.set_ylabel('Portfolio Value ($)')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    
    plt.close()


def plot_returns_distribution(
    returns: pd.Series,
    title: str = "Returns Distribution",
    save_path: Optional[str] = None
) -> None:
    """
    Plot distribution of returns with histogram and KDE.
    
    Parameters
    ----------
    returns : pd.Series
        Series of returns
    title : str, default='Returns Distribution'
        Plot title
    save_path : str, optional
        Path to save the figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram with KDE
    axes[0].hist(returns, bins=50, density=True, alpha=0.7, edgecolor='black')
    returns.plot(kind='kde', ax=axes[0], linewidth=2, color='red')
    axes[0].set_xlabel('Returns')
    axes[0].set_ylabel('Density')
    axes[0].set_title('Distribution of Returns')
    axes[0].grid(True, alpha=0.3)
    
    # Q-Q plot
    from scipy import stats
    stats.probplot(returns, dist="norm", plot=axes[1])
    axes[1].set_title('Q-Q Plot')
    axes[1].grid(True, alpha=0.3)
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    
    plt.close()


def plot_drawdown(
    drawdown: pd.Series,
    title: str = "Drawdown Over Time",
    save_path: Optional[str] = None
) -> None:
    """
    Plot drawdown series.
    
    Parameters
    ----------
    drawdown : pd.Series
        Series of drawdown values
    title : str, default='Drawdown Over Time'
        Plot title
    save_path : str, optional
        Path to save the figure
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    
    ax.fill_between(drawdown.index, drawdown.values, 0, alpha=0.3, color='red')
    ax.plot(drawdown.index, drawdown.values, color='red', linewidth=1.5)
    
    ax.set_xlabel('Date')
    ax.set_ylabel('Drawdown')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Format y-axis as percentage
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    
    plt.close()


def plot_portfolio_weights(
    weights: pd.DataFrame,
    title: str = "Portfolio Weights",
    save_path: Optional[str] = None,
    top_n: int = 10
) -> None:
    """
    Plot portfolio weights as a bar chart.
    
    Parameters
    ----------
    weights : pd.DataFrame
        DataFrame with 'Asset' and 'Weight' columns
    title : str, default='Portfolio Weights'
        Plot title
    save_path : str, optional
        Path to save the figure
    top_n : int, default=10
        Number of top weights to display
    """
    # Sort by weight and take top N
    weights_sorted = weights.sort_values('Weight', ascending=False).head(top_n)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bars = ax.barh(weights_sorted['Asset'], weights_sorted['Weight'])
    
    # Color bars
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(bars)))
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    ax.set_xlabel('Weight')
    ax.set_ylabel('Asset')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Format x-axis as percentage
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1%}'))
    
    # Add value labels
    for i, (asset, weight) in enumerate(zip(weights_sorted['Asset'], weights_sorted['Weight'])):
        ax.text(weight, i, f' {weight:.2%}', va='center')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    
    plt.close()


def plot_weights_evolution(
    weights_history: pd.DataFrame,
    title: str = "Portfolio Weights Evolution",
    save_path: Optional[str] = None,
    top_n: int = 5
) -> None:
    """
    Plot evolution of portfolio weights over time.
    
    Parameters
    ----------
    weights_history : pd.DataFrame
        DataFrame with datetime index and asset columns
    title : str, default='Portfolio Weights Evolution'
        Plot title
    save_path : str, optional
        Path to save the figure
    top_n : int, default=5
        Number of top assets to display
    """
    # Select top N assets by average weight
    avg_weights = weights_history.mean().sort_values(ascending=False)
    top_assets = avg_weights.head(top_n).index
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Plot stacked area chart
    weights_history[top_assets].plot.area(ax=ax, alpha=0.7, stacked=True)
    
    ax.set_xlabel('Date')
    ax.set_ylabel('Weight')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.grid(True, alpha=0.3)
    
    # Format y-axis as percentage
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    
    plt.close()


def plot_rolling_metrics(
    rolling_metrics: pd.DataFrame,
    title: str = "Rolling Performance Metrics",
    save_path: Optional[str] = None
) -> None:
    """
    Plot rolling performance metrics.
    
    Parameters
    ----------
    rolling_metrics : pd.DataFrame
        DataFrame with rolling metrics
    title : str, default='Rolling Performance Metrics'
        Plot title
    save_path : str, optional
        Path to save the figure
    """
    n_metrics = len(rolling_metrics.columns)
    fig, axes = plt.subplots(n_metrics, 1, figsize=(14, 4 * n_metrics))
    
    if n_metrics == 1:
        axes = [axes]
    
    for ax, col in zip(axes, rolling_metrics.columns):
        ax.plot(rolling_metrics.index, rolling_metrics[col], linewidth=2)
        ax.set_xlabel('Date')
        ax.set_ylabel(col.replace('_', ' ').title())
        ax.set_title(col.replace('_', ' ').title())
        ax.grid(True, alpha=0.3)
    
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.001)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    
    plt.close()


def plot_correlation_matrix(
    returns: pd.DataFrame,
    title: str = "Asset Correlation Matrix",
    save_path: Optional[str] = None
) -> None:
    """
    Plot correlation matrix heatmap.
    
    Parameters
    ----------
    returns : pd.DataFrame
        DataFrame of asset returns
    title : str, default='Asset Correlation Matrix'
        Plot title
    save_path : str, optional
        Path to save the figure
    """
    corr = returns.corr()
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    sns.heatmap(
        corr,
        annot=True,
        fmt='.2f',
        cmap='coolwarm',
        center=0,
        square=True,
        linewidths=1,
        cbar_kws={"shrink": 0.8},
        ax=ax
    )
    
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    
    plt.close()


def plot_strategy_comparison(
    metrics_df: pd.DataFrame,
    metrics_to_plot: Optional[List[str]] = None,
    title: str = "Strategy Comparison",
    save_path: Optional[str] = None
) -> None:
    """
    Plot comparison of multiple strategies.
    
    Parameters
    ----------
    metrics_df : pd.DataFrame
        DataFrame with strategies as rows and metrics as columns
    metrics_to_plot : List[str], optional
        List of metrics to plot. If None, plots all.
    title : str, default='Strategy Comparison'
        Plot title
    save_path : str, optional
        Path to save the figure
    """
    if metrics_to_plot is None:
        metrics_to_plot = ['sharpe_ratio', 'sortino_ratio', 'calmar_ratio', 
                          'max_drawdown', 'cagr', 'volatility']
    
    # Filter available metrics
    available_metrics = [m for m in metrics_to_plot if m in metrics_df.columns]
    
    if not available_metrics:
        print("No metrics available to plot")
        return
    
    n_metrics = len(available_metrics)
    n_cols = 2
    n_rows = (n_metrics + 1) // 2
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows))
    axes = axes.flatten() if n_metrics > 1 else [axes]
    
    for i, metric in enumerate(available_metrics):
        ax = axes[i]
        
        data = metrics_df[metric].sort_values(ascending=False)
        bars = ax.barh(data.index, data.values)
        
        # Color bars
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(bars)))
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        ax.set_xlabel(metric.replace('_', ' ').title())
        ax.set_title(metric.replace('_', ' ').title())
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for j, (strategy, value) in enumerate(zip(data.index, data.values)):
            ax.text(value, j, f' {value:.3f}', va='center')
    
    # Hide unused subplots
    for i in range(len(available_metrics), len(axes)):
        axes[i].set_visible(False)
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    
    plt.close()


def plot_cumulative_returns(
    returns_dict: Dict[str, pd.Series],
    title: str = "Cumulative Returns Comparison",
    save_path: Optional[str] = None
) -> None:
    """
    Plot cumulative returns for multiple strategies.
    
    Parameters
    ----------
    returns_dict : Dict[str, pd.Series]
        Dictionary mapping strategy names to return series
    title : str, default='Cumulative Returns Comparison'
        Plot title
    save_path : str, optional
        Path to save the figure
    """
    fig, ax = plt.subplots(figsize=(14, 7))
    
    for name, returns in returns_dict.items():
        cumulative = (1 + returns).cumprod()
        ax.plot(cumulative.index, cumulative.values, label=name, linewidth=2)
    
    ax.set_xlabel('Date')
    ax.set_ylabel('Cumulative Return')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    
    plt.close()
