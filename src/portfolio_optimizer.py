"""
Portfolio Optimizer Module

This module implements various portfolio optimization strategies including:
- Hierarchical Risk Parity (HRP) using skfolio
- Mean-Variance Optimization using skfolio
- Custom Risk Parity implementation
"""

from typing import Optional, Dict, Any
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.model_selection import train_test_split

# Using skfolio — Context7 confirmed
from skfolio import RiskMeasure
from skfolio.optimization import HierarchicalRiskParity, MeanRisk, ObjectiveFunction
from skfolio.distance import PearsonDistance, MutualInformation
from skfolio.preprocessing import prices_to_returns


class PortfolioOptimizer:
    """
    Portfolio optimizer implementing multiple optimization strategies.
    """
    
    def __init__(self, returns: pd.DataFrame):
        """
        Initialize the portfolio optimizer.
        
        Parameters
        ----------
        returns : pd.DataFrame
            DataFrame of asset returns with datetime index
        """
        self.returns = returns
        self.n_assets = returns.shape[1]
        self.asset_names = returns.columns.tolist()
        self.weights = None
        self.method = None
    
    def hierarchical_risk_parity(
        self,
        risk_measure: str = 'variance',
        distance_estimator: str = 'pearson'
    ) -> np.ndarray:
        """
        Compute portfolio weights using Hierarchical Risk Parity.
        
        Uses skfolio HierarchicalRiskParity — Context7 confirmed
        
        Parameters
        ----------
        risk_measure : str, default='variance'
            Risk measure to use ('variance', 'semi_deviation', 'cvar')
        distance_estimator : str, default='pearson'
            Distance estimator ('pearson', 'mutual_information')
            
        Returns
        -------
        np.ndarray
            Array of portfolio weights
        """
        # Map string to RiskMeasure enum
        risk_map = {
            'variance': RiskMeasure.VARIANCE,
            'semi_deviation': RiskMeasure.SEMI_DEVIATION,
            'cvar': RiskMeasure.CVAR,
            'cdar': RiskMeasure.CDAR
        }
        
        # Map string to distance estimator
        distance_map = {
            'pearson': PearsonDistance(),
            'mutual_information': MutualInformation()
        }
        
        risk_measure_obj = risk_map.get(risk_measure, RiskMeasure.VARIANCE)
        distance_est = distance_map.get(distance_estimator, PearsonDistance())
        
        # Using skfolio HierarchicalRiskParity — Context7 confirmed
        model = HierarchicalRiskParity(
            risk_measure=risk_measure_obj,
            distance_estimator=distance_est
        )
        
        model.fit(self.returns)
        self.weights = model.weights_
        self.method = 'HRP'
        
        return self.weights
    
    def mean_variance_optimization(
        self,
        objective: str = 'max_sharpe',
        risk_measure: str = 'variance',
        risk_free_rate: float = 0.02
    ) -> np.ndarray:
        """
        Compute portfolio weights using Mean-Variance Optimization.
        
        Uses skfolio MeanRisk — Context7 confirmed
        
        Parameters
        ----------
        objective : str, default='max_sharpe'
            Optimization objective ('max_sharpe', 'min_risk', 'max_return')
        risk_measure : str, default='variance'
            Risk measure to use
        risk_free_rate : float, default=0.02
            Annual risk-free rate
            
        Returns
        -------
        np.ndarray
            Array of portfolio weights
        """
        # Map string to ObjectiveFunction enum
        objective_map = {
            'max_sharpe': ObjectiveFunction.MAXIMIZE_RATIO,
            'min_risk': ObjectiveFunction.MINIMIZE_RISK,
            'max_return': ObjectiveFunction.MAXIMIZE_RETURN
        }
        
        risk_map = {
            'variance': RiskMeasure.VARIANCE,
            'semi_deviation': RiskMeasure.SEMI_DEVIATION,
            'cvar': RiskMeasure.CVAR,
            'cdar': RiskMeasure.CDAR
        }
        
        objective_func = objective_map.get(objective, ObjectiveFunction.MAXIMIZE_RATIO)
        risk_measure_obj = risk_map.get(risk_measure, RiskMeasure.VARIANCE)
        
        # Using skfolio MeanRisk — Context7 confirmed
        model = MeanRisk(
            objective_function=objective_func,
            risk_measure=risk_measure_obj,
            min_weights=0.0,  # Long-only constraint
            max_weights=1.0
        )
        
        model.fit(self.returns)
        self.weights = model.weights_
        self.method = 'Mean-Variance'
        
        return self.weights
    
    def risk_parity(self, method: str = 'iterative') -> np.ndarray:
        """
        Compute portfolio weights using Risk Parity.
        
        Custom implementation — Context7 found no library equivalent
        Risk Parity aims to equalize risk contribution across assets.
        
        Parameters
        ----------
        method : str, default='iterative'
            Optimization method ('iterative' or 'analytical')
            
        Returns
        -------
        np.ndarray
            Array of portfolio weights
        """
        # Calculate covariance matrix
        cov_matrix = self.returns.cov().values
        
        if method == 'analytical':
            # Analytical solution: inverse volatility weighting
            # This is an approximation when assets are uncorrelated
            volatilities = np.sqrt(np.diag(cov_matrix))
            weights = 1.0 / volatilities
            weights = weights / weights.sum()
        
        elif method == 'iterative':
            # Numerical optimization to equalize risk contributions
            # Custom implementation — Context7 found no library equivalent
            
            def risk_contribution(weights, cov_matrix):
                """Calculate risk contribution of each asset."""
                portfolio_vol = np.sqrt(weights @ cov_matrix @ weights)
                marginal_contrib = cov_matrix @ weights
                risk_contrib = weights * marginal_contrib / portfolio_vol
                return risk_contrib
            
            def risk_parity_objective(weights, cov_matrix):
                """
                Objective function: minimize variance of risk contributions.
                
                When all assets contribute equally to risk, the variance
                of risk contributions is minimized.
                """
                risk_contrib = risk_contribution(weights, cov_matrix)
                target_risk = np.ones(len(weights)) / len(weights)
                return np.sum((risk_contrib - target_risk) ** 2)
            
            # Constraints: weights sum to 1, all weights >= 0
            constraints = [
                {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
            ]
            bounds = tuple((0.0, 1.0) for _ in range(self.n_assets))
            
            # Initial guess: equal weights
            x0 = np.ones(self.n_assets) / self.n_assets
            
            # Optimize
            result = minimize(
                risk_parity_objective,
                x0,
                args=(cov_matrix,),
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000}
            )
            
            if not result.success:
                print(f"Warning: Optimization did not converge: {result.message}")
            
            weights = result.x
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        self.weights = weights
        self.method = 'Risk Parity'
        
        return self.weights
    
    def equal_weight(self) -> np.ndarray:
        """
        Compute equal-weighted portfolio (1/N strategy).
        
        Returns
        -------
        np.ndarray
            Array of equal weights
        """
        self.weights = np.ones(self.n_assets) / self.n_assets
        self.method = 'Equal Weight'
        return self.weights
    
    def get_weights_dataframe(self) -> pd.DataFrame:
        """
        Get portfolio weights as a DataFrame.
        
        Returns
        -------
        pd.DataFrame
            DataFrame with asset names and weights
        """
        if self.weights is None:
            raise ValueError("No weights computed. Run an optimization method first.")
        
        return pd.DataFrame({
            'Asset': self.asset_names,
            'Weight': self.weights,
            'Method': self.method
        }).sort_values('Weight', ascending=False)
    
    def calculate_portfolio_statistics(
        self,
        weights: Optional[np.ndarray] = None,
        risk_free_rate: float = 0.02
    ) -> Dict[str, float]:
        """
        Calculate portfolio statistics for given weights.
        
        Parameters
        ----------
        weights : np.ndarray, optional
            Portfolio weights. If None, uses self.weights
        risk_free_rate : float, default=0.02
            Annual risk-free rate
            
        Returns
        -------
        Dict[str, float]
            Dictionary of portfolio statistics
        """
        if weights is None:
            if self.weights is None:
                raise ValueError("No weights available")
            weights = self.weights
        
        # Portfolio returns
        portfolio_returns = (self.returns @ weights)
        
        # Annualization factor (assuming daily returns)
        annual_factor = 252
        
        # Calculate statistics
        mean_return = portfolio_returns.mean() * annual_factor
        volatility = portfolio_returns.std() * np.sqrt(annual_factor)
        sharpe_ratio = (mean_return - risk_free_rate) / volatility if volatility > 0 else 0.0
        
        # Downside deviation (semi-deviation)
        downside_returns = portfolio_returns[portfolio_returns < 0]
        downside_vol = downside_returns.std() * np.sqrt(annual_factor) if len(downside_returns) > 0 else 0.0
        sortino_ratio = (mean_return - risk_free_rate) / downside_vol if downside_vol > 0 else 0.0
        
        # Maximum drawdown
        cumulative = (1 + portfolio_returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        return {
            'annual_return': mean_return,
            'annual_volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'downside_volatility': downside_vol
        }


def optimize_portfolio(
    returns: pd.DataFrame,
    method: str = 'hrp',
    **kwargs
) -> tuple[np.ndarray, Dict[str, float]]:
    """
    Convenience function to optimize portfolio and return weights and statistics.
    
    Parameters
    ----------
    returns : pd.DataFrame
        DataFrame of asset returns
    method : str, default='hrp'
        Optimization method ('hrp', 'mean_variance', 'risk_parity', 'equal_weight')
    **kwargs
        Additional arguments passed to the optimization method
        
    Returns
    -------
    tuple[np.ndarray, Dict[str, float]]
        Tuple of (weights, statistics)
    """
    optimizer = PortfolioOptimizer(returns)
    
    if method == 'hrp':
        weights = optimizer.hierarchical_risk_parity(**kwargs)
    elif method == 'mean_variance':
        weights = optimizer.mean_variance_optimization(**kwargs)
    elif method == 'risk_parity':
        weights = optimizer.risk_parity(**kwargs)
    elif method == 'equal_weight':
        weights = optimizer.equal_weight()
    else:
        raise ValueError(f"Unknown method: {method}")
    
    stats = optimizer.calculate_portfolio_statistics(weights)
    
    return weights, stats
