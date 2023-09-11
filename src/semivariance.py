import os
import warnings
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Tuple, Dict
from tqdm import tqdm
from cvxpy import SolverError
from pandas import DataFrame, Series
from pypfopt import risk_models as rm
from pypfopt import black_litterman, expected_returns
from pypfopt.efficient_frontier import EfficientSemivariance
from pypfopt.exceptions import OptimizationError


def get_covariance_matrix(
        historical_prices: pd.DataFrame,
        method: str = 'sample_cov') -> pd.DataFrame:
    """
    Calculate the covariance matrix using a specified method and fix non-positive semidefinite issues.

    Parameters:
    - historical_prices (pd.DataFrame): Historical price data for the assets.
    - method (str, optional): Method for covariance matrix calculation.
        Options: 'sample_cov', 'semicovariance', 'exp_cov', 'ledoit_wolf', 'ledoit_wolf_constant_variance',
        'ledoit_wolf_single_factor', 'ledoit_wolf_constant_correlation', 'oracle_approximating'.
        Defaults to 'sample_cov'.

    Returns:
    - pd.DataFrame: Calculated covariance matrix.

    Raises:
    - NotImplementedError: If the method is not one of the acceptable options.
    """
    # Calculate the covariance matrix using the specified method
    covariance_matrix = rm.risk_matrix(historical_prices, method=method)

    # Fix non-positive semidefinite issues if any
    covariance_matrix = rm.fix_nonpositive_semidefinite(covariance_matrix)

    return covariance_matrix


def get_market_implied_metrics(
        market_prices: pd.Series,
        market_caps: pd.Series,
        covariance_matrix: pd.DataFrame,
        risk_free_rate: float) -> Tuple[float, pd.Series]:
    """
    Calculate market-implied risk aversion and expected returns using the Black-Litterman model.

    Parameters:
    - market_prices (pd.Series): Series of market prices.
    - market_caps (pd.Series): Series of market capitalizations for each asset.
    - covariance_matrix (pd.DataFrame): Covariance matrix of asset returns.
    - risk_free_rate (float): Risk-free rate.

    Returns:
        Tuple[float, pd.Series]:
            - float: Market-implied risk aversion (market_risk_aversion).
            - pd.Series: Market-implied expected returns for each asset.
    """

    market_risk_aversion = black_litterman.market_implied_risk_aversion(market_prices)
    market_implied_expected_returns = black_litterman.market_implied_prior_returns(
        market_caps, market_risk_aversion, covariance_matrix, risk_free_rate
    ).astype(float).round(4)

    return market_risk_aversion, market_implied_expected_returns


def get_black_litterman_params(
        covariance_matrix: DataFrame,
        portfolio_tickers: List[str],
        investor_views: Dict[str, float],
        risk_aversion: float,
        market_caps: Series,
        risk_free_rate: float,
        view_confidences: Dict[str, float],
        market_implied_expected_returns: Series) -> Tuple[DataFrame, Series]:
    """
    Compute the Black-Litterman model parameters.

    Parameters:
    - covariance_matrix (DataFrame): Covariance matrix of the portfolio securities.
    - portfolio_tickers (List[str]): List of tickers in the portfolio.
    - investor_views (Dict[str, float]): Dictionary containing the investor's views.
    - risk_aversion (float): Risk aversion parameter (delta).
    - market_caps (Series): Series containing the market caps of the securities.
    - risk_free_rate (float): Risk-free rate.
    - view_confidences (Dict[str, float]): Dictionary containing view confidences.
    - market_implied_expected_returns (Series): Series containing market-implied expected returns.

    Returns:
    - Tuple[DataFrame, Series]: Posterior covariance matrix and posterior expected returns.
    """
    # Ensure the keys in investor_views and view_confidences are the same and sorted
    assert set(investor_views.keys()) == set(
        view_confidences.keys()), "Keys of investor_views and view_confidences must match."
    sorted_keys = sorted(investor_views.keys())

    # Create Pandas Series from the sorted dictionaries
    investor_views = pd.Series({k: investor_views[k] for k in sorted_keys})
    view_confidences = pd.Series({k: view_confidences[k] for k in sorted_keys})

    bl = black_litterman.BlackLittermanModel(
        covariance_matrix,
        pi='market',
        absolute_views=investor_views,
        risk_aversion=risk_aversion,
        market_caps=market_caps.loc[portfolio_tickers],
        risk_free_rate=risk_free_rate,
        omega='idzorek',
        view_confidences=view_confidences
    )

    posterior_covariance_matrix = bl.bl_cov()
    posterior_expected_returns = bl.bl_returns()

    # Clip posterior expected returns between the minimum and maximum of market-implied and investor views
    min_values = pd.concat([market_implied_expected_returns, Series(investor_views)], axis=1).min(axis=1)
    max_values = pd.concat([market_implied_expected_returns, Series(investor_views)], axis=1).max(axis=1)
    posterior_expected_returns = round(posterior_expected_returns.clip(lower=min_values, upper=max_values), 4)

    return posterior_covariance_matrix, posterior_expected_returns


def get_min_risk_portfolio(
        posterior_expected_returns: pd.Series,
        historical_prices: pd.DataFrame,
        weight_bounds: Tuple[float, float],
        risk_free_rate: float) -> Tuple[float, pd.DataFrame]:
    """
    Find the minimum risk portfolio.

    Parameters:
    - posterior_expected_returns (pd.Series): Posterior expected returns.
    - historical_prices (pd.DataFrame): Historical price data for the assets.
    - weight_bounds (Tuple[float, float]): Weight bounds for the portfolio.
    - risk_free_rate (float): Risk-free rate.

    Returns:
        Tuple[float, pd.DataFrame]:
            - float: Minimum risk found.
            - pd.DataFrame: Weights for the minimum risk portfolio.
    """
    try:
        es = EfficientSemivariance(posterior_expected_returns,
                                   expected_returns.returns_from_prices(historical_prices),
                                   weight_bounds=weight_bounds)
        es.min_semivariance()
        min_risk = es.portfolio_performance(risk_free_rate=risk_free_rate)[1]
        min_risk_weights = pd.DataFrame(list(es.clean_weights().items()), columns=['symbol', 'weight']).set_index(
            'symbol')
        return min_risk, min_risk_weights

    except OptimizationError:
        print("Failed to find minimum risk portfolio.")
        return 0.0025, pd.DataFrame()


def get_max_risk(
        posterior_expected_returns: pd.Series,
        historical_prices: pd.DataFrame,
        weight_bounds: Tuple[float, float],
        risk_free_rate: float,
        min_risk: float) -> float:
    max_risk = 0  # Initialize to zero
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_path = os.path.join('../logs', f'get_max_risk_warnings_{current_time}.txt')

    with open(log_file_path, 'w') as f:
        with warnings.catch_warnings(record=True) as w:
            warnings.filterwarnings("always")

            for i, target_risk in tqdm(enumerate(np.linspace(min_risk + 0.0005, 1.0, 10))):
                while True:  # Infinite loop to keep retrying with incremented target_risk
                    try:
                        historical_returns = expected_returns.returns_from_prices(historical_prices)
                        es = EfficientSemivariance(posterior_expected_returns, historical_returns,
                                                   weight_bounds=weight_bounds)
                        es.efficient_risk(target_risk)
                        performance = es.portfolio_performance(risk_free_rate=risk_free_rate)

                        # Update max_risk if this portfolio has higher risk
                        max_risk = max(max_risk, performance[1])

                        # If no error, break the infinite loop
                        break

                    except SolverError:  # Catch the specific SolverError
                        f.write("SolverError encountered. Incrementing target_risk.\n")
                        target_risk += 0.0005  # increment target_risk by 0.0005
                        continue  # continue the while loop

                    except OptimizationError:  # Catch the specific OptimizationError
                        f.write("OptimizationError encountered. Skipping to next target_risk.\n")
                        break  # break the infinite loop and continue with the next 'target_risk' value

            for warning in w:
                f.write(str(warning.message) + '\n')

    return max_risk


def optimize_portfolio(
        posterior_expected_returns: pd.Series,
        historical_prices: pd.DataFrame,
        weight_bounds: Tuple[float, float],
        risk_free_rate: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Optimizes a portfolio by computing portfolios along the efficient frontier.

    Parameters:
    - posterior_expected_returns (pd.Series): Series containing the posterior expected returns for each asset
    - historical_prices (pd.DataFrame): DataFrame containing historical price data for assets
    - weight_bounds (Tuple[float, float]): Tuple representing the lower and upper bounds for asset weights
    - risk_free_rate (float): Risk-free rate used for calculating Sortino ratios

    Returns:
        - Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing two DataFrames:
            - DataFrame with performance metrics ('Expected annual return', 'Annual semi-deviation', 'Sortino ratio')
                for each portfolio along the efficient frontier
            - DataFrame containing the weights for each asset in each portfolio along the efficient frontier
    """

    min_risk, min_risk_weights = get_min_risk_portfolio(
        posterior_expected_returns, historical_prices, weight_bounds, risk_free_rate
    )
    max_risk = get_max_risk(
        posterior_expected_returns,
        historical_prices,
        weight_bounds,
        risk_free_rate,
        min_risk
    )

    performance_df_optimal = pd.DataFrame(columns=['Expected annual return', 'Annual semi-deviation', 'Sortino ratio'])
    es = EfficientSemivariance(
        posterior_expected_returns,
        expected_returns.returns_from_prices(historical_prices),
        weight_bounds=weight_bounds
    )
    es.set_weights(min_risk_weights['weight'].to_dict())
    performance = es.portfolio_performance(risk_free_rate=risk_free_rate)
    performance_df_optimal.loc[1] = performance  # First row is the min risk portfolio

    weights_df_optimal = min_risk_weights.copy()
    weights_df_optimal.columns = [1]  # First column is the min risk portfolio

    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_path = os.path.join('../logs', f'optimize_portfolio_warnings_{current_time}.txt')

    with open(log_file_path, 'w') as f:
        with warnings.catch_warnings(record=True) as w:
            warnings.filterwarnings("always")

            successful_runs = 1 # Initialize to 1 because the first row is the min risk portfolio
            risk_range = np.linspace(min_risk + 0.0005, max_risk, 99)
            for i, target_risk in tqdm(enumerate(risk_range)):
                if successful_runs >= 100:  # Stop if 100 successful portfolios are generated
                    break

                while True:  # Infinite loop to keep retrying with incremented target_risk
                    try:
                        # Run the optimization
                        historical_returns = expected_returns.returns_from_prices(historical_prices)
                        es = EfficientSemivariance(posterior_expected_returns, historical_returns,
                                                   weight_bounds=weight_bounds)
                        es.efficient_risk(target_risk)
                        weights = es.clean_weights()

                        # Store the performance
                        performance = es.portfolio_performance(risk_free_rate=risk_free_rate)
                        performance_df_optimal.loc[successful_runs + 1] = performance

                        # Convert weights to dataframe and store
                        temp_weights_df = pd.DataFrame(list(weights.items()), columns=['symbol', 'weight']).set_index(
                            'symbol')
                        temp_weights_df.columns = [successful_runs + 1]
                        weights_df_optimal = pd.concat([weights_df_optimal, temp_weights_df], axis=1)

                        successful_runs += 1

                        # If no error, break the infinite loop
                        break

                    except SolverError:  # Catch the specific SolverError
                        f.write("SolverError encountered. Incrementing target_risk.\n")
                        target_risk += 0.0005  # increment target_risk by 0.0005
                        continue  # continue the while loop

                    except OptimizationError:  # Catch the specific OptimizationError
                        f.write("OptimizationError encountered. Skipping to next target_risk.\n")
                        break  # break the infinite loop and continue with the next 'target_risk' value

            for warning in w:
                f.write(str(warning.message) + '\n')

    performance_df_optimal = performance_df_optimal.T.round(4)
    weights_df_optimal = weights_df_optimal.round(4)

    return performance_df_optimal, weights_df_optimal


def get_optimal_portfolio(
        performance_df: pd.DataFrame,
        weights_df: pd.DataFrame,
        print_output: bool = False) -> Tuple[int, pd.Series, pd.Series]:
    """
    Find the portfolio with the maximum Sortino Ratio and return its details.

    Parameters:
    - performance_df (pd.DataFrame): DataFrame containing portfolio performance metrics.
    - weights_df (pd.DataFrame): DataFrame containing portfolio weights.
    - print_output (bool, optional): Whether to print the details of the optimal portfolio.

    Returns:
        Tuple[int, pd.Series, pd.Series]:
            - int: Index of the optimal portfolio.
            - pd.Series: Performance details of the optimal portfolio.
            - pd.Series: Weights of the optimal portfolio.
    """
    # Find the portfolio with the maximum Sortino Ratio
    optimal_portfolio_index = performance_df.loc['Sortino ratio'].idxmax() + 1  # +1 because portfolio index starts at 1

    # Extracting the details of the optimal portfolio
    optimal_portfolio_details = round(performance_df.iloc[:, optimal_portfolio_index - 1], 4)  # -1 because DataFrame
    # index starts at 0
    optimal_weights = weights_df[optimal_portfolio_index]

    if print_output:
        print(f"The optimal portfolio is Portfolio {optimal_portfolio_index} with the following details:")
        print(optimal_portfolio_details)
        print("\n")
        print(f"The optimal weights in Portfolio {optimal_portfolio_index}: ")
        print(optimal_weights)

    return optimal_portfolio_index, optimal_portfolio_details, optimal_weights


def get_ef_portfolios(
        names: pd.DataFrame,
        weights_df: pd.DataFrame) -> pd.DataFrame:
    """
    Concatenate the asset names and weights DataFrames.

    Parameters:
    - names (pd.DataFrame): DataFrame containing asset names.
    - weights_df (pd.DataFrame): DataFrame containing portfolio weights.

    Returns:
    - pd.DataFrame: Concatenated DataFrame.
    """
    return pd.concat([names, weights_df], axis=1)


def generate_random_portfolios(
        historical_prices: DataFrame,
        posterior_expected_returns: Series,
        risk_free_rate: float,
        weight_bounds: List[Tuple[float, float]],
        n_portfolios: int = 1000) -> Tuple[DataFrame, DataFrame, Series]:
    """
    Generate random portfolios based on Efficient Semivariance and return the optimal one based on Sortino ratio.

    Parameters:
    - historical_prices (DataFrame): Historical price data for the assets.
    - posterior_expected_returns (Series): Posterior expected returns.
    - risk_free_rate (float): Risk-free rate.
    - weight_bounds (List[Tuple[float, float]]): Weight bounds for the portfolios.
    - n_portfolios (int, optional): Number of random portfolios to generate.

    Returns:
        Tuple[DataFrame, DataFrame, Series]:
            - pd.DataFrame: Performance DataFrame.
            - pd.DataFrame: Weights DataFrame.
            - pd.Series: Details of the optimal portfolio.
    """
    performance_df_mc = DataFrame(columns=['Expected annual return', 'Annual semi-deviation', 'Sortino ratio'])
    weights_df_mc = DataFrame()

    for portfolio_idx in range(n_portfolios):  # Changed i to portfolio_idx to avoid shadowing
        try:
            random_weights = random_weights_with_bounds(len(historical_prices.columns), weight_bounds)
            random_weights_dict: Dict[str, float] = dict(zip(historical_prices.columns, random_weights))

            es = EfficientSemivariance(posterior_expected_returns,
                                       expected_returns.returns_from_prices(historical_prices))
            es.set_weights(random_weights_dict)
            performance = es.portfolio_performance(risk_free_rate=risk_free_rate)

            performance_df_mc.loc[portfolio_idx] = performance

            temp_weights_df = DataFrame(random_weights, index=historical_prices.columns, columns=[portfolio_idx])
            if weights_df_mc.empty:
                weights_df_mc = temp_weights_df
            else:
                weights_df_mc = pd.concat([weights_df_mc, temp_weights_df], axis=1)
        except OptimizationError:
            continue

    weights_df_mc = weights_df_mc.round(4)

    optimal_portfolio_index_mc = performance_df_mc['Sortino ratio'].idxmax()
    optimal_portfolio_mc = performance_df_mc.iloc[optimal_portfolio_index_mc].round(4)

    return performance_df_mc, weights_df_mc, optimal_portfolio_mc


def random_weights_with_bounds(
        n: int,
        weight_bounds: List[Tuple[float, float]]) -> np.ndarray:
    """
    Generate an array of random weights subject to provided bounds.

    Parameters:
    - n (int): The number of assets for which to generate random weights.
    - weight_bounds (List[Tuple[float, float]]): A list of tuples specifying the lower and upper bounds
        for each asset's weight. Each tuple is in the form of (lower_bound, upper_bound).

    Returns:
    - np.ndarray: An array of randomly generated asset weights that sum to 1, subject to the specified bounds.
    """
    lower_bounds, upper_bounds = zip(*weight_bounds)
    lower_bounds = np.array(lower_bounds)
    upper_bounds = np.array(upper_bounds)

    weights = np.zeros(n)
    remaining = 1.0  # Start with a sum of 1 to allocate

    for i in range(n - 1):  # Loop until the second last element
        low = max(lower_bounds[i], 0)  # Lower bound for this asset
        high = min(upper_bounds[i], remaining - np.sum(
            lower_bounds[i + 1:]))  # Upper bound while ensuring future lower bounds can be met

        if high > low:
            w = np.random.uniform(low, high)
        else:
            w = low  # high and low can be the same when tight bounds exist

        weights[i] = w
        remaining -= w

    # For the last weight, make sure it falls within the bounds.
    weights[-1] = remaining
    if weights[-1] > upper_bounds[-1]:
        weights[-1] = upper_bounds[-1]
    elif weights[-1] < lower_bounds[-1]:
        weights[-1] = lower_bounds[-1]

    return weights


def generate_optimal_portfolio(
        historical_prices: pd.DataFrame,
        posterior_expected_returns: pd.Series,
        weight_bounds: List[Tuple[float, float]],
        risk_free_rate: float,
        n_portfolios: int = 10000) -> Tuple[pd.DataFrame, int, pd.Series, pd.DataFrame]:
    """
    Generate optimal portfolio based on Monte Carlo simulations using Efficient Semivariance optimization.

    Parameters:
    - historical_prices (pd.DataFrame): DataFrame containing historical prices for each asset.
    - posterior_expected_returns (pd.Series): Series containing posterior expected returns for each asset.
    - weight_bounds (List[Tuple[float, float]]): List of tuples containing the lower and upper bounds for each asset.
    - risk_free_rate (float): Risk-free rate to use in performance calculation.
    - n_portfolios (int, optional): Number of random portfolios to generate. Default is 10000.

    Returns:
        Tuple[pd.DataFrame, int, pd.Series, pd.DataFrame]:
            - pd.DataFrame: DataFrame containing portfolio weights for each portfolio.
            - int: Index of the optimal portfolio.
            - pd.Series: Details of the optimal portfolio.
            - pd.DataFrame: DataFrame containing performance metrics for each portfolio.
    """
    # Initialize dataframes to store portfolio performances and weights
    performance_df_mc = pd.DataFrame(columns=['Expected annual return', 'Annual semi-deviation', 'Sortino ratio'])
    weights_df_mc = pd.DataFrame()

    # Loop through and generate portfolios
    for i in tqdm(range(n_portfolios)):
        try:
            # Generate random weights that obey the weight_bounds
            random_weights = random_weights_with_bounds(len(historical_prices.columns), weight_bounds)

            # Convert random_weights to dictionary
            random_weights_dict = dict(zip(historical_prices.columns, random_weights))

            es = EfficientSemivariance(posterior_expected_returns,
                                       expected_returns.returns_from_prices(historical_prices))
            es.set_weights(random_weights_dict)  # Set the weights here
            performance = es.portfolio_performance(risk_free_rate=risk_free_rate)

            # Store the performance
            performance_df_mc.loc[i] = performance

            # Convert weights to dataframe and store
            temp_weights_df = pd.DataFrame(random_weights, index=historical_prices.columns, columns=[i])
            if weights_df_mc.empty:
                weights_df_mc = temp_weights_df
            else:
                weights_df_mc = pd.concat([weights_df_mc, temp_weights_df], axis=1)
        except OptimizationError:
            continue
    weights_df_mc = weights_df_mc.round(4)

    # Find portfolio with highest Sortino ratio
    optimal_portfolio_index_mc = performance_df_mc['Sortino ratio'].idxmax()
    optimal_portfolio_mc = performance_df_mc.iloc[optimal_portfolio_index_mc].round(4)

    return weights_df_mc, optimal_portfolio_index_mc, optimal_portfolio_mc, performance_df_mc
