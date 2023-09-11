import bt
import numpy as np
import pandas as pd
from typing import Dict, Union


def optimal_strategy(
        weights_df_optimal: pd.DataFrame,
        optimal_portfolio_index: str,
        historical_prices: pd.DataFrame,
        strategy_name: str = 'Optimal Portfolio') -> bt.Backtest:
    """
    Creates and returns a backtest for an optimal strategy.

    Parameters:
    - weights_df_optimal (pd.DataFrame): DataFrame containing optimal weights
    - optimal_portfolio_index (str): Index name for the optimal portfolio in weights_df_optimal
    - historical_prices (pd.DataFrame): DataFrame containing historical prices
    - strategy_name (str, optional): Name of the strategy

    Returns:
    - bt.Backtest: Backtest object for the optimal strategy
    """
    target_weights: Dict = weights_df_optimal[optimal_portfolio_index].to_dict()
    portfolio_strategy = bt.Strategy(strategy_name, [
        bt.algos.RunMonthly(),
        bt.algos.WeighSpecified(**target_weights),
        bt.algos.Rebalance()
    ])
    return bt.Backtest(
        portfolio_strategy,
        historical_prices,
        strategy_name
    )


def benchmark_strategy(
        strategy_name: str,
        benchmark_portfolio: Dict[str, float],
        benchmark_prices: pd.DataFrame) -> bt.Backtest:
    """
    Creates and returns a backtest for a benchmark strategy.

    Parameters:
    - strategy_name (str): Name of the strategy
    - benchmark_portfolio (Dict[str, float]): Dictionary of asset weights
    - benchmark_prices (pd.DataFrame): DataFrame containing benchmark prices

    Returns:
    - bt.Backtest: Backtest object for the benchmark strategy
    """
    target_weights = benchmark_portfolio
    portfolio_strategy = bt.Strategy(strategy_name, [
        bt.algos.RunMonthly(),
        bt.algos.WeighSpecified(**target_weights),
        bt.algos.Rebalance()
    ])
    return bt.Backtest(
        portfolio_strategy,
        benchmark_prices,
        strategy_name
    )


def generate_random_strategy(strategy_name: str = 'Random Strategy') -> bt.Strategy:
    """
    Generates and returns a random strategy.

    Parameters:
    - strategy_name (str, optional): Name of the random strategy

    Returns:
    - bt.Strategy: Strategy object for the random strategy
    """
    return bt.Strategy(strategy_name,
                       [bt.algos.RunMonthly(),
                        bt.algos.SelectRandomly(),
                        bt.algos.WeighRandomly(),
                        bt.algos.Rebalance()])


class WeighRandomlyWithBounds(bt.Algo):
    """
    A class that extends bt.Algo to implement an algorithm for random weight allocation within bounds.
    """
    def __init__(self, weight_bounds: Union[list, np.ndarray], *args, **kwargs):
        """
        Initialize the algorithm with weight bounds.

        Parameters:
        - weight_bounds (Union[list, np.ndarray]): Weight bounds for the assets
        """
        self.weight_bounds = weight_bounds
        super(WeighRandomlyWithBounds, self).__init__(*args, **kwargs)

    def generate_random_weights(self, selected: list) -> pd.Series:
        """
        Generate random weights for selected assets within specified bounds.

        Parameters:
        - selected (list): List of selected asset names

        Returns:
        - pd.Series: Randomly generated weights for each selected asset
        """
        n = len(selected)

        weight_bounds_selected = self.weight_bounds[:n]
        lower_bounds, upper_bounds = zip(*weight_bounds_selected)

        lower_bounds = np.array(lower_bounds)
        upper_bounds = np.array(upper_bounds)

        weights = np.zeros(n)
        remaining = 1.0

        for i in range(n - 1):
            low = max(lower_bounds[i], 0)
            high = min(upper_bounds[i], remaining - np.sum(lower_bounds[i + 1:]))

            if high > low:
                w = np.random.uniform(low, high)
            else:
                w = low

            weights[i] = w
            remaining -= w

        weights[-1] = remaining

        return pd.Series(weights, index=selected)

    def __call__(self, target) -> bool:
        """
        Calls the algorithm with the current target (bt.Target).

        Parameters:
        - target (bt.Target): Current portfolio target

        Returns:
        - bool: Whether the algorithm succeeded
        """
        selected = target.temp['selected']
        if len(selected) == 0:
            target.temp['weights'] = pd.Series(dtype=float)
        else:
            weights = self.generate_random_weights(selected)
            target.temp['weights'] = weights
        return True


def generate_random_strategy_with_bounds(
        weight_bounds: Union[list, np.ndarray],
        strategy_name: str = 'Random Strategy with Bounds') -> bt.Strategy:
    """
    Generates and returns a random strategy with weight bounds.

    Parameters:
    - weight_bounds (Union[list, np.ndarray]): Weight bounds for the assets
    - strategy_name (str, optional): Name of the random strategy with bounds

    Returns:
    - bt.Strategy: Strategy object for the random strategy with bounds
    """
    return bt.Strategy(strategy_name,
                       [bt.algos.RunMonthly(),
                        bt.algos.SelectRandomly(),
                        WeighRandomlyWithBounds(weight_bounds),
                        bt.algos.Rebalance()])


def run_backtest_with_benchmark(
        strategy_backtest: bt.Backtest,
        benchmark_backtest: bt.Backtest,
        average_risk_free_rate: float) -> bt.backtest.Result:
    """
    Run backtests for the provided strategy and set of random benchmarks. Also, set the risk-free rate for the
    resulting backtests.

    Parameters:
    - strategy_backtest (bt.Backtest): Backtest object for the strategy
    - random_strategy (bt.Strategy): Strategy object for the random strategy
    - average_risk_free_rate (float): Average risk-free rate

    Returns:
    - bt.Result: Backtest result object
    """
    results_with_benchmark = bt.run(strategy_backtest, benchmark_backtest)
    results_with_benchmark.set_riskfree_rate(average_risk_free_rate)
    return results_with_benchmark


def run_backtest_with_random_benchmarks(
        strategy_backtest: bt.Backtest,
        random_strategy: bt.Strategy,
        average_risk_free_rate: float = 0.02,
        nsim: int = 1000) -> bt.backtest.Result:
    """
    Run backtests for the provided strategy and set of random benchmarks. Also, set the risk-free rate for the
    resulting backtests.

    Parameters:
    - strategy_backtest (bt.Backtest): bt.Backtest object for the strategy you want to test
    - random_strategy (bt.Strategy): bt.Strategy object for the random strategy you want to use for comparison
    - average_risk_free_rate (float): The risk-free rate to be set for the backtests
    - nsim (int, optional): Number of simulations for the random benchmarks

    Returns:
    - results_with_random_benchmarks: bt run results with random benchmarks and the risk-free rate set
    """
    results_with_random_benchmarks = bt.backtest.benchmark_random(strategy_backtest, random_strategy, nsim=nsim)
    results_with_random_benchmarks.set_riskfree_rate(average_risk_free_rate)
    return results_with_random_benchmarks


def display_results(results_object: bt.backtest.Result) -> None:
    """
    Displays the backtest results.

    Parameters:
    - results_object (bt.Result): Backtest result object

    Returns:
    - None
    """
    results_object.display()


def get_series_from_object(
        results_object: bt.backtest.Result,
        freq: str = 'd') -> pd.DataFrame:
    """
    Retrieves series from a backtest result object.

    Parameters:
    - results_object (bt.backtest.Result): Backtest result object
    - freq (str, optional): Frequency of the series

    Returns:
    - pd.DataFrame: Series from the result object
    """
    return results_object._get_series(freq)[1:]
