import ffn
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from scipy.stats import norm
from typing import Any, Tuple
from . import backtesting


def format_with_percent(x: Any, pos: Any) -> str:
    """
    Formats a number as a percentage string.

    Parameters:
    - x (Any): The number to be formatted.
    - pos (Any): The position (unused, but required by FuncFormatter).

    Returns:
    - str: The formatted percentage string.
    """
    return '{:,.0%}'.format(x)


def format_with_commas(x: Any, pos: Any) -> str:
    """
    Formats a number with commas as thousand separators.

    Parameters:
    - x (Any): The number to be formatted.
    - pos (Any): The position (unused, but required by FuncFormatter).

    Returns:
    - str: The formatted string.
    """
    return '${:,.0f}'.format(x)


def plot_average_return(
        average_historical_return: pd.Series,
        historical_prices: pd.DataFrame,
        **kwargs) -> None:
    """
    Plots the average historical returns as a bar chart.

    Parameters:
    - average_historical_return (pd.Series): The average return data.
    - historical_prices (pd.DataFrame): The historical price data for the date range in the title.

    Returns:
    - None
    """
    average_historical_return = average_historical_return.sort_values(ascending=False)
    with plt.style.context('dark_background'):
        fig, ax = plt.subplots()

        ax.bar(list(average_historical_return.index), average_historical_return.to_list(), **kwargs)
        formatter = FuncFormatter(format_with_percent)
        ax.yaxis.set_major_formatter(formatter)

        plt.title('Average Historical Return (%)\n{} to {}'.format(
            historical_prices.index[0].strftime('%Y-%m-%d'),
            historical_prices.index[-1].strftime('%Y-%m-%d')
        ))

        plt.grid()
        plt.show()


def plot_rebased_cumulative_value(
        historical_prices: pd.DataFrame,
        logy: bool = True,
        figsize: Tuple[int, int] = (15, 5),
        color_palette: str = 'colorblind',
        **kwargs) -> None:
    """
    Plots the rebased cumulative value of historical prices.

    Parameters:
    - historical_prices (pd.DataFrame): The historical price data.
    - figsize (tuple): The size of the figure. Defaults to (15, 5).
    - logy (bool): Whether to use a logarithmic scale on the y-axis. Defaults to True.
    - color_palette (str): The color palette to use for the plot. Defaults to 'colorblind'.

    Returns:
    - None
    """
    with plt.style.context('dark_background'):
        sns.set_palette(color_palette)
        ax = ffn.core.rebase(historical_prices).plot(
            title='Rebased Cumulative Value ($100 Initial Value)\n{} to {}'.format(
                historical_prices.index[0].strftime('%Y-%m-%d'),
                historical_prices.index[-1].strftime('%Y-%m-%d')
            ),
            logy=logy,
            figsize=figsize,
            **kwargs
        )

        formatter = FuncFormatter(format_with_commas)
        ax.yaxis.set_major_formatter(formatter)
        ax.set_xlabel('Date')
        plt.legend(title='Symbol')

        plt.grid()
        plt.show()


def plot_market_caps(
        market_caps: pd.Series,
        historical_prices: pd.DataFrame,
        color_palette: str = 'colorblind',
        **kwargs) -> None:
    """
    Plots the market capitalizations as a bar chart.

    Parameters:
    - market_caps (pd.Series): The market capitalizations data.
    - historical_prices (pd.DataFrame): The historical price data for the date range in the title.
    - color_palette (str): The color palette to use for the plot. Defaults to 'colorblind'.

    Returns:
    - None
    """
    market_caps = market_caps.sort_values(ascending=False)
    with plt.style.context('dark_background'):
        fig, ax = plt.subplots()

        sns.set_palette(color_palette)
        ax.bar(list(market_caps.index), market_caps.to_list(), **kwargs)
        formatter = FuncFormatter(format_with_commas)
        ax.yaxis.set_major_formatter(formatter)

        plt.title('Market Cap ($Billions)\nas of {}'.format(historical_prices.index[-1].strftime('%Y-%m-%d')))

        plt.grid()
        plt.show()


def plot_market_weights(
        market_weights: pd.Series,
        historical_prices: pd.DataFrame,
        color_palette: str = 'colorblind',
        **kwargs: Any) -> None:
    """
    Plots the market weights as a bar chart.

    Parameters:
    - market_weights (pd.Series): The market weights data.
    - historical_prices (pd.DataFrame): The historical price data for the date range in the title.
    - color_palette (str): The color palette to use for the plot. Defaults to 'colorblind'.

    Returns:
    - None
    """
    market_weights = market_weights.sort_values(ascending=False)
    with plt.style.context('dark_background'):
        fig, ax = plt.subplots()

        sns.set_palette(color_palette)
        ax.bar(list(market_weights.index), market_weights.to_list(), **kwargs)
        formatter = FuncFormatter(format_with_percent)
        ax.yaxis.set_major_formatter(formatter)

        plt.title('Market Weights (%)\nas of {}'.format(
            historical_prices.index[-1].strftime('%Y-%m-%d')
        ))

        plt.grid()
        plt.show()


def plot_correlation_matrix(
        covariance_matrix: pd.DataFrame,
        historical_prices: pd.DataFrame,
        color_scheme: str = 'white',
        title: str = 'Correlation Matrix') -> None:
    """
    Plots the correlation matrix from a given covariance matrix.

    Parameters:
    - covariance_matrix (pd.DataFrame): The covariance matrix.
    - historical_prices (pd.DataFrame): The historical price data for the date range in the title.
    - color_scheme (str): The color scheme to use for the plot. Defaults to 'white'.
    -

    Returns:
    - None
    """
    # Calculate the correlation matrix from the covariance matrix
    diagonal_sqrt = np.sqrt(np.diag(covariance_matrix))
    correlation_matrix = covariance_matrix / (diagonal_sqrt[:, None] * diagonal_sqrt[None, :])

    with plt.style.context('dark_background'):
        plt.figure(figsize=(8, 6))
        sns.set(font_scale=1)
        ax = sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5, fmt=".2f",
                         cbar_kws={'label': 'Correlation'})
        ax.figure.axes[-1].yaxis.label.set_color(color_scheme)  # Set color for colorbar label
        ax.figure.axes[-1].tick_params(axis='y', colors=color_scheme)  # Set color for colorbar tick marks
        plt.title(
            '{}\nas of {}'.format(
                title,
                historical_prices.index[-1].strftime('%Y-%m-%d')
            ), color=color_scheme)
        plt.xticks(color=color_scheme)
        plt.yticks(color=color_scheme)
        plt.xlabel('Tickers', color=color_scheme)
        plt.ylabel('Tickers', color=color_scheme)
        plt.show()


def plot_market_implied_expected_returns(
        market_implied_expected_returns: pd.Series,
        historical_prices: pd.DataFrame,
        color_palette: str = 'colorblind',
        **kwargs: Any) -> None:
    """
    Plots the market-implied expected returns as a bar chart.

    Parameters:
    - market_implied_expected_returns (pd.Series): The market-implied expected return data.
    - historical_prices (pd.DataFrame): The historical price data for the date range in the title.
    - color_palette (str): The color palette to use for the plot. Defaults to 'colorblind'.

    Returns:
    - None
    """
    market_implied_expected_returns = market_implied_expected_returns.sort_values(ascending=False)
    with plt.style.context('dark_background'):
        fig, ax = plt.subplots()

        sns.set_palette(color_palette)
        ax.bar(
            list(market_implied_expected_returns.index),
            market_implied_expected_returns.to_list(),
            **kwargs
        )

        formatter = FuncFormatter(format_with_percent)
        ax.yaxis.set_major_formatter(formatter)

        plt.title('Market-Implied Expected Return (%)\nas of {}'.format(
            historical_prices.index[0].strftime('%Y-%m-%d'),
            historical_prices.index[-1].strftime('%Y-%m-%d')
        ))

        plt.grid()
        plt.show()


def plot_posterior_expected_returns(
        posterior_expected_returns: pd.Series,
        historical_prices: pd.DataFrame,
        **kwargs) -> None:
    """
    Plots the posterior expected returns as a bar chart.

    Parameters:
    - posterior_expected_returns (pd.Series): The posterior expected return data.
    - historical_prices (pd.DataFrame): The historical price data for the date range in the title.

    Returns:
    - None
    """
    posterior_expected_returns = posterior_expected_returns.sort_values(ascending=False)
    with plt.style.context('dark_background'):
        fig, ax = plt.subplots()

        ax.bar(
            list(posterior_expected_returns.index),
            posterior_expected_returns.to_list(),
            **kwargs
        )

        formatter = FuncFormatter(format_with_percent)
        ax.yaxis.set_major_formatter(formatter)

        plt.title('Posterior Expected Return (%)\nas of {}'.format(
            historical_prices.index[0].strftime('%Y-%m-%d'),
            historical_prices.index[-1].strftime('%Y-%m-%d')
        ))

        plt.grid()
        plt.show()


def plot_efficient_frontier(
        performance_df: pd.DataFrame,
        optimal_portfolio_index: int,
        risk_free_rate: float,
        optimal_color: str = 'lime',
        figsize: tuple[float, float] = (12, 8)) -> None:
    """
    Plots the efficient frontier, capital market line, and highlights the optimal portfolio.

    Parameters:
    - performance_df (pd.DataFrame): DataFrame containing portfolio performances.
    - optimal_portfolio_index (int): The index of the optimal portfolio.
    - risk_free_rate (float): The risk-free rate used to plot the Capital Market Line.
    - optimal_color (str): The color to use for the optimal portfolio. Defaults to 'lime'.
    - figsize (tuple): The size of the figure. Defaults to (12, 8).

    Returns:
    - None
    """
    with plt.style.context('dark_background'):
        plt.figure(figsize=figsize)

        # Find the optimal portfolio
        optimal_portfolio = performance_df[optimal_portfolio_index]

        # Plot the efficient frontier
        plt.plot(
            performance_df.loc['Annual semi-deviation'],
            performance_df.loc['Expected annual return'],
            c='cyan', label='Efficient Frontier', zorder=2
        )
        plt.xlabel('Annual Semi-Deviation')
        plt.ylabel('Expected Annual Return')
        plt.title('Efficient Frontier and Capital Market Line')

        # Plot the optimal portfolio
        plt.scatter(
            optimal_portfolio['Annual semi-deviation'],
            optimal_portfolio['Expected annual return'],
            c=optimal_color, marker='*', label='Optimal Portfolio', zorder=3, s=200
        )

        # Plot the Capital Market Line
        x_range = np.linspace(0, max(performance_df.loc['Annual semi-deviation']), 100)
        y_cml = risk_free_rate + (
                (optimal_portfolio['Expected annual return'] - risk_free_rate) /
                optimal_portfolio['Annual semi-deviation']) * x_range
        plt.plot(x_range, y_cml, color='red', linestyle='--', label='Capital Market Line', zorder=1)

        # Format x and y-axis tick labels as percentages
        plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1%}'))
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))

        plt.legend()
        plt.grid()
        plt.show()


def plot_portfolios(
        performance_df_mc: pd.DataFrame,
        optimal_portfolio: pd.Series,
        performance_df_optimal: pd.DataFrame,
        risk_free_rate: float) -> None:
    """
    Plots the efficient frontier, random portfolios, the optimal portfolio, and the Capital Market Line.

    Parameters:
    - performance_df_mc (pd.DataFrame): DataFrame containing the performance metrics of Monte Carlo simulated
                                        portfolios.
    - optimal_portfolio (pd.Series): Series containing the performance metrics of the optimal portfolio.
    - performance_df_optimal (pd.DataFrame): DataFrame containing the efficient frontier.
    - risk_free_rate (float): The risk-free rate.

    Returns:
    - None
    """
    with plt.style.context('dark_background'):
        plt.figure(figsize=(12, 8))

        # Plot random portfolios
        plt.scatter(
            performance_df_mc['Annual semi-deviation'],
            performance_df_mc['Expected annual return'],
            c='gold', label='Random Portfolios', zorder=2, s=15
        )

        # Plot Optimal Portfolio
        plt.scatter(
            optimal_portfolio['Annual semi-deviation'],
            optimal_portfolio['Expected annual return'],
            c='purple', marker='*', label='Optimal Portfolio', zorder=4, s=200
        )

        # Plot Capital Market Line
        x_range = np.linspace(0, max(performance_df_mc['Annual semi-deviation']), 100)
        y_cml = risk_free_rate + (
                (optimal_portfolio['Expected annual return'] - risk_free_rate) /
                optimal_portfolio['Annual semi-deviation']) * x_range
        plt.plot(x_range, y_cml, color='red', linestyle='--', label='Capital Market Line', zorder=1)

        # Plot Efficient Frontier
        plt.plot(
            performance_df_optimal.loc['Annual semi-deviation'],
            performance_df_optimal.loc['Expected annual return'],
            c='cyan', label='Efficient Frontier', zorder=3, linewidth=3
        )

        plt.xlabel('Annual Semi-Deviation')
        plt.ylabel('Expected Annual Return')
        plt.title('Efficient Frontier, Random Portfolios, and Capital Market Line')
        plt.legend()
        plt.grid()
        plt.show()


def plot_backtest(
        results_object: object,
        logy: bool = False,
        figsize: Tuple[int, int] = (15, 5),
        color_palette: str = 'colorblind',
        color_scheme: str = 'white',
        title: str = 'Backtest Performance',
        **kwargs: Any) -> None:
    """
    Plots the backtest performance based on the provided results_object.

    Parameters:
    - results_object (object): The results object containing the backtest data.
    - logy (bool, optional): Whether to use a logarithmic scale on the y-axis. Defaults to False.
    - figsize (Tuple[int, int], optional): The size of the figure. Defaults to (15, 5).
    - color_palette (str, optional): The color palette to use for the plot. Defaults to 'colorblind'.
    - color_scheme (str, optional): The color scheme for the plot elements. Defaults to 'white'.
    - title (str, optional): The title for the plot. Defaults to 'Backtest Performance'.
    - **kwargs (Any, optional): Additional keyword arguments for the plotting function.

    Returns:
    - None
    """
    data = backtesting.get_series_from_object(results_object)

    with plt.style.context('dark_background'):
        sns.set_palette(color_palette)
        ax = data.plot(
            title=f'{title}\n{data.index[0].strftime("%Y-%m-%d")} to {data.index[-1].strftime("%Y-%m-%d")}',
            logy=logy,
            figsize=figsize,
            **kwargs
        )
        for spine in ax.spines.values():
            spine.set_edgecolor(color_scheme)
        ax.tick_params(axis='both', colors=color_scheme)
        ax.tick_params(axis='both', which='minor', colors=color_scheme)
        ax.set_title(f'{title}\n{data.index[0].strftime("%Y-%m-%d")} to {data.index[-1].strftime("%Y-%m-%d")}', color=color_scheme)
        formatter = FuncFormatter(format_with_commas)
        ax.yaxis.set_major_formatter(formatter)
        ax.set_xlabel('Date', color=color_scheme)
        plt.legend(labelcolor=color_scheme)
        plt.grid(color=color_scheme)
        plt.show()


def plot_security_weights(
        results_object: object,
        backtest: int = 1,
        title: str = 'Security Weights (%)',
        logy: bool = False,
        figsize: Tuple[int, int] = (15, 5),
        color_palette: str = 'colorblind',
        color_scheme: str = 'white',
        **kwargs: Any) -> None:
    """
    Plots the security weights.

    Parameters:
    - results_object (object): Object containing security weight results.
    - backtest (int, default=1): The backtest to plot. Defaults to 1.
    - title (str, default='Security Weights (%)'): Title of the plot.
    - logy (bool): Whether to use a logarithmic scale on the y-axis. Defaults to False.
    - figsize (tuple): The size of the figure. Defaults to (15, 5).
    - color_palette (str): The color palette to use for the plot. Defaults to 'colorblind'.
    - color_scheme (str): The color scheme to use for the plot. Defaults to 'white'.

    Returns:
    - None
    """
    if results_object is not None:
        data = results_object.get_security_weights(backtest - 1).iloc[1:]

        with plt.style.context('dark_background'):
            sns.set_palette(color_palette)
            ax = data.plot(
                logy=logy,
                figsize=figsize,
                **kwargs
            )
            ax.set_title(f'{title}\n{data.index[0].strftime("%Y-%m-%d")} to {data.index[-1].strftime("%Y-%m-%d")}',
                         color=color_scheme)
            formatter = FuncFormatter(format_with_percent)
            ax.yaxis.set_major_formatter(formatter)
            ax.set_xlabel('Date', color=color_scheme)
            ax.legend(loc='upper left', labelcolor=color_scheme)
            ax.tick_params(axis='both', colors=color_scheme)

            for spine in ax.spines.values():
                spine.set_edgecolor(color_scheme)

            plt.grid(color=color_scheme)
            plt.show()
    else:
        print("No results to plot security weights.")


def plot_normalized_histogram(
        results_object: Any,  # Replace with the specific type you expect
        statistic: str = "monthly_sharpe",
        figsize: Tuple[int, int] = (15, 5),
        bins: int = 20,
        alpha: float = 0.05,
        color_scheme: str = 'white',
        **kwargs: Any) -> None:
    """
    Plots the distribution of a given statistic with normalized x-axis.

    Args:
    - results_object (Any): Object containing backtest results.
    - statistic (str): Statistic to plot.
    - figsize (Tuple[int, int]): Figure size (width, height).
    - bins (int): Number of histogram bins.
    - alpha (float): Significance level for hypothesis test.
    - color_scheme (str): Color scheme for the plot.
    - kwargs (dict): Additional keyword arguments passed to the histogram function.

    Returns:
    - None
    """
    # Title lookup dictionary
    titles = {
        'daily_sharpe': 'Daily Normalized Sharpe Histogram',
        'daily_sortino': 'Daily Normalized Sortino Histogram',
        'monthly_sharpe': 'Monthly Normalized Sharpe Histogram',
        'monthly_sortino': 'Monthly Normalized Sortino Histogram',
        'yearly_sharpe': 'Yearly Normalized Sharpe Histogram',
        'yearly_sortino': 'Yearly Normalized Sortino Histogram'
    }

    r_stats = results_object.stats.iloc[:, 1:]
    b_stats = results_object.stats.iloc[:, :1].squeeze()

    with plt.style.context('dark_background'):
        if statistic not in r_stats.index:
            raise ValueError(
                "Invalid statistic. Valid statistics are the statistics in stats"
            )

        title = titles.get(statistic, f"{statistic} histogram")

        fig, ax = plt.subplots(figsize=figsize)  # Create a figure and axis

        ser = r_stats.loc[statistic]

        mean = ser.mean()
        std = ser.std()

        normalized_values = (ser - mean) / std

        ax.hist(normalized_values, bins=bins, density=True, zorder=1, edgecolor='black', **kwargs)
        ax.set_title(title, color=color_scheme)

        normalized_benchmark_statistic = (b_stats[statistic] - mean) / std
        ax.axvline(normalized_benchmark_statistic, linewidth=2, color="red", label="Portfolio Z-Score", zorder=4)

        ax.set_xlabel("Standard Deviations from Mean", color=color_scheme)
        ax.set_ylabel("Density", color=color_scheme)

        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_color(color_scheme)

        ax.tick_params(axis='both', colors=color_scheme)

        z = normalized_benchmark_statistic
        critical_value = norm.ppf(1 - alpha)

        ax.axvline(critical_value, linestyle='dashed', color='blue', label='Critical Z-Score', linewidth=2,
                   zorder=3)

        x_vals = np.linspace(normalized_values.min(), normalized_values.max(), 100)
        y_vals = norm.pdf(x_vals, 0, 1)
        ax.plot(x_vals, y_vals, label="Normal Distribution", color="green", linewidth=2, zorder=2)

        ax.annotate("Reject H0" if abs(z) > critical_value else "Fail to Reject H0",
                    xy=(z, 0), xytext=(z + 0.5 if abs(z) > critical_value else z - 2, 0.10),
                    arrowprops=dict(arrowstyle="->", color='magenta'), fontsize=10, color='magenta')

        explanation = "Reject H0: Portfolio outperforms random." if abs(
            z) > critical_value else "Fail to Reject H0: Portfolio does not outperform random."

        ax.text(0.02, 0.9, f"Alpha: {alpha}", transform=ax.transAxes, fontsize=10, color=color_scheme)
        ax.text(0.02, 0.85, f"Portfolio Z-Score: {z:.2f}", transform=ax.transAxes, fontsize=10, color=color_scheme)
        ax.text(0.02, 0.8, f"Critical Z-Score: {critical_value:.2f}", transform=ax.transAxes, fontsize=10,
                color=color_scheme)
        ax.text(0.02, 0.75, explanation, transform=ax.transAxes, fontsize=10, color=color_scheme,
                verticalalignment="top", bbox=dict(facecolor='black', alpha=0.5))

        ax.legend(loc='upper right', facecolor='black')
        for text in ax.legend().get_texts():
            text.set_color(color_scheme)

        for spine in ax.spines.values():
            spine.set_edgecolor(color_scheme)

        plt.show()


def plot_random_portfolios(
        performance_df_mc: pd.DataFrame,
        optimal_portfolio: pd.Series,
        performance_df_optimal: pd.DataFrame,
        risk_free_rate: float,
        optimal_color: str = 'lime',
        figsize: tuple[float, float] = (12, 8)) -> None:
    """
    Plot the Efficient Frontier, Random Portfolios, Optimal Portfolio, and the Capital Market Line.

    Parameters:
    - performance_df_mc (pd.DataFrame): DataFrame containing the performance metrics of the Monte Carlo portfolios.
    - optimal_portfolio (pd.Series): Series containing the performance metrics of the optimal portfolio.
    - performance_df_optimal (pd.DataFrame): DataFrame containing the performance metrics along the efficient frontier.
    - risk_free_rate (float): The risk-free rate to be used for the Capital Market Line.
    - optimal_color (str): The color to be used for the optimal portfolio. Defaults to 'lime'.
    - figsize (tuple): The size of the figure. Defaults to (12, 8).

    Returns:
    - None: This function plots the graph and does not return any value.
    """
    with plt.style.context('dark_background'):
        plt.figure(figsize=figsize)

        # Plot random portfolios
        plt.scatter(
            performance_df_mc['Annual semi-deviation'],
            performance_df_mc['Expected annual return'],
            c='gold', label='Random Portfolios', zorder=2, s=15
        )

        # Plot Optimal Portfolio
        plt.scatter(
            optimal_portfolio['Annual semi-deviation'],
            optimal_portfolio['Expected annual return'],
            c=optimal_color, marker='*', label='Optimal Portfolio', zorder=4, s=200
        )

        # Plot Capital Market Line
        x_range = np.linspace(0, max(performance_df_optimal.loc['Annual semi-deviation']), 100)
        y_cml = risk_free_rate + (
                (optimal_portfolio['Expected annual return'] - risk_free_rate) /
                optimal_portfolio['Annual semi-deviation']) * x_range
        plt.plot(x_range, y_cml, color='red', linestyle='--', label='Capital Market Line', zorder=1)

        # Plot Efficient Frontier
        plt.plot(
            performance_df_optimal.loc['Annual semi-deviation'],
            performance_df_optimal.loc['Expected annual return'],
            c='cyan', label='Efficient Frontier', zorder=3, linewidth=3)

        # Format x and y-axis tick labels as percentages
        plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1%}'))
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))

        # Finish formatting the plot
        plt.xlabel('Annual Semi-Deviation')
        plt.ylabel('Expected Annual Return')
        plt.title('Efficient Frontier, Random Portfolios, and Capital Market Line')
        plt.legend()
        plt.grid()
        plt.show()


def plot_dev_chart(
        results_with_random_benchmarks: Any,  # Replace with the specific type you expect
        statistic: str = 'monthly_sharpe',
        backtest: int = 1,
        plot: str = 'security_weights') -> None:
    """
    Plots either security weights or cumulative return based on sort order of some statistic.

    Args:
    - results_with_random_benchmarks (Any): Object containing backtest results.
    - statistic (str): The statistic to sort by. Default is 'monthly_sharpe'.
    - backtest (int): The backtest index to plot.
    - plot (str): The type of plot to display ('security_weights' or 'cumulative_return').

    Returns:
    - None
    """
    random_sortinos = results_with_random_benchmarks.stats.loc[statistic].sort_values(ascending=False)
    print(random_sortinos)

    if plot == 'security_weights':
        plot_security_weights(results_with_random_benchmarks, title='Security Weights (%)', backtest=backtest)
        plt.grid()
        plt.show()
    elif plot == 'cumulative_return':
        plt.plot((backtesting.get_series_from_object(results_with_random_benchmarks)[random_sortinos.index[0]]))
        plt.grid()
        plt.show()
    else:
        print("Invalid 'plot' parameter. Choose either 'security_weights' or 'cumulative_return'.")


def plot_optimal_portfolio(
        optimal_weights: pd.Series,
        color_palette: str = 'colorblind',
        **kwargs) -> None:
    """
    Plots the optimal portfolio as a bar chart.

    Parameters:
    - optimal_weights (pd.Series): The optimal portfolio weightings.
    - color_palette (str): The color palette to use for the plot. Defaults to 'colorblind'.

    Returns:
    - None
    """
    optimal_weights = optimal_weights.sort_values(ascending=False)
    with plt.style.context('dark_background'):
        fig, ax = plt.subplots()

        sns.set_palette(color_palette)
        ax.bar(list(optimal_weights.index), optimal_weights.to_list(), **kwargs)
        formatter = FuncFormatter(format_with_percent)
        ax.yaxis.set_major_formatter(formatter)

        plt.title('Optimal Portfolio Weightings (%)')

        plt.grid()
        plt.show()
