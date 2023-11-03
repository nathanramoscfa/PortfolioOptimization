import pandas as pd
from yahooquery import Ticker
from typing import List, Dict, Tuple, Optional
from pypfopt import expected_returns

# Constants
BILLION = 1_000_000_000


def get_historical_prices(
        tickers: List[str],
        period: str = 'max',
        start_date: Optional[str] = None,
        end_date: Optional[str] = None) -> pd.DataFrame:
    """Retrieve historical prices for a list of tickers.

    Parameters:
    - tickers (List[str]): List of stock tickers to retrieve data for.
    - period (str, optional): Time period to retrieve data for. Defaults to 'max'.
    - start_date (str, optional): Start date for data retrieval. Defaults to None.
    - end_date (str, optional): End date for data retrieval. Defaults to None.

    Returns:
    - pd.DataFrame: DataFrame containing historical prices.
    """
    ticker_string = ' '.join(tickers)
    data = Ticker(ticker_string).history(period=period, start=start_date, end=end_date)['adjclose'].reset_index()
    return format_historical_data(data)


def format_historical_data(data: pd.DataFrame) -> pd.DataFrame:
    """Format the historical data DataFrame.

    Parameters:
    - data (pd.DataFrame): Raw historical data.

    Returns:
    - pd.DataFrame: Formatted historical data.
    """
    df = data.pivot(index='date', columns='symbol', values='adjclose').dropna()
    df.index = pd.to_datetime(df.index).tz_localize(None)  # Make timestamps tz-naive
    return df.sort_index(axis=1)


def get_summary_profile(tickers: List[str]) -> pd.DataFrame:
    """
    Fetch summary profile for a list of stock tickers.

    Parameters:
    - tickers (List[str]): List of stock tickers to fetch summary profile for.

    Returns:
    - pd.DataFrame: A DataFrame containing summary profile for each ticker,
        sorted by ticker symbol.
    """
    ticker_string = ' '.join(tickers)
    data_dict = Ticker(ticker_string).summary_profile
    summary_profile = pd.DataFrame.from_dict(data_dict, orient='index')
    return summary_profile.transpose().sort_index(axis=1)


def get_summary_details(tickers: List[str]) -> pd.DataFrame:
    """
    Fetch summary details for a list of stock tickers.

    Parameters:
    - tickers (List[str]): List of stock tickers to fetch details for.

    Returns:
    - pd.DataFrame: A DataFrame containing summary details for each ticker, sorted by ticker symbol.
    """
    ticker_string = ' '.join(tickers)
    data_dict = Ticker(ticker_string).summary_detail
    summary_detail = pd.DataFrame.from_dict(data_dict, orient='index')
    return summary_detail.transpose().sort_index(axis=1)


def get_key_stats(tickers: List[str]) -> pd.DataFrame:
    """
    Fetch key statistics for a list of tickers.

    Parameters:
    - tickers (List[str]): List of tickers to fetch details for.

    Returns:
    - pd.DataFrame: A DataFrame containing key statistics for each ticker, sorted by ticker symbol.
    """
    ticker_string = ' '.join(tickers)
    data_dict = Ticker(ticker_string).key_stats
    key_stats = pd.DataFrame.from_dict(data_dict, orient='index')
    return key_stats.transpose().sort_index(axis=1)


def get_earnings_trend(tickers: List[str]) -> dict:
    """
    Extracts 'earningsEstimate', 'revenueEstimate', 'epsTrend', and 'epsRevisions'
    for each ticker into their own DataFrames, structured into a dictionary with keys
    indicating the timeframe and data section.

    Parameters:
    - tickers (List[str]): List of stock tickers.

    Returns:
    - dict: A dictionary containing DataFrames for each of the specified sections
      and ticker, keyed by timeframe and data section.
    """

    def extract_section(section_data, section_name, period):
        data = [item.get(section_name, {}) for item in section_data if section_name in item]
        if period < len(data):
            return pd.DataFrame(data[period], index=[0])
        return pd.DataFrame(columns=["avg", "low", "high", "yearAgoEps", "numberOfAnalysts", "growth"])

    def download_earnings_trend() -> pd.DataFrame:
        ticker_string = ' '.join(tickers)
        data_dict = Ticker(ticker_string).earnings_trend
        earnings_trend = pd.DataFrame.from_dict(data_dict, orient='index')
        return earnings_trend.transpose().sort_index(axis=1)

    # Download earnings trend data
    df = download_earnings_trend()

    # Initialize the dictionary to hold the DataFrames
    dataframe_dict = {}

    for ticker in tickers:
        ticker_series = df.get(ticker)

        if isinstance(ticker_series, pd.Series):
            ticker_data = ticker_series.to_dict()
        else:
            continue

        if 'trend' in ticker_data and isinstance(ticker_data['trend'], list):
            ticker_trend_data = ticker_data['trend']
        else:
            print(f"No 'trend' data for ticker: {ticker}")
            continue

        # Each section is split into current quarter/year and next quarter/year
        periods = ['0q', '+1q', '0y', '+1y']
        for period_idx, period_prefix in enumerate(periods):
            for section in ['earningsEstimate', 'revenueEstimate', 'epsTrend', 'epsRevisions']:
                # Key is constructed as per the required format
                key = f'{period_prefix}_{section}'
                section_df = extract_section(ticker_trend_data, section, period_idx)
                if not section_df.empty:
                    section_df['ticker'] = ticker
                    # Combine with any existing DataFrame for the same section and period
                    if key in dataframe_dict:
                        dataframe_dict[key] = pd.concat([dataframe_dict[key], section_df])
                    else:
                        dataframe_dict[key] = section_df

    # Set the index for each DataFrame in the dictionary
    for key, df in dataframe_dict.items():
        df.set_index(['ticker'], inplace=True)

    return dataframe_dict


def get_revisions(dataframe_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """
    Get the revisions for each ticker in a dictionary of dataframes.

    Parameters:
    - dataframe_dict (Dict[str, pd.DataFrame]): Dictionary of dataframes containing various financial metrics.

    Returns:
    - Dict[str, pd.DataFrame]: Dictionary of dataframes containing revisions for each ticker.
    """
    eps_trend_dfs = {key: df for key, df in dataframe_dict.items() if 'epsTrend' in key}

    revision_dfs = {}

    for key, df in eps_trend_dfs.items():
        revisions = df.loc[:, '7daysAgo':].apply(lambda x: (df['current'] - x) / x)
        revisions['ticker'] = df.index  # Add the ticker index as a column
        revisions.set_index('ticker', inplace=True)  # Set the ticker as the index
        revision_dfs[key] = revisions

    return revision_dfs


def get_current_prices(tickers: List[str]) -> pd.DataFrame:
    """
    Fetch the current prices for a list of stock tickers.

    Parameters:
    - tickers (List[str]): List of stock tickers to fetch prices for.

    Returns:
    - pd.DataFrame: A DataFrame containing the current prices for each ticker, sorted by ticker symbol.
    """
    ticker_string = ' '.join(tickers)
    data_dict = Ticker(ticker_string).price
    current_prices = pd.DataFrame.from_dict(data_dict, orient='index')
    current_prices = current_prices.transpose()
    return current_prices.sort_index(axis=1)


def get_risk_free_rate(ticker: str = '^TNX') -> Tuple[float, str]:
    """
    Fetch the risk-free rate from a specific ticker, typically a Treasury note yield.

    Parameters:
    - ticker (str, optional): The ticker symbol for the risk-free rate, defaults to '^TNX' for 10-year Treasury note
                              yield.

    Returns:
    - Tuple[float, str]: A tuple containing the risk-free rate as a float and the long name of the risk-free rate
                         source.
    """
    data_dict = Ticker(ticker).price
    risk_free_rate = pd.DataFrame.from_dict(data_dict, orient='index').transpose()
    risk_free_rate_name = risk_free_rate.loc['longName'].squeeze()
    risk_free_rate = round(risk_free_rate.loc['regularMarketPrice'].squeeze() / 100, 4)
    return risk_free_rate, risk_free_rate_name


def get_historical_risk_free_rate(ticker: str = '^TNX', period: str = 'max', start_date: str = None,
                                  end_date: str = None) -> Tuple[pd.DataFrame, str]:
    """
    Fetch historical risk-free rates for a specific period, start date, and end date.

    Parameters:
    - ticker (str, optional): The ticker symbol for the risk-free rate, defaults to '^TNX' for 10-year Treasury note
                              yield.
    - period (str, optional): The period for fetching historical data, defaults to 'max'.
    - start_date (str, optional): The start date for fetching historical data. Defaults to None.
    - end_date (str, optional): The end date for fetching historical data. Defaults to None.

    Returns:
    - Tuple[pd.DataFrame, str]: A DataFrame containing historical risk-free rates and the name of the risk-free rate
                                source.
    """
    if start_date is None:
        historical_risk_free_rate = get_historical_prices([ticker], period) / 100
    else:
        historical_risk_free_rate = get_historical_prices([ticker], start_date=start_date, end_date=end_date) / 100

    df = pd.DataFrame.from_dict(Ticker(ticker).price, orient='index').transpose()
    risk_free_rate_name = df.loc['longName'].squeeze()

    historical_risk_free_rate.index = historical_risk_free_rate.index.normalize()
    return historical_risk_free_rate, risk_free_rate_name


def get_historical_data(
        historical_prices: pd.DataFrame,
        benchmark_prices: pd.DataFrame,
        historical_risk_free_rate: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Process historical data to align time frames and synchronize data points for prices, benchmarks, and
    risk-free rates.

    Parameters:
    - historical_prices (pd.DataFrame): DataFrame containing historical prices.
    - benchmark_prices (pd.DataFrame): DataFrame containing benchmark prices.
    - historical_risk_free_rate (pd.DataFrame): DataFrame containing historical risk-free rates.

    Returns:
    - Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Tuple containing DataFrames for processed historical prices,
    - benchmark prices, and historical risk-free rates.
    """
    # Convert the index to a datetime object with only the date component
    historical_prices.index = pd.to_datetime(historical_prices.index).date
    benchmark_prices.index = pd.to_datetime(benchmark_prices.index).date
    historical_risk_free_rate.index = pd.to_datetime(historical_risk_free_rate.index).date

    # Find the common dates between all three dataframes and convert to a list
    common_dates = list(
        set(historical_prices.index) & set(benchmark_prices.index) & set(historical_risk_free_rate.index))

    # Filter the dataframes to keep only the common dates
    historical_prices_filtered = historical_prices.loc[common_dates]
    benchmark_prices_filtered = benchmark_prices.loc[common_dates]
    historical_risk_free_rate_filtered = historical_risk_free_rate.loc[common_dates]

    # Sort the dataframes by the index (date)
    historical_prices_filtered = historical_prices_filtered.sort_index()
    benchmark_prices_filtered = benchmark_prices_filtered.sort_index()
    historical_risk_free_rate_filtered = historical_risk_free_rate_filtered.sort_index()

    # Convert the index to a DatetimeIndex and keep only the date component
    historical_prices_filtered.index = pd.to_datetime(historical_prices_filtered.index)
    benchmark_prices_filtered.index = pd.to_datetime(benchmark_prices_filtered.index)
    historical_risk_free_rate_filtered.index = pd.to_datetime(historical_risk_free_rate_filtered.index)

    return historical_prices_filtered, benchmark_prices_filtered, historical_risk_free_rate_filtered


def get_weight_bounds(portfolio_tickers: List[str], weight_bounds: List[Tuple[float, float]]) -> \
        List[Tuple[float, float]]:
    """
    Process and align weight bounds to portfolio tickers.

    Parameters:
    - portfolio_tickers (List[str]): List of ticker symbols in the portfolio.
    - weight_bounds (List[Tuple[float, float]]): List of weight bounds corresponding to portfolio tickers.

    Returns:
    - List[Tuple[float, float]]: A list of weight bounds, sorted in the order of sorted portfolio tickers.
    """
    ticker_to_bounds = dict(zip(portfolio_tickers, weight_bounds))
    portfolio_tickers = sorted(portfolio_tickers)
    return [ticker_to_bounds[ticker] for ticker in portfolio_tickers]


def get_average_risk_free_rate(historical_risk_free_rate: pd.DataFrame) -> float:
    """
    Calculate the average risk-free rate based on historical risk-free rates.

    Parameters:
    - historical_risk_free_rate (pd.DataFrame): DataFrame containing historical risk-free rates.

    Returns:
    - float: The average risk-free rate, rounded to 4 decimal places.
    """
    return float(round(historical_risk_free_rate.mean().squeeze(), 4))


def get_market_caps(summary_detail: pd.DataFrame) -> pd.Series:
    """
    Extract market capitalizations from summary detail DataFrame.

    Parameters:
    - summary_detail (pd.DataFrame): DataFrame containing various financial metrics including market capitalization.

    Returns:
    - pd.Series: A Series containing market capitalizations for each ticker, sorted by ticker symbol.
    """
    market_caps = summary_detail.copy().loc['marketCap']
    for ticker in market_caps.index:
        if market_caps[ticker] == {}:
            market_caps[ticker] = summary_detail.loc['totalAssets', ticker]
    return (market_caps.astype('float64') / 1000000000).sort_index()


def get_market_cap_weights(market_caps: pd.Series) -> pd.Series:
    """
    Calculate market capitalization-based weights for a portfolio.

    Parameters:
    - market_caps (pd.Series): A Series containing the market capitalizations for each ticker.

    Returns:
    - pd.Series: A Series containing the weight of each ticker based on its market capitalization, rounded to 4
                 decimal places.
    """
    return (market_caps / market_caps.sum()).astype(float).round(4)


def get_market_prices(
        historical_prices: pd.DataFrame,
        market_weights: pd.Series) -> pd.Series:
    """
    Calculate the market prices based on the historical prices and market cap weights.

    Parameters:
    - historical_prices (pd.DataFrame): DataFrame containing historical prices for various tickers.
    - market_weights (pd.Series): A Series containing the weight of each ticker based on its market capitalization.

    Returns:
    - pd.Series: A Series containing the calculated market prices over time.
    """
    market_prices = historical_prices / historical_prices.iloc[0] * 100
    return (market_weights * market_prices).sum(axis=1)


def get_average_historical_return(historical_prices: pd.DataFrame) -> float:
    return round(expected_returns.mean_historical_return(historical_prices), 4)


def get_names(current_prices: pd.DataFrame) -> List[str]:
    """
    Sorts the 'shortName' of tickers based on their index in a given DataFrame.

    Parameters:
    - current_prices (pd.DataFrame): DataFrame containing 'shortName' as one of the columns.

    Returns:
    - List[str]: List of sorted 'shortName'.
    """
    return current_prices.loc['shortName'].sort_index()


def get_benchmark_portfolio(benchmark_portfolio: Dict[str, float]) -> Dict[str, float]:
    """
    Sorts the given benchmark portfolio dictionary by its keys.

    Parameters:
    - benchmark_portfolio (Dict[str, float]): A dictionary containing the portfolio's tickers and their respective
                                              weights.

    Returns:
    - Dict[str, float]: Sorted dictionary by ticker name.
    """
    sorted_benchmark = {key: value for key, value in sorted(benchmark_portfolio.items())}
    return sorted_benchmark
