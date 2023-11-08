import time
import requests
import pandas as pd
from bs4 import BeautifulSoup
from yahooquery import Ticker
from typing import List, Dict, Tuple, Optional
from pypfopt import expected_returns
from tqdm import tqdm

# Constants
BILLION = 1_000_000_000
SAFE_LIMIT = 20  # Safe number of tickers to process per minute to stay below the rate limit


def get_historical_prices(
        tickers: List[str],
        period: str = 'max',
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
) -> pd.DataFrame:
    """Retrieve historical prices for a list of tickers while considering rate limits.

    Parameters:
    - tickers (List[str]): List of stock tickers to retrieve data for.
    - period (str, optional): Time period to retrieve data for. Defaults to 'max'. Options include: '1d', '5d', '7d',
        '60d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max'
    - start_date (str, optional): Start date for data retrieval. Defaults to None.
    - end_date (str, optional): End date for data retrieval. Defaults to None.

    Returns:
    - pd.DataFrame: DataFrame containing historical prices.
    """

    wait_time = 60 / SAFE_LIMIT  # time to wait between requests

    all_data = []

    for chunk in tqdm([tickers[i:i + SAFE_LIMIT] for i in range(0, len(tickers), SAFE_LIMIT)], desc='Downloading'):
        ticker_string = ' '.join(chunk)
        data = Ticker(ticker_string).history(period=period, start=start_date, end=end_date)['adjclose'].reset_index()
        all_data.append(data)

        # Wait before making the next request if not processing the last chunk
        if len(chunk) == SAFE_LIMIT and len(tickers) > SAFE_LIMIT:
            time.sleep(wait_time)

    # Concatenate all the dataframes into one
    final_data = pd.concat(all_data, ignore_index=True)
    df = final_data.pivot(index='date', columns='symbol', values='adjclose').dropna()
    df.index = pd.to_datetime(df.index).tz_localize(None)  # Make timestamps tz-naive
    return df.sort_index(axis=1)


def get_summary_profile(
        tickers: List[str]
) -> pd.DataFrame:
    """
    Fetch summary profile for a list of stock tickers with rate limiting.

    Parameters:
    - tickers (List[str]): List of stock tickers to fetch summary profile for.

    Returns:
    - pd.DataFrame: A DataFrame containing summary profile for each ticker,
        sorted by ticker symbol.
    """

    wait_time = 60 / SAFE_LIMIT  # time to wait between requests
    all_data = []

    # Process in chunks according to the safe limit
    for chunk in tqdm(
            [tickers[i:i + SAFE_LIMIT] for i in range(0, len(tickers), SAFE_LIMIT)], desc='Fetching Profiles'
    ):
        ticker_string = ' '.join(chunk)
        data_dict = Ticker(ticker_string).summary_profile
        summary_profile = pd.DataFrame.from_dict(data_dict, orient='index')
        all_data.append(summary_profile)

        # Wait before making the next request if not processing the last chunk
        if len(chunk) == SAFE_LIMIT and len(tickers) > SAFE_LIMIT:
            time.sleep(wait_time)

    # Concatenate all the dataframes into one
    final_data = pd.concat(all_data)

    return final_data.sort_index(axis=1)


def get_summary_details(
        tickers: List[str]
) -> pd.DataFrame:
    """
    Fetch summary details for a list of stock tickers with rate limiting.

    Parameters:
    - tickers (List[str]): List of stock tickers to fetch details for.

    Returns:
    - pd.DataFrame: A DataFrame containing summary details for each ticker, sorted by ticker symbol.
    """

    wait_time = 60 / SAFE_LIMIT  # time to wait between requests
    all_data = []

    # Process in chunks according to the safe limit
    for chunk in tqdm([tickers[i:i + SAFE_LIMIT] for i in range(0, len(tickers), SAFE_LIMIT)], desc='Fetching Details'):
        ticker_string = ' '.join(chunk)
        data_dict = Ticker(ticker_string).summary_detail
        summary_detail = pd.DataFrame.from_dict(data_dict, orient='index').transpose()
        all_data.append(summary_detail)

        # Wait before making the next request if not processing the last chunk
        if len(chunk) == SAFE_LIMIT and len(tickers) > SAFE_LIMIT:
            time.sleep(wait_time)

    # Concatenate all the dataframes into one
    final_data = pd.concat(all_data)

    return final_data.sort_index(axis=1)


def get_key_stats(
        tickers: List[str]
) -> pd.DataFrame:
    """
    Fetch key statistics for a list of tickers with rate limiting.

    Parameters:
    - tickers (List[str]): List of tickers to fetch key statistics for.

    Returns:
    - pd.DataFrame: A DataFrame containing key statistics for each ticker, sorted by ticker symbol.
    """

    wait_time = 60 / SAFE_LIMIT  # time to wait between requests
    all_data = []

    # Process in chunks according to the safe limit
    for chunk in tqdm(
            [tickers[i:i + SAFE_LIMIT] for i in range(0, len(tickers), SAFE_LIMIT)], desc='Fetching Key Stats'
    ):
        ticker_string = ' '.join(chunk)
        data_dict = Ticker(ticker_string).key_stats
        key_stats = pd.DataFrame.from_dict(data_dict, orient='index').transpose()
        all_data.append(key_stats)

        # Wait before making the next request if not processing the last chunk
        if len(chunk) == SAFE_LIMIT and len(tickers) > SAFE_LIMIT:
            time.sleep(wait_time)

    # Concatenate all the dataframes into one
    final_data = pd.concat(all_data)

    return final_data.sort_index(axis=1)


def get_earnings_trend(
        tickers: List[str]
) -> dict:
    """
    Extracts 'earningsEstimate', 'revenueEstimate', 'epsTrend', and 'epsRevisions'
    for each ticker into their own DataFrames, structured into a dictionary with keys
    indicating the timeframe and data section, while respecting rate limits.

    Parameters:
    - tickers (List[str]): List of stock tickers.

    Returns:
    - dict: A dictionary containing DataFrames for each of the specified sections
      and ticker, keyed by timeframe and data section.
    """

    wait_time = 60 / SAFE_LIMIT  # time to wait between requests
    dataframe_dict = {}

    for chunk in tqdm(
            [tickers[i:i + SAFE_LIMIT] for i in range(0, len(tickers), SAFE_LIMIT)], desc='Fetching Earnings Trend'
    ):
        ticker_string = ' '.join(chunk)
        data_dict = Ticker(ticker_string).earnings_trend

        # Process each ticker's earnings trend data
        for ticker in chunk:
            earnings_trend = data_dict.get(ticker, {}).get('trend', [])
            # Each section is split into current quarter/year and next quarter/year
            periods = ['0q', '+1q', '0y', '+1y']
            for period_idx, period_prefix in enumerate(periods):
                for section in ['earningsEstimate', 'revenueEstimate', 'epsTrend', 'epsRevisions']:
                    # Key is constructed as per the required format
                    key = f'{period_prefix}_{section}'
                    section_df = extract_section(earnings_trend, section, period_idx)
                    if not section_df.empty:
                        section_df['ticker'] = ticker
                        # Combine with any existing DataFrame for the same section and period
                        if key in dataframe_dict:
                            dataframe_dict[key] = pd.concat([dataframe_dict[key], section_df])
                        else:
                            dataframe_dict[key] = section_df

        # Wait before making the next request if not processing the last chunk
        if len(chunk) == SAFE_LIMIT and len(tickers) > SAFE_LIMIT:
            time.sleep(wait_time)

    # Set the index for each DataFrame in the dictionary
    for key, df in dataframe_dict.items():
        df.set_index(['ticker'], inplace=True)

    return dataframe_dict


def extract_section(
        section_data: List[dict],
        section_name: str,
        period: int
) -> pd.DataFrame:
    """
    Extract a section from the earnings trend data for a specific period.

    Parameters:
    - section_data (List[dict]): List of dictionaries containing earnings trend data.
    - section_name (str): Name of the section to extract.
    - period (int): Index of the period to extract.

    Returns:
    - pd.DataFrame: DataFrame containing the specified section for the specified period.
    """
    data = [item.get(section_name, {}) for item in section_data if section_name in item]
    if period < len(data):
        return pd.DataFrame(data[period], index=[0])
    return pd.DataFrame(columns=["avg", "low", "high", "yearAgoEps", "numberOfAnalysts", "growth"])


def get_revisions(
        dataframe_dict: Dict[str, pd.DataFrame]
) -> Dict[str, pd.DataFrame]:
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


def get_current_prices(
        tickers: List[str]
) -> pd.DataFrame:
    """
    Fetch the current prices for a list of stock tickers with rate limiting.

    Parameters:
    - tickers (List[str]): List of stock tickers to fetch prices for.

    Returns:
    - pd.DataFrame: A DataFrame containing the current prices for each ticker, sorted by ticker symbol.
    """

    wait_time = 60 / SAFE_LIMIT  # time to wait between requests if necessary
    all_data = []

    # Process in chunks according to the safe limit
    for chunk in tqdm(
            [tickers[i:i + SAFE_LIMIT] for i in range(0, len(tickers), SAFE_LIMIT)], desc='Fetching Current Prices'
    ):
        ticker_string = ' '.join(chunk)
        data_dict = Ticker(ticker_string).price
        current_prices = pd.DataFrame.from_dict(data_dict, orient='index').transpose()
        all_data.append(current_prices)

        # Wait before making the next request if not processing the last chunk
        if len(chunk) == SAFE_LIMIT and len(tickers) > SAFE_LIMIT:
            time.sleep(wait_time)

    # Concatenate all the dataframes into one and then transpose
    final_data = pd.concat(all_data).transpose()

    return final_data.sort_index(axis=1)


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


def get_historical_risk_free_rate(
        ticker: str = '^TNX',
        period: str = 'max',
        start_date: str = None,
        end_date: str = None
) -> Tuple[pd.DataFrame, str]:
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
        historical_risk_free_rate: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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


def get_weight_bounds(
        portfolio_tickers: List[str],
        weight_bounds: List[Tuple[float, float]]
) -> [Tuple[float, float]]:
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


def get_average_risk_free_rate(
        historical_risk_free_rate: pd.DataFrame
) -> float:
    """
    Calculate the average risk-free rate based on historical risk-free rates.

    Parameters:
    - historical_risk_free_rate (pd.DataFrame): DataFrame containing historical risk-free rates.

    Returns:
    - float: The average risk-free rate, rounded to 4 decimal places.
    """
    return float(round(historical_risk_free_rate.mean().squeeze(), 4))


def get_market_caps(
        summary_detail: pd.DataFrame
) -> pd.Series:
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


def get_market_cap_weights(
        market_caps: pd.Series
) -> pd.Series:
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
        market_weights: pd.Series
) -> pd.Series:
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


def get_average_historical_return(
        historical_prices: pd.DataFrame
) -> float:
    """
    Calculate the average historical return based on historical prices.

    Parameters:
    - historical_prices(pd.DataFrame): DataFrame containing historical prices for various tickers.

    Returns:
    - float: The average historical return, rounded to 4 decimal places.
    """
    return round(expected_returns.mean_historical_return(historical_prices), 4)


def get_names(
        current_prices: pd.DataFrame
) -> List[str]:
    """
    Sorts the 'shortName' of tickers based on their index in a given DataFrame.

    Parameters:
    - current_prices (pd.DataFrame): DataFrame containing 'shortName' as one of the columns.

    Returns:
    - List[str]: List of sorted 'shortName'.
    """
    return current_prices.loc['shortName'].sort_index()


def get_benchmark_portfolio(
        benchmark_portfolio: Dict[str, float]
) -> Dict[str, float]:
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


def get_sp100_tickers():
    """
    Fetches the list of S&P 100 tickers from Wikipedia.

    Returns:
    - List[str]: List of S&P 100 tickers.
    """
    # URL of the Wikipedia page containing S&P 100 tickers
    url = 'https://en.wikipedia.org/wiki/S%26P_100'

    # Send a GET request to the Wikipedia page
    response = requests.get(url)

    # Check if the request was successful
    if response.status_code == 200:
        # Parse the HTML content
        soup = BeautifulSoup(response.content, 'html.parser')

        # Find the table that contains the list of tickers
        # Usually, this is the first table on the page, but you might need to check if it changes
        table = soup.find('table', {'class': 'wikitable sortable'})

        # Find all rows in the table, skip the header row
        rows = table.findAll('tr')[1:]

        # Extract the ticker symbol, which is in the first column of the table
        # Replace any '.' with '-' in the ticker symbols
        tickers = [row.find('td').text.strip().replace('.', '-') for row in rows]

        return tickers
    else:
        # If the request was not successful, print the error code
        print(f"Failed to retrieve Wikipedia page: Status code {response.status_code}")
        return []
