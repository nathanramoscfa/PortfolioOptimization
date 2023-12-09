import pandas as pd
import numpy as np
from scipy.stats import zscore


def calculate_value_score(summary_details: pd.DataFrame,
                          summary_profile: pd.DataFrame,
                          current_prices: pd.DataFrame,
                          group_by: str = 'sector') -> pd.DataFrame:
    """
    Calculate winsorized z-scores for value scores grouped by industry or sector and include the group and individual
    factor z-scores in the final DataFrame, along with the long name of the ticker.

    Args:
        summary_details (pd.DataFrame): DataFrame containing financial ratios with tickers as columns.
        summary_profile (pd.DataFrame): DataFrame containing company profile information with tickers as columns.
        current_prices (pd.Series): Series containing long names for tickers.
        group_by (str): 'sector' or 'industry' to specify the grouping criterion for z-score calculation.

    Returns:
        pd.DataFrame: DataFrame with ticker long name, 'sector', 'industry', grouped 'value_zscores', and individual
            z-scores for each value factor.
    """
    # Combine the relevant fields from summary_details
    ratios = {
        'forwardPE': summary_details,
        'dividendYield': summary_details.fillna(0)
    }
    inverted_ratios = ['forwardPE']  # Higher is typically less desirable

    # Initialize DataFrame to store z-scores for each ratio
    zscores_df = pd.DataFrame(index=summary_details.columns)

    # Compute z-scores for each ratio and add them to zscores_df
    for ratio, df in ratios.items():
        ratio_series = pd.to_numeric(df.loc[ratio], errors='coerce')

        # Invert the ratio if it is one of the ratios where higher values are less desirable
        if ratio in inverted_ratios:
            ratio_series = 1 / ratio_series

        # Replace infinite values with NaN
        ratio_series.replace([np.inf, -np.inf], np.nan, inplace=True)

        # Drop NaNs from the ratio series
        non_nan_ratio_series = ratio_series.dropna()

        # Calculate the z-scores for the non-NaN values
        zscores = zscore(non_nan_ratio_series, nan_policy='omit')

        # Winsorize the z-scores
        zscores_winsorized = np.clip(zscores, -3, 3)

        # Store the winsorized z-scores in the corresponding DataFrame column
        zscores_df[ratio + '_zscore'] = pd.Series(zscores_winsorized, index=non_nan_ratio_series.index).fillna(0)

        # Group the z-scores by the specified group and store in the DataFrame
        grouped_zscores = zscores_df.groupby(summary_profile[group_by])[ratio + '_zscore'].transform(
            lambda x: zscore(x, nan_policy='omit')
        )
        zscores_df[ratio + '_grouped_zscore'] = np.clip(grouped_zscores, -3, 3)

    # Calculate the mean grouped z-score across all ratios for each ticker
    grouped_zscore_columns = [ratio + '_grouped_zscore' for ratio in ratios.keys()]
    zscores_df['value_zscore'] = zscores_df[grouped_zscore_columns].mean(axis=1)

    # Join the sector and industry columns from the summary_profile to the zscores_df DataFrame
    zscores_df = zscores_df.join(summary_profile[['sector', 'industry']])

    # Add the long name of the ticker from current_prices Series
    zscores_df['longName'] = current_prices.loc['longName']

    # Select the columns to be displayed in the final DataFrame
    columns_to_display = ['longName', 'sector', 'industry', 'value_zscore'] + grouped_zscore_columns
    df = zscores_df[columns_to_display]

    # Add index name
    df.index.name = 'symbol'

    return df


def calculate_momentum_score(historical_prices: pd.DataFrame,
                             summary_profile: pd.DataFrame,
                             current_prices: pd.DataFrame,
                             group_by: str = 'sector') -> pd.DataFrame:
    """
    Calculate winsorized z-scores for momentum scores grouped by industry or sector and include the group and individual
    factor z-scores in the final DataFrame, along with the long name of the ticker.

    Args:
        historical_prices (pd.DataFrame): DataFrame containing historical prices with tickers as columns.
        summary_profile (pd.DataFrame): DataFrame containing company profile information with tickers as columns.
        current_prices (pd.DataFrame): DataFrame containing current prices with tickers as columns.
        group_by (str): 'sector' or 'industry' to specify the grouping criterion for z-score calculation.

    Returns:
        pd.DataFrame: DataFrame with ticker long name, 'sector', 'industry', grouped 'momentum_zscore', and individual
            z-scores for each momentum factor.
    """
    # Define trading days in a month and year for financial data
    trading_days_in_month = 21
    trading_days_in_year = 252

    # Calculate the 12-1 month return, 6-1 month return, and 3-1 month return
    momentum_periods = {
        '12_1_month_return': trading_days_in_year,
        '6_1_month_return': trading_days_in_month * 6,
        '3_1_month_return': trading_days_in_month * 3
    }

    # Calculate returns for different periods
    prices_12_1 = historical_prices.iloc[-momentum_periods['12_1_month_return']:-21]
    prices_6_1 = historical_prices.iloc[-momentum_periods['6_1_month_return']:-21]
    prices_3_1 = historical_prices.iloc[-momentum_periods['3_1_month_return']:-21]

    returns_12_1 = prices_12_1.iloc[-1] / prices_12_1.iloc[0] - 1
    returns_6_1 = prices_6_1.iloc[-1] / prices_6_1.iloc[0] - 1
    returns_3_1 = prices_3_1.iloc[-1] / prices_3_1.iloc[0] - 1

    # Join sector and industry information
    sector_and_industry = summary_profile[['sector', 'industry']]
    returns_12_1 = returns_12_1.to_frame('returns').join(sector_and_industry)
    returns_6_1 = returns_6_1.to_frame('returns').join(sector_and_industry)
    returns_3_1 = returns_3_1.to_frame('returns').join(sector_and_industry)

    # Join sector and industry information
    ticker_names = current_prices.loc['longName']
    returns_12_1 = returns_12_1.join(ticker_names)
    returns_6_1 = returns_6_1.join(ticker_names)
    returns_3_1 = returns_3_1.join(ticker_names)

    # Function to compute z-scores and clip them
    def compute_clipped_zscores(group):
        z_scores = zscore(group['returns'])
        z_scores_clipped = np.clip(z_scores, -3, 3)
        group['momentum_zscores'] = z_scores_clipped
        return group[['longName', 'sector', 'industry', 'momentum_zscores']]

    # Apply the function and reset the index
    zscores_12_1 = returns_12_1.groupby(group_by).apply(compute_clipped_zscores).reset_index(level=0, drop=True)
    zscores_6_1 = returns_6_1.groupby(group_by).apply(compute_clipped_zscores).reset_index(level=0, drop=True)
    zscores_3_1 = returns_3_1.groupby(group_by).apply(compute_clipped_zscores).reset_index(level=0, drop=True)

    # Join the three z-scores DataFrames on the ticker index
    combined_zscores_df = pd.DataFrame({
        'longName': zscores_12_1['longName'],
        'zscores_12_1': zscores_12_1['momentum_zscores'],
        'zscores_6_1': zscores_6_1['momentum_zscores'],
        'zscores_3_1': zscores_3_1['momentum_zscores'],
        'sector': zscores_12_1['sector'],
        'industry': zscores_12_1['industry']
    })

    # Calculate the combined z-score
    combined_zscores_df['momentum_zscore'] = combined_zscores_df[[
        'zscores_12_1', 'zscores_6_1', 'zscores_3_1'
    ]].mean(axis=1)

    # Add the long name of the ticker from current_prices Series
    combined_zscores_df['longName'] = current_prices.loc['longName']

    # Select the relevant columns to return
    final_scores = combined_zscores_df[[
        'longName', 'sector', 'industry', 'momentum_zscore', 'zscores_12_1', 'zscores_6_1', 'zscores_3_1'
    ]]

    # Add index name
    final_scores.index.name = 'symbol'

    return final_scores


def calculate_profitability_score(financial_data: pd.DataFrame,
                                  summary_profile: pd.DataFrame,
                                  current_prices: pd.DataFrame,
                                  group_by: str = 'sector') -> pd.DataFrame:
    """
    Calculate winsorized z-scores for profitability scores grouped by industry or sector and include the group and
    individual factor z-scores in the final DataFrame, along with the long name of the ticker.

    Args:
        financial_data (pd.DataFrame): DataFrame containing financial ratios with tickers as columns.
        summary_profile (pd.DataFrame): DataFrame containing company profile information with tickers as columns.
        current_prices (pd.DataFrame): DataFrame containing current prices with tickers as columns.
        group_by (str): 'sector' or 'industry' to specify the grouping criterion for z-score calculation.

    Returns:
        pd.DataFrame: DataFrame with ticker long name, 'sector', 'industry', grouped 'profitability_zscore', and
            individual z-scores for each profitability factor.
    """
    # Define the profitability metrics we're interested in
    profitability_metrics = ['profitMargins', 'returnOnAssets', 'returnOnEquity']

    # Initialize DataFrame to store z-scores for each profitability metric
    profitability_scores_df = pd.DataFrame(index=financial_data.columns)

    # Calculate z-scores for each profitability metric
    for metric in profitability_metrics:
        metric_series = pd.to_numeric(financial_data.loc[metric], errors='coerce')
        non_nan_metric_series = metric_series.dropna()
        zscores = zscore(non_nan_metric_series, nan_policy='omit')
        zscores_winsorized = np.clip(zscores, -3, 3)
        profitability_scores_df[metric + '_zscore'] = pd.Series(zscores_winsorized, index=non_nan_metric_series.index).fillna(0)

        # Group the z-scores by the specified group and store in the DataFrame
        grouped_zscores = profitability_scores_df.groupby(summary_profile[group_by])[metric + '_zscore'].transform(
            lambda x: zscore(x, nan_policy='omit')
        )
        profitability_scores_df[metric + '_grouped_zscore'] = np.clip(grouped_zscores, -3, 3)

    # Calculate the mean grouped z-score across all profitability metrics for each ticker
    grouped_zscore_columns = [metric + '_grouped_zscore' for metric in profitability_metrics]
    profitability_scores_df['profitability_zscore'] = profitability_scores_df[grouped_zscore_columns].mean(axis=1)

    # Join the sector, industry, and ticker long name
    profitability_scores_df = profitability_scores_df.join(summary_profile[['sector', 'industry']])
    profitability_scores_df['longName'] = current_prices.loc['longName']

    # Select the columns to be displayed in the final DataFrame
    columns_to_display = ['longName', 'sector', 'industry', 'profitability_zscore'] + grouped_zscore_columns
    df = profitability_scores_df[columns_to_display]

    # Add index name
    df.index.name = 'symbol'

    return df


def calculate_revisions_score(revisions: dict,
                              summary_profile: pd.DataFrame,
                              current_prices: pd.DataFrame,
                              group_by: str = 'sector') -> pd.DataFrame:
    """
    Calculate winsorized z-scores for revisions scores grouped by industry or sector for multiple revision trends and
    combine into a single score.

    Args:
        revisions (dict): Dictionary containing revision trends DataFrames with revisions data.
        summary_profile (pd.DataFrame): DataFrame containing company profile information.
        current_prices (pd.DataFrame): DataFrame containing current prices with long names for tickers.
        group_by (str): Column name to group by, default is 'sector'.

    Returns:
        pd.DataFrame: A DataFrame containing the ticker name, sector, industry, winsorized combined revisions
            z-scores, and individual z-scores for each ticker.
    """
    # Initialize a DataFrame to store the mean revisions and z-scores for each trend
    revisions_df = pd.DataFrame(index=summary_profile.index)

    # Loop through each trend in the revisions dictionary
    for trend_name, trend_data in revisions.items():
        mean_revision = trend_data.mean(axis=1)
        revisions_df[trend_name + '_mean_revision'] = mean_revision

        # Calculate individual z-scores for each trend and winsorize them
        zscore_column = trend_name + '_zscore'
        zscores = zscore(mean_revision, nan_policy='omit')
        zscores_winsorized = np.clip(zscores, -3, 3)
        revisions_df[zscore_column] = pd.Series(zscores_winsorized, index=mean_revision.index).fillna(0)

        # Group the z-scores by the specified group and store in the DataFrame
        grouped_zscores = revisions_df.groupby(summary_profile[group_by])[zscore_column].transform(
            lambda x: zscore(x, nan_policy='omit')
        )
        revisions_df[zscore_column + '_grouped'] = np.clip(grouped_zscores, -3, 3)

    # Calculate the mean grouped z-score across all trends for each ticker
    grouped_zscore_columns = [trend_name + '_zscore_grouped' for trend_name in revisions.keys()]
    revisions_df['revisions_zscore'] = revisions_df[grouped_zscore_columns].mean(axis=1)

    # Join the sector, industry, and ticker names
    revisions_df = revisions_df.join(summary_profile[['sector', 'industry']])
    revisions_df = revisions_df.join(current_prices.loc['longName'])

    # Select the columns to be displayed in the final DataFrame
    columns_order = ['longName', 'sector', 'industry', 'revisions_zscore'] + grouped_zscore_columns
    df = revisions_df[columns_order]

    # Add index name
    df.index.name = 'symbol'

    return df


def calculate_reversal_score(historical_prices: pd.DataFrame,
                             summary_profile: pd.DataFrame,
                             current_prices: pd.DataFrame,
                             group_by: str = 'sector') -> pd.DataFrame:
    """
    Calculate winsorized z-scores for reversal scores grouped by industry or sector and include the group and individual
    factor z-scores in the final DataFrame, along with the long name of the ticker.

    Args:
        historical_prices (pd.DataFrame): DataFrame containing historical prices with tickers as columns.
        summary_profile (pd.DataFrame): DataFrame containing company profile information with tickers as columns.
        current_prices (pd.DataFrame): DataFrame containing current prices with tickers as columns.
        group_by (str): 'sector' or 'industry' to specify the grouping criterion for z-score calculation.

    Returns:
        pd.DataFrame: DataFrame with ticker long name, 'sector', 'industry', grouped 'reversal_zscore', and individual
            z-scores for each reversal factor.
    """
    # Define trading days for a week and a month for financial data
    trading_days_in_week = 5
    trading_days_in_month = 21

    # Calculate the 1-week and 1-month returns
    weekly_return = historical_prices.pct_change(periods=trading_days_in_week)
    monthly_return = historical_prices.pct_change(periods=trading_days_in_month)

    # Invert the scores because a negative return has a positive long-term reversal potential
    inverse_weekly_return = -weekly_return.iloc[-trading_days_in_week]
    inverse_monthly_return = -monthly_return.iloc[-trading_days_in_month]

    # Join sector and industry information
    sector_and_industry = summary_profile[['sector', 'industry']]
    inverse_weekly_return = inverse_weekly_return.to_frame('returns').join(sector_and_industry)
    inverse_monthly_return = inverse_monthly_return.to_frame('returns').join(sector_and_industry)

    # Add the long name of the ticker from current_prices
    inverse_weekly_return['longName'] = current_prices.loc['longName']
    inverse_monthly_return['longName'] = current_prices.loc['longName']

    # Function to compute z-scores and clip them
    def compute_clipped_zscores(group):
        z_scores = zscore(group['returns'])
        z_scores_clipped = np.clip(z_scores, -3, 3)
        group['reversal_zscores'] = z_scores_clipped
        return group[['longName', 'sector', 'industry', 'reversal_zscores']]

    # Apply the function and reset the index
    zscores_weekly = inverse_weekly_return.groupby(group_by).apply(compute_clipped_zscores).reset_index(
        level=0, drop=True
    )
    zscores_monthly = inverse_monthly_return.groupby(group_by).apply(compute_clipped_zscores).reset_index(
        level=0, drop=True
    )

    # Join the weekly and monthly z-scores on the ticker index
    combined_zscores_df = pd.DataFrame({
        'longName': zscores_weekly['longName'],
        'zscores_weekly': zscores_weekly['reversal_zscores'],
        'zscores_monthly': zscores_monthly['reversal_zscores'],
        'sector': zscores_weekly['sector'],
        'industry': zscores_weekly['industry']
    })

    # Calculate the combined z-score
    combined_zscores_df['reversal_zscore'] = combined_zscores_df[['zscores_weekly', 'zscores_monthly']].mean(axis=1)

    # Select the relevant columns to return
    final_scores = combined_zscores_df[[
        'longName', 'sector', 'industry', 'reversal_zscore', 'zscores_weekly', 'zscores_monthly'
    ]]

    # Add index name
    final_scores.index.name = 'symbol'

    return final_scores


def calculate_multifactor_model(summary_details: pd.DataFrame,
                                historical_prices: pd.DataFrame,
                                financial_data: pd.DataFrame,
                                summary_profile: pd.DataFrame,
                                revisions: dict,
                                current_prices: pd.DataFrame,
                                group_by: str = 'sector') -> pd.DataFrame:
    """
    Calculate the multifactor model for each ticker and return a DataFrame with the combined score and individual
    factor scores.

    Args:
        summary_details (pd.DataFrame): DataFrame containing financial ratios with tickers as columns.
        key_stats (pd.DataFrame): DataFrame containing financial ratios with tickers as columns.
        historical_prices (pd.DataFrame): DataFrame containing historical prices with tickers as columns.
        financial_data (pd.DataFrame): DataFrame containing financial ratios with tickers as columns.
        summary_profile (pd.DataFrame): DataFrame containing company profile information with tickers as columns.
        revisions (dict): Dictionary containing revision trends DataFrames with revisions data.
        current_prices (pd.DataFrame): DataFrame containing current prices with tickers as columns.
        group_by (str): 'sector' or 'industry' to specify the grouping criterion for z-score calculation.

    Returns:
        pd.DataFrame: DataFrame with ticker long name, 'sector', 'industry', grouped 'combined_zscore', and individual
            z-scores for each factor.
    """
    # Calculate individual factor scores
    value_scores = calculate_value_score(
        summary_details, summary_profile, current_prices, group_by)['value_zscore']
    momentum_scores = calculate_momentum_score(
        historical_prices, summary_profile, current_prices, group_by)['momentum_zscore']
    profitability_scores = calculate_profitability_score(
        financial_data, summary_profile, current_prices, group_by)['profitability_zscore']
    revisions_scores = calculate_revisions_score(
        revisions, summary_profile, current_prices, group_by)['revisions_zscore']
    reversal_scores = calculate_reversal_score(
        historical_prices, summary_profile, current_prices, group_by)['reversal_zscore']

    # Combine factor scores into a combined score dataframe
    combined_scores = pd.DataFrame({
        'Value': value_scores,
        'Momentum': momentum_scores,
        'Profitability': profitability_scores,
        'Revisions': revisions_scores,
        'Reversal': reversal_scores
    })

    # Add longName, sector, and industry to the combined_scores dataframe
    combined_scores['longName'] = current_prices.loc['longName']
    combined_scores['Sector'] = summary_profile['sector']
    combined_scores['Industry'] = summary_profile['industry']

    # Reorder the columns to have longName, sector, and industry first
    columns_order = ['longName', 'Sector', 'Industry', 'Value', 'Momentum', 'Profitability', 'Revisions', 'Reversal']
    combined_scores = combined_scores[columns_order]

    # Simple average for demonstration purposes to create a combined score
    combined_scores['Combined'] = combined_scores[[
        'Value', 'Momentum', 'Profitability', 'Revisions', 'Reversal'
    ]].mean(axis=1)

    # Rank stocks by the combined score
    combined_scores['Rank'] = combined_scores['Combined'].rank(ascending=False)

    # Return the ranked tickers in a dataframe, sorted by rank
    ranked_tickers = combined_scores.sort_values('Rank').round(2)

    # Add index name
    ranked_tickers.index.name = 'symbol'

    return ranked_tickers
