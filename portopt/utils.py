import json
import pandas as pd
from . import backtesting
from . import plotting
import matplotlib.pyplot as plt
from typing import Dict


def read_portfolio_config(file_path):
    with open(file_path, 'r') as f:
        config_data = json.load(f)

    # Extract and transform portfolio data
    portfolio_data = config_data['portfolio']
    portfolio_tickers = [x['ticker'] for x in portfolio_data]
    weight_bounds = [(x['min_weight'], x['max_weight']) for x in portfolio_data]
    investor_views = {x['ticker']: x.get('view', None) for x in portfolio_data}
    view_confidences = {x['ticker']: x.get('confidence', None) for x in portfolio_data}

    # Filter out None values from investor_views and view_confidences
    investor_views = {k: v for k, v in investor_views.items() if v is not None}
    view_confidences = {k: v for k, v in view_confidences.items() if v is not None}

    # Extract other config data
    start_date = config_data['start_date']
    end_date = config_data['end_date']
    n_portfolios = config_data['n_portfolios']
    nsim = config_data['nsim']

    return {
        'portfolio_tickers': portfolio_tickers,
        'weight_bounds': weight_bounds,
        'investor_views': investor_views,
        'view_confidences': view_confidences,
        'start_date': start_date,
        'end_date': end_date,
        'n_portfolios': n_portfolios,
        'nsim': nsim
    }


def process_json(json_data: Dict):
    portfolio_data = json_data['portfolio']
    portfolio_tickers = [x['ticker'] for x in portfolio_data]
    weight_bounds = [(x['min_weight'], x['max_weight']) for x in portfolio_data]
    investor_views = {x['ticker']: x.get('view', None) for x in portfolio_data}
    view_confidences = {x['ticker']: x.get('confidence', None) for x in portfolio_data}

    # Filter out None values from investor_views and view_confidences
    investor_views = {k: v for k, v in investor_views.items() if v is not None}
    view_confidences = {k: v for k, v in view_confidences.items() if v is not None}

    # Extract other config data
    start_date = json_data['start_date']
    end_date = json_data['end_date']
    n_portfolios = json_data['n_portfolios']
    n_simulations = json_data['n_simulations']

    return {
        'portfolio_tickers': portfolio_tickers,
        'weight_bounds': weight_bounds,
        'investor_views': investor_views,
        'view_confidences': view_confidences,
        'start_date': start_date,
        'end_date': end_date,
        'n_portfolios': n_portfolios,
        'n_simulations': n_simulations
    }


def hypothesis_test_parameters(results_obj: object, statistic: str = 'monthly_sortino'):
    r_stats = results_obj.stats.iloc[:, 1:]
    b_stats = results_obj.stats.iloc[:, :1].squeeze()
    random_stats = results_obj.stats.loc[statistic].sort_values(ascending=False)
    print('Random Portfolios\' Stats')
    print(random_stats[random_stats >= random_stats.loc['Optimal Portfolio']])
    print('\nOptimal Portfolio: {}'.format(statistic))
    print(round(random_stats.loc['Optimal Portfolio'], 4))
    print('\nRandom Portfolios\' Mean, Standard Deviation, and Mean + Standard Deviation')
    print(round(random_stats.mean(), 4))
    print(round(random_stats.std(), 4))
    print(round(random_stats.mean() + random_stats.std(), 4))
    print('\nOptimal Portfolio Z-Score')
    print(round((random_stats.loc['Optimal Portfolio'] - random_stats.mean()) / random_stats.std(), 4))
    return r_stats, b_stats, random_stats


def plot_hypothesis_test(results_obj: object, random_stats: pd.Series, chart_num: int = 0):
    if chart_num == 0:
        plt.plot((backtesting.get_series_from_object(results_obj)[random_stats.index[0]]))
        plt.title(random_stats.index[0])
        plt.grid()
        plt.show()

    elif chart_num == 1:
        plotting.plot_security_weights(results_obj, backtest=int(random_stats.index[0].split('_')[1]),
                                       title='Security Weights (%): {}'.format(random_stats.index[0]))


def display_market_caps(benchmark_portfolio, benchmark_name):
    pd.DataFrame.from_dict(benchmark_portfolio, orient='index', columns=[benchmark_name]).squeeze().sort_values(
        ascending=False)
