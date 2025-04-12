"""
Computes realized variance, and its decomposition into a continuous and a jump part, from intraday stock price data.
Code is based on the RealizedQuantities repo by Sebastian Bayer, https://github.com/BayerSe/RealizedQuantities/.
FirstRate Data does not permit the distribution of its intraday stock data sets.
Hence, data includes a sample to run this function and the output of this function for the entire dataset.
"""


import pickle
import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import gamma
from config import DATA_DIR, BLD_data


def collect_realized_quantity_data():
    """
    Cleans intraday stock prices and computes realized variance and its continuous and jump component.
    """
    folder_path = BLD_data / 'Application'
    if not folder_path.is_dir():
        folder_path.mkdir(parents=True, exist_ok=True)

    data_path = DATA_DIR / 'SPX_full_5min_sample.csv'
    data = pd.read_csv(data_path)
    intraday_prices = clean_intraday_data(data)
    rv_data = get_realized_quantities(intraday_prices)

    file_path = folder_path / 'realized_quantities_sample.pkl'
    with open(file_path, "wb") as f:
        pickle.dump(rv_data, f)


def clean_intraday_data(data):
    """
    Sets index of the stock price DataFrame and sets the last closing price as the final opening price.
    """
    df = data.copy()
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    df = df.set_index('DateTime')
    last_interval_map = df.index.time == pd.to_datetime('16:05:00').time()
    df.loc[last_interval_map, 'Open'] = df.loc[last_interval_map, 'Close']
    return df['Open']


def get_realized_quantities(data):
    """
    Computes realized quantities based in intraday stock price data and stores them in a pd.DataFrame.
    """
    trading_seconds = 6.5*3600 # 6.5 trading hours x 3600 seconds per hour
    sampling_frequency = 5*60 # One observation every three hundred seconds
    M = trading_seconds / sampling_frequency + 1 # Observations per day, corrected for 79th value.
    MU_1 = np.sqrt((2 / np.pi))
    MU_43 = 2 ** (2 / 3) * gamma(7 / 6) * gamma(1 / 2) ** (-1)

    intraday_returns = data.groupby(pd.Grouper(freq="1d")).apply(lambda x: np.log(x / x.shift(1))).dropna().reset_index(level=0, drop=True)
    index = data.groupby(pd.Grouper(freq="1d")).first().dropna().index

    def realized_quantity(fun):
        """Applies the function 'fun' to each day separately"""
        return intraday_returns.groupby(pd.Grouper(freq="1d")).apply(fun)[index]

    prices_open = data.resample('D').first()[index]
    prices_close = data.resample('D').last()[index]

    return_close_to_close = pd.Series(np.log(prices_close / prices_close.shift(1)))
    return_open_to_close = pd.Series(np.log(prices_close / prices_open))

    realized_variance = realized_quantity(lambda x: (x ** 2).sum())

    bipower_variation = MU_1 ** (-2) * realized_quantity(lambda x: (x.abs() * x.shift(1).abs()).sum())
    tripower_quarticity = M * MU_43 ** (-3) * realized_quantity(
        lambda x: (x.abs() ** (4 / 3) * x.shift(1).abs() ** (4 / 3) * x.shift(2).abs() ** (4 / 3)).sum()
    )

    jump_test = (np.log(realized_variance) - np.log(bipower_variation)) / \
        ((MU_1 ** -4 + 2 * MU_1 ** -2 - 5) / (M * tripower_quarticity * bipower_variation ** -2)) ** 0.5
    jump = jump_test.abs() >= stats.norm.ppf(0.999)

    continuous_var = pd.Series(0.0, index=index)
    continuous_var[jump] = bipower_variation[jump]
    continuous_var[~jump] = realized_variance[~jump]

    jump_var = pd.Series(0.0, index=index)
    jump_var[jump] = realized_variance[jump] - bipower_variation[jump]
    jump_var[jump_var < 0] = 0

    out = pd.DataFrame({
        'Open to Close Return': return_open_to_close,
        'Close to Close Returns': return_close_to_close,
        'Realized Variance': realized_variance,
        'Jump Variance': jump_var,
        'Continuous Variance': continuous_var
    },
        index=index
    )

    return out

if __name__ == '__main__':
    collect_realized_quantity_data()
