"""
Gathers macroeconomic data and release dates from the data files
and stores them in a dict that is stored in a pickle file.
"""

import pandas as pd
import numpy as np
import pickle
from config import DATA_DIR, BLD_data

def collect_macro_inputs():
    """
    Calls all functions gathering macroeconomic variables and stores them in a dict that is saved in a pickle file.
    """

    folder_path = BLD_data / 'Application'
    if not folder_path.is_dir():
        folder_path.mkdir(parents=True, exist_ok=True)

    macroeconomic_inputs = {
        'inflation': get_cpi_data(),
        'fed_meetings': get_fed_meeting_dates(),
        'fed_rate': get_fed_funds_rate_data(),
        'gdp': get_gdp_data(),
        'unemployment': get_employment_situation_data(),
    }

    file_path = folder_path / 'macroeconomic_inputs.pkl'
    with open(file_path, "wb") as f:
        pickle.dump(macroeconomic_inputs, f)


def get_gdp_data(start_date='2007-04-01'):
    """
    Collects GDP data, computes growth rates and matches it with release dates.
    Additionally, for release dates, a dummy variable is included.
    Returns a pd.DataFrame with GDP level, GDP growth, and a release date dummy.
    """
    sources_dict = {
        'dates': 'BEA_GDP_Release_Dates.xlsx',
        'values': 'Quarterly_GDP.xlsx'
    }
    rename_dict = {
        'dates': 'Release Dates',
        'values': 'observation_date'
    }
    sheet_dict = {
        'dates': 'Release Dates',
        'values': 'GDP Values'
    }
    data_dict = {}
    for name, source in sources_dict.items():
        data_dict[name] = pd.read_excel(DATA_DIR / source, sheet_name=sheet_dict[name])
        data_dict[name] = data_dict[name].rename(columns={rename_dict[name]: 'Date'})
        data_dict[name]['Date'] = pd.to_datetime(data_dict[name]['Date'])
        data_dict[name] = data_dict[name][data_dict[name]['Date'] >= start_date].reset_index(drop=True)

    gdp_df = pd.DataFrame()
    gdp_df['Date'] = data_dict['dates']['Date']

    def get_previous_quarter_values(release_date):
        """
        Matches GDP data to the release date in the previous quarter, to mimic real-time observable data.
        """
        previous_quarter_date = release_date - pd.DateOffset(months=3)

        gdp_data = data_dict['values'][data_dict['values']['Date'] <= previous_quarter_date]

        if not gdp_data.empty:
            closest_date = gdp_data.iloc[-1]
            return closest_date['GDP_20250130']
        else:
            return np.nan

    gdp_df['GDP Values'] = gdp_df['Date'].map(get_previous_quarter_values)
    gdp_df['GDP Growth'] = gdp_df['GDP Values'].pct_change()
    gdp_df['GDP Growth'] = gdp_df['GDP Growth'].replace(0.0, np.nan)
    gdp_df['GDP Growth'] = gdp_df['GDP Growth'].ffill()
    gdp_df = gdp_df.set_index('Date')
    gdp_df['GDP Release Dummy'] = pd.Series(np.ones(len(gdp_df)), index=gdp_df.index).astype(int)

    return gdp_df

def get_cpi_data(start_date='2006-04-01'):
    """
    Collects CPI data, computes inflation rates and matches them with CPI release dates.
    Additionally, for release dates, a dummy variable is included.
    Returns a pd.DataFrame with the price index, YoY inflation, and date dummies.
    """
    sources_dict = {
        'dates': 'CPI_release_dates_10.xlsx',
        'values': 'CPIAUCSL.xlsx'
    }
    rename_dict = {
        'dates': 'Release Dates',
        'values': 'observation_date'
    }
    sheet_dict = {
        'dates': 'Release Dates',
        'values': 'Monthly'
    }
    data_dict = {}
    for name, source in sources_dict.items():
        data_dict[name] = pd.read_excel(DATA_DIR / source, sheet_name=sheet_dict[name])
        data_dict[name] = data_dict[name].rename(columns={rename_dict[name]: 'Date'})
        data_dict[name]['Date'] = pd.to_datetime(data_dict[name]['Date'])
        data_dict[name] = data_dict[name][data_dict[name]['Date'] >= start_date].reset_index(drop=True)
    data_dict['values']['YoY Inflation'] = data_dict['values']['CPIAUCSL'] / data_dict['values']['CPIAUCSL'].shift(12) - 1

    cpi_df = pd.DataFrame()
    cpi_df['Date'] = data_dict['dates']['Date']

    def get_previous_month_values(release_date):
        """
        Matches CPI inflation data to the release date in the previous quarter, to mimic real-time observable data.
        """
        previous_month_date = release_date - pd.DateOffset(months=1)
        cpi_data = data_dict['values'][data_dict['values']['Date'] <= previous_month_date]

        if not cpi_data.empty:
            closest_date = cpi_data.iloc[-1]  # Get the latest available data before release
            return pd.Series([closest_date['CPIAUCSL'], closest_date['YoY Inflation']],
                             index=['CPI Values', 'YoY Inflation'])
        else:
            return pd.Series([np.nan, np.nan], index=['CPI Values', 'YoY Inflation'])

    # Apply function to each row
    cpi_df[['CPI Values', 'YoY Inflation']] = cpi_df['Date'].apply(get_previous_month_values)
    cpi_df = cpi_df.set_index('Date')
    cpi_df['CPI Release Dummy'] = pd.Series(np.ones(len(cpi_df)), index=cpi_df.index).astype(int)

    return cpi_df

def get_employment_situation_data(start_date='2007-11-01'):
    """
    Collects unemployment data and economic situation release dates.
    Returns a pd.DataFrame with the unemployment rate and release date dummies.
    """
    sources_dict = {
        'dates': 'BLS_employment_situation_release_dates_50.xlsx',
        'values': 'BLS_unemployment_rate.xlsx'
    }
    rename_dict = {
        'dates': 'Release Dates',
        'values': 'observation_date'
    }
    sheet_dict = {
        'dates': 'Release Dates',
        'values': 'Monthly'
    }
    data_dict = {}
    for name, source in sources_dict.items():
        data_dict[name] = pd.read_excel(DATA_DIR / source, sheet_name=sheet_dict[name])
        data_dict[name] = data_dict[name].rename(columns={rename_dict[name]: 'Date'})
        data_dict[name]['Date'] = pd.to_datetime(data_dict[name]['Date'])
        data_dict[name] = data_dict[name][data_dict[name]['Date'] >= start_date].reset_index(drop=True)

    unemployment_df = pd.DataFrame()
    unemployment_df['Date'] = data_dict['dates']['Date']

    def get_previous_month_values(release_date):
        """
        Matches unemployment rate with the release date in the previous month, to mimic real-time observable data.
        """

        previous_month_date = release_date - pd.DateOffset(months=1)
        data = data_dict['values'][data_dict['values']['Date'] <= previous_month_date]

        if not data.empty:
            closest_date = data.iloc[-1]
            return closest_date['UNRATE_20250207']
        else:
            return np.nan

    unemployment_df['Unemployment Values'] = unemployment_df['Date'].map(get_previous_month_values)
    unemployment_df = unemployment_df.set_index('Date')
    unemployment_df['Unemployment Release Dummy'] = pd.Series(np.ones(len(unemployment_df)), index=unemployment_df.index).astype(
        int)

    return unemployment_df



def get_fed_funds_rate_data(start_date='2008-01-02'):
    """
    Collects data on the Federal Reserve Effective Rate and stores it in a pd.DataFrame.
    """

    source = DATA_DIR / 'DFF.xlsx'
    sheet_name = 'Daily, 7-Day'

    fed_rate_df = pd.read_excel(source, sheet_name=sheet_name)
    fed_rate_df = fed_rate_df.rename(columns={'observation_date': 'Date', 'DFF': 'Effective Funds Rate'})
    fed_rate_df = fed_rate_df[fed_rate_df['Date'] >= start_date].reset_index(drop=True)
    fed_rate_df = fed_rate_df.set_index('Date')

    return fed_rate_df

def get_fed_meeting_dates():
    """
    Creates date dummies for FOMC meetings. If press releases occurred a day after the meeting,
    which happened twice in 2008, the date of the press release is used.
    """
    source = DATA_DIR / 'Fed_Meetings.xlsx'
    sheet_name = 'Dates'

    fed_meetings_df = pd.read_excel(source, sheet_name=sheet_name)
    fed_meetings_df.loc[pd.notna(fed_meetings_df['Later Releases']), 'Date'] = fed_meetings_df['Later Releases']
    fed_meetings_df['Fed Meeting Dummy'] = pd.Series(np.ones(len(fed_meetings_df)), index=fed_meetings_df.index).astype(int)
    fed_meetings_df = fed_meetings_df.drop(columns='Later Releases').set_index('Date')

    return fed_meetings_df


if __name__ == '__main__':
    collect_macro_inputs()













