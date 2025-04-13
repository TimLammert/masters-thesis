"""
Creates datasets for forecasts of S&P500 realized variance from the previously generated realized quantities
and the collected macroeconomic variables.

"""
import pandas as pd
import pickle
from config import BLD_data, DATA_DIR
import copy


def get_all_specifications():
    """
    Builds all datasets used in the thesis: Three forecast horizons for square_root=False,
    only one-step-ahead for square-root=True. Other horizons would be possible for both versions.
    """
    start_date = '2008-02-04'
    end_date = '2024-12-31'
    for sqrt in [True, False]:
        for step in [1, 5, 22] if not sqrt else [1]:
            create_datasets(steps_ahead=step, square_root=sqrt, start_date=start_date, end_date=end_date)


def build_har_data(steps_ahead:int, square_root:bool):
    """
    Builds dataset containing daily realized variance and weekly and monthly averages.
    Depending on the forecast horizon RV_t is set to daily realized variance, or its weekly or monthly average.
    This slightly abuses notation, but does not harm the rest of the code.

    Arguments
    ---------
    steps_ahead : int
        determines the forecast horizon and thus whether the dependent variable is daily realized variance or its
        weekly or monthly average.
    square_root : bool
        determines whether the square root is taken for all variables derived from realized variance.

    Returns
    -------
    har_rv_data : pd.DataFrame
        Dataset containing daily realized variance and weekly and monthly averages
        or its square roots for one forecast horizon.

    """

    power = 0.5 if square_root else 1

    with open(DATA_DIR / 'realized_quantities.pkl', 'rb') as f:
        volatility_data = pickle.load(f)

    har_rv_data = pd.DataFrame(index=volatility_data.index)
    har_rv_data['RV_t'] = copy.deepcopy(volatility_data['Realized Variance'])
    har_rv_data['RV_t-1'] = har_rv_data['RV_t'].shift(1)

    for lag in [5, 22]:
        har_rv_data[f'ave_RV_t-1_t-{lag}'] = har_rv_data[f'RV_t-1'].rolling(window=lag).mean()

    if steps_ahead > 1:
        har_rv_data['RV_t'] = har_rv_data['RV_t'].shift(-(steps_ahead-1)).rolling(window=steps_ahead).mean()
        # For multiple step-ahead forecasts, we forecast average realized variance.
        # Multi-step ahead forecasts of the average were only implemented later, which is why the implementation
        # is a bit wacky here.
        # For multi-step ahead forecasts, RV_t is actually RV_t|t+steps_ahead-1,
        # i.e. the average of t,..., t+(steps_ahead-1).
        # Names were left unchanged for simplicity.
    return har_rv_data ** power

def build_har_cj_data(steps_ahead, square_root):
    """Builds dataset containing daily realized variance, and decomposed daily variance as well as
     weekly and monthly averages. Depending on the forecast horizon RV_t is set to daily realized variance,
     or its weekly or monthly average.

    Arguments
    ---------
    steps_ahead : int
        determines the forecast horizon and thus whether the dependent variable is daily realized variance or its
        weekly or monthly average.
    square_root : bool
        determines whether the square root is taken for all variables derived from realized variance.

    Returns
    -------
    har_rv_cj_data : pd.DataFrame
        Dataset containing dependent and independent variables of the HAR-RV-CJ model
        or its square roots for one forecast horizon.

    """

    power = 0.5 if square_root else 1

    folder_path = BLD_data / 'application'
    with open(DATA_DIR / 'realized_quantities.pkl', 'rb') as f:
        volatility_data = pickle.load(f)


    har_rv_cj_data = pd.DataFrame(index=volatility_data.index)
    har_rv_cj_data['RV_t'] = copy.deepcopy(volatility_data['Realized Variance'])
    for var_type, column in {'C': 'Continuous Variance', 'J': 'Jump Variance'}.items():
        har_rv_cj_data[f'{var_type}_t-1'] = volatility_data[column].shift(1)
        for lag in [5, 22]:
            har_rv_cj_data[f'ave_{var_type}_t-1_t-{lag}'] = har_rv_cj_data[f'{var_type}_t-1'].rolling(window=lag).mean()
    if steps_ahead > 1:
        har_rv_cj_data['RV_t'] = har_rv_cj_data['RV_t'].shift(-(steps_ahead-1)).rolling(window=steps_ahead).mean()
        # See explanation above!
    return har_rv_cj_data ** power

def build_macroeconomic_data():
    """
    Collects macroeconomic variables, indexes, forward fills
    and shifts them to align with the realized variance data.
    """
    folder_path = BLD_data / 'application'

    with open(folder_path / 'macroeconomic_inputs.pkl', 'rb') as f:
        macro_inputs = pickle.load(f)

    with open(DATA_DIR / 'realized_quantities.pkl', 'rb') as f:
        volatility_data = pickle.load(f)

    macroeconomic_data = pd.DataFrame(index=volatility_data.index)

    macroeconomic_data['Fed Effective Rate'] = macro_inputs['fed_rate']['Effective Funds Rate']
    macroeconomic_data['Inflation'] = macro_inputs['inflation']['YoY Inflation']
    macroeconomic_data['GDP Growth'] = macro_inputs['gdp']['GDP Growth']
    macroeconomic_data['Unemployment'] = macro_inputs['unemployment']['Unemployment Values']

    for col in ['Fed Effective Rate', 'Inflation', 'Unemployment', 'GDP Growth']:
        if col != 'Fed Effective Rate':
            macroeconomic_data[col] = macroeconomic_data[col].ffill()
        macroeconomic_data[col] = macroeconomic_data[col].shift(1)

    macroeconomic_data['Unemployment Release'] = macro_inputs['unemployment']['Unemployment Release Dummy']
    macroeconomic_data['GDP Release'] = macro_inputs['gdp']['GDP Release Dummy']
    macroeconomic_data['Inflation Release'] = macro_inputs['inflation']['CPI Release Dummy']
    macroeconomic_data['Fed Meetings'] = macro_inputs['fed_meetings']['Fed Meeting Dummy']

    for col in ['Unemployment Release', 'GDP Release', 'Inflation Release', 'Fed Meetings']:
        macroeconomic_data[col] = macroeconomic_data[col].fillna(0).astype(int)

    return macroeconomic_data

def create_datasets(steps_ahead=1, start_date='2008-02-04', end_date='2024-12-31', square_root:bool=False):
    """
    Creates a dict with four datasets, containing inputs for the HAR-RV model, the HAR-RV-CJ model, and each with added
    macroeconomic variables, for one forecast horizon,taking the square root of the realized variance figures or not.
    """

    datasets = {
        'RV': build_har_data(steps_ahead=steps_ahead, square_root=square_root),
        'CJ': build_har_cj_data(steps_ahead=steps_ahead, square_root=square_root)
    }

    macro_data = build_macroeconomic_data()
    datasets['RV-M'] = pd.concat([datasets['RV'], macro_data], axis=1)
    datasets['CJ-M'] = pd.concat([datasets['CJ'], macro_data], axis=1)

    for name, df in datasets.items():
        datasets[name] = df.loc[start_date:end_date]

    folder_path = BLD_data / 'application'
    file_name  = (f'datasets_{steps_ahead}_step_ahead_square_root.pkl'
                  if square_root else f'datasets_{steps_ahead}_step_ahead.pkl')
    file_path = folder_path / file_name
    with open(file_path, "wb") as f:
        pickle.dump(datasets, f)


if __name__ == "__main__":
    get_all_specifications()
