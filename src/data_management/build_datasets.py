import pandas as pd
import numpy as np
import pickle
from src.template_project.config import BLD_data

def build_har_data(steps_ahead, square_root):

    power = 0.5 if square_root else 1

    folder_path = BLD_data / 'Application'
    with open(folder_path / 'realized_quantities.pkl', 'rb') as f:
        volatility_data = pickle.load(f)

    har_rv_data = pd.DataFrame(index=volatility_data.index)
    har_rv_data['RV_t'] = volatility_data['Realized Variance']
    har_rv_data[f'RV_t-{steps_ahead}'] = har_rv_data['RV_t'].shift(steps_ahead)

    for lag in [5, 22]:
        har_rv_data[f'ave_RV_t_{lag}'] = har_rv_data[f'RV_t-{steps_ahead}'].rolling(window=lag).mean()

    return har_rv_data ** power

def build_har_cj_data(steps_ahead, square_root):

    power = 0.5 if square_root else 1

    folder_path = BLD_data / 'Application'
    with open(folder_path / 'realized_quantities.pkl', 'rb') as f:
        volatility_data = pickle.load(f)


    har_rv_cj_data = pd.DataFrame(index=volatility_data.index)
    har_rv_cj_data['RV_t'] = volatility_data['Realized Variance']

    for var_type, column in {'C': 'Continuous Variance', 'J': 'Jump Variance'}.items():
        har_rv_cj_data[f'{var_type}_t-{steps_ahead}'] = volatility_data[column].shift(steps_ahead)
        for lag in [5, 22]:
            har_rv_cj_data[f'ave_{var_type}_t_{lag}'] = har_rv_cj_data[f'{var_type}_t-{steps_ahead}'].rolling(window=lag).mean()

    return har_rv_cj_data ** power

def build_macroeconomic_data(steps_ahead):
    folder_path = BLD_data / 'Application'

    with open(folder_path / 'macroeconomic_inputs.pkl', 'rb') as f:
        macro_inputs = pickle.load(f)

    with open(folder_path / 'realized_quantities.pkl', 'rb') as f:
        volatility_data = pickle.load(f)

    macroeconomic_data = pd.DataFrame(index=volatility_data.index)

    macroeconomic_data['Fed Effective Rate'] = macro_inputs['fed_rate']['Effective Funds Rate']
    macroeconomic_data['Inflation'] = macro_inputs['inflation']['YoY Inflation']
    macroeconomic_data['GDP Growth'] = macro_inputs['gdp']['GDP Growth']
    macroeconomic_data['Unemployment'] = macro_inputs['unemployment']['Unemployment Values']

    for col in ['Fed Effective Rate', 'Inflation', 'Unemployment', 'GDP Growth']:
        if col != 'Fed Effective Rate':
            macroeconomic_data[col] = macroeconomic_data[col].ffill()
        macroeconomic_data[col] = macroeconomic_data[col].shift(steps_ahead)


    # Fill first
    #macroeconomic_data['Inflation'].loc['2008-01-02'] = macro_inputs['inflation']['YoY Inflation'].loc['2007-12-14']
    #macroeconomic_data['GDP Growth'].loc['2008-01-02'] = macro_inputs['gdp']['GDP Growth'].loc['2007-12-20']
    #macroeconomic_data['Unemployment'].loc['2008-01-02'] = macro_inputs['unemployment']['Unemployment Values'].loc[
    #    '2007-12-07']


    macroeconomic_data['Unemployment Release'] = macro_inputs['unemployment']['Unemployment Release Dummy']
    macroeconomic_data['GDP Release'] = macro_inputs['gdp']['GDP Release Dummy']
    macroeconomic_data['Inflation Release'] = macro_inputs['inflation']['CPI Release Dummy']
    macroeconomic_data['Fed Meetings'] = macro_inputs['fed_meetings']['Fed Meeting Dummy']

    for col in ['Unemployment Release', 'GDP Release', 'Inflation Release', 'Fed Meetings']:
        macroeconomic_data[col] = macroeconomic_data[col].fillna(0).astype(int)

    return macroeconomic_data

def create_datasets(steps_ahead=1, start_date='2008-02-04', end_date='2024-12-31', square_root:bool=False):

    datasets = {'RV': build_har_data(steps_ahead=steps_ahead, square_root=square_root),
                'CJ': build_har_cj_data(steps_ahead=steps_ahead, square_root=square_root),}

    macro_data = build_macroeconomic_data(steps_ahead=steps_ahead)
    datasets['RV-M'] = pd.concat([datasets['RV'], macro_data], axis=1)
    datasets['CJ-M'] = pd.concat([datasets['CJ'], macro_data], axis=1)

    for name, df in datasets.items():
        datasets[name] = df.loc[start_date:end_date]

    folder_path = BLD_data / 'Application'
    file_name  = f'datasets_{steps_ahead}_step_ahead_square_root.pkl' if square_root else f'datasets_{steps_ahead}_step_ahead.pkl'
    file_path = folder_path / file_name
    with open(file_path, "wb") as f:
        pickle.dump(datasets, f)

if __name__ == "__main__":
    create_datasets(steps_ahead=1, start_date= '2008-02-04', end_date='2024-12-31', square_root=True)

    folder_path = BLD_data / 'Application'
    file_path = folder_path / 'datasets_1_step_ahead.pkl'
    with open(file_path, 'rb') as f:
        datasets = pickle.load(f)








