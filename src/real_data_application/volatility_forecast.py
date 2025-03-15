import pandas as pd
import numpy as np
import pickle
import statsmodels.api as sm

from bagged_tree_model import BaggedTree
from dm_test import dm_test
from MC_simulation.df_to_table import df_to_latex
from src.template_project.config import BLD_data, BLD_final
from sklearn.metrics import mean_squared_error


def split_testing_and_training_x_and_y(dataset, training_obs=3000):

    data = {}
    training = dataset[:training_obs]
    test = dataset[training_obs:]

    data['training'] = {
        'x': training.drop(['RV_t'],axis=1).reset_index(drop=True),
        'y': training['RV_t'].reset_index(drop=True),
    }
    data['testing'] = {
        'x': test.drop(['RV_t'], axis=1).reset_index(drop=True),
        'y': test['RV_t'].reset_index(drop=True),
    }

    return data


def fit_models_and_compute_forecast(
        data,
        tuning_grid: dict|None = None,
        fixed_and_rolling:bool = True
):


    if tuning_grid is None:
        tuning_grid = {
            "estimator__max_depth": [2], # , 6, 8, 10, None],
            "estimator__min_samples_split": [2],# 10, 15, 20, 30, 40, 50, 100],
            'n_estimators': [30],  # , 500],
            'max_samples': [3] #, 5, 10, 12, 14, 20]  # block length
        }
        # tuning_grid = {
        #     "estimator__max_depth": [2, 4, 6, 8, 10, None],
        #     "estimator__min_samples_split": [2, 5, 10, 15, 20, 30, 40, 50, 100],
        #     'n_estimators': [50, 100, 200, 500, 1000],  # , 500],
        #     'max_samples': [1, 2, 5, 10, 12, 14, 20]  # block length
        # }

    tree_model = BaggedTree(
        time_series=True,
        block_bootstrap=True,
        circular=True
    )

    tree_model.fit(data['training']['x'], data['training']['y'], param_grid=tuning_grid, cv_splits=10)
    # tree_model.fit(data['training']['x'], data['training']['y'])


    tree_forecast = {}
    har_forecast = {}

    additional_info = {}

    tree_forecast['fixed'] = tree_model.predict(data['testing']['x'])
    har_forecast['fixed'], additional_info['ols_summary'] = get_ols_forecast(data, rolling=False)

    if fixed_and_rolling:

        tree_forecast['rolling'] = tree_model.rolling_predict(
            data['training']['x'],
            data['training']['y'],
            data['testing']['x'],
            data['testing']['y'],
            refit=True
        )
        har_forecast['rolling'], _ = get_ols_forecast(data, rolling=True)

    additional_info['permutation importance'] = tree_model.feature_importance(
        data['testing']['x'],
        data['testing']['y']
    )
    additional_info['best_parameters'] = tree_model.best_parameters

    return tree_forecast, har_forecast, additional_info


def get_ols_forecast(data, rolling=False, window_size=None):

    if window_size is None:
        window_size = len(data['training']['x'])

    x_train = sm.add_constant(data['training']['x'])
    x_test = sm.add_constant(data['testing']['x'])

    if window_size is None:
        window_size = len(x_train)

    if rolling:
        rolling_predictions = []
        x_train_rolling= x_train.copy()
        y_train_rolling = data['testing']['y'].copy()

        for i in range(len(x_test)):

            x_train_window = x_train_rolling.iloc[-window_size:]
            y_train_window = y_train_rolling[-window_size:]

            model = sm.OLS.fit(x_train_window, y_train_window)

            y_pred = model.predict(x_test.iloc[[i]])[0]
            rolling_predictions.append(y_pred)

            x_train_rolling = pd.concat([x_train_rolling, x_test.iloc[[i]]], axis=0)
            y_train_rolling = np.append(y_train_rolling, data['testing']['y'][i])

        model_forecast = np.array(rolling_predictions)

    else:
        model = sm.OLS(data['training']['y'], x_train).fit()
        model_forecast = model.predict(x_test)

    return model_forecast, (model.summary() if not rolling else None)


def do_all_type_forecasts(steps_ahead=1, training_obs=3000, square_root=False):

    input_folder_path = BLD_data / 'Application'
    input_file_name = f'datasets_{steps_ahead}_step_ahead_square_root.pkl' if square_root else f'datasets_{steps_ahead}_step_ahead.pkl'
    input_file_path = input_folder_path / input_file_name

    with open(input_file_path, 'rb') as f:
        datasets = pickle.load(f)

    model_types = ['RV', 'CJ', 'RV-M', 'CJ-M']
    model_results = {}
    training_testing_data = {}

    for model in model_types:
        training_testing_data[model] = split_testing_and_training_x_and_y(datasets[model], training_obs)
        (
            model_results[f'BT-{model}'],
            model_results[f'HAR-{model}'],
            model_results[f'{model} additional_info']
        ) = fit_models_and_compute_forecast(
            data = training_testing_data[model],
            tuning_grid = None,
            fixed_and_rolling = False # SET BACK TO TRUE ##########################################################################################
        )

    folder_path = BLD_data / 'Test Application' ############# CHANGE TO APPLICATION #############################################################
    if not folder_path.is_dir():
        folder_path.mkdir(exist_ok=True, parents=True)

    forecast_file_name = f'real_data_forecasts_{steps_ahead}_step_ahead_square_root.pkl' if square_root else f'real_data_forecasts_{steps_ahead}_step_ahead.pkl'
    data_file_name = f'real_data_data_{steps_ahead}_step_ahead_square_root.pkl' if square_root else f'real_data_data_{steps_ahead}_step_ahead.pkl'

    forecast_path = folder_path / forecast_file_name
    data_path = folder_path / data_file_name
    for file, path in [[model_results, forecast_path],[training_testing_data, data_path]]:
        with open(path, "wb") as f:
            pickle.dump(file, f)

if __name__ == '__main__':
    do_all_type_forecasts(steps_ahead=1, training_obs=100, square_root=True)











