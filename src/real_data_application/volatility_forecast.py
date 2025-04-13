"""
Computes forecasts of S&P500 realized variance for different specifications.
"""
import pandas as pd
import numpy as np
import pickle
import statsmodels.api as sm
from bagged_tree_class.bagged_tree_model import BaggedTree
from config import BLD_data
import time
import copy


def compute_all_forecasts(training_observations=3000):
    """
    Computes all forecasts discussed in the thesis: 1, 5, and 22-step ahead forecasts of realized variance,
    and a 1-step-ahead forecast for the square root of realized variance.
    """
    np.random.seed(291)
    for sqrt in [True, False]:
        for step in [1] if sqrt else [1, 5, 22]:
            start = time.time()
            print(f'Starting forecasts for {step} steps, square_root {sqrt}', flush=True)
            do_one_specification_forecast_for_all_datasets(
                steps_ahead=step,
                training_obs=training_observations,
                square_root=sqrt,
                fixed_and_rolling=True
            )
            end = time.time()
            print(f'Forecasts for {step} steps, square_root {sqrt} took {end - start} seconds.', flush=True)


def split_testing_and_training_x_and_y(dataset, training_obs=3000):
    """
    Splits data into a training and into a test set, based on a given length of the training set.
    """
    data = {}
    training = dataset[:training_obs]
    test = dataset[training_obs:]
    test_index = dataset.index[training_obs:]

    data['training'] = {
        'x': training.drop(['RV_t'],axis=1).reset_index(drop=True),
        'y': training['RV_t'].reset_index(drop=True),
    }
    data['testing'] = {
        'x': test.drop(['RV_t'], axis=1).reset_index(drop=True),
        'y': test['RV_t'].reset_index(drop=True),
    }

    return data, test_index


def fit_models_and_compute_forecast(
        data,
        tuning_grid: dict|None = None,
        fixed_and_rolling:bool = True
):
    """
    Tunes the tree model, estimate parameters for the HAR model and computes the forecast from both.
    By default, a fixed and a rolling forecast is computed. Returns the forecast series for both the bagged tree
    and the HAR model as well as additional information containing hyperparameters and importance measures.
    The BaggedTree class is used to compute the bagged tree forecast.
    """

    if tuning_grid is None:
        tuning_grid = {
            "estimator__max_depth": [6, 8, 10, None],
            "estimator__min_samples_split": [2, 5, 10, 20],
            'n_estimators': [50, 100, 150, 200],
            'max_samples': [1, 2, 5, 7, 10, 14]
        }

    tree_model = BaggedTree(
        time_series=True,
        block_bootstrap=True,
        circular=True
    )
    tuning_start = time.time()
    tree_model.fit(data['training']['x'], data['training']['y'], param_grid=tuning_grid, cv_splits=5, n_jobs=2)
    tuning_time = time.time() - tuning_start
    print(f'Cross-Validation took {tuning_time} seconds.', flush=True)


    tree_forecast = {}
    har_forecast = {}

    additional_info = {}
    tree_forecast['fixed'] = tree_model.predict(data['testing']['x'])
    har_forecast['fixed'], additional_info['ols_summary'] = get_ols_forecast(data, rolling=False)

    if fixed_and_rolling:
         tree_forecast['rolling'], additional_info['feature_importances'] = tree_model.rolling_predict(
                data['training']['x'],
                data['training']['y'],
                data['testing']['x'],
                data['testing']['y'],
                feature_importance=True
         )
         har_forecast['rolling'], _ = get_ols_forecast(data, rolling=True)

    additional_info['permutation importance'] = pd.Series(
        tree_model.permutation_importance(data['testing']['x'], data['testing']['y'])
    )

    additional_info['best_parameters'] = tree_model.best_parameters

    return tree_forecast, har_forecast, additional_info


def get_ols_forecast(data, rolling=False, window_size=None):
    """
    Estimates parameters of an ordinary least squares model from the training set and computes predictions of the test
    set based on the estimated model. Returns the predictions as well as the model summary.
    """

    if window_size is None:
        window_size = len(data['training']['x'])

    x_train = sm.add_constant(copy.deepcopy(data['training']['x']))
    x_test = sm.add_constant(copy.deepcopy(data['testing']['x']))

    if window_size is None:
        window_size = len(x_train)

    if rolling:
        rolling_predictions = []
        x_train_rolling= copy.deepcopy(x_train)
        y_train_rolling = copy.deepcopy(data['training']['y'])

        for i in range(len(x_test)):
            iter_time = time.time()
            x_train_window = x_train_rolling.iloc[-window_size:].reset_index(drop=True)
            y_train_window = y_train_rolling[-window_size:]

            model = sm.OLS(y_train_window, x_train_window).fit(cov_type='HAC', cov_kwds={'maxlags': None})

            y_pred = model.predict(x_test.iloc[[i]]).reset_index(drop=True)[0]
            rolling_predictions.append(y_pred)

            x_train_rolling = pd.concat([x_train_rolling, x_test.iloc[[i]]], axis=0).reset_index(drop=True)
            y_train_rolling = np.append(y_train_rolling, data['testing']['y'][i])
            if np.mod(i, 100) == 0:
                print(f'Rolling OLS forecast step {i} finished, took {time.time()-iter_time} seconds.', flush=True)

        model_forecast = np.array(rolling_predictions)

    else:
        model = sm.OLS(data['training']['y'], x_train).fit(cov_type='HAC',cov_kwds={'maxlags': None})
        model_forecast = model.predict(x_test)

    return model_forecast, (model if not rolling else None)


def do_one_specification_forecast_for_all_datasets(
        steps_ahead=1,
        training_obs=3000,
        square_root=False,
        fixed_and_rolling=True
):
    """
    Calls functions to compute forecasts from all four datasets for one horizon and one value of square_root.
    Stores the results in a pickle file.
    """
    sqrt_str = 'square_root' if square_root else ''

    input_folder_path = BLD_data / 'application'
    input_file_name = (
        f'datasets_{steps_ahead}_step_ahead_square_root.pkl'
        if square_root else
        f'datasets_{steps_ahead}_step_ahead.pkl'
    )
    input_file_path = input_folder_path / input_file_name

    output_folder_path = BLD_data / 'application' / 'results'
    if not output_folder_path.is_dir():
        output_folder_path.mkdir(exist_ok=True, parents=True)

    with open(input_file_path, 'rb') as f:
        datasets = pickle.load(f)

    model_types = ['RV', 'CJ', 'RV-M', 'CJ-M']
    model_results = {}
    training_testing_data = {}

    for model in model_types:
        try:
            model_start = time.time()
            training_testing_data[model], test_index = split_testing_and_training_x_and_y(
                datasets[model],
                training_obs
            )
            (
                model_results[f'BT-{model}'],
                model_results[f'HAR-{model}'],
                model_results[f'{model} additional_info']
            ) = fit_models_and_compute_forecast(
                data = training_testing_data[model],
                tuning_grid = None,
                fixed_and_rolling = fixed_and_rolling
            )
            model_time = time.time() - model_start
            print(f'{steps_ahead}-step-ahead {sqrt_str} model, {model} dataset took {model_time} seconds.', flush=True)
            model_results[f'{model} additional_info']['test_index'] = test_index

        except Exception as e:
            print(f"Error on model {model}, {steps_ahead} step, square_root {square_root}: {e}", flush=True)
            continue

    forecast_file_name = (
        f'real_data_forecasts_{steps_ahead}_step_ahead_square_root.pkl'
        if square_root else
        f'real_data_forecasts_{steps_ahead}_step_ahead.pkl'
    )
    data_file_name = (
        f'real_data_data_{steps_ahead}_step_ahead_square_root.pkl'
        if square_root else
        f'real_data_data_{steps_ahead}_step_ahead.pkl'
    )

    forecast_path = output_folder_path / forecast_file_name
    data_path = output_folder_path / data_file_name
    for file, path in [[model_results, forecast_path],[training_testing_data, data_path]]:
        with open(path, "wb") as f:
            pickle.dump(file, f)


if __name__ == '__main__':
    compute_all_forecasts(training_observations=3000)
