"""
Script that contains inside functions for Monte Carlo simulations on three types of bagged tree models
(standard bootstrap,block bootstrap and circular block bootstrap) and three different time series processes
(AR, GARCH and Random Walk). The function allows for different settings, e.g. passing fixed training and test sets
or building them inside the function, passing hyperparameters or performing hyperparameter tuning.

Helpers to create simulated datasets  and tune hyperparameters are below the main function.

Quite a few features were included in the code early on (garch predictions with noise added to variance,
larger variance for error term in AR/RW process, multi-step ahead forecast), but not used in the thesis.

Functions that perform specific simulations, as well as functions that unpack and process the simulation results are
stored in different scripts.
"""

import pandas as pd
import numpy as np
import time
from diebold_mariano_test_function.dm_test import dm_test
from sklearn.metrics import mean_squared_error
from arch.univariate import GARCH
from bagged_tree_class.bagged_tree_model import BaggedTree
from statsmodels.tsa.arima_process import ArmaProcess
import copy


def do_mc_simulation(
        process_type:str,
        mc_iterations:int=100,
        steps_ahead:int=1,
        set_sizes=None,
        parameters:list|None=None,
        ar_sigma:float|int=1,
        window:int|None=None,
        garch_variance_noise:float|int = 0,
        fixed_and_rolling:bool=False,
        tune:bool=True,
        tuning_grids:dict=None,
        hyperparameters:dict=None,
        fixed_data_sets:dict=None,
        sample_size_simulation:bool=False,
        size_simulation_full_training_length:int=None
):
    """
    Inside function for monte carlo simulations of different time series processes and three different bagged tree
    model configurations (regular bootstrap, block bootstrap and circular block bootstrap).

    Function returns mean squared error for all iterations, Diebold-Mariano test results comparing the different models
    and the true forecast, the hyperparameters determined in cross-validation and the forecast and data of the
    last iteration.
    Arguments
    ---------
    process_type : str
        determines which process is used to generate the data
    mc_iterations : int
        number of Monte Carlo iterations
    steps_ahead : int
        forecast horizon
    set_sizes : dict
        sizes of the training and the test set
    parameters : list|None
        parameters of the process used to generate data
    ar_sigma : float|int
        variance of the error term for simulated autoregressive data
    window : int|None
        window length or rolling window simulation
    garch_variance_noise : float|int
        variance of noise added to simulated variance (no noise if zero)
    fixed_and_rolling : bool
        determines whether simulation should include a rolling forecast
    tune : bool
        determines whether hyperparameters should be tuned before starting the simulation
    tuning_grids : dict
        grids of possible hyperparameter values to search over in cross-validation
    hyperparameters : dict
        hyperparameters used in tree models if they are not tuned
    fixed_data_sets : dict
        dataset passed to function to be used for the simulation, if data is not supposed to be generated during the
        simulation
    sample_size_simulation : bool
        determines whether a simulation over different sample sizes is performed, in which case a specified amount
        of observations is taken as training set from the fixed data sets
    size_simulation_full_training_length : int
        size of training sets in set size simulation, used for checking whether pandas handles datasets properly
        without changing them

    Returns
    -------
    results : dict
        contains results for evaluation, with mean squared error values and Diebold-Mariano test results,
        as well as tuned hyperparameters

    """
    if hyperparameters is None and not tune:
        msg = 'Set tuning = True or add hyperparameters!'
        raise ValueError(msg)

    if tune and hyperparameters is not None:
        msg = 'Tuning activated and hyperparameters supplied, choose one!'
        raise ValueError(msg)

    if set_sizes is None:
        set_sizes = {'training': 1000, 'testing': 1000}

    if window is None:
        window = set_sizes['training']

    if fixed_and_rolling:
        modulus_number = 2
    else:
        if mc_iterations >= 500:
            modulus_number = 50
        else:
            modulus_number = 10


    types = ['rolling', 'fixed'] if fixed_and_rolling else ['fixed']
    base_models = ['block', 'no_block', 'circular_block']
    evaluation = {
        fc_type: {
            mod: {
                'mse': [],
                'dm': {
                    m: {
                        'test': [],
                        'p_value': []
                    } for m in (base_models) if m != mod
                }
            }
            for mod in base_models
        }
        for fc_type in types
    }

    #evaluation['true_model_mse'] = []
    if fixed_and_rolling:
        evaluation['rolling_fixed_dm'] =  {
            mod: {
                'test': [],
                'p_value': []
            } for mod in base_models}

    if tune:
        tuning_data = get_simulation_data(
            process_type,
            set_sizes,
            parameters,
            steps_ahead,
            garch_variance_noise,
            ar_sigma
        )
        hyper_start = time.time()
        hyperparameters = tune_hyperparameters(tuning_data, tuning_grids)
        hyper_time = time.time() - hyper_start
        print(f'Model {process_type}, hyperparameter tuning finished after {hyper_time:.2f} seconds.', flush=True)

    if fixed_data_sets is not None:
        training_set_length = set_sizes['training']
    for i in range(mc_iterations):
        try:
            start = time.time()
            if fixed_data_sets is not None:

                data = copy.deepcopy(fixed_data_sets[f'{i}'])

                if sample_size_simulation:
                    if len(data['training']['x']) != size_simulation_full_training_length:
                        msg = (
                            f'Training sets should have length {training_set_length}, have length {len(
                                data['training']['x'])}. Check Implementation!')
                        raise ValueError(msg)

                    data['training']['x'] = data['training']['x'].iloc[-set_sizes['training']:].reset_index(drop=True)
                    data['training']['y'] = data['training']['y'][-set_sizes['training']:]
            else:
                data = get_simulation_data(
                    process_type,
                    set_sizes,
                    parameters,
                    steps_ahead,
                    garch_variance_noise,
                    ar_sigma
                )

            randstate = np.random.randint(0, 2**32)
            models = {
                'block': BaggedTree(
                    random_state=randstate,
                    n_estimators=hyperparameters['block']['n_estimators'],
                    time_series=True,
                    block_bootstrap=True,
                    max_depth=hyperparameters['block']['estimator__max_depth'],
                    min_samples_split=hyperparameters['block']['estimator__min_samples_split'],
                    block_size=hyperparameters['block']['max_samples']
                ),
                'no_block': BaggedTree(
                    random_state=randstate,
                    n_estimators=hyperparameters['no_block']['n_estimators'],
                    time_series=True,
                    block_bootstrap=False,
                    max_depth=hyperparameters['no_block']['estimator__max_depth'],
                    min_samples_split=hyperparameters['no_block']['estimator__min_samples_split']
                ),
                'circular_block': BaggedTree(
                    random_state=randstate + 1,
                    n_estimators=hyperparameters['circular_block']['n_estimators'],
                    time_series=True,
                    block_bootstrap=True,
                    max_depth=hyperparameters['circular_block']['estimator__max_depth'],
                    min_samples_split=hyperparameters['circular_block']['estimator__min_samples_split'],
                    block_size=hyperparameters['circular_block']['max_samples'],
                    circular=True
                )
            }

            for model_name, model_obj in models.items():
                model_obj.fit(data['training']['x'], data['training']['y'])

            forecasts = {
                fc_type: {} for fc_type in types
            }

            for model_name, mod in models.items():
                forecasts['fixed'][model_name] = mod.predict(data['testing']['x'])
                if fixed_and_rolling:
                    forecasts['rolling'][model_name], _ = mod.rolling_predict(
                        data['training']['x'],
                        data['training']['y'],
                        data['testing']['x'],
                        data['testing']['y'],
                        window
                 )

            # forecasts['true_model'] = get_true_model_forecast(
            #     model_type=process_type,
            #     data=data,
            #     window=window,
            #     parameters=parameters,
            #     steps_ahead=steps_ahead
            # )

            # evaluation['true_model_mse'].append(
            #     mean_squared_error(data['testing']['y'], forecasts['true_model']) / np.var(data['testing']['y'])
            # )
            for fc_type in types:
                for mod in base_models:

                    evaluation[fc_type][mod]['mse'].append(
                        mean_squared_error(data['testing']['y'], forecasts[fc_type][mod]) / np.var(data['testing']['y'])
                    )

                    for m in base_models:
                        if m != mod:
                            dm_result = dm_test(data['testing']['y'], forecasts[fc_type][mod], forecasts[fc_type][m])
                            evaluation[fc_type][mod]['dm'][m]['test'].append(dm_result.DM)
                            evaluation[fc_type][mod]['dm'][m]['p_value'].append(dm_result.p_value)

            if fixed_and_rolling:
                for mod in base_models:
                    dm_result = dm_test(data['testing']['y'], forecasts['rolling'][mod], forecasts['fixed'][mod])
                    evaluation['rolling_fixed_dm'][mod]['test'].append(dm_result.DM)
                    evaluation['rolling_fixed_dm'][mod]['p_value'].append(dm_result.p_value)

            duration = time.time() - start

            if np.mod(i, modulus_number) == 0:
                print(f'Model {process_type}, finished iteration {i}, time elapsed: {duration:.2f} seconds', flush=True)
        except Exception as e:
            print(f"Error on iteration {i}: {e}", flush=True)
            continue

    results = {
        'hyperparameters': hyperparameters,
        'evaluation': evaluation
    }

    return results


def get_simulation_data(process_type, set_sizes, parameters, steps_ahead=1, garch_variance_noise=0, ar_sigma=1):
    """ Collects simulated data given a determined process. """
    if process_type == 'GARCH':
        data = get_garch_data(
            set_sizes,
            parameters=parameters,
            steps_ahead=steps_ahead,
            noise_scale=garch_variance_noise
        )
    elif process_type == 'AR':
        data = get_ar_data(set_sizes, parameters, steps_ahead, ar_sigma)

    elif process_type == 'RW':
        data = get_ar_data(set_sizes, [1], steps_ahead, ar_sigma)
    else:
        msg = 'Choose either AR, RW, or GARCH'
        raise ValueError(msg)

    return data


def simulate_AR(observations:int, parameters:list, sigma:float=1):
    """Simulates autoregressive process for a given number of observations and list of autoregressive parameters."""

    ar_lag_polynomial = np.r_[1, -np.array(parameters)]
    ma_lag_polynomial = np.array([1])
    return pd.DataFrame(ArmaProcess(ar_lag_polynomial, ma_lag_polynomial).generate_sample(observations, scale=sigma))


def simulate_GARCH(observations: int, parameters: list):
    """Simulates a GARCH(1,1) process for a given number of observations and list of parameters."""

    gm = GARCH(p=1, o=0, q=1)
    rng = lambda n: np.random.normal(loc=0, scale=1, size=n)
    simulated = gm.simulate(
        parameters=parameters,
        nobs=observations,
        rng=rng,
        burn=5000
    )
    return pd.DataFrame(simulated).T


def get_garch_data(set_sizes: dict, parameters: list, steps_ahead: int = 1, noise_scale=0):
    """ Builds training and testing datasets from a GARCH(1,1) process."""

    total_obs = sum(set_sizes.values())
    variance_noise = pd.Series(np.random.normal(loc=0, scale=noise_scale, size=total_obs))  # Noise series

    simulation = simulate_GARCH(observations=total_obs, parameters=parameters)

    noisy_variance = (simulation[1] + variance_noise)

    full_data = pd.DataFrame({'residuals': simulation[0]}).reset_index(drop=True)
    full_data['noisy_variance'] = noisy_variance.reset_index(drop=True)

    training_size = set_sizes['training']
    data = {
        'training': {'x': full_data.iloc[:training_size].reset_index(drop=True),
                     'y': simulation[1].iloc[steps_ahead:steps_ahead + training_size].values},
        'testing': {'x': full_data.iloc[training_size:-steps_ahead].reset_index(drop=True),
                    'y': simulation[1].iloc[steps_ahead + training_size:].values}
    }

    return data


def get_ar_data(set_sizes: dict, parameters: list, steps_ahead: int = 1, sigma: int | float = 1):
    """
    Builds training and testing datasets from an AR (or Random Walk) process.
    y contains the series, x the lagged values.
    """

    order = len(parameters)
    lags = range(1, order + 1)
    discard_obs = order - 1

    total_obs = sum(set_sizes.values())
    simulation = simulate_AR(observations=total_obs, parameters=parameters, sigma=sigma)

    y = simulation[0].iloc[discard_obs + steps_ahead:].reset_index(drop=True)
    x = pd.DataFrame({
        f'y_t_{lag}': simulation[0].shift(steps_ahead + (lag - 1)).iloc[discard_obs + steps_ahead:]
        for lag in lags
    }).reset_index(drop=True)

    train_size = set_sizes['training']
    data = {
        'training': {
            'y': y.iloc[:train_size].values,
            'x': x.iloc[:train_size].reset_index(drop=True)
        },
        'testing': {
            'y': y.iloc[train_size:].values,
            'x': x.iloc[train_size:].reset_index(drop=True)
        }
    }

    return data


def tune_hyperparameters(tuning_data, hyperparameter_grids: dict | None = None):
    """ Tunes hyperparameters of the bootstrap, block bootstrap and circular block bootstrap bagged tree models."""

    optimal_block_size = int(round(len(tuning_data['training']['y']) ** (1 / 3)))
    if hyperparameter_grids is None:
        hyperparameter_grids = {
            'block': {
                "estimator__max_depth": [6, 8, 10, None],
                "estimator__min_samples_split": [2, 5, 10, 20],
                'n_estimators': [50, 100, 150, 200],
                'max_samples': [2, 5, optimal_block_size - 1, optimal_block_size, optimal_block_size + 1] # block length
            },
            'no_block': {
                "estimator__max_depth": [6, 8, 10, None],
                "estimator__min_samples_split": [2, 5, 10, 20],
                'n_estimators': [50, 100, 150, 200],
            },
            'circular_block': {
                "estimator__max_depth": [6, 8, 10, None],
                "estimator__min_samples_split": [2, 5, 10, 20],
                'n_estimators': [50, 100, 150, 200],
                'max_samples': [2, 5, optimal_block_size - 1, optimal_block_size, optimal_block_size + 1]
            },
        }

    hyperparameters = {}
    for tree_type, type_bools in {'block': [True, False], 'no_block': [False, False],
                                  'circular_block': [True, True]}.items():
        tuning_tree = BaggedTree(time_series=True, block_bootstrap=type_bools[0], circular=type_bools[1])
        tuning_tree.fit(
            tuning_data['training']['x'],
            tuning_data['training']['y'],
            param_grid=hyperparameter_grids[tree_type]
        )
        hyperparameters[tree_type] = tuning_tree.best_parameters

    return hyperparameters


# def get_true_model_forecast(model_type:str, parameters, data, window=None, steps_ahead=1):
#     """ Forecasts the test set based on the model used to generate the data."""
#
#     if model_type == 'AR':
#         if len(parameters) > 1 and steps_ahead > 1:
#             msg = 'Function can only handle steps_ahead > 1 for AR(1) or AR(p>1) for steps_ahead=1.'
#             raise ValueError(msg)
#
#         AR_params = np.array(parameters)
#         model_forecast = data['testing']['x'] @ (AR_params ** steps_ahead)
#
#     elif model_type == 'GARCH':
#         model_forecast = do_garch_forecast(data, window=window)
#     elif model_type == 'RW':
#         model_forecast = pd.Series(data['testing']['x'].iloc[:,0])
#
#     else:
#         msg = 'Choose either GARCH, AR or RW'
#         raise ValueError(msg)
#
#     return model_forecast


# def do_garch_forecast(data: dict, steps_ahead:int = 1, window=None):
#     """ Forecasts the test set using a GARCH(1,1) model."""
#
#     if window is None:
#         window = len(data['training']['x'])
#
#     x_train = data['training']['x']['residuals']
#     x_test = data['testing']['x']['residuals']
#
#     var_bounds = np.column_stack([np.full(window, 1e-6), np.full(window, 5 * np.var(x_train))])
#
#     garch_fit = arch_model(x_train, vol='GARCH', p=1, q=1, mean='Zero').fit(disp="off")
#     omega, alpha, beta = garch_fit.params[['omega', 'alpha[1]', 'beta[1]']]
#
#     garch_model = GARCH(p=1, q=1)
#
#     last_variance = garch_fit.conditional_volatility.iloc[-1] ** 2
#
#     garch_forecast = []
#     window_set = x_train.tolist()
#
#     for obs in x_test:
#         window_set.append(obs)
#         window_set = window_set[-window:]
#
#         rolling = garch_model.forecast(
#             parameters=np.array([omega, alpha, beta]),
#             resids=np.array(window_set[-window:]),
#             backcast=last_variance,
#             var_bounds=var_bounds,
#             horizon=steps_ahead
#         )
#         garch_forecast.append(float(rolling.forecasts[0, -1]))
#         last_variance = garch_forecast[-1]
#     return np.array(garch_forecast)
