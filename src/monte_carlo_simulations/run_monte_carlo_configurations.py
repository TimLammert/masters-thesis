"""
Functions build different Monte Carlo Simulations on the number of trees, training set size, regression tree versus
bagged tree, fixed window forecasts, and fixed & rolling forecast. For some simulations, datasets are generated in
advance, to ensure comparability across different settings. Results are stored in pickle files.

Functions that process the results and the function that performs the simulation are stored in different scripts.
"""
from config import BLD_data
from sklearn.metrics import mean_squared_error
from monte_carlo_simulations.monte_carlo_inner_functions import do_mc_simulation, get_simulation_data
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
import numpy as np
import pandas as pd
import time
import pickle
import copy

def do_rolling_fixed_simulation(settings:dict|None = None, set_sizes:dict|None=None, mc_iterations:int=200):
    """
    Performs simulation that compares rolling and fixed window forecasts across model types and time series processes.
    """
    np.random.seed(317)
    folder_path = BLD_data / 'rolling_fixed'
    if not folder_path.is_dir():
        folder_path.mkdir(parents=True, exist_ok=True)

    if settings is None:
        settings = {
            'RW': [1],
            'GARCH': [0.022, 0.068, 0.898],
            'AR': [0.7]
        }

    if set_sizes is None:
        set_sizes = {'training': 250, 'testing': 500}

    fixed_and_rolling = True
    steps_ahead = 1

    for p_type, params in settings.items():

        type_start = time.time()

        np.random.seed(np.random.randint(0, 2 ** 32))
        results = do_mc_simulation(
            process_type=p_type,
            mc_iterations=mc_iterations,
            steps_ahead=steps_ahead,
            set_sizes=set_sizes,
            parameters=params,
            ar_sigma=1,
            window=set_sizes['training'],
            fixed_and_rolling=fixed_and_rolling,
            tune=True,
            tuning_grids=None,
            hyperparameters=None,
            fixed_data_sets=None
        )

        file_path = folder_path / f'{p_type}_{steps_ahead}_fixed_and_rolling_simulation.pkl'
        with open(file_path, "wb") as f:
            pickle.dump(results, f)

        type_end = time.time() - type_start
        print(f'Type {p_type} took {type_end} seconds.')


def do_number_of_trees_simulation(settings:dict|None = None, set_sizes:dict|None = None, mc_iterations:int = 1000):
    """
    Performs simulation on the number of trees that make up the bagged tree, going from one tree up to 500 trees.
    """

    folder_path = BLD_data / 'number_of_trees_simulation'
    if not folder_path.is_dir():
        folder_path.mkdir(parents=True, exist_ok=True)

    np.random.seed(317)
    if settings is None:
        settings = {
            'GARCH': [0.022, 0.068, 0.898],
            'AR': [0.7],
            'RW': [1]
        }

    if set_sizes is None:
        set_sizes = {'training': 1000, 'testing': 1000}

    optimal_block_length = int(round(set_sizes['training'] ** (1/3)))

    models = ['block', 'circular_block', 'no_block']

    hyperparameter_grids = {
        'block': {
            "estimator__max_depth": [6, 8, 10, None],
            "estimator__min_samples_split": [2, 5, 10, 20],
            'n_estimators': [50],
            'max_samples': [2, 5, optimal_block_length - 1, optimal_block_length, optimal_block_length + 1]
        },
        'no_block': {
            "estimator__max_depth": [6, 8, 10, None],
            "estimator__min_samples_split": [2, 5, 10, 20],
            'n_estimators': [50]
        },
        'circular_block': {
            "estimator__max_depth": [2, 4, 10, None],
            "estimator__min_samples_split": [2, 5, 10, 20],
            'n_estimators': [50],
            'max_samples': [2, 5, optimal_block_length - 1, optimal_block_length, optimal_block_length + 1]
        },
    }
    steps_ahead = 1
    np.random.seed(np.random.randint(0, 2 ** 32))

    dfs = {}
    bootstrap_reps = [1, 2, 5, 10, 25, 50, 100, 150, 200, 300, 400, 500]
    for p_type, params in settings.items():

        type_start = time.time()
        mse_dict = {}
        mse_dict[p_type] = {
            model: [] for model in models
        }

        tuning_results = do_mc_simulation(
            process_type=p_type,
            mc_iterations=1,
            steps_ahead=steps_ahead,
            set_sizes=set_sizes,
            parameters=params,
            ar_sigma=1,
            window=set_sizes['training'],
            fixed_and_rolling=False,
            tune=True,
            tuning_grids=hyperparameter_grids
        )
        hyperparameters = tuning_results['hyperparameters']
        fixed_data_sets = {}
        for j in range(mc_iterations):
            data = get_simulation_data(
                process_type=p_type,
                set_sizes=set_sizes,
                parameters=params,
                steps_ahead=steps_ahead,
                garch_variance_noise=False,
                ar_sigma=1
            )
            fixed_data_sets[f'{j}'] = data

        for i in bootstrap_reps:
            rep_start = time.time()
            for model in models:
                hyperparameters[model]['n_estimators'] = i

            results = do_mc_simulation(
                process_type=p_type,
                mc_iterations=mc_iterations,
                steps_ahead=steps_ahead,
                set_sizes=set_sizes,
                parameters=params,
                ar_sigma=1,
                window=set_sizes['training'],
                fixed_and_rolling=False,
                tune=False,
                hyperparameters=hyperparameters,
                fixed_data_sets=fixed_data_sets
            )
            for model in models:
                mse_dict[p_type][model].append(np.mean(results['evaluation']['fixed'][model]['mse']))
            print(f'Model {p_type}, replications {i} finished, took {time.time() - rep_start} seconds.', flush=True)

        dfs[p_type] = pd.DataFrame(mse_dict[p_type])
        dfs[p_type].index = bootstrap_reps


        type_end = time.time() - type_start

        file_path = folder_path / f'number_of_bootstrap_reps_simulation_{p_type}.pkl'
        with open(file_path, "wb") as f:
            pickle.dump(dfs[p_type], f)

        file_path = folder_path / f'{p_type}_hyperparameters.pkl'
        with open(file_path, "wb") as f:
            pickle.dump(hyperparameters, f)

        print(f'Type {p_type} took {type_end} seconds.', flush=True)


def do_fixed_window_simulation(
        settings: dict | None = None,
        set_sizes: dict | None = None,
        mc_iterations:int=1000,
        steps_ahead=1
):
    """
    Performs simulation comparing forecast accuracy of the different bagged tree models over a fixed window forecast
    of three different processes.
    """

    np.random.seed(317)

    folder_path = BLD_data / 'fixed_simulation'
    if not folder_path.is_dir():
        folder_path.mkdir(parents=True, exist_ok=True)

    if settings is None:
        settings = {
            'GARCH': [0.022, 0.068, 0.898],
            'AR': [0.7],
            'RW': [1]
        }

    if set_sizes is None:
        set_sizes = {'training': 2000, 'testing': 1000}

    for p_type, params in settings.items():

        type_start = time.time()
        np.random.seed(np.random.randint(0, 2 ** 32))
        results = do_mc_simulation(
            process_type=p_type,
            mc_iterations=mc_iterations,
            steps_ahead=1,
            set_sizes=set_sizes,
            parameters=params,
            ar_sigma=1,
            window=set_sizes['training'],
            fixed_and_rolling=False,
            tune=True
        )

        file_path = folder_path / f'fixed_estimation_simulation_results_{p_type}_{steps_ahead}_step.pkl'
        with open(file_path, "wb") as f:
            pickle.dump(results, f)

        type_end = time.time() - type_start
        print(f'Type {p_type} took {type_end} seconds.')


def do_training_set_size_simulation(
        settings: dict | None = None,
        test_set_size:int=1000,
        mc_iterations:int=1000
):
    """
    Performs simulation evaluating bagged trees for different training set sizes.
    """
    folder_path = BLD_data / 'set_size_simulation'
    if not folder_path.is_dir():
        folder_path.mkdir(parents=True, exist_ok=True)

    if settings is None:
        settings = {
            'AR': [0.7],
            'GARCH': [0.022, 0.068, 0.898],
            'RW': [1]
        }

    np.random.seed(317)
    if not BLD_data.is_dir():
        BLD_data.mkdir(parents=True, exist_ok=True)

    models = ['block', 'no_block', 'circular_block']

    hyperparameter_grids = {

        'block': {
            "estimator__max_depth": [6, 8, 10, None],
            "estimator__min_samples_split": [2, 5, 10, 20],
            'n_estimators': [50, 100, 150, 200],
            'max_samples': [3]  # block length
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
            'max_samples': [3]  # block length
        }
    }
    steps_ahead = 1
    np.random.seed(np.random.randint(0, 2 ** 32))

    dfs = {}
    training_size = [25, 50, 75, 100, 200, 300, 400, 500, 750, 1000, 1500, 2000, 3000, 5000, 10000]
    fixed_set_sizes = {'training': max(training_size), 'testing': test_set_size}
    tuning_set_sizes = {'training': 1000, 'testing': test_set_size}
    mse_dict = {}
    for p_type, params in settings.items():

        type_start = time.time()
        mse_dict[p_type] = {
            model: [] for model in models
        }

        tuning_results  = do_mc_simulation(
            process_type=p_type,
            mc_iterations=1,
            steps_ahead=steps_ahead,
            set_sizes=tuning_set_sizes,
            parameters=params,
            ar_sigma=1,
            window=tuning_set_sizes['training'],
            fixed_and_rolling=False,
            tune=True,
            tuning_grids=hyperparameter_grids,
            fixed_data_sets=None,
            sample_size_simulation=False
        )
        hyperparameters = tuning_results['hyperparameters']
        fixed_data_sets = {}
        for j in range(mc_iterations):
            data = get_simulation_data(
                process_type=p_type,
                set_sizes=fixed_set_sizes,
                parameters=params,
                steps_ahead=steps_ahead,
                garch_variance_noise=0,
                ar_sigma=1
            )
            fixed_data_sets[f'{j}'] = data


        for i in training_size:

            rep_start = time.time()
            for model in ['block', 'circular_block']:
                hyperparameters[model]['max_samples'] = int(round(i ** (1/3)))

            set_sizes = {'training': i, 'testing': test_set_size}

            np.random.seed(184)
            results = do_mc_simulation(
                process_type=p_type,
                mc_iterations=mc_iterations,
                steps_ahead=steps_ahead,
                set_sizes=set_sizes,
                parameters=params,
                ar_sigma=1,
                window=set_sizes['training'],
                fixed_and_rolling=False,
                tune=False,
                hyperparameters=hyperparameters,
                fixed_data_sets=fixed_data_sets,
                sample_size_simulation=True,
                size_simulation_full_training_length=fixed_set_sizes['training']
            )

            for model in models:
                mse_dict[p_type][model].append(np.mean(results['evaluation']['fixed'][model]['mse']))

            print(f'Model {p_type}, sample_size {i} finished, took {time.time() - rep_start} seconds.', flush=True)

        dfs[p_type] = pd.DataFrame(mse_dict[p_type])
        dfs[p_type].index = training_size

        file_path = folder_path / f'training_set_size_{p_type}.pkl'
        with open(file_path, "wb") as f:
            pickle.dump(dfs[p_type], f)

        file_path = folder_path / f'{p_type}_hyperparameters.pkl'
        with open(file_path, "wb") as f:
            pickle.dump(hyperparameters, f)

        type_end = time.time() - type_start
        print(f'Type {p_type} took {type_end} seconds.', flush=True)

def do_one_versus_hundred_trees_simulation(settings:dict|None = None, test_set_size:int=1000, mc_iterations:int=1000):

    folder_path = BLD_data / 'one_versus_hundred_trees'
    if not folder_path.is_dir():
        folder_path.mkdir(parents=True, exist_ok=True)

    if settings is None:
        settings = {
            'AR': [0.7],
            'GARCH': [0.022, 0.068, 0.898],
            'RW': [1]
        }

    np.random.seed(317)
    if not BLD_data.is_dir():
        BLD_data.mkdir(parents=True, exist_ok=True)

    models = ['block', 'no_block', 'circular_block']

    hyperparameter_grids = {
        'one_tree': {
            "max_depth": [6, 8, 10, None],
            "min_samples_split": [2, 5, 10, 20]
        },
        'hundred_trees': {
            'block': {
                "estimator__max_depth": [6, 8, 10, None],
                "estimator__min_samples_split": [2, 5, 10, 20],
                'n_estimators': [100],
                'max_samples': [10]  # block length
            },
            'no_block': {
                "estimator__max_depth": [6, 8, 10, None],
                "estimator__min_samples_split": [2, 5, 10, 20],
                'n_estimators': [100],
            },
            'circular_block': {
                "estimator__max_depth": [6 ,8, 10, None],
                "estimator__min_samples_split": [2, 5, 10, 20],
                'n_estimators': [100],
                'max_samples': [10]  # block length
            }
        }
    }
    steps_ahead = 1
    np.random.seed(np.random.randint(0, 2 ** 32))

    dfs = {}
    tuning_set_sizes = {'training': 1000, 'testing': test_set_size}

    for p_type, params in settings.items():

        training_size = [25, 50, 100, 200, 300, 400, 500, 1000, 2000, 3000, 5000, 10000
        ] if p_type == 'GARCH' else [25, 50, 100, 200, 300, 400, 500, 1000, 2000, 3000, 5000, 10000, 50000, 100000]

        fixed_set_sizes = {'training': max(training_size), 'testing': test_set_size}
        mse_dict = {}
        dfs[p_type] = {}
        type_start = time.time()
        mse_dict[p_type] = {
            tree_number: {
                model: [] for model in models
            }
            for tree_number in hyperparameter_grids
        }

        fixed_data_sets = {}
        for j in range(mc_iterations):
            data = get_simulation_data(
                process_type=p_type,
                set_sizes=fixed_set_sizes,
                parameters=params,
                steps_ahead=steps_ahead,
                garch_variance_noise=0,
                ar_sigma=1
            )
            fixed_data_sets[f'{j}'] = data

        one_tree_tuning_data = get_simulation_data(
            process_type=p_type,
            set_sizes=tuning_set_sizes,
            parameters=params,
            steps_ahead=steps_ahead,
            garch_variance_noise=0,
            ar_sigma=1
        )

        one_tree_hyperparams = tune_regression_tree_hyperparameters(
            data=one_tree_tuning_data,
            param_grid=hyperparameter_grids['one_tree'],
            n_splits=5,
            n_jobs=1
        )
        tuning_results = do_mc_simulation(
            process_type=p_type,
            mc_iterations=1,
            steps_ahead=steps_ahead,
            set_sizes=tuning_set_sizes,
            parameters=params,
            ar_sigma=1,
            window=tuning_set_sizes['training'],
            fixed_and_rolling=False,
            tune=True,
            tuning_grids=hyperparameter_grids['hundred_trees'],
            fixed_data_sets=None,
            sample_size_simulation=False
        )

        hyperparameters = tuning_results['hyperparameters']

        for tree_number in hyperparameter_grids:

            for i in training_size:
                rep_start = time.time()
                for model in ['block', 'circular_block']:
                    hyperparameters[model]['max_samples'] = int(round(i ** (1/3)))
                    pass
                set_sizes = {'training': i, 'testing': test_set_size}

                if tree_number == 'hundred_trees':
                    np.random.seed(184)
                    results = do_mc_simulation(
                        process_type=p_type,
                        mc_iterations=mc_iterations,
                        steps_ahead=steps_ahead,
                        set_sizes=set_sizes,
                        parameters=params,
                        ar_sigma=1,
                        window=set_sizes['training'],
                        fixed_and_rolling=False,
                        tune=False,
                        hyperparameters=hyperparameters,
                        fixed_data_sets=fixed_data_sets,
                        sample_size_simulation=True,
                        size_simulation_full_training_length=fixed_set_sizes['training']
                    )
                    for model in models:
                        mse_dict[p_type][tree_number][model].append(
                            np.mean(results['evaluation']['fixed'][model]['mse'])
                        )

                elif tree_number == 'one_tree':
                    one_tree_mse = do_single_regression_tree_forecast(
                        fixed_data_sets=fixed_data_sets,
                        mc_iterations=mc_iterations,
                        training_set_size=i,
                        hyperparameters=one_tree_hyperparams
                    )
                    for model in models:
                        mse_dict[p_type][tree_number][model].append(np.mean(one_tree_mse))


                print(
                    f'Model {p_type}, {tree_number}, sample_size {i} finished, took {time.time() - rep_start} seconds.',
                    flush=True
                )

            dfs[p_type][tree_number] = pd.DataFrame(mse_dict[p_type][tree_number])
            dfs[p_type][tree_number].index = training_size

            file_path = folder_path / f'{tree_number}_{p_type}.pkl'
            with open(file_path, "wb") as f:
                pickle.dump(dfs[p_type][tree_number], f)

        file_path = folder_path / f'one_and_hundred_{p_type}.pkl'
        with open(file_path, "wb") as f:
            pickle.dump(dfs[p_type], f)

        file_path = folder_path / f'{p_type}_hyperparameters.pkl'
        with open(file_path, "wb") as f:
            pickle.dump(hyperparameters, f)

        type_end = time.time() - type_start
        print(f'Type {p_type} took {type_end} seconds.', flush=True)


def do_block_length_simulation(settings:dict|None = None, set_sizes:dict|None=None, mc_iterations:int=1000):
    """ Performs simulation on the length of blocks for the block and the circular block bootstrap."""

    folder_path = BLD_data / 'block_length_simulation'
    if not folder_path.is_dir():
        folder_path.mkdir(parents=True, exist_ok=True)

    np.random.seed(317)
    if settings is None:
        settings = {
            'AR': [0.7],
            'RW': [1],
            'GARCH': [0.022, 0.068, 0.898]
        }

    if set_sizes is None:
        set_sizes = {'training': 1000, 'testing': 1000}

    optimal_block_length = int(round(set_sizes['training'] ** (1/3)))

    models = ['block', 'circular_block']

    hyperparameter_grids = {
        'block': {
            "estimator__max_depth": [6, 8, 10, None],
            "estimator__min_samples_split": [2, 5, 10, 20],
            'n_estimators': [50, 100, 150, 200],
            'max_samples': [optimal_block_length]  # block length
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
            'max_samples': [optimal_block_length]  # block length
        },
    }
    steps_ahead = 1
    np.random.seed(np.random.randint(0, 2 ** 32))

    dfs = {}
    block_length = [1, 2, 3, 4, 5, 6, 8, 10, 15, 20]

    for p_type, params in settings.items():

        type_start = time.time()
        mse_dict = {}
        mse_dict[p_type] = {
            model: [] for model in models
        }

        tuning_results = do_mc_simulation(
            process_type=p_type,
            mc_iterations=1,
            steps_ahead=steps_ahead,
            set_sizes=set_sizes,
            parameters=params,
            ar_sigma=1,
            window=set_sizes['training'],
            fixed_and_rolling=False,
            tune=True,
            tuning_grids=hyperparameter_grids
        )
        hyperparameters = tuning_results['hyperparameters']
        fixed_data_sets = {}
        for j in range(mc_iterations):
            data = get_simulation_data(
                process_type=p_type,
                set_sizes=set_sizes,
                parameters=params,
                steps_ahead=steps_ahead,
                garch_variance_noise=0,
                ar_sigma=1
            )
            fixed_data_sets[f'{j}'] = data

        for i in block_length:
            rep_start = time.time()
            for model in models:
                hyperparameters[model]['max_samples'] = i

            results = do_mc_simulation(
                process_type=p_type,
                mc_iterations=mc_iterations,
                steps_ahead=steps_ahead,
                set_sizes=set_sizes,
                parameters=params,
                ar_sigma=1,
                window=set_sizes['training'],
                fixed_and_rolling=False,
                tune=False,
                hyperparameters=hyperparameters,
                fixed_data_sets=fixed_data_sets
            )
            for model in models:
                mse_dict[p_type][model].append(np.mean(results['evaluation']['fixed'][model]['mse']))
            print(f'Model {p_type}, replications {i} finished, took {time.time() - rep_start} seconds.', flush=True)

        dfs[p_type] = pd.DataFrame(mse_dict[p_type])
        dfs[p_type].index = block_length

        file_path = folder_path / f'block_length_simulation_{p_type}.pkl'
        with open(file_path, "wb") as f:
            pickle.dump(dfs[p_type], f)

        file_path = folder_path / f'{p_type}_hyperparameters.pkl'
        with open(file_path, "wb") as f:
            pickle.dump(hyperparameters, f)

        type_end = time.time() - type_start
        print(f'Type {p_type} took {type_end} seconds.', flush=True)

def tune_regression_tree_hyperparameters(data, param_grid, n_splits=5, n_jobs=1):
    """ Tunes regression tree hyperparameters for the one versus hundred tree simulation."""

    model = DecisionTreeRegressor()
    tscv = TimeSeriesSplit(n_splits=n_splits)
    grid_search = GridSearchCV(model, param_grid, cv=tscv, scoring='neg_mean_squared_error', n_jobs=n_jobs)
    grid_search.fit(data['training']['x'], data['training']['y'])

    return grid_search.best_params_


def do_single_regression_tree_forecast(fixed_data_sets, mc_iterations, training_set_size, hyperparameters):
    """
    Does forecast using an individual regression tree on a slice of a large data set.
    The slice is determined by the training set size.
    """
    mse_list = []
    for j in range(mc_iterations):

        data = copy.deepcopy(fixed_data_sets[f'{j}'])
        data['training']['x'] = data['training']['x'].iloc[-training_set_size:,:].reset_index(drop=True)
        data['training']['y'] = data['training']['y'][-training_set_size:]

        tree = DecisionTreeRegressor(
            max_depth=hyperparameters['max_depth'], min_samples_split=hyperparameters['min_samples_split']
        )
        tree.fit(data['training']['x'], data['training']['y'])
        predictions = tree.predict(data['testing']['x'])
        mse_list.append(mean_squared_error(data['testing']['y'], predictions) / np.var(data['testing']['y']))

    return mse_list



if __name__ == '__main__':
    do_fixed_window_simulation(mc_iterations=1000)
    do_one_versus_hundred_trees_simulation(test_set_size=1000, mc_iterations=1000)
    do_training_set_size_simulation(test_set_size=1000, mc_iterations=1000)
    do_block_length_simulation(mc_iterations=1000)
    do_number_of_trees_simulation(mc_iterations=1000)
    do_rolling_fixed_simulation(settings={'RW': [1]}, mc_iterations=200)



