from arch.bootstrap import optimal_block_length

from MC_backup import do_mc_simulation
from src.template_project.config import BLD_data
from MC_backup import _get_simulation_data
import numpy as np
import pandas as pd
import time
import pickle


def do_rolling_fixed_simulation(settings:dict|None = None, set_sizes:dict|None=None, mc_iterations:int=100):
    np.random.seed(317)
    folder_path = BLD_data / 'rolling_fixed'
    if not folder_path.is_dir():
        BLD_data.mkdir(parents=True, exist_ok=True)

    if settings is None:
        settings = {
            'GARCH': [0.022, 0.068, 0.898],
            'AR': [0.7],
            'RW': [1]
        }

    if set_sizes is None:
        set_sizes = {'training': 500, 'testing': 250}

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
            garch_resid_only=False,
            garch_variance_noise=0,
            fixed_and_rolling=fixed_and_rolling
        )

        file_path = folder_path / f'{p_type}_{steps_ahead}_fixed_and_rolling_simulation.pkl'
        with open(file_path, "wb") as f:
            pickle.dump(results, f)

        type_end = time.time() - type_start
        print(f'Type {p_type} took {type_end} seconds.')


def do_bootstrap_number_simulation(settings:dict|None = None, set_sizes:dict|None=None, mc_iterations:int=1000):

    folder_path = BLD_data / 'Bootstrap_simulation'
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
        set_sizes = {'training': 1000, 'testing': 250}

    optimal_block_length = int(round(set_sizes['training'] ** (1/3)))

    models = ['block', 'circular_block', 'no_block']

    hyperparameter_grids = {
        'block': {
            "estimator__max_depth": [2, 4, 10, None],
            "estimator__min_samples_split": [2, 10, 20, 50],
            'n_estimators': [50],  # , 400],
            'max_samples': [2, 5, optimal_block_length - 1, optimal_block_length, optimal_block_length + 1]  # block length
        },
        'no_block': {
            "estimator__max_depth": [2, 4, 10, None],
            "estimator__min_samples_split": [2, 10, 20, 50],
            'n_estimators': [50],  # , 400],
        },
        'circular_block': {
            "estimator__max_depth": [2, 4, 10, None],
            "estimator__min_samples_split": [2, 10, 20, 50],
            'n_estimators': [50],  # , 400],
            'max_samples': [2, 5, optimal_block_length - 1, optimal_block_length, optimal_block_length + 1]  # block length
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

        tuning_results,_ = do_mc_simulation(
            process_type=p_type,
            mc_iterations=1,
            steps_ahead=steps_ahead,
            set_sizes=set_sizes,
            parameters=params,
            ar_sigma=1,
            window=set_sizes['training'],
            garch_resid_only=False,
            garch_variance_noise=0,
            fixed_and_rolling=False,
            tune=True,
            tuning_grids=hyperparameter_grids
        )
        hyperparameters = tuning_results['hyperparameters']

        fixed_data_sets = {}
        for j in range(mc_iterations):
            data = _get_simulation_data(
                p_type,
                set_sizes,
                params,
                steps_ahead,
                False,
                0,
                1
            )
            fixed_data_sets[f'{j}'] = data


        for i in bootstrap_reps:
            rep_start = time.time()
            for model in models:
                hyperparameters[model]['n_estimators'] = i


            np.random.seed(184)
            results,_ = do_mc_simulation(
                process_type=p_type,
                mc_iterations=mc_iterations,
                steps_ahead=steps_ahead,
                set_sizes=set_sizes,
                parameters=params,
                ar_sigma=1,
                window=set_sizes['training'],
                garch_resid_only=False,
                garch_variance_noise=0,
                fixed_and_rolling=False,
                tune=False,
                hyperparameters=hyperparameters
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

        print(f'Type {p_type} took {type_end} seconds.', flush=True)

    file_path = folder_path / 'number_of_bootstrap_reps_simulation_full.pkl'
    with open(file_path, "wb") as f:
        pickle.dump(dfs, f)


def do_fixed_simulation(settings:dict|None = None, set_sizes:dict|None=None, mc_iterations:int=1000):
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
        set_sizes = {'training': 1000, 'testing': 500}

    fixed_and_rolling = False
    steps_ahead = 1

    for p_type, params in settings.items():

        type_start = time.time()
        np.random.seed(np.random.randint(0, 2 ** 32))
        results, garch_params = do_mc_simulation(
            process_type=p_type,
            mc_iterations=mc_iterations,
            steps_ahead=steps_ahead,
            set_sizes=set_sizes,
            parameters=params,
            ar_sigma=1,
            window=set_sizes['training'],
            garch_resid_only=False,
            garch_variance_noise=0,
            fixed_and_rolling=fixed_and_rolling,
            tune=True
        )

        file_path = folder_path / f'fixed_estimation_simulation_results_{p_type}_{steps_ahead}_step.pkl'
        with open(file_path, "wb") as f:
            pickle.dump(results, f)

        if p_type == 'GARCH':
            file_path = folder_path / 'GARCH_parameters.pkl'
            with open(file_path, "wb") as f:
                pickle.dump(garch_params, f)

        type_end = time.time() - type_start
        print(f'Type {p_type} took {type_end} seconds.')


def do_population_size_simulation(settings:dict|None = None, test_set_size:int=500, mc_iterations:int=1000):

    folder_path = BLD_data / 'set_size_simulation'
    if not folder_path.is_dir():
        folder_path.mkdir(parents=True, exist_ok=True)

    if settings is None:
        settings = {
            'GARCH': [0.022, 0.068, 0.898],
            'AR': [0.7],
            'RW': [1]
        }

    set_sizes = {'training': 100, 'testing': test_set_size}

    np.random.seed(317)
    if not BLD_data.is_dir():
        BLD_data.mkdir(parents=True, exist_ok=True)

    models = ['block', 'circular_block', 'no_block']

    hyperparameter_grids = {
        'block': {
            "estimator__max_depth": [2],
            "estimator__min_samples_split": [2],
            'n_estimators': [50],  # , 400],
            'max_samples': [10]  # block length
        },
        'no_block': {
            "estimator__max_depth": [2],
            "estimator__min_samples_split": [2],
            'n_estimators': [50],  # , 400],
        },
        'circular_block': {
            "estimator__max_depth": [2],
            "estimator__min_samples_split": [2],
            'n_estimators': [50],  # , 400],
            'max_samples': [10]  # block length
        }
        #'block': {
        #    "estimator__max_depth": [2, 4, 10, None],
        #    "estimator__min_samples_split": [2, 5, 10, 20],
        #    'n_estimators': [50, 100, 150, 200],  # , 400],
        #    'max_samples': [10]  # block length
        #},
        #'no_block': {
        #    "estimator__max_depth": [2, 4, 10, None],
        #    "estimator__min_samples_split": [2, 10, 20, 5],
        #    'n_estimators': [50, 100, 150, 200],  # , 400],
        #},
        #'circular_block': {
        #    "estimator__max_depth": [2, 4, 10, None],
        #    "estimator__min_samples_split": [2, 10, 20, 5],
        #    'n_estimators': [50, 100, 150, 200],  # , 400],
        #    'max_samples': [10]  # block length
        #}
    }
    steps_ahead = 1
    np.random.seed(np.random.randint(0, 2 ** 32))

    dfs = {}
    training_size = [25, 50, 75, 100, 125, 150, 200, 250, 300, 400, 500, 750, 1000, 1500, 2000, 3000]
    for p_type, params in settings.items():

        type_start = time.time()
        mse_dict = {}
        mse_dict[p_type] = {
            model: [] for model in models
        }

        tuning_results,_ = do_mc_simulation(
            process_type=p_type,
            mc_iterations=1,
            steps_ahead=steps_ahead,
            set_sizes=set_sizes,
            parameters=params,
            ar_sigma=1,
            window=set_sizes['training'],
            garch_resid_only=False,
            garch_variance_noise=0,
            fixed_and_rolling=False,
            tune=True,
            tuning_grids=hyperparameter_grids,
            fixed_data_sets=None,
            fixed_test_sets=None,
        )
        hyperparameters = tuning_results['hyperparameters']


        fixed_test_sets = {}
        for j in range(mc_iterations):
            data = _get_simulation_data(
                p_type,
                set_sizes,
                params,
                steps_ahead,
                False,
                0,
                1
            )
            fixed_test_sets[f'{j}'] = data['testing']



        for i in training_size:

            rep_start = time.time()
            for model in ['block', 'circular_block']:
                hyperparameters[model]['max_samples'] = int(round(i ** (1/3)))

            set_sizes['training'] = i

            np.random.seed(184)
            results,_ = do_mc_simulation(
                process_type=p_type,
                mc_iterations=mc_iterations,
                steps_ahead=steps_ahead,
                set_sizes=set_sizes,
                parameters=params,
                ar_sigma=1,
                window=set_sizes['training'],
                garch_resid_only=False,
                garch_variance_noise=0,
                fixed_and_rolling=False,
                tune=False,
                hyperparameters=hyperparameters,
                fixed_test_sets=fixed_test_sets
            )
            for model in models:
                mse_dict[p_type][model].append(np.mean(results['evaluation']['fixed'][model]['mse']))
            print(f'Model {p_type}, sample_size {i} finished, took {time.time() - rep_start} seconds.', flush=True)

        dfs[p_type] = pd.DataFrame(mse_dict[p_type])
        dfs[p_type].index = training_size
        type_end = time.time() - type_start

        file_path = folder_path / f'training_set_size_{p_type}.pkl'
        with open(file_path, "wb") as f:
            pickle.dump(dfs[p_type], f)

        print(f'Type {p_type} took {type_end} seconds.', flush=True)

    file_path = folder_path / 'training_set_size_full.pkl'
    with open(file_path, "wb") as f:
        pickle.dump(dfs, f)


if __name__ == '__main__':
    do_population_size_simulation()
    #do_fixed_simulation()
    #do_bootstrap_number_simulation()
    #do_rolling_fixed_simulation()


