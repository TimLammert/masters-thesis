import pandas as pd
import statsmodels.api as sm
import numpy as np
from dm_test import dm_test
from sklearn.metrics import mean_squared_error
from MC_simulation.monte_carlo_functions import tune_hyperparameters, do_garch_forecast
from MC_simulation.process_simulation import simulate_AR, simulate_GARCH
from bagged_tree_model import BaggedTree
from template_project.config import BLD_data
import pickle
import time




def get_garch_data(set_sizes: dict, parameters: list, steps_ahead: int = 1, resid_only: bool = False, noise_scale=1):
    total_obs = sum(set_sizes.values())
    variance_noise = pd.Series(np.random.normal(loc=0, scale=noise_scale, size=total_obs))  # Noise series


    simulation = simulate_GARCH(observations=total_obs, parameters=parameters)

    residuals_shifted = simulation[0].shift(steps_ahead).iloc[steps_ahead:]
    variance_shifted = simulation[1].shift(steps_ahead).iloc[steps_ahead:]
    noisy_variance = (variance_shifted + variance_noise.iloc[steps_ahead:])

    full_data = pd.DataFrame({'residuals': residuals_shifted}).reset_index(drop=True)
    if not resid_only:
        full_data['noisy_variance'] = noisy_variance.reset_index(drop=True)

    training_size = set_sizes['training']
    data = {
        'training': {'x': full_data.iloc[:training_size].reset_index(drop=True),
                     'y': simulation[1].iloc[steps_ahead:steps_ahead + training_size].values},
        'testing': {'x': full_data.iloc[training_size:].reset_index(drop=True),
                    'y': simulation[1].iloc[steps_ahead + training_size:].values}
    }

    return data


def get_har_data_from_garch_process(set_sizes=None, parameters: list | None = None, steps_ahead: int = 1):

    if set_sizes is None:
        set_sizes = {'training': 1000, 'testing': 1000}
    if parameters is None:
        parameters = [0.022, 0.068, 0.898]

    if set_sizes['training'] < 22 or set_sizes['testing'] < 22:
        msg = 'Chosen set size too small to account for lag order of HAR-model'
        raise ValueError(msg)

    lags = [1, 5, 22]
    discard_lags = lags[-1] - 1

    total_obs = sum(set_sizes.values()) + discard_lags
    simulation = simulate_GARCH(observations=total_obs, parameters=parameters)

    y = simulation[1].iloc[discard_lags + steps_ahead:].reset_index(drop=True)
    x = pd.DataFrame({
        f'variance_t_{lag}': simulation[1].shift(steps_ahead + (lag - 1)).iloc[discard_lags + steps_ahead:]
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


def get_ar_data(set_sizes: dict, parameters: list, steps_ahead: int = 1, sigma: int | float = 1):
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



def do_mc_simulation(
        process_type:str,
        mc_iterations:int=100,
        steps_ahead:int=1,
        set_sizes=None,
        parameters:list|None=None,
        ar_sigma=1,
        window=None,
        garch_resid_only:bool=False,
        garch_variance_noise:float|int = 0.2
):
    if set_sizes is None:
        set_sizes = {'training': 200, 'testing': 200}
    if window is None:
        window = set_sizes['training']

    types = ['rolling', 'fixed']
    base_models = ['block', 'no_block', 'circular_block']

    evaluation = {
        fc_type: {
            mod: {
                'mse': [],
                'dm': {
                    m: {
                        'test': [],
                        'p_value': []
                    } for m in (base_models + ['true_model']) if m != mod
                }
            }
            for mod in base_models
        }
        for fc_type in types
    }

    evaluation['true_model_mse'] = []

    evaluation['rolling_fixed_dm'] =  {
        mod: {
            'test': [],
            'p_value': []
        } for mod in base_models}

    tuning_data = _get_simulation_data(
        process_type,
        set_sizes,
        parameters,
        steps_ahead,
        garch_resid_only,
        garch_variance_noise,
        ar_sigma
    )

    hyperparameters = tune_hyperparameters(tuning_data)

    for i in range(mc_iterations):
        try:
            start = time.time()
            data = _get_simulation_data(
                process_type,
                set_sizes,
                parameters,
                steps_ahead,
                garch_resid_only,
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
                    n_estimators=hyperparameters['no block']['n_estimators'],
                    time_series=True,
                    block_bootstrap=False,
                    max_depth=hyperparameters['no block']['estimator__max_depth'],
                    min_samples_split=hyperparameters['no block']['estimator__min_samples_split']
                ),
                'circular_block': BaggedTree(
                    random_state=randstate,
                    n_estimators=hyperparameters['circular block']['n_estimators'],
                    time_series=True,
                    block_bootstrap=True,
                    max_depth=hyperparameters['circular block']['estimator__max_depth'],
                    min_samples_split=hyperparameters['circular block']['estimator__min_samples_split'],
                    block_size=hyperparameters['circular block']['max_samples'],
                    circular=True
                )
            }

            for model_name, model_obj in models.items():
                model_obj.fit(data['training']['x'], data['training']['y'])

            forecasts = {
                'rolling': {},
                'fixed': {}
            }
            for model_name, mod in models.items():
                forecasts['rolling'][model_name] = mod.rolling_predict(
                    data['training']['x'],
                    data['training']['y'],
                    data['testing']['x'],
                    data['testing']['y'],
                    window
                )
                forecasts['fixed'][model_name] = mod.predict(data['testing']['x'])

            forecasts['true_model'] = get_true_model_forecast(
                type=process_type,
                data=data,
                window=window
            )

            evaluation['true_model_mse'].append(mean_squared_error(data['testing']['y'], forecasts['true_model']))
            for fc_type in types:
                for mod in base_models:
                    evaluation[fc_type][mod]['mse'].append(mean_squared_error(data['testing']['y'], forecasts[fc_type][mod]))
                    for m in base_models + ['true_model']:
                        if m != mod:
                            if m == 'true_model':
                                dm_result = dm_test(data['testing']['y'], forecasts[fc_type][mod], forecasts[m])
                            else:
                                dm_result = dm_test(data['testing']['y'], forecasts[fc_type][mod], forecasts[fc_type][m])
                            evaluation[fc_type][mod]['dm'][m]['test'].append(dm_result.DM)
                            evaluation[fc_type][mod]['dm'][m]['p_value'].append(dm_result.p_value)


            for mod in base_models:
                dm_result = dm_test(data['testing']['y'], forecasts['rolling'][mod], forecasts['fixed'][mod])
                evaluation['rolling_fixed_dm'][mod]['test'].append(dm_result.DM)
                evaluation['rolling_fixed_dm'][mod]['p_value'].append(dm_result.p_value)

            duration = time.time() - start
            if np.mod(i, 10) == 0:
                print(f'Model {process_type}, finished iteration {i}, time elapsed: {duration:.2f} seconds', flush=True)
        except Exception as e:
            print(f"Error on iteration {i}: {e}", flush=True)
            continue

    results = {
        'hyperparameters': hyperparameters,
        'evaluation': evaluation,
        'last_iteration_forecast': forecasts,
        'last_iteration_data': data
    }
    return results

def evaluate_model_simulation():
    pass


def evaluate_simulation(eval:dict):
    pass






        # return pd.DataFrame({'true': data['testing']['y'], 'lm': lm_forecast, 'block': block_forecast, 'no_block': no_block_forecast, 'circular_block': circular_block_forecast})



def get_true_model_forecast(type:str, data, window=None):
    """ Forecasts the test set based on the model used to generate the data."""

    if type == 'GARCH-HAR' or type == 'AR':
        for set in ['training', 'testing']:
            lm_data = data
            # lm_data[set]['x'] = sm.add_constant(lm_data[set]['x'])
        linear_model = sm.OLS(lm_data['training']['y'], lm_data['training']['x'])
        linear_model = linear_model.fit()
        model_forecast = linear_model.predict(data['testing']['x'])
    elif type == 'GARCH':
        model_forecast = do_garch_forecast(data, window=window)
    elif type == 'RW':
        first_value = np.array([data['testing']['x'].iloc[0, -1]])  # Convert scalar to array
        model_forecast = pd.Series(np.concatenate([first_value, data['testing']['y'][:-1]]))

    else:
        msg = 'Choose either GARCH, GARCH-HAR, AR or RW'
        raise ValueError(msg)

    return model_forecast

def _get_simulation_data(process_type, set_sizes, parameters, steps_ahead, garch_resid_only, garch_variance_noise, ar_sigma):

    if process_type == 'GARCH-HAR':
        if set_sizes['training'] < 22 or set_sizes['testing'] < 22:
            msg = (
                f'Set sizes are {set_sizes['training']} for training and {set_sizes['testing']} for testing. \\'
                'HAR uses x_t-22 as an input, need a larger set size.'
                   )
        data = get_har_data_from_garch_process(set_sizes, parameters=parameters, steps_ahead=steps_ahead)
    elif process_type == 'GARCH':
        data = get_garch_data(
            set_sizes,
            parameters=parameters,
            steps_ahead=steps_ahead,
            resid_only=garch_resid_only,
            noise_scale=garch_variance_noise
        )
    elif process_type == 'AR':
        data = get_ar_data(set_sizes, parameters, steps_ahead, ar_sigma)
    elif process_type == 'RW':
        data = get_ar_data(set_sizes, [1], steps_ahead, ar_sigma)
    else:
        msg = 'Choose either AR, RW, GARCH or GARCH-HAR'
        raise ValueError(msg)

    return data


def unpack_evaluation(evaluation, base_models, fc_types):
    dfs = {
        mod: pd.DataFrame() for mod in base_models
    }
    dfs['mse'] = pd.DataFrame()

    for fc_type in fc_types:
        for base in base_models :
            dfs['mse'][f'{fc_type}_{base}'] = evaluation[fc_type][base]['mse']
            comparison_models = [comp for comp in base_models + ['true_model'] if base != comp]
            for comp in comparison_models:
                dfs[base][f'{fc_type}_{comp}_test'] = evaluation[fc_type][base]['dm'][comp]['test']
                dfs[base][f'{fc_type}_{comp}_p_value'] = evaluation[fc_type][base]['dm'][comp]['p_value']
            if base in base_models:
                dfs[base]['rolling_fixed_comparison_test'] = evaluation['rolling_fixed_dm'][base]['test']
                dfs[base]['rolling_fixed_comparison_p_value'] = evaluation['rolling_fixed_dm'][base]['p_value']
    dfs['mse']['true_model'] = evaluation['true_model_mse']

    return dfs

def take_scores(dfs, base_models=['block', 'no_block', 'circular_block'], forecast_types=['rolling', 'fixed']):
    mse_means = dfs['mse'].mean()

    test_results = {
    }

    signif = [0.1, 0.05, 0.01, 0.001]
    for fc_type in forecast_types:
        models = base_models
        for lvl in signif:
            for model in models:
                test_results[model] = {}
                for comp in [m for m in models if m != model]:
                    test_results[model][f'{fc_type}_compared to_true_model_{int(lvl * 100)}%'] = pd.Series(
                        ((dfs[model][f'{fc_type}_true_model_test'] > 0) &
                         (dfs[model][f'{fc_type}_true_model_p_value'] < lvl)).mean()
                    )

                    test_results[model][f'{fc_type}_{model}_compared_to_{comp}_{int(lvl * 100)}%'] = pd.Series(
                        ((dfs[model][f'{fc_type}_{comp}_test'] > 0) & (
                                dfs[model][f'{fc_type}_{comp}_p_value'] < lvl)
                         ).mean()
                    )

    return test_results, mse_means



if __name__ == '__main__':
    np.random.seed(317)
    if not BLD_data.is_dir():
        BLD_data.mkdir(parents=True, exist_ok=True)

    settings = {
        'GARCH': [0.022, 0.068, 0.898],
        'AR': [0.7],
        'RW': [1],
        'GARCH-HAR': [0.022, 0.068, 0.898]
    }
    for p_type, params in settings.items():
        file_path = BLD_data / f'simulation_results_{p_type}.pkl'
        #xyz = pd.DataFrame({'true': data['testing']['y'], 'true_forecast': forecasts['true_model']['forecast'],
        #                    'tree_data': data['testing']['x']['y_t_1']})
        type_start = time.time()
        np.random.seed(np.random.randint(0, 2**32))
        set_sizes = {'training': 400, 'testing': 400}
        steps = 1
        results = do_mc_simulation(
            process_type=p_type,
            mc_iterations=100,
            steps_ahead=1,
            set_sizes=set_sizes,
            parameters=params,
            ar_sigma=1,
            window=set_sizes['training'],
            garch_resid_only=False,
            garch_variance_noise=0
        )
        with open(file_path, "wb") as f:
            pickle.dump(results, f)


        #with open("my_dict.pkl", "rb") as f:
        #    loaded_dict = pickle.load(f)

        #rolling_df = pd.DataFrame(forecasts['rolling'])
        #rolling_df['true'] = forecasts['true_model']
        #fixed_df = pd.DataFrame(forecasts['fixed'])
        #fixed_df['true'] = forecasts['true_model']

        #for fc_type, fc_df in {'fixed': fixed_df, 'rolling': rolling_df}.items():

        #        fig = make_subplots(rows=fc_df.shape[1], cols=1, shared_xaxes=True, subplot_titles=fc_df.columns)

         #   for i, col in enumerate(fc_df.columns):
          #      fig.add_trace(go.Scatter(y=fc_df[col], mode='lines', name=col), row=i+1, col=1)
    #
     #           fig.update_layout(
      #              height=800,
       #             width=800,
        #            title_text=f"Comparison of {fc_type.capitalize()} Predictions",
         #           showlegend=False
          #      )
          #  fig.show()
        type_end = time.time() - type_start
        print(f'Type {p_type} took {type_end} seconds.')

        #fc_types = ['rolling', 'fixed']
        #base_models = ['block', 'no_block', 'circular_block']
        #dfs = unpack_evaluation(eval, base_models, fc_types)












