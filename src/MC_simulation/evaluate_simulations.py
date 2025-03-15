"""
Contains functions that unpack and process results from the Monte Carlo Simulations.
Results are unpacked, reorganized and summarized in plots and tables.
The code storing the results in the MC simulations developed into something rather clumsy over time,
so the code here is not that nice to look at.
"""

import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
from src.template_project.config import DATA_DIR, BLD_data, BLD_final, BLD_figures
from df_to_table import df_to_latex


def unpack_evaluation(evaluation, base_models=None, fc_types=None):
    if fc_types is None:
        fc_types = ['rolling', 'fixed']
    if base_models is None:
        base_models = ['block', 'no_block', 'circular_block']
    dfs = {
        mod: pd.DataFrame() for mod in base_models
    }
    dfs['mse'] = pd.DataFrame()

    for fc_type in fc_types:
        for base in base_models:
            dfs['mse'][f'{fc_type}_{base}'] = evaluation[fc_type][base]['mse']
            comparison_models = [comp for comp in base_models + ['true_model'] if base != comp]
            for comp in comparison_models:
                dfs[base][f'{fc_type}_{comp}_test'] = evaluation[fc_type][base]['dm'][comp]['test']
                dfs[base][f'{fc_type}_{comp}_p_value'] = evaluation[fc_type][base]['dm'][comp]['p_value']
            if base in base_models and 'rolling' in fc_types:
                dfs[base]['rolling_fixed_comparison_test'] = evaluation['rolling_fixed_dm'][base]['test']
                dfs[base]['rolling_fixed_comparison_p_value'] = evaluation['rolling_fixed_dm'][base]['p_value']
    dfs['mse']['true_model'] = evaluation['true_model_mse']

    return dfs

def take_scores(dfs, base_models=None, forecast_types=None, significance_levels=None):

    if significance_levels is None:
        significance_levels = [0.05]
    if forecast_types is None:
        forecast_types = ['rolling', 'fixed']
    if base_models is None:
        base_models = ['block', 'no_block', 'circular_block']
    mse_df = pd.DataFrame()
    mse_df['Mean MSE'] = dfs['mse'].mean()
    mse_df['Standard Deviation'] = dfs['mse'].std()
    mse_df['Count of lowest MSE'] = (dfs['mse'].eq(dfs['mse'].min(axis=1), axis=0)).sum()
    mse_df['Count of highest MSE'] = (dfs['mse'].eq(dfs['mse'].max(axis=1), axis=0)).sum()

    models = base_models

    test_results = {}
    for fc_type in forecast_types:
        for lvl in significance_levels:
            temp_df = pd.DataFrame()
            for col_model in models:
                temp_series = pd.Series()
                for row_model in models:
                    if col_model != row_model:
                        temp_series.loc[f'{row_model}'] = (
                            (dfs[row_model][f'{fc_type}_{col_model}_test'] < 0) &
                            (dfs[row_model][f'{fc_type}_{col_model}_p_value'] < lvl)
                     ).mean()
                    else:
                        temp_series.loc[f'{row_model}'] = np.nan

                temp_df[f'{col_model}'] = temp_series

            true_temp_series = pd.Series()

            for row_model in models:
                true_temp_series.loc[f'{row_model}'] = (
                        (dfs[row_model][f'{fc_type}_true_model_test'] < 0) &
                        (dfs[row_model][f'{fc_type}_true_model_p_value'] < lvl)
                ).mean()

            temp_df['true_model'] = true_temp_series
            test_results[f'{fc_type}_{lvl*100}%'] = temp_df

    if 'rolling' in forecast_types:
        rolling_fixed = pd.DataFrame()
        fixed_better = pd.Series()
        rolling_better = pd.Series()
        for lvl in significance_levels:
            for model in models:
                rolling_better.loc[model] = (
                        (dfs[model]['rolling_fixed_comparison_test'] < 0) &
                        (dfs[model]['rolling_fixed_comparison_p_value'] < lvl)
                ).mean()

                fixed_better.loc[model] = (
                        (dfs[model]['rolling_fixed_comparison_test'] > 0) &
                        (dfs[model]['rolling_fixed_comparison_p_value'] < lvl)
                ).mean()

            rolling_fixed[f'Rolling better than fixed {lvl*100}%'] = fixed_better.copy()
            rolling_fixed[f'Fixed better than rolling {lvl*100}%'] = rolling_better.copy()
    else:
        rolling_fixed = 'No rolling forecasts included in results'

    return test_results, mse_df, rolling_fixed


def process_fixed_simulations(processes:list|None = None):

    simulation_name = 'fixed_only'
    folder_path = BLD_final / simulation_name
    if not folder_path.is_dir():
        BLD_final.mkdir(parents=True, exist_ok=True)

    if processes is None:
        processes = ['AR', 'GARCH', 'RW']


    evaluation = {}
    unpacked = {}

    for p_type in processes:

        file_path = BLD_data / f'fixed_estimation_simulation_results_{p_type}_1_step.pkl'
        with open(file_path, 'rb') as f:
            loaded_file = pickle.load(f)
        data = loaded_file['last_iteration_data']
        forecasts = loaded_file['last_iteration_forecast']
        evaluation[p_type] = unpack_evaluation(loaded_file['evaluation'], fc_types=['fixed'])
        unpacked[p_type] = {}

        unpacked[p_type]['test_results'], unpacked[p_type]['mse_df'], _ = take_scores(evaluation[p_type], forecast_types=['fixed'])

        for df, location in {'DM Test Results': unpacked[p_type]['test_results']['fixed_5.0%'], 'Mean Squared Error': unpacked[p_type]['mse_df']}.items():
            table_file_path = folder_path / f'fixed_estimation_simulation_results_{p_type}_{df}.txt'
            table_name = f'{p_type}-model {df}, fixed window'
            table_label = f'fixed-only-{p_type}-{df}'
            latex_str = df_to_latex(location, table_name, table_label)
            with open(table_file_path, "w") as f:
                f.write(latex_str)

        plot_forecast_and_true_vals(
            process=p_type,
            simulation_name=simulation_name,
            data=data,
            forecasts=forecasts,
            include_true=True
        )

def process_fixed_and_rolling_simulations(processes:list|None = None):
    folder_path = BLD_final / 'fixed_and_rolling'
    if not folder_path.is_dir():
        folder_path.mkdir(parents=True, exist_ok=True)

    if processes is None:
        processes = ['AR', 'GARCH', 'RW']


    evaluation = {}
    unpacked = {}
    rolling_fixed_comparison = {}

    for p_type in processes:

        file_path = BLD_data / f'simulation_results_{p_type}_f_and_r_1_step.pkl'
        with open(file_path, 'rb') as f:
            loaded_file = pickle.load(f)
        data = loaded_file['last_iteration_data']
        forecasts = loaded_file['last_iteration_forecast']
        evaluation[p_type] = unpack_evaluation(loaded_file['evaluation'], fc_types=['fixed', 'rolling'])
        unpacked[p_type] = {}

        unpacked[p_type]['test_results'], unpacked[p_type]['mse_df'], rolling_fixed_comparison[p_type]= take_scores(evaluation[p_type], forecast_types=['fixed', 'rolling'])

        for df_name, location in {
            'Fixed DM Test Results': unpacked[p_type]['test_results']['fixed_5.0%'],
            'Rolling DM Test Results': unpacked[p_type]['test_results']['rolling_5.0%'],
            'Rolling vs. Fixed Comparison DM Test Results': rolling_fixed_comparison[p_type],
            'Mean Squared Error': unpacked[p_type]['mse_df'],
        }.items():

            table_file_path = folder_path / f'F_and_R_{p_type}_{df_name}.txt'
            table_name = f'{p_type}-model {df_name}'
            table_label = f'{p_type}-{df_name}'
            latex_str = df_to_latex(location, table_name, table_label)
            with open(table_file_path, "w") as f:
                f.write(latex_str)

        plot_forecast_and_true_vals(
            process=p_type,
            simulation_name='fixed_and_rolling',
            data=data,
            forecasts=forecasts,
            include_true=True
        )



def process_bootstrap_analysis(processes:list|None = None):
    if processes is None:
        processes = ['AR', 'GARCH', 'RW']

    folder_path = BLD_final / 'Bootstrap_simulation'
    if not folder_path.is_dir():
        folder_path.mkdir(parents=True, exist_ok=True)

    for p_type in processes:
        path = BLD_data / f'number_of_bootstrap_reps_simulation_{p_type}.pkl'
        with open(path, 'rb') as f:
            loaded_file = pickle.load(f)

        plot_bootstrap_replications_simulation(p_type, loaded_file)

        df = loaded_file.T
        table_file_path = folder_path / f'Bootstrap_Reps_{p_type}.txt'
        table_name = f'{p_type}-Bootstrap-Replications'
        table_label = f'{p_type}-bootstraps'
        latex_str = df_to_latex(df, table_name, table_label)
        with open(table_file_path, "w") as f:
            f.write(latex_str)


def plot_bootstrap_replications_simulation(process, data):
    folder_path = BLD_figures / 'bootstrap_rep_evolution'
    if not folder_path.is_dir():
        folder_path.mkdir(parents=True, exist_ok=True)

    name_dict ={
        'no_block': 'Regular Bootstrap',
        'block': 'Block Bootstrap',
        'circular_block': 'Circular Block Bootstrap'
    }
    fig = go.Figure()

    for col in data.columns:
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data[col],
            mode='lines+markers',
            line=dict(width=1),
            name=f'{name_dict[col]}'
        ))

        fig.update_layout(
            xaxis=dict(
                type="log",
                title="t",
                tickvals=[1, 2, 5, 10, 25, 50, 100, 150, 200, 300, 400, 500],
                ticktext=[r"1", r"2", r"5", r"10", r"25", r"50", r"100", r"150", r"200", r"300", r"400", r"500"],
                # Custom labels
                tickmode="array"
            ),
            yaxis_title='Forecast',
            legend=dict(
                x=0.05,
                y=0.9,
                bgcolor="rgba(255,255,255,0.5)"
            ),
            width=700,
            height=500,
            template='simple_white',
            margin=dict(
                l=20,
                r=20,
                t=20,
                b=20
            )
        )
        plot_image_path = folder_path / f'{process}_bootstrap_simulation_plot.png'
        plot_pkl_path = folder_path / f'{process}_bootstrap_simulation_plot.pkl'
        fig.write_image(plot_image_path, scale=3)
        with open(plot_pkl_path, "wb") as f:
            pickle.dump(fig, f)

def process_set_size_analysis(processes:list|None = None):
    if processes is None:
        processes = ['AR', 'GARCH', 'RW']

    folder_path = BLD_final / 'set_size_simulation'
    if not folder_path.is_dir():
        folder_path.mkdir(parents=True, exist_ok=True)

    for p_type in processes:
        path = BLD_data / 'set_size_simulation' / f'training_set_size_{p_type}.pkl'
        with open(path, 'rb') as f:
            loaded_file = pickle.load(f)

        plot_set_size_simulation(p_type, loaded_file)

        df = loaded_file.T
        table_file_path = folder_path / f'training_set_size_{p_type}.txt'
        table_name = f'{p_type}-Sample-Size'
        table_label = f'{p_type}-sample-size'
        latex_str = df_to_latex(df, table_name, table_label)
        with open(table_file_path, "w") as f:
            f.write(latex_str)


def plot_set_size_simulation(process, data):
    folder_path = BLD_figures / 'set_size_simulation'
    if not folder_path.is_dir():
        folder_path.mkdir(parents=True, exist_ok=True)

    name_dict ={
        'no_block': 'Regular Bootstrap',
        'block': 'Block Bootstrap',
        'circular_block': 'Circular Block Bootstrap'
    }
    fig = go.Figure()
    x_values = list(data.index)
    for col in data.columns:
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data[col],
            mode='lines+markers',
            line=dict(width=1),
            name=f'{name_dict[col]}'
        ))
        fig.update_layout(
            xaxis=dict(
                type="log",
                title="t",
                tickvals=x_values,
                ticktext=[rf'{i}' for i in x_values],
                # Custom labels
                tickmode="array"
            ),
            yaxis_title='Forecast',
            legend=dict(
                x=0.05,
                y=0.9,
                bgcolor="rgba(255,255,255,0.5)"
            ),
            width=700,
            height=500,
            template='simple_white',
            margin=dict(
                l=20,
                r=20,
                t=20,
                b=20
            )
        )
        plot_image_path = folder_path / f'{process}_sample_size_simulation_plot.png'
        plot_pkl_path = folder_path / f'{process}_sample_size_simulation_plot.pkl'
        fig.write_image(plot_image_path, scale=3)
        with open(plot_pkl_path, "wb") as f:
            pickle.dump(fig, f)

def plot_forecast_and_true_vals(process, simulation_name, data, forecasts, include_true:bool=False):

    folder_path = BLD_figures / simulation_name
    if not folder_path.is_dir():
        folder_path.mkdir(parents=True, exist_ok=True)

    name_dict ={
        'no_block': 'Regular Bootstrap',
        'block': 'Block Bootstrap',
        'circular_block': 'Circular Block Bootstrap'
    }

    exclude = 'true_model'
    for key, val in forecasts.items():
        if key != exclude:

            plot_df = pd.DataFrame()
            plot_df['True Values'] = data['testing']['y']
            for model, model_forecast in val.items():
                plot_df[name_dict[model]] = model_forecast

            if include_true:
                plot_df['True Model Forecast'] = forecasts['true_model']

            fig = go.Figure()

            for col in plot_df.columns:
                fig.add_trace(go.Scatter(
                    x=plot_df.index,
                    y=plot_df[col],
                    mode='lines',
                    line=dict(width=1),
                    name=f'{col}'
                ))
            fig.update_layout(
                # title='Nonlinear Relationship with a Bump Shape',
                xaxis_title='t',
                yaxis_title='Forecast',
                legend=dict(
                    x=0.05,
                    y=0.9,
                    bgcolor="rgba(255,255,255,0.5)"
                ),
                width=700,
                height=500,
                template='simple_white',
                margin=dict(
                    l=20,
                    r=20,
                    t=20,
                    b=20
                )
            )
            plot_image_path = folder_path / f'{simulation_name}_{process}_{key}_forecast_plot.png'
            plot_pkl_path = folder_path / f'{simulation_name}_{process}_{key}_forecast_plot.pkl'
            fig.write_image(plot_image_path, scale=3)
            with open(plot_pkl_path, "wb") as f:
                pickle.dump(fig, f)







if __name__ == '__main__':
    #file_path = BLD_data / 'simulation_results_GARCH_f_and_r_1_step.pkl'
    #
    #with open(file_path, 'rb') as f:
    #    loaded_dict = pickle.load(f)
    #
    #eval = loaded_dict['evaluation']
    #
    #unpacked_eval = unpack_evaluation(eval, ['block', 'no_block', 'circular_block'], ['rolling', 'fixed'])
    #test_results, mse_df, rolling_fixed = take_scores(unpacked_eval)

    process_set_size_analysis()


