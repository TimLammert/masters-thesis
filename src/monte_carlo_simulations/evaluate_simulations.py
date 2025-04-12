"""
Contains functions that unpack and process results from the Monte Carlo Simulations.
Results are unpacked, reorganized and summarized in functions_for_plots_and_tables and tables.
The code storing the results in the MC simulations developed into something rather clumsy over time,
so the code here is not that nice to look at. Some features, such as tables ranking the MSE
of different model specifications were not used, in the thesis.
"""

import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
from config import BLD_data, BLD_final, BLD_figures
from functions_for_plots_and_tables.df_to_table import df_to_latex
from functions_for_plots_and_tables.update_layout import update_plot_layout
from plotly.subplots import make_subplots


def unpack_evaluation(evaluation, base_models=None, fc_types=None):
    """
    Unpacks the results stored in the MC simulations.
    """
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
            comparison_models = [comp for comp in base_models if base != comp]
            for comp in comparison_models:
                dfs[base][f'{fc_type}_{comp}_test'] = evaluation[fc_type][base]['dm'][comp]['test']
                dfs[base][f'{fc_type}_{comp}_p_value'] = evaluation[fc_type][base]['dm'][comp]['p_value']
            if base in base_models and 'rolling' in fc_types:
                dfs[base]['rolling_fixed_comparison_test'] = evaluation['rolling_fixed_dm'][base]['test']
                dfs[base]['rolling_fixed_comparison_p_value'] = evaluation['rolling_fixed_dm'][base]['p_value']

    return dfs

def take_scores(dfs, base_models=None, forecast_types=None, significance_levels=None):
    """
    Gathers information on MSE of different model specifications and Diebold-Mariano test results
    for the fixed window and the fixed and rolling window forecasts and stores information in pd.DataFrames.
    """

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

            rolling_fixed[f'Rolling better than fixed {lvl*100}%'] = rolling_better.copy()
            rolling_fixed[f'Fixed better than rolling {lvl*100}%'] = fixed_better.copy()
    else:
        rolling_fixed = 'No rolling forecasts included in results'

    return test_results, mse_df, rolling_fixed


def process_fixed_simulations(processes:list|None = None):
    """
    Loads results from the fixed window Monte Carlo simulation and stores results
    on MSE and Diebold-Mariano tests in txt files containing LaTeX Tables.
    """

    simulation_name = 'fixed_simulation'
    folder_path = BLD_final / simulation_name
    if not folder_path.is_dir():
        folder_path.mkdir(parents=True, exist_ok=True)

    if processes is None:
        processes = ['AR', 'GARCH', 'RW']

    evaluation = {}
    unpacked = {}

    for p_type in processes:

        file_path = BLD_data / 'fixed_simulation' / f'fixed_estimation_simulation_results_{p_type}_1_step.pkl'
        with open(file_path, 'rb') as f:
            loaded_file = pickle.load(f)

        evaluation[p_type] = unpack_evaluation(loaded_file['evaluation'], fc_types=['fixed'])
        unpacked[p_type] = {}

        unpacked[p_type]['test_results'], unpacked[p_type]['mse_df'], _ = take_scores(
            evaluation[p_type], forecast_types=['fixed']
        )

        for df, location in {
            'DM Test Results': unpacked[p_type]['test_results']['fixed_5.0%'],
            'Mean Squared Error': unpacked[p_type]['mse_df']
        }.items():
            table_file_path = folder_path / f'fixed_estimation_simulation_results_{p_type}_{df}.txt'
            table_name = f'{p_type}-model {df}, fixed window'
            table_label = f'fixed-only-{p_type}-{df}'
            latex_str = df_to_latex(location, table_name, table_label)
            with open(table_file_path, "w") as f:
                f.write(latex_str)


def process_fixed_and_rolling_simulations(processes:list|None = None):
    """
    Loads results from the fixed and rolling window comparison simulation and stores results
    on MSE and Diebold-Mariano tests in txt files containing LaTeX Tables.
    """

    folder_path = BLD_final / 'fixed_and_rolling'
    if not folder_path.is_dir():
        folder_path.mkdir(parents=True, exist_ok=True)

    if processes is None:
        processes = ['AR', 'GARCH', 'RW']

    evaluation = {}
    unpacked = {}
    rolling_fixed_comparison = {}

    for p_type in processes:

        file_path = BLD_data / 'rolling_fixed' / f'{p_type}_1_fixed_and_rolling_simulation.pkl'
        with open(file_path, 'rb') as f:
            loaded_file = pickle.load(f)
        loaded_file = loaded_file[0]
        evaluation[p_type] = unpack_evaluation(loaded_file['evaluation'], fc_types=['fixed', 'rolling'])
        unpacked[p_type] = {}

        unpacked[p_type]['test_results'], unpacked[p_type]['mse_df'], rolling_fixed_comparison[p_type]= take_scores(
            evaluation[p_type],
            forecast_types=['fixed', 'rolling']
        )

        for df_name, location in {
            'Fixed DM Test Results Row Better Than Column': unpacked[p_type]['test_results']['fixed_5.0%'],
            'Rolling DM Test Results Row Better Than Column': unpacked[p_type]['test_results']['rolling_5.0%'],
            'Rolling vs. Fixed Comparison DM Test Results': rolling_fixed_comparison[p_type],
            'Mean Squared Error': unpacked[p_type]['mse_df']
        }.items():
            table_file_path = folder_path / f'F_and_R_{p_type}_{df_name}.txt'
            table_name = f'{p_type}-model {df_name}'
            table_label = f'{p_type}-{df_name}'
            latex_str = df_to_latex(location, table_name, table_label)
            with open(table_file_path, "w") as f:
                f.write(latex_str)


def process_number_of_trees_simulation(processes: list | None = None):
    """
    Creates tables and plots displaying the results of the Monte Carlo simulation
    studying MSE for different numbers of trees in the bagged tree.
    """

    if processes is None:
        processes = ['AR', 'GARCH', 'RW']

    folder_path = BLD_final / 'number_of_trees_simulation'
    if not folder_path.is_dir():
        folder_path.mkdir(parents=True, exist_ok=True)
    process_results = {}
    for p_type in processes:
        path = BLD_data / 'number_of_trees_simulation' / f'number_of_bootstrap_reps_simulation_{p_type}.pkl'
        with open(path, 'rb') as f:
            loaded_file = pickle.load(f)
        process_results[p_type] = loaded_file

        df = loaded_file
        table_file_path = folder_path / f'Bootstrap_Reps_{p_type}.txt'
        table_name = f'{p_type}-Bootstrap-Replications'
        table_label = f'{p_type}-bootstraps'
        latex_str = df_to_latex(df, table_name, table_label)
        with open(table_file_path, "w") as f:
            f.write(latex_str)

    create_number_of_trees_subplots(process_results)


def create_number_of_trees_subplots(process_results):
    """
    Creates plots that displays the results of the Monte Carlo simulation studying
    MSE for different numbers of trees in the bagged tree.
    """

    folder_path = BLD_figures / 'number_of_trees_simulation'
    if not folder_path.is_dir():
        folder_path.mkdir(parents=True, exist_ok=True)

    name_dict = {
        'no_block': 'Regular Bootstrap',
        'block': 'Block Bootstrap',
        'circular_block': 'Circular Block Bootstrap'
    }
    colour_dict = {
        'no_block': '#FF7F32',
        'block': '#3B5998',
        'circular_block': '#A3C9D6'
    }

    processes = list(process_results.keys())
    fig = make_subplots(
        rows=1, cols=len(processes),
        subplot_titles=processes,
        horizontal_spacing=0.02
    )
    tree_number = [1, 2, 5, 10, 25, 50, 100, 150, 200, 300, 400, 500]
    for col_index, process in enumerate(processes, start=1):
        df = process_results[process]
        for i, model in enumerate(df.columns):
            show_legend = col_index == 1
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df[model],
                mode='lines+markers',
                name=name_dict[model] if show_legend else None,
                line=dict(color=colour_dict[model], width=0.75),
                marker=dict(size=4),
                showlegend=show_legend
            ), row=1, col=col_index)

        fig.update_xaxes(
            type="log",
            tickvals=tree_number,
            ticktext=[f'{i}' for i in tree_number],
            title_text="Number of Individual Trees" if col_index == (len(processes) // 2 + 1) else None,
            row=1, col=col_index
        )

        fig.update_yaxes(
            title_text="MSE" if col_index == 1 else None,
            row=1, col=col_index
        )

    fig.update_layout(
        annotations=[dict(font=dict(size=10)) for a in fig['layout']['annotations']]  # Smaller subplot titles
    )

    fig = update_plot_layout(fig, subplots=True)

    plot_image_path = folder_path / f'subplots_number_of_trees_simulation_plot.png'
    fig.write_image(plot_image_path, scale=3)


def process_set_size_analysis(processes:list|None = None, ar_tuned_with_small_sample=False):
    """
    Create plots and tables displaying the results of the Monte Carlo simulation studying
    MSE over different training set sizes, for all three processes and for the AR case tuned to
    a set with 25 observations (shown in the appendix).
    """
    if ar_tuned_with_small_sample:
        msg = ('Simulation function was changed manually to create data, \n'
               '"25training_set_size_AR.pkl" not available, unless it was actively built, \n'
               'in which case, this error can be deleted.')
        raise NotImplementedError(msg)

    name_str = '25' if ar_tuned_with_small_sample else ''


    if processes is None:
        processes = ['AR', 'GARCH', 'RW'] if not ar_tuned_with_small_sample else ['AR']

    folder_path = BLD_final / 'set_size_simulation'
    if not folder_path.is_dir():
        folder_path.mkdir(parents=True, exist_ok=True)
    process_results = {}
    for p_type in processes:
        path = BLD_data / 'set_size_simulation' / f'{name_str}training_set_size_{p_type}.pkl'
        with open(path, 'rb') as f:
            loaded_file = pickle.load(f)

        process_results[p_type] = loaded_file
        df = loaded_file
        table_file_path = folder_path / f'{name_str}training_set_size_{p_type}.txt'
        table_name = f'{p_type}-Sample-Size'
        table_label = f'{p_type}-sample-size'
        latex_str = df_to_latex(df, table_name, table_label)
        with open(table_file_path, "w") as f:
            f.write(latex_str)
    if not ar_tuned_with_small_sample:
        create_sample_size_subplots(process_results)

    # Section below is commented out, because the function creating the simulation was changed manually
    # for the AR simulation found in the appendix.
    # else:
    #     plot_set_size_simulation('AR', loaded_file)

def plot_set_size_simulation(process, data):
    """
    Plot results of the Monte Carlo simulation on different training set sizes for one process.
    """

    folder_path = BLD_figures / 'set_size_simulation'
    if not folder_path.is_dir():
        folder_path.mkdir(parents=True, exist_ok=True)

    name_dict ={
        'no_block': 'Regular Bootstrap',
        'block': 'Block Bootstrap',
        'circular_block': 'Circular Block Bootstrap'
    }
    colour_dict = {
        'no_block': '#FF7F32',
        'block': '#3B5998',
        'circular_block': '#A3C9D6'
    }
    fig = go.Figure()
    x_values = list(data.index)
    for col in data.columns:
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data[col],
            mode='lines+markers',
            line=dict(width=1, color=colour_dict[col]),
            name=f'{name_dict[col]}'
        ))
        fig.update_layout(
            xaxis=dict(
                type="log",
                title="Training Set Size",
                tickvals=x_values,
                ticktext=[rf'{i}' for i in x_values],
                tickmode="array"
            ),
            yaxis_title='MSE',
        )
        fig = update_plot_layout(fig)

        plot_image_path = folder_path / f'{process}_sample_size_simulation_plot.png'
        plot_pkl_path = folder_path / f'{process}_sample_size_simulation_plot.pkl'
        fig.write_image(plot_image_path, scale=3)
        with open(plot_pkl_path, "wb") as f:
            pickle.dump(fig, f)

def create_sample_size_subplots(process_results):
    """
    Creates plot that displays the results of the Monte Carlo simulation on different training set sizes
    for all processes.
    """
    folder_path = BLD_figures / 'set_size_simulation'
    if not folder_path.is_dir():
        folder_path.mkdir(parents=True, exist_ok=True)

    name_dict = {
        'no_block': 'Regular Bootstrap',
        'block': 'Block Bootstrap',
        'circular_block': 'Circular Block Bootstrap'
    }
    colour_dict = {
        'no_block': '#FF7F32',
        'block': '#3B5998',
        'circular_block': '#A3C9D6'
    }
    sample_sizes = [25, 50, 75, 100, 200, 300, 400, 500, 750, 1000, 1500, 2000, 3000, 5000, 10000]

    processes = list(process_results.keys())
    fig = make_subplots(
        rows=1, cols=len(processes),
        subplot_titles=processes,
        horizontal_spacing=0.02
    )

    for col_index, process in enumerate(processes, start=1):
        df = process_results[process]
        for i, model in enumerate(df.columns):
            show_legend = col_index == 1  # Show legend only for the first subplot
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df[model],
                mode='lines+markers',
                name=name_dict[model] if show_legend else None,  # Only name legend items once
                line=dict(color=colour_dict[model], width=0.75),
                marker=dict(size=4),
                showlegend=show_legend
            ), row=1, col=col_index)

        fig.update_xaxes(
            type="log",
            tickvals=sample_sizes,
            ticktext=[f"{i}" for i in sample_sizes],
            title_text="Training Set Size" if col_index == (len(processes) // 2 + 1) else None,
            row=1, col=col_index
        )

        fig.update_yaxes(
            title_text="MSE" if col_index == 1 else None,
            row=1, col=col_index
        )

    # Adjust subplot title font size
    fig.update_layout(
        annotations=[dict(font=dict(size=12)) for a in fig['layout']['annotations']]  # Smaller subplot titles
    )

    fig = update_plot_layout(fig, subplots=True)

    plot_image_path = folder_path / f'subplots_sample_size.png'
    plot_pkl_path = folder_path / f'subplots_sample_size.pkl'
    fig.write_image(plot_image_path, scale=3)
    with open(plot_pkl_path, "wb") as f:
        pickle.dump(fig, f)

def process_one_versus_one_hundred_trees_simulation(processes:list|None = None):
    """
    Builds plots and tables that display the results of the Monte Carlo simulation studying
    MSE difference between a regression tree and a bagged tree for all processes.
    """
    if processes is None:
        processes = ['AR', 'GARCH', 'RW']

    folder_path = BLD_final / 'one_versus_hundred_trees'
    if not folder_path.is_dir():
        folder_path.mkdir(parents=True, exist_ok=True)

    all_process_results = {}
    for p_type in processes:
        path = BLD_data / 'one_versus_hundred_trees' / f'one_and_hundred_{p_type}.pkl'
        with open(path, 'rb') as f:
            loaded_file = pickle.load(f)

        all_process_results[p_type] = pd.DataFrame()
        for model in ['no_block', 'block', 'circular_block']:
            all_process_results[p_type][model] = (
                                         loaded_file['one_tree'][model] - loaded_file['hundred_trees'][model]
                                                 ).iloc[2:]

        df = all_process_results[p_type]
        table_file_path = folder_path / f'one_versus_hundred_{p_type}.txt'
        table_name = f'{p_type}-One-Versus-Hundred-Trees'
        table_label = f'{p_type}-One-Versus-Hundred-Trees'
        latex_str = df_to_latex(df, table_name, table_label)
        with open(table_file_path, "w") as f:
            f.write(latex_str)
    create_one_versus_hundred_subplots(all_process_results)


def create_one_versus_hundred_subplots(process_results):
    """
    Creates plots that displays the results of the Monte Carlo simulation studying
    MSE difference between a regression tree and a bagged tree for all processes.
    """

    folder_path = BLD_figures / 'one_versus_hundred_trees'
    if not folder_path.is_dir():
        folder_path.mkdir(parents=True, exist_ok=True)

    colour_dict = {
        'no_block': '#FF7F32',
        'block': '#3B5998',
        'circular_block': '#A3C9D6'
    }

    processes = list(process_results.keys())
    fig = make_subplots(
        rows=1, cols=len(processes),
        subplot_titles=processes,
        horizontal_spacing=0.05
    )

    for col_index, process in enumerate(processes, start=1):
        df = process_results[process]
        for model in ['block', 'no_block', 'circular_block']:
            show_legend = False
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df[f'{model}'],
                mode='lines+markers',
                name=None,
                line=dict(color=colour_dict[model], width=0.75),
                marker=dict(size=4),
                showlegend=show_legend
            ), row=1, col=col_index)

        fig.update_xaxes(
            type="log",
            tickvals=df.index,
            ticktext=[f"{i}" for i in df.index],
            title_text="Sample Size" if col_index == (len(processes) // 2 + 1) else None,
            row=1, col=col_index
        )

        fig.update_yaxes(
            title_text="Regression Tree MSE - Bagged Tree MSE" if col_index == 1 else None,
            row=1, col=col_index
        )

    for model, color in colour_dict.items():
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='lines',
            line=dict(color=color, width=1),
            name=model.replace('_', ' ').title(),
            showlegend=True
        ))

    fig.update_layout(
        annotations=[dict(font=dict(size=10)) for a in fig['layout']['annotations']],  # Smaller subplot titles
        legend_title_text=""
    )

    fig = update_plot_layout(fig, subplots=True)

    plot_image_path = folder_path / f'subplots_one_versus_hundred_plot.png'
    fig.write_image(plot_image_path, scale=3)


def process_block_size_analysis(processes:list|None = None):
    """
    Create plots and tables displaying results of the Monte Carlo simulation on different
    block lengths for the moving block bootstrap and the circular block bootstrap.
    """

    if processes is None:
        processes = ['AR', 'GARCH', 'RW']

    folder_path = BLD_final / 'block_size_simulation'
    if not folder_path.is_dir():
        folder_path.mkdir(parents=True, exist_ok=True)

    process_results = {}
    for p_type in processes:
        path = BLD_data / 'block_length_simulation' / f'block_length_simulation_{p_type}.pkl'
        with open(path, 'rb') as f:
            loaded_file = pickle.load(f)
        process_results[p_type] = loaded_file

        df = loaded_file
        table_file_path = folder_path / f'block_length{p_type}.txt'
        table_name = f'{p_type}-block_length'
        table_label = f'{p_type}-block_length'
        latex_str = df_to_latex(df, table_name, table_label)
        with open(table_file_path, "w") as f:
            f.write(latex_str)

    create_block_size_subplots(process_results)


def create_block_size_subplots(process_results):
    """
    Creates plots displaying the results of the Monte Carlo simulation on different
    block lengths for the moving block bootstrap and the circular block bootstrap.
    """

    folder_path = BLD_figures / 'block_size_simulation'
    if not folder_path.is_dir():
        folder_path.mkdir(parents=True, exist_ok=True)

    name_dict = {
        'no_block': 'Regular Bootstrap',
        'block': 'Block Bootstrap',
        'circular_block': 'Circular Block Bootstrap'
    }
    colour_dict = {
        'no_block': '#FF7F32',
        'block': '#3B5998',
        'circular_block': '#A3C9D6'
    }

    processes = list(process_results.keys())
    fig = make_subplots(
        rows=1, cols=len(processes),
        subplot_titles=processes,
        horizontal_spacing=0.05
    )

    for col_index, process in enumerate(processes, start=1):
        df = process_results[process]
        for i, model in enumerate(df.columns):
            show_legend = col_index == 1
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df[model],
                mode='lines+markers',
                name=name_dict[model] if show_legend else None,
                line=dict(color=colour_dict[model], width=0.75),
                marker=dict(size=5),
                showlegend=show_legend
            ), row=1, col=col_index)

        fig.update_xaxes(
            tickvals=df.index,
            ticktext=[f"{i}" for i in df.index],
            title_text="Block Length" if col_index == (len(processes) // 2 + 1) else None,
            row=1, col=col_index
        )

        fig.update_yaxes(
            title_text="MSE" if col_index == 1 else None,
            row=1, col=col_index
        )

    fig.update_layout(
        annotations=[dict(font=dict(size=10)) for a in fig['layout']['annotations']]
    )

    fig = update_plot_layout(fig, subplots=True)

    plot_image_path = folder_path / f'subplots_block_length_plot.png'
    fig.write_image(plot_image_path, scale=3)


def evaluate_all_simulations():
    """ Gathers results for all simulations as displayed in the thesis."""

    for ar_bool in [False]: # To obtain the AR simulation in the appendix, this has to be set to "True",
                            # but the function creating the simulation,  do_training_size_simulation(),
                            # has to be changed manually by changing the tuning set size and renaming the output pickle
        process_set_size_analysis(ar_tuned_with_small_sample=ar_bool)
    process_fixed_and_rolling_simulations(processes=['RW']) # Simulation was only performed on random walk
    process_fixed_simulations()
    process_number_of_trees_simulation()
    process_block_size_analysis()
    process_one_versus_one_hundred_trees_simulation()

if __name__ == '__main__':
    evaluate_all_simulations()
