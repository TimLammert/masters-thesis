"""
Builds tables from the results of the volatility forecast, comparing the different models and
displaying feature importances, hyperparameters, etc. Some tables were not used in the thesis.
Other were created separately in this script and later combined manually.
"""

import pandas as pd
import pickle
import numpy as np
from diebold_mariano_test_function.dm_test import dm_test
from functions_for_plots_and_tables.df_to_table import df_to_latex, df_to_latex_regression_table_with_two_columns
from config import BLD_data, BLD_final
from sklearn.metrics import mean_squared_error


def create_all_tables():
    """
    Creates all tables for all forecasts discussed in the thesis.
    """
    for sqrt in [False]:
        for step in [1] if sqrt else [1, 5, 22]:
            create_tables_for_one_setup(steps_ahead=step, square_root=sqrt)


def create_tables_for_one_setup(steps_ahead:int, square_root:bool):
    """
    Creates tables for one setup, that is one forecqst horizon and one value for square_root, but for both fixed
    and rolling forecasts. Loads forecast results, and calls several functions that create LaTeX tables for this setup.
    """
    folder_path = BLD_final / 'application' / f'{steps_ahead}_step_ahead'
    if not folder_path.is_dir():
        folder_path.mkdir(exist_ok=True, parents=True)


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
    input_folder_path = BLD_data / 'application' / 'results'
    forecast_path = input_folder_path / forecast_file_name
    data_path = input_folder_path / data_file_name

    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    with open(forecast_path, 'rb') as f:
        results = pickle.load(f)

    models = ['HAR-', 'BT-']
    setups = ['RV', 'RV-M', 'CJ', 'CJ-M']
    setup_model_strings =  [ f'{model}{setup}' for model in models for setup in setups ]
    variable_rename_dict = {
        'RV_t-1': r'$(RV_t^{(1)})^{1/2}$' if square_root else r'$RV_t^{(1)}$',
        'ave_RV_t-1_t-5': r'$(RV_t^{(5)})^{1/2}$' if square_root else r'$RV_t^{(5)}$',
        'ave_RV_t-1_t-22': r'$(RV_t^{(22)})^{1/2}$' if square_root else r'$RV_t^{(22)}$',

        'C_t-1': r'$(C_t^{(1)})^{1/2}$' if square_root else r'$C_t^{(1)}$',
        'ave_C_t-1_t-5': r'$(C_t^{(5)})^{1/2}$' if square_root else r'$C_t^{(5)}$',
        'ave_C_t-1_t-22': r'$(C_t^{(22)})^{1/2}$' if square_root else r'$C_t^{(22)}$',

        'J_t-1': r'$(J_t^{(1)})^{1/2}$' if square_root else r'$J_t^{(1)}$',
        'ave_J_t-1_t-5': r'$(J_t^{(5)})^{1/2}$' if square_root else r'$J_t^{(5)}$',
        'ave_J_t-1_t-22': r'$(J_t^{(22)})^{1/2}$' if square_root else r'$J_t^{(22)}$',

        'Inflation': 'CPI Inflation',
        'Unemployment': 'Unemployment Rate',
        'Unemployment Release': 'Employment Situation Release',
        'GDP Growth': 'GDP Growth',
        'Fed Effective Rate': 'Fed Effective Rate',
        'Inflation Release': 'CPI Release',
        'GDP Release': 'GDP Release',
        'Fed Meetings': 'FOMC Meetings',
        'const': 'Constant'
        }
    if not square_root:
        create_importance_and_hyperparameter_tables(
            results=results,
            setups=setups,
            steps_ahead=steps_ahead,
            square_root=square_root,
            variable_rename_dict=variable_rename_dict
        )
    overview_tables = create_evaluation_dfs(
        data=data,
        results=results,
        setup_model_strings=setup_model_strings,
        forecast_types=['rolling'] if square_root or not steps_ahead == 1 else ['fixed', 'rolling']
    )
    create_forecast_comparison_table(
        tables=overview_tables,
        steps_ahead=steps_ahead,
        square_root=square_root,
        forecast_types=['rolling'] if square_root or not steps_ahead == 1 else ['fixed', 'rolling']
    )
    if steps_ahead == 1:
        create_ols_summary_tables(
            results=results,
            square_root=square_root,
            steps_ahead=steps_ahead,
            variable_rename_dict=variable_rename_dict,
            round=4
        )


def create_evaluation_dfs(data, results, setup_model_strings, significance_levels=None, forecast_types=None):
    """
    Creates pd.DataFrames for each the fixed and rolling window forecast containing all results
    for one forecast horizon and value of square_root.
    pd.DataFrames contain relative MSE, R-squared and Diebold-Mariano test results for different levels.
    """
    if significance_levels is None:
        significance_levels = [0.1, 0.05, 0.01]
    if forecast_types is None:
        forecast_types = ['fixed', 'rolling']
    true_series = data['RV']['testing']['y']

    tables = {}
    out = {}
    for fc_type in forecast_types:
        tables[fc_type] = {}
        tables[fc_type]['mean squared error'] = pd.DataFrame()
        tables[fc_type]['diebold_mariano'] = {
            f'{lvl*100}%': pd.DataFrame() for lvl in significance_levels
        }

        mses = pd.DataFrame({
            model: mean_squared_error(true_series, results[model][fc_type]) for model in setup_model_strings
        }, index = [0]).T

        r_squared = pd.DataFrame({
            model: 1 - mean_squared_error(
                true_series, results[model][fc_type]
            ) / np.var(true_series) for model in setup_model_strings
        }, index = [0]).T

        for col_model in setup_model_strings:
            tables[fc_type]['mean squared error'][col_model] = mses / mses.loc[col_model]

        for col_model in setup_model_strings:
            for lvl in significance_levels:
                dm_series = pd.Series(index=tables[fc_type]['mean squared error'].index)
                for row_model in setup_model_strings:
                    if row_model != col_model:
                        dm_result = dm_test(true_series, results[row_model][fc_type], results[col_model][fc_type])
                        if dm_result.DM < 0 and (dm_result.p_value/2) < lvl: # Adjustment for one-sided test!
                            dm_series.loc[row_model] = 1
                        else:
                            dm_series.loc[row_model] = 0
                    else:
                        dm_series.loc[row_model] = pd.NA

                tables[fc_type]['diebold_mariano'][f'{lvl*100}%'][col_model] = dm_series

        r_squared_string = r_squared.apply(lambda col: col.map(lambda x: f"{x:.2f}"))
        mse_string_df = tables[fc_type]['mean squared error'].apply(lambda col: col.map(lambda x: f"{x:.2f}"))
        significance_star_df = sum(tables[fc_type]["diebold_mariano"].values()).fillna(0).apply(lambda col: col.map(lambda x: "*" * int(x)))
        out[fc_type] = pd.concat([mse_string_df + significance_star_df, r_squared_string], axis=1)

    return out


def create_forecast_comparison_table(tables, steps_ahead, square_root, forecast_types=None):
    """
    Combines DataFrames containing relative MSE, R-squared and Diebold-Mariano test results for different significance
    levels for one forecast horizon, square_root value and either a fixed or rolling forecast
    into one DataFrame of strings, which is then stored as a LaTeX table in a txt file.
    """
    folder_path = BLD_final / 'application' / f'{steps_ahead}_step_ahead'
    if not folder_path.is_dir():
        folder_path.mkdir(parents=True, exist_ok=True)

    if forecast_types is None:
        forecast_types = ['fixed', 'rolling'] 
    for fc_type in forecast_types:
        df = tables[fc_type]
        table_str = df_to_latex(df, f'{fc_type} forecast comparison', f'{fc_type}-comparison')

        table_file_name = (
            f'square_root_{fc_type}_{steps_ahead}_step_ahead_comparison_table.txt'
            if square_root else
            f'{fc_type}_{steps_ahead}_step_ahead_comparison_table.txt'
        )
        file_path = folder_path / table_file_name
        with open(file_path, "w") as f:
            f.write(table_str)


def create_importance_and_hyperparameter_tables(results, setups, square_root, steps_ahead, variable_rename_dict):
    """
    Creates tables displaying in-sample feature importance averaged over the rolling forecasts and hyperparameters
    for one forecast horizon and value of square_root.
    """
    folder_path = BLD_final / 'application' / f'{steps_ahead}_step_ahead' /'permutation_importance_and_hyperparameters'
    if not folder_path.is_dir():
        folder_path.mkdir(parents=True, exist_ok=True)

    label_id = 'vol' if square_root else 'var'

    if not square_root:
        for no_macro, macro in [['RV', 'RV-M'], ['CJ', 'CJ-M']]:
            no_macro_df = pd.DataFrame(results[f'{no_macro} additional_info']['feature_importances'])
            macro_df = pd.DataFrame(results[f'{macro} additional_info']['feature_importances'])
            no_macro_mean = no_macro_df.mean(axis=1)
            macro_mean = macro_df.mean(axis=1)

            importance_df = pd.concat([macro_mean, no_macro_mean], axis=1)
            importance_df = importance_df.rename(
                columns = {0: f'{macro}', 1: f'{no_macro}'}
            ).sort_values(by=macro, ascending=False)
            importance_df = importance_df[[no_macro, macro]].rename(index=variable_rename_dict)
            importance_df /= importance_df.max()

            table_file_name = f'{no_macro}_{steps_ahead}_step_feature_importance.txt'
            table_name = f'{macro} {steps_ahead} step feature importance'
            table_label = f'{no_macro} {steps_ahead} featimp'
            feature_table = df_to_latex(importance_df, table_name, table_label, round=4)
            file_path = folder_path / table_file_name
            with open(file_path, "w") as f:
                f.write(feature_table)

    hyperparameter_dfs = {}
    for setup in setups:
        hyperparameter_dfs[setup] = pd.DataFrame(results[f'{setup} additional_info']['best_parameters'], index=[0]).T

        # permutation_series = results[f'{setup} additional_info']['permutation importance']
        # permutation_df = pd.DataFrame(permutation_series.rename(variable_rename_dict))
        # permutation_df /= permutation_df.max()
        #
        # importance_table = df_to_latex(
        #    df=permutation_df,
        #    table_name=f'{setup} Permutation Importance ({rv_name})',
        #    table_label=f'{label_id}_{setup}_permimp',
        #    round=4
        # )
        # hyperparameter_dfs[setup] = pd.DataFrame(results[f'{setup} additional_info']['best_parameters'],
        #                                         index=[0]).T
        #
        # for file_name, table in [
        #    [f'{setup}_{label_id}_{steps_ahead}_step_permutation_importance.txt', importance_table]
        # ]:
        #    file_path = folder_path / file_name
        #    with open(file_path, "w") as f:
        #        f.write(table)

    setups = ['RV', 'RV-M', 'CJ', 'CJ-M']
    hyperparameter_df = pd.concat([hyperparameter_dfs[setup] for setup in setups], axis=1)
    hyperparameter_df.columns = setups
    index_rename_dict = {
        'estimator__max_depth': 'Maximum Depth',
        'estimator__min_samples_split': 'Minimum Observations for a Split',
        'max_samples': 'Block Length',
        'n_estimators': 'Number of Trees'
    }
    hyperparameter_df = hyperparameter_df.rename(index=index_rename_dict)
    hyperparameter_table = df_to_latex(
        df=hyperparameter_df,
        table_name=f'{setup} Hyperparameters ({label_id})',
        table_label=f'{label_id}_{setup}_params',
        round=0
    )

    file_path = folder_path / f'{steps_ahead}_{label_id}_hyperparameter_table.txt'
    with open(file_path, "w") as f:
        f.write(hyperparameter_table)


def create_ols_summary_tables(results, square_root, steps_ahead, variable_rename_dict, round=4):
    """
    Creates a table displaying the OLS coefficients estimated from the training set.
    """

    folder_path = BLD_final / 'application' / f'{steps_ahead}_step_ahead' /'ols_tables'
    if not folder_path.is_dir():
        folder_path.mkdir(parents=True, exist_ok=True)
    rv_name = '$RV^{1/2}$' if square_root else '$RV$'
    label_id = 'sqrt_var' if square_root else 'var'
    variable_order = {
        'RV': ['const', 'RV_t-1', 'ave_RV_t-1_t-5', 'ave_RV_t-1_t-22', 'Fed Effective Rate', 'GDP Growth', 'Inflation',
               'Unemployment', 'Fed Meetings', 'GDP Release', 'Inflation Release', 'Unemployment Release'],
        'CJ': ['const', 'C_t-1', 'ave_C_t-1_t-5', 'ave_C_t-1_t-22', 'J_t-1', 'ave_J_t-1_t-5', 'ave_J_t-1_t-22', 'Fed Effective Rate',
               'GDP Growth', 'Inflation', 'Unemployment', 'Fed Meetings', 'GDP Release', 'Inflation Release',
               'Unemployment Release'],

    }
    for no_macro, macro in [['RV', 'RV-M'], ['CJ', 'CJ-M']]:
        no_macro_ols = results[f'{no_macro} additional_info']['ols_summary']
        macro_ols = results[f'{macro} additional_info']['ols_summary']


        ols_df = pd.concat([
            ols_summary_to_df(macro_ols).rename(
                columns={'Coefficient': f'{macro} Coefficient', 'Std Error': f'{macro} Std Error'}
            ),
            ols_summary_to_df(no_macro_ols).rename(
                columns={'Coefficient': f'{no_macro} Coefficient', 'Std Error': f'{no_macro} Std Error'}
            )
        ]).groupby(level=0).first()

        ols_df = ols_df.loc[variable_order[no_macro]]

        ols_df = ols_df.rename(index=variable_rename_dict)
        ols_df = ols_df.round(round)
        ols_table = df_to_latex_regression_table_with_two_columns(
            ols_df=ols_df,
            table_name=f'HAR {no_macro} Coefficients ({rv_name})',
            macro=macro,
            no_macro=no_macro
        )

        file_path = folder_path / f'{no_macro}_{label_id}_{steps_ahead}_step_ols_table.txt'
        with open(file_path, "w") as f:
            f.write(ols_table)


def ols_summary_to_df(ols_summary):
    """
    Converts an OLS summary to a DataFrame with coefficients and standard errors.
    """

    coefficients = ols_summary.params
    std_errors = ols_summary.bse

    ols_df = pd.DataFrame({
        'Coefficient': coefficients,
        'Std Error': std_errors
    })

    ols_df.index.name = 'Variable'

    return ols_df


if __name__ == '__main__':
    create_all_tables()
