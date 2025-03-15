import pandas as pd
import pickle
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dm_test import dm_test
from MC_simulation.df_to_table import df_to_latex
from src.template_project.config import BLD_data, BLD_final, BLD_figures
from sklearn.metrics import mean_squared_error

def create_real_data_tables_and_plots(steps_ahead, square_root=False):

    folder_path = BLD_data / 'Test Application'  ############# CHANGE TO APPLICATION #############################################################
    forecast_file_name = f'real_data_forecasts_{steps_ahead}_step_ahead_square_root.pkl' if square_root else f'real_data_forecasts_{steps_ahead}_step_ahead.pkl'
    data_file_name = f'real_data_data_{steps_ahead}_step_ahead_square_root.pkl' if square_root else f'real_data_data_{steps_ahead}_step_ahead.pkl'

    forecast_path = folder_path / forecast_file_name
    data_path = folder_path / data_file_name

    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    with open(forecast_path, 'rb') as f:
        results = pickle.load(f)

    models = ['HAR-', 'BT-']
    setups = ['RV', 'RV-M', 'CJ', 'CJ-M']
    setup_model_strings =  [ f'{setup} {model}' for setup in setups for model in models ]

    create_forecast_plots(data=data, results=results, models=models, setups=setups, square_root=square_root, steps_ahead=steps_ahead)
    overview_tables = create_evaluation_dfs(data, results, setup_model_strings)
    create_forecast_comparison_table(overview_tables, steps_ahead, square_root)




def create_evaluation_dfs(data, results, setup_model_strings, significance_levels=None, forecast_types=None):

    if significance_levels is None:
        significance_levels = [0.1, 0.05, 0.01]
    if forecast_types is None:
        forecast_types = ['fixed'] # ADD ROLLING!!!! ##########################################################################################
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


        for col_model in setup_model_strings:
            tables[fc_type]['mean squared error'][col_model] = mses / mses.loc[col_model]

            for lvl in significance_levels:
                dm_series = pd.Series(index=tables[fc_type]['mean squared error'].index)
                for row_model in setup_model_strings:
                    if row_model != col_model:
                        dm_result = dm_test(true_series, results[col_model][fc_type], results[row_model][fc_type])
                        if dm_result.DM < 0 and dm_result.p_value < lvl:
                            dm_series.loc[row_model] = 1
                        else:
                            dm_series.loc[row_model] = 0
                    else:
                        dm_series.loc[row_model] = pd.NA

                tables[fc_type]['diebold_mariano'][f'{lvl*100}%'][col_model] = dm_series

        tables[fc_type]['mean squared error'] = tables[fc_type]['mean squared error'].T
        mse_string_df = tables[fc_type]['mean squared error'].apply(lambda col: col.map(lambda x: f"{x:.2f}"))
        significance_star_df = sum(tables[fc_type]["diebold_mariano"].values()).fillna(0).apply(lambda col: col.map(lambda x: "*" * int(x)))
        out[fc_type] = mse_string_df + significance_star_df

    return out

def create_forecast_comparison_table(tables, steps_ahead, square_root, forecast_types=None):
    if not BLD_final.is_dir():
        BLD_final.mkdir(exist_ok=True, parents=True)
    if forecast_types is None:
        forecast_types = ['fixed'] ###### ADD ROLLING!!!! ##########################################################################################
    for fc_type in forecast_types:
        df = tables[fc_type]
        table_str = df_to_latex(df, f'{fc_type} forecast comparison', f'{fc_type}-comparison')

        table_file_name = f'square_root_{fc_type}_{steps_ahead}_step_ahead_comparison_table.txt' if square_root else f'{fc_type}_{steps_ahead}_step_ahead_comparison_table.txt'
        file_path = BLD_final / table_file_name
        with open(file_path, "w") as f:
            f.write(table_str)

def create_forecast_plots(data, results, models:list[str], setups:list[str], steps_ahead:int, square_root, forecast_types=None):

    if forecast_types is None:
        forecast_types = ['fixed'] ###### ADD ROLLING!!!! ##########################################################################################

    folder_path = BLD_figures / 'application'
    if not folder_path.is_dir():
        folder_path.mkdir(parents=True, exist_ok=True)

    plot_rows = 2
    plot_cols = 2

    colour_dict = {
        'True Series': 'gray',
        'Mean': 'tomato',
        'HAR': 'orange',
        'BT': 'navy'
    }

    fig = make_subplots(
        rows=plot_rows,
        cols=plot_cols,
        subplot_titles=setups,
        vertical_spacing=0.1,
        horizontal_spacing=0.05
    )
    for fc_type in forecast_types:
        for i in range(plot_rows):
            for j in range(plot_cols):
                plot_df = pd.DataFrame()
                plot_df['True Series'] = data['RV']['testing']['y']
                plot_df['Mean'] = plot_df['True Series'].rolling(window=252, min_periods=1).mean()
                plot_df['HAR'] = results[f'HAR-{setups[i+j]}'][fc_type]
                plot_df['BT'] = results[f'BT-{setups[i+j]}'][fc_type]
                plot_df = plot_df * np.sqrt(252) if square_root else plot_df * 252
                for col in plot_df.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=plot_df.index,
                            y=plot_df[col],
                            mode='lines',
                            line=dict(
                                width= 0.8 if col == 'Mean' or col == 'True Series' else 0.5,
                                color=colour_dict[col]
                            ),
                            name=f'{col}',
                            showlegend=False,
                            opacity=0.5 if col == 'True Series' else 1
                        ),
                        row=i+1,
                        col=j+1
                    )
                #fig.update_xaxes(title_text='t', row=i+1, col=j+1)
                #fig.update_yaxes(title_text='Forecast', row=i+1, col=j+1)

        for name, color in colour_dict.items():
            fig.add_trace(
                go.Scatter(
                    x=[None], y=[None],  # Invisible points
                    mode='lines',
                    line=dict(width=2, color=color),
                    name=name,
                    showlegend=True
                )
            )

        fig.update_layout(
            font=dict(
                family="Times New Roman",
                size=12,
                color="black"
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.01,
                xanchor="center",
                x=0.5,
                bgcolor="rgba(255,255,255,0.5)"
            ),
            width=900,
            height=700,
            template='simple_white',
            margin=dict(l=20, r=20, t=40, b=60)
        )

        plot_image_file_name = f'square_root_{fc_type}_{steps_ahead}_step_ahead_forecast.png' if square_root else f'{fc_type}_{steps_ahead}_step_ahead_forecast.png'
        plot_pkl_file_name = f'square_root_{fc_type}_{steps_ahead}_step_ahead_forecast.pkl' if square_root else f'{fc_type}_{steps_ahead}_step_ahead_forecast.pkl'

        plot_image_path = folder_path / plot_image_file_name
        plot_pkl_path = folder_path / plot_pkl_file_name
        fig.write_image(plot_image_path, scale=3)
        with open(plot_pkl_path, "wb") as f:
            pickle.dump(fig, f)



if __name__ == '__main__':
    create_real_data_tables_and_plots(steps_ahead=1)
    # out = create_evaluation_dfs(steps_ahead=1)
    # create_forecast_comparison_table(out, steps_ahead=1)
    #create_forecast_plots(steps_ahead=1)