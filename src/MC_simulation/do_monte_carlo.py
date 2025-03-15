import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from monte_carlo_functions import get_garch_data, do_garch_forecast, tune_hyperparameters
from sklearn.metrics import mean_squared_error
from bagged_tree_model import BaggedTree
from plotly.subplots import make_subplots
import time

steps = 5
start = time.time()
garch_parameters = [0.022, 0.068, 0.898] # (omega, alpha, beta), Andersen & Bollerslev: Answering the skeptics
set_sizes = {
    'training': 1000,
    'testing': 1000
}
np.random.seed(99)
tuning_data = get_garch_data(parameters=garch_parameters, set_sizes=set_sizes, steps_ahead=steps, resid_only=False, noise_scale=0.1)
data = get_garch_data(parameters=garch_parameters, set_sizes=set_sizes, steps_ahead=steps, resid_only=False, noise_scale=0.1)

hyperparameters = tune_hyperparameters(tuning_data)

block_model = BaggedTree(
            n_estimators= hyperparameters['block']['n_estimators'],
            time_series=True,
            block_bootstrap=True,
            max_depth=hyperparameters['block']['estimator__max_depth'],
            min_samples_split=hyperparameters['block']['estimator__min_samples_split'],
            block_size=hyperparameters['block']['max_samples']
        )

no_block_model = BaggedTree(
            n_estimators= hyperparameters['no block']['n_estimators'],
            time_series=True,
            block_bootstrap=False,
            max_depth=hyperparameters['no block']['estimator__max_depth'],
            min_samples_split=hyperparameters['no block']['estimator__min_samples_split'],
        )

tuning_time = time.time() - start





block_model.fit(data['training']['x'], data['training']['y'])
no_block_model.fit(data['training']['x'], data['training']['y'])

block_prediction = block_model.predict(data['testing']['x'])
no_block_prediction = no_block_model.predict(data['testing']['x'])

garch_prediction = do_garch_forecast(data['training']['x']['residuals'], data['testing']['x']['residuals'], steps_ahead=steps)

fitting_time = time.time()  - start - tuning_time

x = pd.DataFrame({
    'True': np.sqrt(data['testing']['y'])*np.sqrt(252),
    'rolling_garch': np.sqrt(garch_prediction['rolling'])*np.sqrt(252),
    'direct_garch': np.sqrt(garch_prediction['direct'])*np.sqrt(252),
    'Block Predicted': np.sqrt(block_prediction)*np.sqrt(252),
    'No Block Predicted': np.sqrt(no_block_prediction)*np.sqrt(252)
})

print({
    'Block Tree': mean_squared_error(x['True'], x['Block Predicted']),
    'No Block Tree': mean_squared_error(x['True'], x['No Block Predicted']),
    'Rolling GARCH': mean_squared_error(x['True'], x['rolling_garch']),
    'Direct GARCH': mean_squared_error(x['True'], x['direct_garch']),
       })

fig = make_subplots(rows=5, cols=1, shared_xaxes=True, subplot_titles=x.columns)

for i, col in enumerate(x.columns):
    fig.add_trace(go.Scatter(y=x[col], mode='lines', name=col), row=i+1, col=1)

fig.update_layout(height=800, width=800, title_text="Comparison of Predictions", showlegend=False)
fig.show()

print(f'Tuning Time: {tuning_time}, Fitting Time: {fitting_time}')
p