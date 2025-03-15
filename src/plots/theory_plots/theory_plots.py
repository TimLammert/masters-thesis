from src.template_project.config import BLD_figures
import plotly.graph_objects as go
import numpy as np
from sklearn.tree import DecisionTreeRegressor
import pandas as pd


def create_overfitting_depth_plot():

    if not BLD_figures.is_dir():
        BLD_figures.mkdir()

    max_depth = [2, 4, 6]
    colours = ['red', 'green', 'orange']
    nobs = 500

    error = np.random.normal(loc=0, scale=0.03, size=nobs)
    X = np.random.normal(loc=2, scale=0.5, size=nobs)
    mu1, mu2 = 1.8, 2.2
    s = 0.1
    Y = (1 / (1 + np.exp(-(X - mu1) / s)) - 1 / (1 + np.exp(-(X - mu2) / s))) + error
    X[-1] = 2.98
    Y[-1] = 0.6
    X_outlier = [2.98]
    Y_outlier = [0.6]

    X = np.append(X, X_outlier)
    Y = np.append(Y, Y_outlier)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=X,
        y=Y,
        mode='markers',
        marker=dict(size=4, opacity=0.2),
        showlegend=False
    ))

    for depth, colour in zip(max_depth, colours):
        model = DecisionTreeRegressor(max_depth=depth)
        model.fit(X.reshape(-1, 1), Y.reshape(-1, 1))
        prediction = model.predict(X.reshape(-1, 1))
        fitted_values = pd.DataFrame({'X': X, 'prediction': prediction})
        fitted_values = fitted_values.sort_values(by='X', ascending=False)

        fig.add_trace(go.Scatter(
            x=fitted_values['X'],
            y=fitted_values['prediction'],
            mode='lines',
            line=dict(width=1, color=colour),
            name=f'Max depth = {depth}'
        ))

    fig.update_layout(
        # title='Nonlinear Relationship with a Bump Shape',
        xaxis_title='X',
        yaxis_title='Y',
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

    fig.write_image(BLD_figures / 'overfitting_depth_plot.png', scale=3)

if __name__ == '__main__':
    create_overfitting_depth_plot()