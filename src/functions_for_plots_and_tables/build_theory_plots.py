""" Functions creating plots used in the theory part of the thesis"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor, plot_tree
from plotly.subplots import make_subplots
import matplotlib.patches as patches
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import norm
from functions_for_plots_and_tables.update_layout import update_plot_layout
from sklearn.ensemble import BaggingRegressor
from config import BLD_figures
from monte_carlo_simulations.monte_carlo_inner_functions import get_simulation_data
from bagged_tree_class.bagged_tree_model import BaggedTree

def create_rectangular_partition_and_tree_plots(folder_path):
    """
    Creates combined plot of a two-dimensional space of inputs partitioned into rectangular regions and a depiction of
    the same partition as a regression tree.
    """
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['mathtext.fontset'] = 'stix'
    np.random.seed(42)
    X = np.squeeze(np.array([1 + np.random.rand(200, 1) * 9, 1 + np.random.rand(200, 1)*3])).T
    y = np.log(X[:, 0]) + 0.1*np.exp(X[:, 1]) + np.random.randn(200) * 1


    reg_tree = DecisionTreeRegressor(max_depth=None, max_leaf_nodes=4)
    reg_tree.fit(X, y)


    plt.figure(figsize=(12, 9))
    plot_tree(
        reg_tree,
        feature_names=[r"$X_1$", r"$X_2$"],
        filled=False,
        rounded=False,
        fontsize=28,
        label='all',
        impurity=False,
        proportion=False
    )
    plt.tight_layout()
    plt.savefig(folder_path / 'tree.png', bbox_inches='tight', pad_inches=0)

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))

    feature_range = (x_min, x_max, y_min, y_max)
    boxes = get_boxes(reg_tree.tree_, feature_range)

    fig, ax = plt.subplots(figsize=(12, 9))

    for (x0, x1, y0, y1, pred) in boxes:
        rect = patches.Rectangle((x0, y0), x1 - x0, y1 - y0, edgecolor='black', facecolor="white", alpha=0.5)
        ax.add_patch(rect)
        ax.text((x0 + x1) / 2, (y0 + y1) / 2, f"{pred:.3f}", ha='center', va='center', fontsize=28, color="black")

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel(r"$X_1$", fontdict={'family': 'Times New Roman', 'size': 18})
    ax.set_ylabel(r"$X_2$", fontdict={'family': 'Times New Roman', 'size': 18})
    plt.tight_layout()
    plt.savefig(folder_path / 'rectangles.png', bbox_inches='tight')

def get_boxes(tree, feature_range):
    """
    Creates boxes for rectangular partition plot.
    """
    left = tree.children_left
    right = tree.children_right
    threshold = tree.threshold
    value = tree.value
    boxes = []

    def recurse(node, x0, x1, y0, y1):
        if left[node] == -1:  # Leaf node
            pred = value[node].flatten()[0]
            boxes.append((x0, x1, y0, y1, pred))
        else:
            feat = tree.feature[node]
            thresh = threshold[node]

            if feat == 0:
                recurse(left[node], x0, thresh, y0, y1)
                recurse(right[node], thresh, x1, y0, y1)
            elif feat == 1:
                recurse(left[node], x0, x1, y0, thresh)
                recurse(right[node], x0, x1, thresh, y1)

    recurse(0, *feature_range)
    return boxes


def create_instability_plot(folder_path):
    """
    Creates a plot of predictions of the same test set by five bagged trees trained on different training sets
    from the same distribution.
    """
    size = 200
    sets = [str(i) for i in range(1, 5)]
    sets.append('oos')
    seeds = [184, 6887, 976, 77, 80]
    data = {}
    trees = {}
    np.random.seed(49)
    for set, seed in zip(sets, seeds):
        np.random.seed(seed)
        sample_size = 2000 if set == 'oos' else size
        data[set] = {}
        data[set]['x'] = np.array([1 + np.random.rand(sample_size, 1) * 4]).reshape(-1, 1)
        data[set]['y'] = np.squeeze(
            np.exp(data[set]['x']) + np.random.randn(sample_size).reshape(-1, 1) * 2 * data[set]['x']
        )
        if set != 'oos':
            trees[set] = DecisionTreeRegressor(max_depth=None, max_leaf_nodes=8, min_samples_split=2)
            trees[set].fit(data[set]['x'], data[set]['y'])

    df = pd.DataFrame({number: trees[number].predict(data['oos']['x']) for number in sets if number != 'oos'})

    df['X'] = np.squeeze(data['oos']['x'])
    df['y'] = data['oos']['y']

    df = df.set_index('X')
    df = df.sort_index(ascending=True)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=df.index, y=df['y'], mode='markers', marker=dict(size=2, color='lightgray', opacity=0.5)))
    for set in sets:
        if set != 'oos':
            fig.add_trace(go.Scatter(x=df.index, y=df[set], mode='lines', name='Tree 1', line=dict(width=1)))
    # fig.add_trace(go.Scatter(x=df.index, y=df['tree_2'], mode='lines', name='Tree 2', line=dict(width=1)))
    fig.update_layout(template="plotly_white")
    fig = update_plot_layout(fig)
    fig.update_layout(showlegend=False,
                      xaxis_title='X',
                      yaxis_title='Y')
    fig.write_image(folder_path / 'two_plot.png', scale=3)


def create_overfitting_depth_plot(folder_path):
    """
    Creates plot of predictions by bagged trees of different depth.
    """

    if not BLD_figures.is_dir():
        BLD_figures.mkdir()

    max_depth = [2, 4, 6]
    colours = ['#A8DADC', '#FF6F61', '#2B8A3E']
    nobs = 1000

    error = np.random.normal(loc=0, scale=0.04, size=nobs)
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
        marker=dict(size=4, opacity=0.2, color='gray'),
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
            name=f'Depth = {depth}'
        ))

    fig.update_layout(
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
    fig = update_plot_layout(fig)
    fig.write_image(folder_path / 'overfitting_depth_plot.png', scale=3)


def create_combined_plot(folder_path):
    """
    Creates combined plot of the indicator example by Bühlmann & Yu
    and a plot of an individual regression tree's as well as a bagged tree's prediction.
    """

    x = np.linspace(-5, 5, 10001)
    df1 = pd.DataFrame(index=x)
    df1['unbagged'] = 0
    df1.loc[df1.index <= 0, 'unbagged'] = 1
    df1['bagged'] = norm.cdf(-df1.index)

    np.random.seed(99)
    sample_size = 2000
    X = np.array([1 + np.random.rand(sample_size, 1) * 5]).reshape(-1, 1)
    y = np.squeeze(np.exp(X) + np.random.randn(sample_size).reshape(-1, 1) * 3 * X)
    reg_tree = DecisionTreeRegressor(max_depth=None, max_leaf_nodes=12, min_samples_split=2)
    bag_tree = BaggingRegressor(estimator=reg_tree, n_estimators=200)
    for obj in (reg_tree, bag_tree):
        obj.fit(X, y)

    df2 = pd.DataFrame({'X': np.squeeze(X), 'y': y, 'bagged': bag_tree.predict(X), 'unbagged': reg_tree.predict(X)})
    df2 = df2.set_index('X').sort_index()

    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.05)


    colors = {'Unbagged': '#FF7F32', 'Bagged': '#3B5998'}

    # First subplot
    fig.add_trace(
        go.Scatter(
            x=df1.index,
            y=df1['unbagged'],
            mode='lines',
            name='Unbagged',
            line=dict(width=0.8, color=colors['Unbagged']),
           showlegend=True if 'Unbagged' not in [trace.name for trace in fig.data] else False
        ),
        row=1,
        col=1
    )
    fig.add_trace(
        go.Scatter(
            x=df1.index,
            y=df1['bagged'],
            mode='lines',
            name='Bagged',
            line=dict(width=0.8, color=colors['Bagged']),
            showlegend=True if 'Bagged' not in [trace.name for trace in fig.data] else False
        ),
        row=1,
        col=1
    )

    # Second subplot
    fig.add_trace(
        go.Scatter(x=df2.index, y=df2['y'], mode='markers', marker=dict(size=2, color='lightgray', opacity=0.999),
                   showlegend=False), row=1, col=2)
    fig.add_trace(
        go.Scatter(
            x=df2.index,
            y=df2['unbagged'],
            mode='lines',
            name='Unbagged',
            line=dict(width=0.8, color=colors['Unbagged']),
            showlegend=False
        ),
        row=1,
        col=2
    )
    fig.add_trace(
        go.Scatter(
            x=df2.index,
            y=df2['bagged'],
            mode='lines',
            name='Bagged',
            line=dict(width=0.8, color=colors['Bagged']),
            showlegend=False
        ),
        row=1,
        col=2
    )

    fig.update_layout(
        template="plotly_white",
        height=500,
        showlegend=True,
        legend=dict(orientation="h", y=-0.2, x=0.5, xanchor="center", yanchor="top"),
    )
    fig.update_xaxes(title_text="Z", row=1, col=1)
    fig.update_xaxes(title_text="X", row=1, col=2)
    fig = update_plot_layout(fig)
    fig.write_image(folder_path / 'combined_bagged_unbagged_plot.png', scale=3)

def create_random_walk_failure_plot(folder_path):
    """
    Creates plot of a fixed window, rolling window forecast and the true values of a random walk.
    """
    np.random.seed(777)
    data = get_simulation_data(
        process_type='RW',
        set_sizes= {'training': 500, 'testing': 500},
        parameters=[1],
        steps_ahead=1,
        garch_variance_noise=0,
        ar_sigma=1
    )
    BT = BaggedTree(
        n_estimators=100,
        time_series=True,
        block_bootstrap=False,
        max_depth=None,
        min_samples_split=2
    )
    BT.fit(data['training']['x'], data['training']['y'])
    df = pd.DataFrame({
        'True Series': data['testing']['y']
    })
    df['Rolling Forecast'], _ = BT.rolling_predict(
        data['training']['x'],
        data['training']['y'],
        data['testing']['x'],
        data['testing']['y']
    )
    BT.fit(data['training']['x'], data['training']['y'])

    df['Fixed Forecast'] = BT.predict(data['testing']['x'])
    fig = go.Figure()
    for col in df.columns:
        fig.add_trace(
            go.Scatter(
                line=dict(width=0.9),
                x=df.index,
                y=df[col],
                name=col
            )
        )
    fig.update_layout(
        xaxis_title='t',
        yaxis_title='Y'
    )
    fig = update_plot_layout(fig)
    fig.write_image(folder_path / 'random_walk_failure_plot.png', scale=3)

def create_all_theory_plots():
    """ Calls all functions to create theory plots. """
    folder_path = BLD_figures / 'theory_plots'
    if not folder_path.is_dir():
        folder_path.mkdir(exist_ok=True, parents=True)

    create_rectangular_partition_and_tree_plots(folder_path)
    create_overfitting_depth_plot(folder_path)
    create_instability_plot(folder_path)
    create_random_walk_failure_plot(folder_path)
    create_combined_plot(folder_path)


if __name__ == '__main__':
    create_all_theory_plots()
