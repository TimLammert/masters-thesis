""" Function used to unify the layout of all plots."""

import plotly.graph_objects as go

def update_plot_layout(fig:go.Figure, subplots=False, width=None, height=None):
    """
    Updates layout of a plot.
    """
    if height is None:
        height = 400 if subplots else 400
    if width is None:
        width = 1100 if subplots else 900

    fig.update_layout(
                font=dict(
                    family="Times New Roman",
                    size=14,
                    color="black"
                ),
                legend=dict(
                    orientation="h",
                    yanchor="top",
                    y=-0.3 if subplots else -0.2,
                    xanchor="center",
                    x=0.5,
                    bgcolor="rgba(255,255,255,0.5)"
                ),
                width=width,
                height=height,
                template='simple_white',
                margin=dict(l=1, r=50, t=30 if subplots else 20, b=30)
            )
    return fig
