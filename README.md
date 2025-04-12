# Bagged Trees: Theoretical Foundations and Applications in Volatility Forecasting
## Master's Thesis - M.Sc. Economics - University of Bonn
### Tim Lammert

To get started, create and activate the environment with

```console
$ conda/mamba env create
$ conda activate fpl
```

This project was created with the
[econ-project-templates](https://github.com/OpenSourceEconomics/econ-project-templates).

## Description

This repo contains all codes used throughout my master's thesis on bagged regression trees.
Thse codes build plots for the theory section, perform Monte Carlo simulations, and 
compute a real data forecast of S&P 500 volatility.
Some tables created by this code were not used in the final version of the thesis.

The bagged tree model was built based on scikit-learn's BaggingRegressor and DecisionTreeRegressor

Functions computing realized quantities from intraday stock data are based on the 
RealizedQuantities repo by Sebastian Bayer (https://github.com/BayerSe/RealizedQuantities).

The Diebold-Mariano test function is taken from John Tsang (https://github.com/johntwk/Diebold-Mariano-Test)


