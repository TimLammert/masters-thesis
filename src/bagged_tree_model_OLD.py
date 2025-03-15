from sklearn.tree import DecisionTreeRegressor
from BlockBaggedRegressor import BlockBaggingRegressor
from sklearn.ensemble import BaggingRegressor, _bagging
from sklearn.utils import check_random_state
from sklearn.model_selection import TimeSeriesSplit, KFold, GridSearchCV, ParameterGrid
from sklearn.base import clone
from sklearn.inspection import permutation_importance
import numpy as np
import pandas as pd
import importlib

_bagging = importlib.import_module('sklearn.ensemble._bagging')

def _generate_circular_block_indices(random_state, n_population, n_samples, block_size):
    """
    Generate indices using a circular moving block bootstrap.
    """
    random_state = check_random_state(random_state)

    n_blocks = int(np.ceil(n_samples / block_size))
    indices = []

    for _ in range(n_blocks):
        start = random_state.randint(0, n_population - 1)
        block = np.arange(start, start + block_size)
        block = np.mod(block, n_population)
        indices.append(block)
    indices = np.concatenate(indices)
    return indices[:n_samples]

def _generate_block_indices(random_state, n_population, n_samples, block_size):
    """
    Generate indices using a moving block bootstrap.
    """
    random_state = check_random_state(random_state)

    n_blocks = int(np.ceil(n_samples / block_size))
    indices = []
    max_start = n_population - block_size
    for _ in range(n_blocks):
        start = random_state.randint(0, max_start + 1)
        block = np.arange(start, start + block_size)
        indices.append(block)
    indices = np.concatenate(indices)
    return indices[:n_samples]

def block_generate_bagging_indices(random_state, bootstrap_features, bootstrap_samples,
                                   n_features, n_samples, max_features, max_samples):
    """
    Helper to alter the bootstrap sampling process to a moving block bootstrap.
    Feature selection remains unchanged.
    max_samples is not used as in the original class, but determines the block size.
    This means that subsampling and block size cannot be implemented simultaneously.
    """
    random_state = check_random_state(random_state)

    feature_indices = _bagging._generate_indices(random_state, bootstrap_features, n_features, max_features)

    sample_indices = _generate_block_indices(
        random_state,
        n_population=n_samples,
        n_samples=n_samples,
        block_size=max_samples
    )
    return feature_indices, sample_indices

def circular_block_generate_bagging_indices(random_state, bootstrap_features, bootstrap_samples,
                                   n_features, n_samples, max_features, max_samples):
    """
    Helper to alter the bootstrap sampling process to a moving block bootstrap.
    Feature selection remains unchanged.
    max_samples is not used as in the original class, but determines the block size.
    This means that subsampling and block size cannot be implemented simultaneously.
    """
    random_state = check_random_state(random_state)

    feature_indices = _bagging._generate_indices(random_state, bootstrap_features, n_features, max_features)

    sample_indices = _generate_circular_block_indices(
        random_state,
        n_population=n_samples,
        n_samples=n_samples,
        block_size=max_samples
    )
    return feature_indices, sample_indices


class BaggedTree:

    def __init__(
            self,
            n_estimators:int=100,
            time_series:bool=False,
            block_bootstrap:bool=False,
            max_depth:int|None=None,
            min_samples_split:int=2,
            min_samples_leaf:int=1,
            max_features:int=None,
            random_state:int|None=None,
            block_size:int=10,
            circular:bool=False,
    ):
        self.time_series = time_series
        self.block_bootstrap = block_bootstrap
        self.circular = circular

        self.base_tree = DecisionTreeRegressor(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=random_state
        )
        if self.block_bootstrap:
            self.model = self.implement_block_bootstrap(
                n_estimators=n_estimators,
                random_state=random_state,
                block_size=block_size
            )
        else:
            self.model = BaggingRegressor(
                estimator=self.base_tree,
                n_estimators=n_estimators,
                bootstrap=True,
                random_state=random_state
            )

        self.best_parameters = None

    def new_fit_doesnt_work(self, X, y, param_grid=None, cv_splits=5, n_jobs=1):
        """Fit the model with optional hyperparameter tuning"""

        if self.time_series:
            cv_method = TimeSeriesSplit(n_splits=cv_splits)
        else:
            cv_method = KFold(n_splits=cv_splits)

        original_function = _bagging._generate_bagging_indices
        if self.block_bootstrap:
            _bagging._generate_bagging_indices = circular_block_generate_bagging_indices if self.circular else block_generate_bagging_indices

        try:
            if param_grid:
                if self.block_bootstrap:
                    # Perform manual cross-validation
                    best_score = float('-inf')
                    best_params = None
                    best_model = None

                    for params in ParameterGrid(param_grid):
                        self.model.set_params(**params)
                        scores = []

                        for train_idx, val_idx in cv_method.split(X):
                            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

                            self.model.fit(X_train, y_train)
                            score = self.model.score(X_val, y_val)
                            scores.append(score)

                        mean_score = np.mean(scores)

                        if mean_score > best_score:
                            best_score = mean_score
                            best_params = params
                            best_model = clone(self.model)
                            best_model.set_params(**params)

                    self.model = best_model
                    self.best_parameters = best_params
                else:
                    # Use GridSearchCV for non-block-bootstrap cases
                    grid_search = GridSearchCV(
                        self.model, param_grid, cv=cv_method, scoring="neg_mean_squared_error", n_jobs=n_jobs
                    )
                    grid_search.fit(X, y)
                    self.model = grid_search.best_estimator_
                    self.best_parameters = grid_search.best_params_
            else:
                self.model.fit(X, y)
        finally:
            _bagging._generate_bagging_indices = original_function



    def feature_importance(self, x_data, y_data):
        """Compute permutation importance"""
        perm_importance = permutation_importance(self.model, x_data, y_data, scoring="neg_mean_squared_error")
        return perm_importance.importances_mean


    def implement_block_bootstrap(self, n_estimators, random_state, block_size):
        """
        Implementation of block bootstrapping for Scikit-learn's BaggingRegressor.
        Monkey-patches the bootstrapping function in _bagging.py with a custom function for block bootstrapping
        and reverses this after initialization of the model.
        Note that the block size is passed to the class as max_samples. This is needed to implement the block bootstrap
        without making major changes in class methods.
        """
        original_function = _bagging._generate_bagging_indices
        _bagging._generate_bagging_indices = circular_block_generate_bagging_indices if self.circular else block_generate_bagging_indices
        try:
            block_bagged_tree = BaggingRegressor(
                estimator=self.base_tree,
                n_estimators= n_estimators,
                bootstrap=True,
                random_state=random_state,
                max_samples=block_size  # Have to use argument max_samples for block size
            )
        finally:
            _bagging._generate_bagging_indices = original_function
        return block_bagged_tree


    def predict(self, x_data):
        """Make predictions"""
        return self.model.predict(x_data)


    def rolling_predict(self, x_train, y_train, x_test, y_test, window_size=None, refit=True):
        """
        Performs a rolling prediction, updating the model at each step.
        If no window-size is specified, then it performs an expanding prediction.

        Parameters:
        - X_train: Training feature data (historical observations)
        - y_train: Training target values (historical target variable)
        - X_test: Test feature data (future periods to predict)
        - window_size: Size of the rolling training window (default: use expanding window)
        - refit: Whether to refit the model at each step (default: True)

        Returns:
        - np.array: Rolling predictions
        """

        if window_size is None:
            window_size = len(x_train)

        rolling_predictions = []
        X_train_rolling, y_train_rolling = x_train.copy(), y_train.copy()

        # tree_preds_df = pd.DataFrame()


        for i in range(len(x_test)):
            if refit:
                x_train_window = X_train_rolling.iloc[-window_size:]
                y_train_window = y_train_rolling[-window_size:]

                self.fit(x_train_window, y_train_window)

            y_pred = self.model.predict(x_test.iloc[[i]])[0]
            rolling_predictions.append(y_pred)

            X_train_rolling = pd.concat([X_train_rolling, x_test.iloc[[i]]], axis=0)
            y_train_rolling = np.append(y_train_rolling, y_test[i])

        return np.array(rolling_predictions)


    def fit(self, X, y, param_grid=None, cv_splits=5, n_jobs=1):
        """Fit the model with optional hyperparameter tuning"""

        if self.time_series:
            cv_method = TimeSeriesSplit(n_splits=cv_splits)
        else:
            cv_method = KFold(n_splits=cv_splits)

        original_function = _bagging._generate_bagging_indices

        if self.block_bootstrap:
            _bagging._generate_bagging_indices = circular_block_generate_bagging_indices if self.circular else block_generate_bagging_indices

        try:
            print(_bagging._generate_bagging_indices, flush=True)
            if param_grid:
                grid_search = GridSearchCV(self.model, param_grid, cv=cv_method, scoring="neg_mean_squared_error",
                                           n_jobs=n_jobs)
                grid_search.fit(X, y)
                print(_bagging._generate_bagging_indices, flush=True)
                self.model = grid_search.best_estimator_
                self.best_parameters = grid_search.best_params_
            else:
                self.model.fit(X, y)
                print(_bagging._generate_bagging_indices, flush=True)
        finally:
            _bagging._generate_bagging_indices = original_function



