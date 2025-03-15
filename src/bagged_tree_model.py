from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor
from BlockBaggedRegressor import BlockBaggingRegressor
from CircularBaggedRegressor import CircularBlockBaggingRegressor
from sklearn.model_selection import TimeSeriesSplit, KFold, GridSearchCV
from sklearn.inspection import permutation_importance
import numpy as np
import pandas as pd


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
        if circular and not block_bootstrap:
            msg = 'circular is enabled, set block_bootstrap=True to enable circular block bootstrap sampling.'
            raise ValueError(msg)

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
            if self.circular:
                self.model = CircularBlockBaggingRegressor(
                    estimator=self.base_tree,
                    n_estimators=n_estimators,
                    bootstrap=True,
                    random_state=random_state,
                    max_samples=block_size
                )
            else:
                self.model = BlockBaggingRegressor(
                    estimator=self.base_tree,
                    n_estimators=n_estimators,
                    bootstrap=True,
                    random_state=random_state,
                    max_samples=block_size
                )
        else:
            self.model = BaggingRegressor(
                estimator=self.base_tree,
                n_estimators=n_estimators,
                bootstrap=True,
                random_state=random_state
            )

        self.best_parameters = None

    def fit(self, X, y, param_grid=None, cv_splits=5, n_jobs=1):
        """Fit the model with optional hyperparameter tuning"""

        if self.time_series:
            cv_method = TimeSeriesSplit(n_splits=cv_splits)
        else:
            cv_method = KFold(n_splits=cv_splits)

        if param_grid:
            grid_search = GridSearchCV(self.model, param_grid, cv=cv_method, scoring="neg_mean_squared_error",
                                       n_jobs=n_jobs)
            grid_search.fit(X, y)

            self.model = grid_search.best_estimator_
            self.best_parameters = grid_search.best_params_
        else:
            self.model.fit(X, y)

    def feature_importance(self, x_data, y_data):
        """Compute permutation importance"""
        perm_importance = permutation_importance(self.model, x_data, y_data, scoring="neg_mean_squared_error")
        return perm_importance.importances_mean

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








