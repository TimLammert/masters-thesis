"""
Class that combines scikit-learn's DecisionTreeRegressor with the scikit-learn's BaggingRegressor or custom subclasses
for block and circular block bagging. Used in Monte Carlo simulations as well as the real data application.
"""


from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor
from bagging_subclasses.block_bagged_regressor import BlockBaggingRegressor
from bagging_subclasses.circular_bagged_regressor import CircularBlockBaggingRegressor
from sklearn.model_selection import TimeSeriesSplit, KFold, GridSearchCV
from sklearn.inspection import permutation_importance
import copy
import numpy as np
import pandas as pd
import time


class BaggedTree:

    def __init__(
            self,
            n_estimators:int=100,
            time_series:bool=False,
            block_bootstrap:bool=False,
            max_depth:int|None=None,
            min_samples_split:int=2,
            min_samples_leaf:int=1,
            max_features:int|None=None,
            random_state:int|None=None,
            block_size:int=10,
            circular:bool=False,
    ):
        """
            Bagged tree model comprised of a regression tree and a bagging component, both based on classes from scikit-learn.
            The class is able to tune hyperparameters, perform fixed and rolling window predictions and compute variable
            importance measures. It can be used with features adapted to time series data, such as moving block and circular
            block bootstrap and time-series cross-validation.

            Parameters
            -----------
            n_estimators : int
                number of trees in the bagged tree
            time_series : bool
                apply time series cross validation if true
            block_bootstrap : bool
                apply moving block bootstrap if true
            max_depth : int|None
                maximum tree depth for the individual trees
            min_samples_split : int|None
                minimum number of samples in a node to perform a split on it
            min_samples_leaf : int|None
                minimum number of samples in a leaf
            max_features: int|str|None
                maximum number of features considered for a split (set to value smaller than one to obtain a random forest)
            random_state: int|None
                controls the randomness in bootstrapping the samples
            block_size: int
                length of blocks for moving block and circular block bootstrap
            circular: bool
                activate circular block bootstrap


            Attributes
            ----------
            time_series : bool
                boolean for time series cross-validation
            block_bootstrap : bool
                boolean for moving block bootstrap
            circular : bool
                circular block bootstrap
            base_tree : DecisionTreeRegressor
                underlying regression tree model
            model : BaggingRegressor | BlockBaggingRegressor | CircularBlockBaggingRegressor
                bagged tree model based on the base_model
            best_parameters : dict
                parameters determined by cross-validation


            """
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
        """
        Fit the model with optional hyperparameter tuning via cross-validation based on a dict determining the grids
        to search over for each hyperparameter.

        """

        if self.time_series:
            cv_method = TimeSeriesSplit(n_splits=cv_splits)
        else:
            cv_method = KFold(n_splits=cv_splits)

        if param_grid:
            grid_search = GridSearchCV(
                self.model,
                param_grid,
                cv=cv_method,
                scoring="neg_mean_squared_error",
                n_jobs=n_jobs
            )
            grid_search.fit(X, y)

            self.model = grid_search.best_estimator_
            self.best_parameters = grid_search.best_params_
        else:
            self.model.fit(X, y)

    def permutation_importance(self, x_data, y_data):
        """Compute permutation importance based on a fitted model and a test set."""

        perm_importance = permutation_importance(
            self.model,
            x_data,
            y_data,
            n_repeats=10,
            scoring="neg_mean_squared_error"
        )

        importance_df = pd.Series(
            perm_importance.importances_mean,
            name='Error reduction',
            index=x_data.columns
        )
        importance_df = importance_df.sort_values(ascending=False)

        return importance_df

    def feature_importance(self, X):
        """
        Compute feature importance based on a fitted model by averaging over the importances of the individual trees.
        """

        if hasattr(self.model, "estimators_"):
            feature_importances = np.mean(
                [tree.feature_importances_ for tree in self.model.estimators_],
                axis=0
            )
            return pd.Series(feature_importances, index=X.columns).sort_values(
                ascending=False)
        else:
            raise ValueError("Model is not fitted yet.")

    def predict(self, x_data):
        """Use fitted model to predict based on a set of inputs"""
        return self.model.predict(x_data)

    def rolling_predict(
            self,
            x_train,
            y_train,
            x_test,
            y_test,
            window_size=None,
            print_info=True,
            feature_importance=False
    ):
        """
        Performs a rolling prediction, updating the model at each step.

        Parameters:
        - x_train: inputs of the training set
        - y_train: outputs of the training set
        - x_test: inputs of the test set
        - y_test: outputs of the test set
        - window_size: size of the rolling window, default is size of the training set
        - print_info: determines whether information on the process should be printed during computations
        - feature_importance: determines whether feature importance should be computed during computations

        Returns:
        - np.array: rolling predictions of the output variable
        - feature_importances: feature importances of the output variable for each iteration

        """

        if window_size is None:
            window_size = len(x_train)

        rolling_predictions = []
        feature_importances = {}
        X_train_rolling = copy.deepcopy(x_train)
        y_train_rolling = copy.deepcopy(y_train)
        for i in range(len(x_test)):
            if print_info:
                one_step_time = time.time()

            x_train_window = X_train_rolling.iloc[-window_size:].reset_index(drop=True)
            y_train_window = y_train_rolling[-window_size:]

            self.fit(x_train_window, y_train_window)
            feature_importances[f'Iteration {i}'] = self.feature_importance(x_train_window)
            y_pred = self.model.predict(x_test.iloc[[i]])[0]
            rolling_predictions.append(y_pred)

            X_train_rolling = pd.concat([X_train_rolling, x_test.iloc[[i]]], axis=0).reset_index(drop=True)
            y_train_rolling = np.append(y_train_rolling, y_test[i])

            if print_info:
                if np.mod(i, 100) == 0:
                    print(
                        f'Rolling Tree Forecast, step {i} took {time.time() - one_step_time} seconds.', flush=True
                    )

        return np.array(rolling_predictions), feature_importances if feature_importance else None








