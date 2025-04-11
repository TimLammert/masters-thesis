"""
Copy/Subclass of scikit-learns BaggingRegressor and custom helper functions applying
the moving block bootstrap to create training set bootstrap samples.
A simple subclass was not possible, because the resampling is performed by helper functions outside of the class.
"""


import itertools
import numbers
from abc import ABCMeta, abstractmethod
from functools import partial
from numbers import Integral
from warnings import warn

import numpy as np

from sklearn.base import ClassifierMixin, RegressorMixin, _fit_context
from sklearn.metrics import accuracy_score, r2_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.utils import (
    Bunch,
    _safe_indexing,
    check_random_state,
    column_or_1d,
)
from sklearn.utils._mask import indices_to_mask
from sklearn.utils._param_validation import HasMethods, Interval, RealNotInt
from sklearn.utils._tags import get_tags
from sklearn.utils.metadata_routing import (
    MetadataRouter,
    MethodMapping,
    _raise_for_params,
    _routing_enabled,
    get_routing_for_object,
    process_routing,
)
from sklearn.utils.metaestimators import available_if
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.parallel import Parallel, delayed
from sklearn.utils.random import sample_without_replacement
from sklearn.utils.validation import (
    _check_method_params,
    _check_sample_weight,
    _deprecate_positional_args,
    _estimator_has,
    check_is_fitted,
    has_fit_parameter,
    validate_data,
)
from sklearn.ensemble._base import BaseEnsemble, _partition_estimators
from sklearn.ensemble import BaggingRegressor

MAX_INT = np.iinfo(np.int32).max

def _generate_indices(random_state, bootstrap, n_population, n_samples):
    """Draw randomly sampled indices."""
    # Draw sample indices
    if bootstrap:
        indices = random_state.randint(0, n_population, n_samples)
    else:
        indices = sample_without_replacement(
            n_population, n_samples, random_state=random_state
        )

    return indices

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

    feature_indices = _generate_indices(random_state, bootstrap_features, n_features, max_features)

    sample_indices = _generate_block_indices(
        random_state,
        n_population=n_samples,
        n_samples=n_samples,
        block_size=max_samples
    )
    return feature_indices, sample_indices

def _parallel_build_estimators(
    n_estimators,
    ensemble,
    X,
    y,
    seeds,
    total_n_estimators,
    verbose,
    check_input,
    fit_params,
):
    """Private function used to build a batch of estimators within a job."""
    # Retrieve settings
    n_samples, n_features = X.shape
    max_features = ensemble._max_features
    max_samples = ensemble._max_samples
    bootstrap = ensemble.bootstrap
    bootstrap_features = ensemble.bootstrap_features
    has_check_input = has_fit_parameter(ensemble.estimator_, "check_input")
    requires_feature_indexing = bootstrap_features or max_features != n_features

    # Build estimators
    estimators = []
    estimators_features = []

    # TODO: (slep6) remove if condition for unrouted sample_weight when metadata
    # routing can't be disabled.
    support_sample_weight = has_fit_parameter(ensemble.estimator_, "sample_weight")
    if not _routing_enabled() and (
        not support_sample_weight and fit_params.get("sample_weight") is not None
    ):
        raise ValueError(
            "The base estimator doesn't support sample weight, but sample_weight is "
            "passed to the fit method."
        )

    for i in range(n_estimators):
        if verbose > 1:
            print(
                "Building estimator %d of %d for this parallel run (total %d)..."
                % (i + 1, n_estimators, total_n_estimators)
            )

        random_state = seeds[i]
        estimator = ensemble._make_estimator(append=False, random_state=random_state)

        if has_check_input:
            estimator_fit = partial(estimator.fit, check_input=check_input)
        else:
            estimator_fit = estimator.fit

        # Draw random feature, sample indices
        features, indices = block_generate_bagging_indices(
            random_state,
            bootstrap_features,
            bootstrap,
            n_features,
            n_samples,
            max_features,
            max_samples,
        )

        fit_params_ = fit_params.copy()

        # TODO(SLEP6): remove if condition for unrouted sample_weight when metadata
        # routing can't be disabled.
        # 1. If routing is enabled, we will check if the routing supports sample
        # weight and use it if it does.
        # 2. If routing is not enabled, we will check if the base
        # estimator supports sample_weight and use it if it does.

        # Note: Row sampling can be achieved either through setting sample_weight or
        # by indexing. The former is more efficient. Therefore, use this method
        # if possible, otherwise use indexing.
        if _routing_enabled():
            request_or_router = get_routing_for_object(ensemble.estimator_)
            consumes_sample_weight = request_or_router.consumes(
                "fit", ("sample_weight",)
            )
        else:
            consumes_sample_weight = support_sample_weight
        if consumes_sample_weight:
            # Draw sub samples, using sample weights, and then fit
            curr_sample_weight = _check_sample_weight(
                fit_params_.pop("sample_weight", None), X
            ).copy()

            if bootstrap:
                sample_counts = np.bincount(indices, minlength=n_samples)
                curr_sample_weight *= sample_counts
            else:
                not_indices_mask = ~indices_to_mask(indices, n_samples)
                curr_sample_weight[not_indices_mask] = 0

            fit_params_["sample_weight"] = curr_sample_weight
            X_ = X[:, features] if requires_feature_indexing else X
            estimator_fit(X_, y, **fit_params_)
        else:
            # cannot use sample_weight, so use indexing
            y_ = _safe_indexing(y, indices)
            X_ = _safe_indexing(X, indices)
            fit_params_ = _check_method_params(X, params=fit_params_, indices=indices)
            if requires_feature_indexing:
                X_ = X_[:, features]
            estimator_fit(X_, y_, **fit_params_)

        estimators.append(estimator)
        estimators_features.append(features)

    return estimators, estimators_features


def _parallel_predict_proba(estimators, estimators_features, X, n_classes):
    """Private function used to compute (proba-)predictions within a job."""
    n_samples = X.shape[0]
    proba = np.zeros((n_samples, n_classes))

    for estimator, features in zip(estimators, estimators_features):
        if hasattr(estimator, "predict_proba"):
            proba_estimator = estimator.predict_proba(X[:, features])

            if n_classes == len(estimator.classes_):
                proba += proba_estimator

            else:
                proba[:, estimator.classes_] += proba_estimator[
                    :, range(len(estimator.classes_))
                ]

        else:
            # Resort to voting
            predictions = estimator.predict(X[:, features])

            for i in range(n_samples):
                proba[i, predictions[i]] += 1

    return proba


def _parallel_predict_log_proba(estimators, estimators_features, X, n_classes):
    """Private function used to compute log probabilities within a job."""
    n_samples = X.shape[0]
    log_proba = np.empty((n_samples, n_classes))
    log_proba.fill(-np.inf)
    all_classes = np.arange(n_classes, dtype=int)

    for estimator, features in zip(estimators, estimators_features):
        log_proba_estimator = estimator.predict_log_proba(X[:, features])

        if n_classes == len(estimator.classes_):
            log_proba = np.logaddexp(log_proba, log_proba_estimator)

        else:
            log_proba[:, estimator.classes_] = np.logaddexp(
                log_proba[:, estimator.classes_],
                log_proba_estimator[:, range(len(estimator.classes_))],
            )

            missing = np.setdiff1d(all_classes, estimator.classes_)
            log_proba[:, missing] = np.logaddexp(log_proba[:, missing], -np.inf)

    return log_proba


def _parallel_decision_function(estimators, estimators_features, X):
    """Private function used to compute decisions within a job."""
    return sum(
        estimator.decision_function(X[:, features])
        for estimator, features in zip(estimators, estimators_features)
    )


def _parallel_predict_regression(estimators, estimators_features, X):
    """Private function used to compute predictions within a job."""
    return sum(
        estimator.predict(X[:, features])
        for estimator, features in zip(estimators, estimators_features)
    )


class BlockBaggingRegressor(BaggingRegressor):
    def __init__(
            self,
            estimator=None,
            n_estimators=10,
            *,
            max_samples=1.0,
            max_features=1.0,
            bootstrap=True,
            bootstrap_features=False,
            oob_score=False,
            warm_start=False,
            n_jobs=None,
            random_state=None,
            verbose=0,
    ):
        super().__init__(
            estimator=estimator,
            n_estimators=n_estimators,
            max_samples=max_samples,
            max_features=max_features,
            bootstrap=bootstrap,
            bootstrap_features=bootstrap_features,
            oob_score=oob_score,
            warm_start=warm_start,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
        )

    def _fit(
        self,
        X,
        y,
        max_samples=None,
        max_depth=None,
        check_input=True,
        **fit_params,
    ):
        """Build a Bagging ensemble of estimators from the training
           set (X, y).

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Sparse matrices are accepted only if
            they are supported by the base estimator.

        y : array-like of shape (n_samples,)
            The target values (class labels in classification, real numbers in
            regression).

        max_samples : int or float, default=None
            Argument to use instead of self.max_samples.

        max_depth : int, default=None
            Override value used when constructing base estimator. Only
            supported if the base estimator has a max_depth parameter.

        check_input : bool, default=True
            Override value used when fitting base estimator. Only supported
            if the base estimator has a check_input parameter for fit function.
            If the meta-estimator already checks the input, set this value to
            False to prevent redundant input validation.

        **fit_params : dict, default=None
            Parameters to pass to the :term:`fit` method of the underlying
            estimator.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        random_state = check_random_state(self.random_state)

        # Remap output
        n_samples = X.shape[0]
        self._n_samples = n_samples
        y = self._validate_y(y)

        # Check parameters
        self._validate_estimator(self._get_estimator())

        if _routing_enabled():
            routed_params = process_routing(self, "fit", **fit_params)
        else:
            routed_params = Bunch()
            routed_params.estimator = Bunch(fit=fit_params)
            if "sample_weight" in fit_params:
                routed_params.estimator.fit["sample_weight"] = fit_params[
                    "sample_weight"
                ]

        if max_depth is not None:
            self.estimator_.max_depth = max_depth

        # Validate max_samples
        if max_samples is None:
            max_samples = self.max_samples
        elif not isinstance(max_samples, numbers.Integral):
            max_samples = int(max_samples * X.shape[0])

        if max_samples > X.shape[0]:
            raise ValueError("max_samples must be <= n_samples")

        # Store validated integer row sampling value
        self._max_samples = max_samples

        # Validate max_features
        if isinstance(self.max_features, numbers.Integral):
            max_features = self.max_features
        elif isinstance(self.max_features, float):
            max_features = int(self.max_features * self.n_features_in_)

        if max_features > self.n_features_in_:
            raise ValueError("max_features must be <= n_features")

        max_features = max(1, int(max_features))

        # Store validated integer feature sampling value
        self._max_features = max_features

        # Other checks
        if not self.bootstrap and self.oob_score:
            raise ValueError("Out of bag estimation only available if bootstrap=True")

        if self.warm_start and self.oob_score:
            raise ValueError("Out of bag estimate only available if warm_start=False")

        if hasattr(self, "oob_score_") and self.warm_start:
            del self.oob_score_

        if not self.warm_start or not hasattr(self, "estimators_"):
            # Free allocated memory, if any
            self.estimators_ = []
            self.estimators_features_ = []

        n_more_estimators = self.n_estimators - len(self.estimators_)

        if n_more_estimators < 0:
            raise ValueError(
                "n_estimators=%d must be larger or equal to "
                "len(estimators_)=%d when warm_start==True"
                % (self.n_estimators, len(self.estimators_))
            )

        elif n_more_estimators == 0:
            warn(
                "Warm-start fitting without increasing n_estimators does not "
                "fit new trees."
            )
            return self

        # Parallel loop
        n_jobs, n_estimators, starts = _partition_estimators(
            n_more_estimators, self.n_jobs
        )
        total_n_estimators = sum(n_estimators)

        # Advance random state to state after training
        # the first n_estimators
        if self.warm_start and len(self.estimators_) > 0:
            random_state.randint(MAX_INT, size=len(self.estimators_))

        seeds = random_state.randint(MAX_INT, size=n_more_estimators)
        self._seeds = seeds

        all_results = Parallel(
            n_jobs=n_jobs, verbose=self.verbose, **self._parallel_args()
        )(
            delayed(_parallel_build_estimators)(
                n_estimators[i],
                self,
                X,
                y,
                seeds[starts[i] : starts[i + 1]],
                total_n_estimators,
                verbose=self.verbose,
                check_input=check_input,
                fit_params=routed_params.estimator.fit,
            )
            for i in range(n_jobs)
        )

        # Reduce
        self.estimators_ += list(
            itertools.chain.from_iterable(t[0] for t in all_results)
        )
        self.estimators_features_ += list(
            itertools.chain.from_iterable(t[1] for t in all_results)
        )

        if self.oob_score:
            self._set_oob_score(X, y)

        return self


    def _get_estimators_indices(self):
        # Get drawn indices along both sample and feature axes
        for seed in self._seeds:
            # Operations accessing random_state must be performed identically
            # to those in `_parallel_build_estimators()`
            feature_indices, sample_indices = block_generate_bagging_indices(
                seed,
                self.bootstrap_features,
                self.bootstrap,
                self.n_features_in_,
                self._n_samples,
                self._max_features,
                self._max_samples,
            )

            yield feature_indices, sample_indices
