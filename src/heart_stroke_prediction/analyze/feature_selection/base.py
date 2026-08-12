"""Base interface for feature selection algorithms.

The classes in this module deliberately contain no model evaluation logic.
Concrete selectors are responsible only for learning a feature subset and
transforming data. Model evaluation belongs to the experiment/evaluation layer.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class BaseFeatureSelector(BaseEstimator, TransformerMixin, ABC):
    """Common interface for filter, wrapper, and embedded selectors.

    Concrete implementations must implement :meth:`_fit` and populate a
    boolean support mask containing one value per input feature.

    The public API intentionally follows the scikit-learn transformer
    convention so selectors can be composed with existing pipelines.
    """

    def fit(self, X: Any, y: Any = None) -> "BaseFeatureSelector":
        """Fit the selector and learn the selected feature subset.

        Parameters
        ----------
        X:
            Training features. A pandas DataFrame is preferred because column
            names can then be preserved by ``get_feature_names_out``.
        y:
            Target values. Unsupervised selectors may ignore this argument.

        Returns
        -------
        BaseFeatureSelector
            The fitted selector.
        """
        self._validate_X(X)

        if y is not None and len(X) != len(y):
            raise ValueError(
                "X and y have inconsistent numbers of samples: "
                f"{len(X)} != {len(y)}."
            )

        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = self._get_feature_names(X)

        support = np.asarray(self._fit(X, y), dtype=bool)
        if support.ndim != 1 or support.shape[0] != self.n_features_in_:
            raise ValueError(
                "A feature selector must return a one-dimensional boolean "
                "support mask with one value per input feature."
            )

        if not support.any():
            raise ValueError("Feature selection produced an empty feature set.")

        self.support_ = support
        self.n_features_selected_ = int(support.sum())
        self.is_fitted_ = True
        return self

    @abstractmethod
    def _fit(self, X: Any, y: Any = None) -> np.ndarray:
        """Learn and return the boolean support mask.

        Parameters
        ----------
        X:
            Training features.
        y:
            Target values.

        Returns
        -------
        numpy.ndarray
            Boolean array of shape ``(n_features,)``. ``True`` means the
            corresponding feature is selected.
        """
        raise NotImplementedError

    def transform(self, X: Any) -> Any:
        """Transform X by retaining only selected features."""
        self._check_is_fitted()
        self._validate_X(X)

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                "X has a different number of features than the data used to "
                f"fit the selector: {X.shape[1]} != {self.n_features_in_}."
            )

        if hasattr(X, "iloc"):
            return X.iloc[:, self.support_]

        X_array = np.asarray(X)
        return X_array[:, self.support_]

    def fit_transform(self, X: Any, y: Any = None, **fit_params: Any) -> Any:
        """Fit the selector and transform X in one operation."""
        return self.fit(X, y).transform(X)

    def get_support(self, indices: bool = False) -> np.ndarray:
        """Return the mask, or selected feature indices, learned during fit."""
        self._check_is_fitted()

        if indices:
            return np.flatnonzero(self.support_)
        return self.support_.copy()

    def get_feature_names_out(
        self, input_features: Optional[Any] = None
    ) -> np.ndarray:
        """Return names of the selected features."""
        self._check_is_fitted()

        if input_features is None:
            input_features = self.feature_names_in_

        if input_features is None:
            input_features = np.asarray(
                [f"x{i}" for i in range(self.n_features_in_)], dtype=object
            )
        else:
            input_features = np.asarray(input_features, dtype=object)
            if input_features.ndim != 1:
                raise ValueError("input_features must be one-dimensional.")
            if len(input_features) != self.n_features_in_:
                raise ValueError(
                    "input_features must contain one name per input feature: "
                    f"{len(input_features)} != {self.n_features_in_}."
                )

        return input_features[self.support_]

    @staticmethod
    def _validate_X(X: Any) -> None:
        """Validate the minimum array-like contract required by selectors."""
        if X is None or not hasattr(X, "shape"):
            raise TypeError("X must be a 2-dimensional array-like object.")

        if len(X.shape) != 2:
            raise ValueError("X must be a 2-dimensional array-like object.")

        if X.shape[1] == 0:
            raise ValueError("X must contain at least one feature.")

    @staticmethod
    def _get_feature_names(X: Any) -> Optional[np.ndarray]:
        """Extract DataFrame column names when available."""
        if hasattr(X, "columns"):
            return np.asarray(X.columns, dtype=object)
        return None

    def _check_is_fitted(self) -> None:
        """Raise a clear error when transform is called before fit."""
        if not getattr(self, "is_fitted_", False):
            raise RuntimeError(
                "This feature selector is not fitted yet. Call fit(X, y) "
                "before using this method."
            )
