import numpy as np
import pandas as pd
import pytest

from heart_stroke_prediction.analyze.feature_selection.base import (
    BaseFeatureSelector,
)


class DummySelector(BaseFeatureSelector):
    """Minimal selector used to test the base contract."""

    def __init__(self, selected=(0, 2)):
        self.selected = selected

    def _fit(self, X, y=None):
        support = np.zeros(X.shape[1], dtype=bool)
        support[list(self.selected)] = True
        return support


def make_data():
    X = pd.DataFrame(
        {
            "age": [20, 30, 40, 50],
            "bmi": [20.1, 22.4, 25.3, 28.1],
            "glucose": [80, 90, 110, 130],
        }
    )
    y = pd.Series([0, 0, 1, 1], name="stroke")
    return X, y


def test_fit_learns_feature_metadata_and_support():
    X, y = make_data()

    selector = DummySelector().fit(X, y)

    assert selector.n_features_in_ == 3
    assert selector.n_features_selected_ == 2
    assert selector.is_fitted_ is True
    np.testing.assert_array_equal(selector.get_support(), [True, False, True])
    np.testing.assert_array_equal(selector.get_support(indices=True), [0, 2])


def test_transform_preserves_dataframe_and_selected_columns():
    X, y = make_data()

    result = DummySelector().fit(X, y).transform(X)

    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["age", "glucose"]
    assert result.shape == (4, 2)


def test_fit_transform_returns_selected_features():
    X, y = make_data()

    result = DummySelector().fit_transform(X, y)

    assert list(result.columns) == ["age", "glucose"]


def test_get_feature_names_out_uses_dataframe_columns():
    X, y = make_data()

    selector = DummySelector().fit(X, y)

    np.testing.assert_array_equal(
        selector.get_feature_names_out(), ["age", "glucose"]
    )


def test_get_feature_names_out_accepts_custom_names():
    X, y = make_data()

    selector = DummySelector().fit(X, y)

    np.testing.assert_array_equal(
        selector.get_feature_names_out(["a", "b", "c"]), ["a", "c"]
    )


def test_transform_supports_numpy_arrays():
    X = np.array([[1, 2, 3], [4, 5, 6]])
    y = np.array([0, 1])

    result = DummySelector().fit(X, y).transform(X)

    np.testing.assert_array_equal(result, [[1, 3], [4, 6]])


def test_transform_before_fit_raises():
    X, _ = make_data()

    with pytest.raises(RuntimeError, match="not fitted"):
        DummySelector().transform(X)


def test_fit_rejects_inconsistent_sample_counts():
    X, _ = make_data()
    y = pd.Series([0, 1])

    with pytest.raises(ValueError, match="inconsistent"):
        DummySelector().fit(X, y)


def test_fit_rejects_invalid_support_length():
    class InvalidSelector(BaseFeatureSelector):
        def _fit(self, X, y=None):
            return np.array([True])

    X, y = make_data()

    with pytest.raises(ValueError, match="one value per input feature"):
        InvalidSelector().fit(X, y)


def test_fit_rejects_empty_feature_set():
    class EmptySelector(BaseFeatureSelector):
        def _fit(self, X, y=None):
            return np.zeros(X.shape[1], dtype=bool)

    X, y = make_data()

    with pytest.raises(ValueError, match="empty feature set"):
        EmptySelector().fit(X, y)


def test_transform_rejects_different_feature_count():
    X, y = make_data()
    selector = DummySelector().fit(X, y)
    invalid_X = X[["age", "bmi"]]

    with pytest.raises(ValueError, match="different number of features"):
        selector.transform(invalid_X)
