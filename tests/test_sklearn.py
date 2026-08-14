import numpy as np
from sklearn.linear_model import LogisticRegression

from kissml.step import step
from kissml.types import CacheConfig


def test_fitted_estimators_hash_by_fitted_state():
    """Same hyperparameters but different fits must not share a cache entry."""
    call_count = 0
    X = np.array([[0.0], [1.0]])

    @step(cache=CacheConfig(version=0))
    def predict(model: LogisticRegression) -> list:
        nonlocal call_count
        call_count += 1
        return model.predict(X).tolist()

    model_a = LogisticRegression().fit(X, [0, 1])
    model_b = LogisticRegression().fit(X, [1, 0])
    assert str(model_a) == str(model_b)  # identical constructor reprs

    result_a = predict(model_a)
    assert call_count == 1

    # Different fit - must miss the cache and give the new fit's answer
    result_b = predict(model_b)
    assert call_count == 2
    assert result_a != result_b

    # Same fit again - cache hit
    assert predict(model_a) == result_a
    assert call_count == 2
