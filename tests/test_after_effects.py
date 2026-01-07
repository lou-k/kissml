import tempfile
from pathlib import Path
from typing import Annotated

import pytest

from kissml.core import close_all_caches
from kissml.settings import settings
from kissml.step import step
from kissml.types import AfterEffect, CacheConfig, EvictionPolicy


@pytest.fixture(autouse=True)
def clean_cache():
    """Clean up cache before and after each test."""
    # Setup: use temporary directory for tests
    with tempfile.TemporaryDirectory() as tmpdir:
        original_cache_dir = settings.cache_directory
        settings.cache_directory = Path(tmpdir)

        yield

        # Teardown: close all caches and restore original directory
        close_all_caches()
        settings.cache_directory = original_cache_dir


class RecordingAfterEffect(AfterEffect):
    """AfterEffect that records when it's called."""

    def __init__(self):
        self.call_count = 0
        self.calls = []

    def __call__(
        self, result, was_cached: bool, func_name: str, execution_time: float
    ):
        self.call_count += 1
        self.calls.append(
            {
                "result": result,
                "was_cached": was_cached,
                "func_name": func_name,
                "execution_time": execution_time,
            }
        )


class FailingAfterEffect(AfterEffect):
    """AfterEffect that always raises an exception."""

    def __call__(
        self, result, was_cached: bool, func_name: str, execution_time: float
    ):
        raise ValueError("AfterEffect intentionally failed")


def test_after_effect_called():
    """Test that AfterEffect is called when present in return annotation."""
    effect = RecordingAfterEffect()

    @step()
    def compute(x: int, y: int) -> Annotated[int, effect]:
        return x + y

    result = compute(3, 4)

    assert result == 7
    assert effect.call_count == 1
    assert effect.calls[0]["result"] == 7
    assert effect.calls[0]["was_cached"] is False
    assert effect.calls[0]["func_name"] == "compute"
    assert isinstance(effect.calls[0]["execution_time"], float)
    assert effect.calls[0]["execution_time"] >= 0


def test_after_effect_called_with_cache():
    """Test that AfterEffect is called on both fresh and cached results."""
    effect = RecordingAfterEffect()

    @step(cache=CacheConfig(version=1, eviction_policy=EvictionPolicy.NONE))
    def compute(x: int) -> Annotated[int, effect]:
        return x * 2

    # First call - fresh result
    result1 = compute(5)
    assert result1 == 10
    assert effect.call_count == 1
    assert effect.calls[0]["result"] == 10
    assert effect.calls[0]["was_cached"] is False

    # Second call - cached result
    result2 = compute(5)
    assert result2 == 10
    assert effect.call_count == 2
    assert effect.calls[1]["result"] == 10
    assert effect.calls[1]["was_cached"] is True


def test_multiple_after_effects_called():
    """Test that multiple AfterEffects are called in left-to-right order."""
    call_order = []

    class OrderTrackingEffect(AfterEffect):
        def __init__(self, name):
            self.name = name

        def __call__(
            self,
            result,
            was_cached: bool,
            func_name: str,
            execution_time: float,
        ):
            call_order.append(self.name)

    effect_a = OrderTrackingEffect("A")
    effect_b = OrderTrackingEffect("B")
    effect_c = OrderTrackingEffect("C")

    @step()
    def compute(x: int) -> Annotated[int, effect_a, effect_b, effect_c]:
        return x * 3

    result = compute(7)

    assert result == 21
    assert call_order == ["A", "B", "C"]


def test_error_on_effect_failure_false():
    """Test that AfterEffect errors are caught when error_on_effect_failure=False."""
    failing_effect = FailingAfterEffect()

    @step(error_on_effect_failure=False)
    def compute(x: int) -> Annotated[int, failing_effect]:
        return x + 10

    # Function should complete successfully despite effect failure
    result = compute(5)
    assert result == 15


def test_error_on_effect_failure_true():
    """Test that AfterEffect errors are raised when error_on_effect_failure=True."""
    failing_effect = FailingAfterEffect()

    @step(error_on_effect_failure=True)
    def compute(x: int) -> Annotated[int, failing_effect]:
        return x + 10

    # Function should raise the effect's exception
    with pytest.raises(ValueError, match="AfterEffect intentionally failed"):
        compute(5)


def test_multiple_effects_one_fails_strict_mode():
    """Test that when one effect fails in strict mode, execution stops."""
    effect1 = RecordingAfterEffect()
    failing_effect = FailingAfterEffect()
    effect2 = RecordingAfterEffect()

    @step(error_on_effect_failure=True)
    def compute(x: int) -> Annotated[int, effect1, failing_effect, effect2]:
        return x + 10

    with pytest.raises(ValueError, match="AfterEffect intentionally failed"):
        compute(5)

    # First effect should have been called
    assert effect1.call_count == 1
    # Second effect should not have been called (execution stopped)
    assert effect2.call_count == 0


def test_multiple_effects_one_fails_permissive_mode():
    """Test that when one effect fails in permissive mode, other effects still run."""
    effect1 = RecordingAfterEffect()
    failing_effect = FailingAfterEffect()
    effect2 = RecordingAfterEffect()

    @step(error_on_effect_failure=False)
    def compute(x: int) -> Annotated[int, effect1, failing_effect, effect2]:
        return x + 10

    result = compute(5)

    assert result == 15
    # All effects should have been called
    assert effect1.call_count == 1
    assert effect2.call_count == 1


def test_no_after_effects():
    """Test that function works normally without any AfterEffects."""

    @step()
    def compute(x: int) -> int:
        return x + 5

    result = compute(10)
    assert result == 15


def test_after_effect_with_non_annotated_type():
    """Test that function works normally when return type is not Annotated."""
    effect = RecordingAfterEffect()

    @step()
    def compute(x: int) -> int:
        return x + 5

    result = compute(10)
    assert result == 15
    assert effect.call_count == 0
