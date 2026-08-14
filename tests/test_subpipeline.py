from typing import Annotated

import pytest

from kissml.step import step, subpipeline
from kissml.types import AfterEffect, CacheConfig, EvictionPolicy


class RecordingAfterEffect(AfterEffect):
    """AfterEffect that records how many times it's called."""

    def __init__(self):
        self.call_count = 0

    def __call__(
        self, result, was_cached: bool, func_name: str, execution_time: float
    ):
        self.call_count += 1


class FailingAfterEffect(AfterEffect):
    """AfterEffect that always raises an exception."""

    def __call__(
        self, result, was_cached: bool, func_name: str, execution_time: float
    ):
        raise ValueError("AfterEffect intentionally failed")


def test_subpipeline_body_runs_every_call_even_with_cached_inner_step():
    """A subpipeline's body always executes, even when the inner @step
    it calls hits its cache."""
    inner_calls = 0
    outer_calls = 0

    @step(cache=CacheConfig(version=0, eviction_policy=EvictionPolicy.NONE))
    def load(x: int) -> int:
        nonlocal inner_calls
        inner_calls += 1
        return x * 2

    @subpipeline()
    def prepare(x: int) -> int:
        nonlocal outer_calls
        outer_calls += 1
        return load(x)

    assert prepare(5) == 10
    assert prepare(5) == 10

    assert inner_calls == 1  # Inner step cache hit on second call
    assert outer_calls == 2  # Subpipeline body always runs


def test_subpipeline_after_effect_runs_every_call():
    """AfterEffects on a subpipeline's return annotation fire on every call."""
    effect = RecordingAfterEffect()

    @subpipeline()
    def prepare(x: int) -> Annotated[int, effect]:
        return x + 1

    prepare(1)
    prepare(1)

    assert effect.call_count == 2


def test_kissml_kind_markers():
    """__kissml_kind__ distinguishes subpipeline- from step-decorated
    functions for introspection tooling."""

    @subpipeline()
    def prepare(x: int) -> int:
        return x

    @step()
    def compute(x: int) -> int:
        return x

    assert prepare.__kissml_kind__ == "subpipeline"
    assert compute.__kissml_kind__ == "step"


def test_subpipeline_error_on_effect_failure():
    """error_on_effect_failure=True raises when a subpipeline's effect fails."""
    failing_effect = FailingAfterEffect()

    @subpipeline(error_on_effect_failure=True)
    def prepare(x: int) -> Annotated[int, failing_effect]:
        return x

    with pytest.raises(ValueError, match="AfterEffect intentionally failed"):
        prepare(1)


def test_subpipeline_returns_value_and_preserves_name():
    """Decorated function returns its value and keeps its original name."""

    @subpipeline()
    def prepare(x: int) -> int:
        return x * 3

    assert prepare(4) == 12
    assert prepare.__name__ == "prepare"
