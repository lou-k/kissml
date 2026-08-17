import pytest

import kissml
from kissml import CacheConfig, step, subpipeline
from tests.conftest import RecordingAfterEffect


@pytest.fixture
def seen(clean_global_effects):
    """Tags seen by a global effect, one entry per step call."""
    effect = RecordingAfterEffect()
    clean_global_effects.append(effect)
    return lambda: [c["tags"] for c in effect.calls]


def test_step_tags_win_over_ambient(seen):
    @step(tags={"layer": 3})
    def f():
        return 1

    f()
    with kissml.tags({"layer": 9, "phase": "train"}):
        f()
    assert seen() == [{"layer": 3}, {"layer": 3, "phase": "train"}]


def test_subpipeline_accepts_tags(seen):
    @subpipeline(tags={"lane": "telemetry"})
    def f():
        return 1

    f()
    assert seen() == [{"lane": "telemetry"}]


def test_nested_blocks_merge_and_unwind(seen):
    @step()
    def f():
        return 1

    f()
    with kissml.tags({"layer": 1}):
        with kissml.tags({"lane": "training"}):
            f()
        with pytest.raises(RuntimeError), kissml.tags({"layer": 9}):
            f()
            raise RuntimeError
        f()
    f()
    assert seen() == [
        {},
        {"layer": 1, "lane": "training"},
        {"layer": 9},
        {"layer": 1},
        {},
    ]


def test_ambient_tags_reach_nested_steps(seen):
    @step()
    def inner():
        return 1

    @step()
    def outer():
        return inner()

    with kissml.tags({"phase": "train"}):
        outer()
    assert seen() == [{"phase": "train"}] * 2


def test_tags_skip_cache_key_and_cache_hits_see_current_tags(seen):
    calls = 0

    def body():
        nonlocal calls
        calls += 1
        return 1

    step(cache=CacheConfig(version=1), tags={"owner": "a"})(body)()
    step(cache=CacheConfig(version=1), tags={"owner": "b"})(body)()
    with kissml.tags({"phase": "x"}):
        step(cache=CacheConfig(version=1))(body)()
    assert calls == 1
    assert seen() == [{"owner": "a"}, {"owner": "b"}, {"phase": "x"}]
