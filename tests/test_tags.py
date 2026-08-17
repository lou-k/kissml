"""Tests for step tags and the ambient ``kissml.tags`` context manager."""

import pytest

import kissml
from kissml import CacheConfig, step, subpipeline
from kissml.tags import current_tags
from tests.conftest import RecordingAfterEffect


@pytest.fixture
def recorder(clean_global_effects):
    """A global RecordingAfterEffect exposing the tags it saw as ``seen``."""

    class TagRecorder(RecordingAfterEffect):
        @property
        def seen(self):
            return [c["tags"] for c in self.calls]

    effect = TagRecorder()
    clean_global_effects.append(effect)
    return effect


def test_no_tags_gives_empty_dict(recorder):
    @step()
    def f():
        return 1

    f()
    assert recorder.seen == [{}]


def test_step_tags_passed_to_effects(recorder):
    @step(tags={"layer": 3})
    def f():
        return 1

    f()
    assert recorder.seen == [{"layer": 3}]


def test_subpipeline_tags_passed_to_effects(recorder):
    @subpipeline(tags={"lane": "telemetry"})
    def f():
        return 1

    f()
    assert recorder.seen == [{"lane": "telemetry"}]


def test_step_tags_win_over_ambient(recorder):
    @step(tags={"layer": 3})
    def f():
        return 1

    with kissml.tags({"layer": 9, "phase": "train"}):
        f()
    assert recorder.seen == [{"layer": 3, "phase": "train"}]


def test_nested_blocks_merge_and_restore():
    assert current_tags() == {}
    with kissml.tags({"layer": 1}):
        with kissml.tags({"lane": "training"}):
            assert current_tags() == {"layer": 1, "lane": "training"}
        assert current_tags() == {"layer": 1}
        with kissml.tags({"layer": 9}):
            assert current_tags() == {"layer": 9}
        assert current_tags() == {"layer": 1}
    assert current_tags() == {}


def test_exception_unwinds_all_blocks():
    with pytest.raises(RuntimeError):
        with kissml.tags({"a": 1}):
            with kissml.tags({"b": 2}):
                raise RuntimeError
    assert current_tags() == {}


def test_ambient_tags_reach_nested_steps(recorder):
    @step()
    def inner():
        return 1

    @step()
    def outer():
        return inner()

    with kissml.tags({"phase": "train"}):
        outer()
    assert recorder.seen == [{"phase": "train"}, {"phase": "train"}]


def test_tags_do_not_affect_cache_key(recorder):
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


def test_cache_hit_sees_current_ambient_tags(recorder):
    @step(cache=CacheConfig(version=1))
    def f():
        return 1

    with kissml.tags({"phase": "train"}):
        f()
    with kissml.tags({"phase": "compare"}):
        f()
    assert recorder.seen == [{"phase": "train"}, {"phase": "compare"}]
