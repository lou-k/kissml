import tempfile
from pathlib import Path

import pytest

from kissml.core import close_all_caches
from kissml.settings import settings
from kissml.types import AfterEffect


class RecordingAfterEffect(AfterEffect):
    """AfterEffect that records every call's arguments."""

    def __init__(self):
        self.call_count = 0
        self.calls = []

    def __call__(self, result, was_cached, func_name, execution_time, tags):
        self.call_count += 1
        self.calls.append(
            {
                "result": result,
                "was_cached": was_cached,
                "func_name": func_name,
                "execution_time": execution_time,
                "tags": tags,
            }
        )


class FailingAfterEffect(AfterEffect):
    """AfterEffect that always raises an exception."""

    def __call__(self, result, was_cached, func_name, execution_time, tags):
        raise ValueError("AfterEffect intentionally failed")


@pytest.fixture
def clean_global_effects():
    """Reset settings.global_after_effects before and after each test."""
    original = list(settings.global_after_effects)
    settings.global_after_effects.clear()
    try:
        yield settings.global_after_effects
    finally:
        settings.global_after_effects.clear()
        settings.global_after_effects.extend(original)


@pytest.fixture(autouse=True)
def clean_cache():
    """Point the cache at a fresh temp directory for each test."""
    with tempfile.TemporaryDirectory() as tmpdir:
        original_cache_dir = settings.cache_directory
        settings.cache_directory = Path(tmpdir)

        yield

        close_all_caches()
        settings.cache_directory = original_cache_dir
