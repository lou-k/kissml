import tempfile
from pathlib import Path

import pytest

from kissml.core import close_all_caches
from kissml.settings import settings


@pytest.fixture(autouse=True)
def clean_cache():
    """Point the cache at a fresh temp directory for each test."""
    with tempfile.TemporaryDirectory() as tmpdir:
        original_cache_dir = settings.cache_directory
        settings.cache_directory = Path(tmpdir)

        yield

        close_all_caches()
        settings.cache_directory = original_cache_dir
