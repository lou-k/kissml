from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

from .types import Tags

_TAGS: ContextVar[Tags] = ContextVar("kissml_tags", default={})


@contextmanager
def tags(new: Tags) -> Iterator[None]:
    """Attach ambient tags to every step called within the block.

    Nested blocks merge, inner winning on conflicts; exiting restores the
    prior state. Ambient tags do not reach joblib/multiprocessing workers.
    """
    token = _TAGS.set({**_TAGS.get(), **new})
    try:
        yield
    finally:
        _TAGS.reset(token)
