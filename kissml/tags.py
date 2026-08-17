import contextvars
from collections.abc import Iterator
from contextlib import contextmanager

from .types import Tags, TagValue

_TAGS: contextvars.ContextVar[dict[str, TagValue]] = contextvars.ContextVar(
    "kissml_tags", default={}
)


def current_tags() -> Tags:
    """The ambient tags in force for the current context (read-only)."""
    return _TAGS.get()


@contextmanager
def tags(new: Tags) -> Iterator[None]:
    """
    Attach ambient tags to every step called within the block.

    Tags are merged with any enclosing ``tags`` blocks (inner wins on a
    conflicting key) and with each step's own ``tags=`` (the step wins).
    Exiting the block restores the exact prior state, even when an
    exception unwinds several blocks at once.

    Ambient tags live in a ``contextvars.ContextVar``, so they propagate
    into nested calls and asyncio tasks but NOT into ``joblib`` or
    ``multiprocessing`` workers. A step executing in a worker sees its
    decorator tags but not the surrounding ``with kissml.tags(...)``.

    Example:
        >>> with kissml.tags({"phase": "train"}):
        ...     build_training_input()  # every step beneath sees phase=train
    """
    # Always build a new dict: the ContextVar default is shared and must
    # never be mutated in place.
    token = _TAGS.set({**_TAGS.get(), **new})
    try:
        yield
    finally:
        _TAGS.reset(token)
