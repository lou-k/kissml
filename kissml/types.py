from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, BinaryIO

from pydantic import BaseModel, Field


class EvictionPolicy(Enum):
    """
    Cache eviction policies for DiskCache.

    Attributes:
        LEAST_RECENTLY_STORED: Default policy. Evicts oldest stored keys first.
            No update required on access. Best for large caches.
        LEAST_RECENTLY_USED: Most common policy. Evicts least recently accessed keys.
            Updates access time on every access (slower due to writes).
        LEAST_FREQUENTLY_USED: Evicts least frequently accessed keys.
            Increments access count on every access (slower due to writes).
        NONE: Disables cache evictions. Cache grows without bound.
            Items still lazily removed if expired.
    """

    LEAST_RECENTLY_STORED = "least-recently-stored"
    LEAST_RECENTLY_USED = "least-recently-used"
    LEAST_FREQUENTLY_USED = "least-frequently-used"
    NONE = "none"


class CacheConfig(BaseModel):
    eviction_policy: EvictionPolicy = Field(
        default=EvictionPolicy.NONE,
        description="The eviction policy for the cache. Defaults to None (keeps forever). See https://grantjenks.com/docs/diskcache/api.html#diskcache.diskcache.EVICTION_POLICY",
    )
    version: int = Field(
        default=0,
        description="The cache version. Change this to invalidate all entries on a re-run.",
    )
    namespace: str | None = Field(
        default=None,
        description="Optional namespace for the cache. Useful for isolating caches between different parts of an application.",
    )


class Serializer(ABC):
    """
    Abstract base class for type-specific serializers.

    Serializers define how to convert Python objects to/from disk storage.
    Each serializer handles one type.
    """

    @abstractmethod
    def serialize(self, value: Any, out: BinaryIO) -> None:
        """
        Write value to output stream at the given position.

        Args:
            value: The object to serialize
            out: The file like object to write to
        """
        pass

    @abstractmethod
    def deserialize(self, input: BinaryIO) -> Any:
        """
        Read value from the bytestream at the given path.

        Args:
            input: The file or byte stream to read from

        Returns:
            The deserialized object
        """
        pass


class AfterEffect(ABC):
    """
    Abstract base class for side effects that run after a step completes.

    AfterEffects are executed after a step function returns, whether the result
    was freshly computed or loaded from cache. They enable visualization, logging,
    validation, and other observability patterns without modifying the step's logic.

    Example:
        Create a custom AfterEffect to log DataFrame info:

        >>> import logging
        >>> from typing import Annotated
        >>> from kissml import step, AfterEffect, CacheConfig
        >>>
        >>> class DataFrameLogger(AfterEffect):
        ...     def __call__(self, result, was_cached, func_name, execution_time):
        ...         logging.info(f"{func_name}: {len(result)} rows, cached={was_cached}")
        >>>
        >>> @step(cache=CacheConfig(version=1))
        ... def load_data() -> Annotated[pd.DataFrame, DataFrameLogger()]:
        ...     return pd.read_csv("data.csv")

        The step decorator will automatically execute all AfterEffects in the
        annotation, passing the result and execution metadata.

    Notes:
        - AfterEffects run on both cached and fresh results
        - Multiple effects are executed left-to-right from the annotation
        - Effects should observe results, not transform them
        - The step decorator handles error handling based on configuration
    """

    @abstractmethod
    def __call__(
        self, result, was_cached: bool, func_name: str, execution_time: float
    ):
        """
        Execute this effect on the function result.

        Args:
            result: The return value from the step function
            was_cached: True if the result was loaded from cache, False if freshly computed
            func_name: The name of the step function that produced this result
            execution_time: Time in seconds taken to produce or load the result
        """
        pass
