from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Any

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


class Serializer(ABC):
    """
    Abstract base class for type-specific serializers.

    Serializers define how to convert Python objects to/from disk storage.
    Each serializer handles one type.
    """

    @abstractmethod
    def serialize(self, value: Any, path: Path) -> None:
        """
        Write value to disk at the given path.

        Args:
            value: The object to serialize
            path: Full path including filename and extension
        """
        pass

    @abstractmethod
    def deserialize(self, path: Path) -> Any:
        """
        Read value from disk at the given path.

        Args:
            path: Full path to the serialized file

        Returns:
            The deserialized object
        """
        pass
