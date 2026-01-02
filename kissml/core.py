from collections import OrderedDict
from typing import Any, Tuple

from diskcache import Cache

from kissml.disk import TypeRoutingDisk
from kissml.settings import settings
from kissml.types import EvictionPolicy


def _hash_value(v: Any) -> str:
    hash_f = settings.hash_by_type.get(type(v), lambda x: str(hash(x)))
    return hash_f(v)


def create_cache_key(**kwargs: dict[str, Any]) -> OrderedDict:
    """
    Creates a deterministic cache key from arbitrary keyword arguments.

    Hashes each argument value using type-specific hash functions (configurable via
    settings.hash_by_type), then returns an ordered dictionary of argument names to
    hash values. The ordering ensures consistent cache keys regardless of argument order.

    Args:
        **kwargs: Arbitrary keyword arguments to hash. Supports custom hash functions
            for types like pandas DataFrames/Series via settings.hash_by_type.

    Returns:
        OrderedDict mapping argument names (sorted alphabetically) to their hash values.
        This can be used directly as a cache key with DiskCache.

    Example:
        >>> key = create_cache_key(a=1, b="test", df=some_dataframe)
        >>> cache.set(key, result)
    """

    # Compute a map of input name -> hash value for the input variables
    hashes = {k: _hash_value(v) for k, v in kwargs.items()}

    # Order the map
    hashes = OrderedDict(sorted(hashes.items()))

    return hashes


_caches: dict[Tuple[str, EvictionPolicy], Cache] = {}


def get_cache(function_name: str, eviction_policy: EvictionPolicy) -> Cache:
    global _caches
    key = (function_name, eviction_policy)
    if key not in _caches:
        cache_directory = (
            settings.cache_directory / function_name / eviction_policy.value
        )
        _caches[key] = Cache(
            directory=str(cache_directory),
            eviction_policy=eviction_policy.value,
            disk=TypeRoutingDisk,
        )

    return _caches[key]


def close_all_caches():
    global _caches
    for cache in _caches.values():
        cache.close()
    _caches.clear()
