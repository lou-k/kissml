import inspect
import logging
import time
from functools import wraps
from types import FunctionType
from typing import Callable, Optional, ParamSpec, TypeVar, cast

from .core import create_cache_key, get_cache
from .types import CacheConfig

P = ParamSpec("P")
R = TypeVar("R")

# Sentinel value to distinguish "not in cache" from "cached None"
_CACHE_MISS = object()


def step(
    log_level: Optional[int] = None, cache: Optional[CacheConfig] = None
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """
    Decorator to track execution time and optionally cache function results.

    This decorator provides two main features:
    1. Execution time tracking with configurable logging
    2. Persistent disk-based caching with version control

    The decorator normalizes function arguments (positional and keyword) to ensure
    consistent cache keys regardless of how the function is called.

    Args:
        log_level: Optional logging level (e.g., logging.INFO, logging.DEBUG).
            If provided, logs execution time for every call. Cached results are
            marked with "(cached)" suffix. If None, no logging is performed.
        cache: Optional cache configuration for persistent result caching.
            If provided, results are cached to disk based on function arguments.
            Cache keys include the version number, allowing easy invalidation.
            Different eviction policies can be configured per function.

    Returns:
        Decorated function that logs execution time and caches results.

    Notes:
        - Execution time includes cache overhead (lookup + deserialization for hits)
        - Arguments are normalized via inspect.signature.bind() for consistent caching
        - Functions with same args in different forms hit the same cache:
          f(1, 2) and f(a=1, b=2) produce identical cache keys
        - Cache is isolated per function name and eviction policy
        - Bumping the version number invalidates old cached results

    Example:
        Basic timing without caching:

        >>> @step(log_level=logging.INFO)
        >>> def compute(x, y):
        ...     return x + y
        >>> compute(1, 2)
        # Logs: "compute completed in 0.0001 seconds"

        With caching enabled:

        >>> from kissml.types import CacheConfig, EvictionPolicy
        >>> @step(
        ...     log_level=logging.INFO,
        ...     cache=CacheConfig(version=1, eviction_policy=EvictionPolicy.NONE)
        ... )
        >>> def expensive_computation(data):
        ...     return process(data)
        >>> expensive_computation(my_data)
        # First call logs: "expensive_computation completed in 5.2341 seconds"
        >>> expensive_computation(my_data)
        # Second call logs: "expensive_computation completed in 0.0023 seconds (cached)"

        Version-based cache invalidation:

        >>> @step(cache=CacheConfig(version=2))  # Bumped from version=1
        >>> def updated_function(x):
        ...     return new_logic(x)
        # Cache miss - version 2 doesn't match version 1 cache
    """

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        # Cast to FunctionType to help type checker understand func has __name__
        func_typed = cast(FunctionType, func)

        # Get function signature once at decoration time
        sig = inspect.signature(func_typed)

        @wraps(func_typed)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            start_time = time.time()

            # Handle caching if enabled
            if cache is not None:
                # Get the cache for this function
                cache_instance = get_cache(
                    func_typed.__name__, cache.eviction_policy
                )

                # Bind arguments to normalize positional and keyword args
                bound = sig.bind(*args, **kwargs)
                bound.apply_defaults()

                # Create cache key from version + normalized arguments
                arg_hash = create_cache_key(**bound.arguments)
                cache_key = (cache.version, arg_hash)

                # Check if result is cached
                # Use sentinel to distinguish "not in cache" from "cached None"
                cached_result = cache_instance.get(
                    cache_key, default=_CACHE_MISS
                )
                if cached_result is not _CACHE_MISS:
                    execution_time = time.time() - start_time
                    if log_level is not None:
                        logging.log(
                            log_level,
                            f"{func_typed.__name__} completed in {execution_time:.4f} seconds (cached)",
                        )
                    return cached_result

            # Execute function
            result = func_typed(*args, **kwargs)
            execution_time = time.time() - start_time

            # Store result in cache if caching is enabled
            if cache is not None:
                cache_instance.set(cache_key, result)

            # Log execution time if logging is enabled
            if log_level is not None:
                logging.log(
                    log_level,
                    f"{func_typed.__name__} completed in {execution_time:.4f} seconds",
                )

            return result

        return wrapper

    return decorator
