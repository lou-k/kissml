import inspect
import logging
import time
from functools import wraps
from types import FunctionType
from typing import Callable, Optional, ParamSpec, TypeVar, cast, get_type_hints

from .core import create_cache_key, get_cache
from .types import AfterEffect, CacheConfig

P = ParamSpec("P")
R = TypeVar("R")

# Sentinel value to distinguish "not in cache" from "cached None"
_CACHE_MISS = object()


def step(
    log_level: Optional[int] = None,
    cache: Optional[CacheConfig] = None,
    error_on_affect_failure: bool = False,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """
    Decorator for machine learning pipeline steps.

    This decorator provides the following features:
    1. Execution time tracking with configurable logging
    2. Persistent disk-based caching with version control
    3. Executes any AfterEffects declared in the function's return type annotation

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
        error_on_affect_failure: If True, AfterEffect failures raise exceptions.
            If False (default), AfterEffect errors are logged but don't stop execution.

    Returns:
        Decorated function that logs execution time and caches results.

    Notes:
        - Execution time includes cache overhead (lookup + deserialization for hits)
        - Arguments are normalized via inspect.signature.bind() for consistent caching
        - Functions with same args in different forms hit the same cache:
          f(1, 2) and f(a=1, b=2) produce identical cache keys
        - Cache is isolated per function name and eviction policy
        - Bumping the version number invalidates old cached results

    Examples:
        Basic timing without caching:

        >>> from kissml import step
        >>> import logging
        >>> @step(log_level=logging.INFO)
        ... def compute(x, y):
        ...     return x + y
        >>> compute(1, 2)
        # Logs: "compute completed in 0.0001 seconds"

        With caching enabled:

        >>> from kissml import step, CacheConfig, EvictionPolicy
        >>> @step(
        ...     log_level=logging.INFO,
        ...     cache=CacheConfig(version=1, eviction_policy=EvictionPolicy.NONE)
        ... )
        ... def expensive_computation(data):
        ...     return process(data)
        >>> expensive_computation(my_data)
        # First call logs: "expensive_computation completed in 5.2341 seconds"
        >>> expensive_computation(my_data)
        # Second call logs: "expensive_computation completed in 0.0023 seconds (cached)"

        Version-based cache invalidation:

        >>> from kissml import step, CacheConfig
        >>> @step(cache=CacheConfig(version=2))  # Bumped from version=1
        ... def updated_function(x):
        ...     return new_logic(x)
        # Cache miss - version 2 doesn't match version 1 cache

        Using AfterEffects for automatic visualization:

        >>> from typing import Annotated
        >>> import mlflow
        >>> from kissml import step, AfterEffect, CacheConfig
        >>>
        >>> class HTMLVisualizer(AfterEffect):
        ...     def __call__(self, result, was_cached, func_name, execution_time):
        ...         result.head(100).to_html(f"{func_name}.html")
        ...         mlflow.log_artifact(f"{func_name}.html")
        >>>
        >>> @step(cache=CacheConfig(version=1))
        ... def load_data() -> Annotated[pd.DataFrame, HTMLVisualizer()]:
        ...     return pd.read_csv("data.csv")
        # AfterEffect runs automatically on both cached and fresh results
    """

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        # Cast to FunctionType to help type checker understand func has __name__
        func_typed = cast(FunctionType, func)

        # Get function signature once at decoration time
        sig = inspect.signature(func_typed)

        @wraps(func_typed)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            start_time = time.time()
            was_cached = False

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
                    result = cached_result
                    was_cached = True
                    execution_time = time.time() - start_time
                    if log_level is not None:
                        logging.log(
                            log_level,
                            f"{func_typed.__name__} completed in {execution_time:.4f} seconds (cached)",
                        )
                else:
                    # Execute function if not cached
                    result = func_typed(*args, **kwargs)
                    execution_time = time.time() - start_time
                    cache_instance.set(cache_key, result)
                    if log_level is not None:
                        logging.log(
                            log_level,
                            f"{func_typed.__name__} completed in {execution_time:.4f} seconds",
                        )
            else:
                # Execute function without caching
                result = func_typed(*args, **kwargs)
                execution_time = time.time() - start_time

                # Log execution time if logging is enabled
                if log_level is not None:
                    logging.log(
                        log_level,
                        f"{func_typed.__name__} completed in {execution_time:.4f} seconds",
                    )

            # Execute AfterEffects from type annotations
            # Lazily resolve and cache type hints to avoid repeated work on each call
            if not hasattr(wrapper, "_type_hints"):
                wrapper._type_hints = get_type_hints(func_typed, include_extras=True)
            hints = cast(dict, wrapper._type_hints)
            if "return" in hints and hasattr(hints["return"], "__metadata__"):
                # Process effects left-to-right
                for effect in hints["return"].__metadata__:
                    if isinstance(effect, AfterEffect):
                        if error_on_affect_failure:
                            effect(
                                result,
                                was_cached,
                                func_typed.__name__,
                                execution_time,
                            )
                        else:
                            try:
                                effect(
                                    result,
                                    was_cached,
                                    func_typed.__name__,
                                    execution_time,
                                )
                            except Exception as e:
                                logging.error(
                                    f"AfterEffect {effect.__class__.__name__} failed for {func_typed.__name__}: {e}"
                                )
            return result

        return wrapper

    return decorator
