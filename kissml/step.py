import inspect
import logging
import time
from collections.abc import Callable, Collection, Iterator, Mapping
from functools import wraps
from itertools import repeat
from types import FunctionType
from typing import (
    Any,
    ParamSpec,
    TypeVar,
    cast,
    get_args,
    get_origin,
    get_type_hints,
)

from .core import create_cache_key, get_cache
from .settings import settings
from .tags import _TAGS
from .types import AfterEffect, CacheConfig, Tags

P = ParamSpec("P")
R = TypeVar("R")

# Sentinel value to distinguish "not in cache" from "cached None"
_CACHE_MISS = object()


def _has_effects(annotation: Any) -> bool:
    """True if an AfterEffect appears anywhere in the annotation tree.

    Deliberately over-approximate: may return True for shapes
    _iter_effects declines to descend into (e.g. union arms). Only used
    to skip walking values that can never yield an effect.
    """
    metadata = getattr(annotation, "__metadata__", ())
    if any(isinstance(meta, AfterEffect) for meta in metadata):
        return True
    return any(map(_has_effects, get_args(annotation)))


def _iter_effects(
    annotation: Any, value: Any
) -> Iterator[tuple[AfterEffect, Any]]:
    """Yield (effect, sub_value) for every AfterEffect in the annotation
    tree, pairing each effect with the part of value it annotates.

    Descends into container annotations (tuple, list, dict, set, ...)
    when the runtime value is a matching Collection, so effects nested
    at any depth fire against their corresponding element. Only
    Collection annotations are entered, which excludes union arms,
    iterators, and generators (iterating those would consume them).
    Dict keys are not descended into, only values.
    """
    if hasattr(annotation, "__metadata__"):
        for meta in annotation.__metadata__:
            if isinstance(meta, AfterEffect):
                yield meta, value
        annotation = annotation.__origin__
    origin, args = get_origin(annotation), get_args(annotation)
    if not (
        args
        and isinstance(origin, type)
        and issubclass(origin, Collection)
        and isinstance(value, Collection)
    ):
        return
    if isinstance(value, Mapping):
        # Pair dict[K, V]'s value type with each mapping value
        args, value = args[-1:], value.values()
    if args[-1] is Ellipsis or len(args) == 1:
        # Homogeneous container: one element type applies to every element
        args = repeat(args[0])
    for sub_annotation, sub_value in zip(args, value):
        yield from _iter_effects(sub_annotation, sub_value)


def step(
    log_level: int | None = None,
    cache: CacheConfig | None = None,
    error_on_effect_failure: bool = False,
    tags: Tags | None = None,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """
    Decorator for machine learning pipeline steps.

    This decorator provides the following features:
    1. Execution time tracking with configurable logging
    2. Persistent disk-based caching with version control
    3. Executes any AfterEffects declared in the function's return type
       annotation, including effects nested in container annotations
       (e.g. on a tuple element's type), which receive the matching
       element of the result rather than the full return value

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
        error_on_effect_failure: If True, AfterEffect failures raise exceptions.
            If False (default), AfterEffect errors are logged but don't stop execution.
        tags: Tags passed to AfterEffects, merged over ambient ``kissml.tags``
            (step wins). Never part of the cache key.

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
        ...     def __call__(self, result, was_cached, func_name, execution_time, tags):
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

        # Return annotation, resolved lazily on first call; None when it
        # declares no AfterEffects so later calls skip walking the result
        return_annotation: Any = _CACHE_MISS
        step_tags = tags or {}

        @wraps(func_typed)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            start_time = time.time()
            was_cached = False

            # Handle caching if enabled
            if cache is not None:
                # Get the cache for this function
                cache_instance = get_cache(
                    func_typed.__name__, cache.eviction_policy, cache.namespace
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

            # Read at call time: a cache hit sees the tags in force now
            effect_tags = {**_TAGS.get(), **step_tags}

            def _run_effect(effect: AfterEffect, value: Any) -> None:
                try:
                    effect(
                        value,
                        was_cached,
                        func_typed.__name__,
                        execution_time,
                        effect_tags,
                    )
                except Exception as e:
                    if error_on_effect_failure:
                        raise
                    logging.error(
                        f"AfterEffect {effect.__class__.__name__} failed for {func_typed.__name__}: {e}"
                    )

            # Execute AfterEffects from type annotations
            nonlocal return_annotation
            if return_annotation is _CACHE_MISS:
                hints = get_type_hints(func_typed, include_extras=True)
                return_annotation = hints.get("return")
                if not _has_effects(return_annotation):
                    return_annotation = None
            if return_annotation is not None:
                # Outer effects run before nested ones, left-to-right
                for effect, value in _iter_effects(return_annotation, result):
                    _run_effect(effect, value)

            for effect in settings.global_after_effects:
                _run_effect(effect, result)

            return result

        wrapper.__kissml_kind__ = "step"  # type: ignore[attr-defined]

        return wrapper

    return decorator


def subpipeline(
    log_level: int | None = None,
    error_on_effect_failure: bool = False,
    tags: Tags | None = None,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """
    Decorator for composing steps into an uncached pipeline.

    A subpipeline is an uncached composition of steps: it calls other
    (typically cached) `@step` functions and returns their results,
    performing no computation of its own. It never caches, because
    caching a composition would skip its body on a cache hit and, with
    it, the calls to the inner steps -- silently suppressing their own
    caching, logging, and AfterEffects.

    Because a subpipeline's body always runs (nothing is ever cached),
    any AfterEffects declared in its `Annotated` return type are
    guaranteed to fire on every call. This is a stronger guarantee than
    a plain, undecorated function gets: `Annotated` metadata on a bare
    function's return type is inert -- Python attaches no runtime
    behavior to it, so those effects would never execute.

    This decorator wraps `step(cache=None, ...)` and additionally stamps
    `__kissml_kind__ = "subpipeline"` on the returned function (as
    opposed to `__kissml_kind__ = "step"`, which `step()` stamps on its
    own wrapper), so tooling can distinguish the two by introspection.

    Args:
        log_level: Optional logging level (e.g., logging.INFO, logging.DEBUG).
            If provided, logs execution time for every call. If None, no
            logging is performed.
        error_on_effect_failure: If True, AfterEffect failures raise exceptions.
            If False (default), AfterEffect errors are logged but don't stop
            execution.
        tags: Tags passed to AfterEffects; see ``step``.

    Returns:
        Decorated function that always executes its body and runs any
        declared AfterEffects, with no caching support.

    Notes:
        - There is no `cache` parameter: caching is deliberately
          unsupported so the composition's body -- and the inner steps
          it calls -- always run.
        - `__kissml_kind__` is set to `"subpipeline"` on the wrapper,
          letting tooling classify decorated functions without
          inspecting their implementation.

    Examples:
        Composing two cached steps, with an AfterEffect on the result:

        >>> from typing import Annotated
        >>> from kissml import step, subpipeline, AfterEffect, CacheConfig
        >>>
        >>> class RowCountLogger(AfterEffect):
        ...     def __call__(self, result, was_cached, func_name, execution_time, tags):
        ...         print(f"{func_name}: {len(result)} rows")
        >>>
        >>> @step(cache=CacheConfig(version=1))
        ... def load_data() -> pd.DataFrame:
        ...     return pd.read_csv("data.csv")
        >>>
        >>> @step(cache=CacheConfig(version=1))
        ... def clean_data(df: pd.DataFrame) -> pd.DataFrame:
        ...     return df.dropna()
        >>>
        >>> @subpipeline()
        ... def prepare_data() -> Annotated[pd.DataFrame, RowCountLogger()]:
        ...     return clean_data(load_data())
        >>> prepare_data()
        # load_data/clean_data may each hit cache, but prepare_data's own
        # body -- and RowCountLogger -- run on every call.
    """
    base = step(
        log_level=log_level,
        cache=None,  # Subpipelines do not support caching
        error_on_effect_failure=error_on_effect_failure,
        tags=tags,
    )

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        wrapper = base(func)
        wrapper.__kissml_kind__ = "subpipeline"  # type: ignore[attr-defined]
        return wrapper

    return decorator
