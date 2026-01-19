import tempfile
from pathlib import Path

import pytest

from kissml.core import close_all_caches
from kissml.settings import settings
from kissml.step import step
from kissml.types import CacheConfig, EvictionPolicy


@pytest.fixture(autouse=True)
def clean_cache():
    """Clean up cache before and after each test."""
    # Setup: use temporary directory for tests
    with tempfile.TemporaryDirectory() as tmpdir:
        original_cache_dir = settings.cache_directory
        settings.cache_directory = Path(tmpdir)

        yield

        # Teardown: close all caches and restore original directory
        close_all_caches()
        settings.cache_directory = original_cache_dir


def test_no_caching():
    """Test that function executes normally when caching is disabled."""
    call_count = 0

    @step()
    def add(x: int, y: int) -> int:
        nonlocal call_count
        call_count += 1
        return x + y

    # First call
    result1 = add(1, 2)
    assert result1 == 3
    assert call_count == 1

    # Second call with same args - should execute again (no caching)
    result2 = add(1, 2)
    assert result2 == 3
    assert call_count == 2


def test_caching_enabled():
    """Test that caching works and second call returns cached result."""
    call_count = 0

    @step(cache=CacheConfig(version=0, eviction_policy=EvictionPolicy.NONE))
    def multiply(x: int, y: int) -> int:
        nonlocal call_count
        call_count += 1
        return x * y

    # First call - cache miss
    result1 = multiply(3, 4)
    assert result1 == 12
    assert call_count == 1

    # Second call with same args - cache hit
    result2 = multiply(3, 4)
    assert result2 == 12
    assert call_count == 1  # Function not executed again

    # Different args - cache miss
    result3 = multiply(5, 6)
    assert result3 == 30
    assert call_count == 2


def test_version_change_invalidates_cache():
    """Test that changing cache version results in cache miss."""
    call_count = 0

    def compute(x: int) -> int:
        nonlocal call_count
        call_count += 1
        return x * 2

    # Version 0
    compute_v0 = step(cache=CacheConfig(version=0))(compute)

    # First call populates cache
    result1 = compute_v0(10)
    assert result1 == 20
    assert call_count == 1

    # Second call hits cache
    result2 = compute_v0(10)
    assert result2 == 20
    assert call_count == 1

    # Reset call count and create version 1
    call_count = 0
    compute_v1 = step(cache=CacheConfig(version=1))(compute)

    # Same args but different version - cache miss
    result3 = compute_v1(10)
    assert result3 == 20
    assert call_count == 1  # Function executed again


def test_argument_normalization():
    """Test that different calling conventions hit the same cache."""
    call_count = 0

    @step(cache=CacheConfig(version=0))
    def divide(a: int, b: int, c: int = 10) -> int:
        nonlocal call_count
        call_count += 1
        return (a + b) // c

    # Call with positional args
    result1 = divide(100, 50)
    assert result1 == 15
    assert call_count == 1

    # Call with keyword args - should hit cache
    result2 = divide(a=100, b=50)
    assert result2 == 15
    assert call_count == 1  # Cache hit

    # Call with mixed args - should hit cache
    result3 = divide(100, b=50, c=10)
    assert result3 == 15
    assert call_count == 1  # Cache hit

    # Call with explicit default - should hit cache
    result4 = divide(100, 50, 10)
    assert result4 == 15
    assert call_count == 1  # Cache hit


def test_cache_none_values():
    """Test that None return values are cached correctly."""
    call_count = 0

    @step(cache=CacheConfig(version=0))
    def returns_none(x: int) -> None:
        nonlocal call_count
        call_count += 1
        return None

    # First call
    result1 = returns_none(1)
    assert result1 is None
    assert call_count == 1

    # Second call - should hit cache even though result is None
    result2 = returns_none(1)
    assert result2 is None
    assert call_count == 1  # Cache hit


def test_cache_falsy_values():
    """Test that falsy values (0, False, empty string, empty list) are cached."""
    call_count = 0

    @step(cache=CacheConfig(version=0))
    def return_falsy(value: int):
        nonlocal call_count
        call_count += 1
        # Return different falsy values based on input
        if value == 0:
            return 0
        elif value == 1:
            return False
        elif value == 2:
            return ""
        elif value == 3:
            return []

    # Test 0
    result = return_falsy(0)
    assert result == 0
    assert call_count == 1
    result = return_falsy(0)
    assert result == 0
    assert call_count == 1  # Cache hit

    # Test False
    result = return_falsy(1)
    assert result is False
    assert call_count == 2
    result = return_falsy(1)
    assert result is False
    assert call_count == 2  # Cache hit

    # Test empty string
    result = return_falsy(2)
    assert result == ""
    assert call_count == 3
    result = return_falsy(2)
    assert result == ""
    assert call_count == 3  # Cache hit

    # Test empty list
    result = return_falsy(3)
    assert result == []
    assert call_count == 4
    result = return_falsy(3)
    assert result == []
    assert call_count == 4  # Cache hit


def test_function_with_no_arguments():
    """Test caching works for functions with no arguments."""
    call_count = 0

    @step(cache=CacheConfig(version=0))
    def no_args() -> int:
        nonlocal call_count
        call_count += 1
        return 42

    # First call
    result1 = no_args()
    assert result1 == 42
    assert call_count == 1

    # Second call - cache hit
    result2 = no_args()
    assert result2 == 42
    assert call_count == 1


def test_function_with_only_args():
    """Test caching works for functions with only *args (no kwargs)."""
    call_count = 0

    @step(cache=CacheConfig(version=0))
    def sum_all(*values: int) -> int:
        nonlocal call_count
        call_count += 1
        return sum(values)

    # First call
    result1 = sum_all(1, 2, 3)
    assert result1 == 6
    assert call_count == 1

    # Second call with same args - cache hit
    result2 = sum_all(1, 2, 3)
    assert result2 == 6
    assert call_count == 1

    # Different args - cache miss
    result3 = sum_all(4, 5)
    assert result3 == 9
    assert call_count == 2


def test_different_eviction_policies_separate_caches():
    """Test that different eviction policies create separate caches."""
    call_count = 0

    def compute(x: int) -> int:
        nonlocal call_count
        call_count += 1
        return x * 2

    # Same function, different eviction policies
    compute_none = step(
        cache=CacheConfig(eviction_policy=EvictionPolicy.NONE)
    )(compute)
    compute_lru = step(
        cache=CacheConfig(eviction_policy=EvictionPolicy.LEAST_RECENTLY_USED)
    )(compute)

    # Call with NONE policy
    result1 = compute_none(5)
    assert result1 == 10
    assert call_count == 1

    # Call with LRU policy - should not hit NONE cache
    result2 = compute_lru(5)
    assert result2 == 10
    assert call_count == 2  # Different cache, so function executed again

    # Second call with LRU - should hit LRU cache
    result3 = compute_lru(5)
    assert result3 == 10
    assert call_count == 2  # Cache hit in LRU cache


def test_namespaces_create_separate_caches():
    """Test that different namespaces create separate caches."""
    call_count = 0

    def compute(x: int) -> int:
        nonlocal call_count
        call_count += 1
        return x * 2

    # Same function name and eviction policy, different namespaces
    compute_ns1 = step(
        cache=CacheConfig(
            version=0, eviction_policy=EvictionPolicy.NONE, namespace="ns1"
        )
    )(compute)
    compute_ns2 = step(
        cache=CacheConfig(
            version=0, eviction_policy=EvictionPolicy.NONE, namespace="ns2"
        )
    )(compute)

    # Call with namespace ns1
    result1 = compute_ns1(5)
    assert result1 == 10
    assert call_count == 1

    # Call with namespace ns2 - should not hit ns1 cache
    result2 = compute_ns2(5)
    assert result2 == 10
    assert call_count == 2

    # Call again with ns1 - should hit ns1 cache
    result3 = compute_ns1(5)
    assert result3 == 10
    assert call_count == 2


def test_exceptions_not_cached():
    """Test that exceptions are not cached."""
    call_count = 0

    @step(cache=CacheConfig(version=0))
    def raises_error(x: int) -> int:
        nonlocal call_count
        call_count += 1
        if x < 0:
            raise ValueError("Negative value")
        return x * 2

    # First call raises exception
    with pytest.raises(ValueError, match="Negative value"):
        raises_error(-1)
    assert call_count == 1

    # Second call with same args - should raise again (not cached)
    with pytest.raises(ValueError, match="Negative value"):
        raises_error(-1)
    assert call_count == 2  # Exception not cached, function executed again

    # Successful call
    result = raises_error(5)
    assert result == 10
    assert call_count == 3

    # Second successful call - cache hit
    result = raises_error(5)
    assert result == 10
    assert call_count == 3


def test_close_all_caches():
    """Test that close_all_caches properly closes all cache instances."""
    from kissml.core import _caches

    # Create some cached functions
    @step(cache=CacheConfig(version=0, eviction_policy=EvictionPolicy.NONE))
    def func1(x: int) -> int:
        return x

    @step(
        cache=CacheConfig(
            version=0, eviction_policy=EvictionPolicy.LEAST_RECENTLY_USED
        )
    )
    def func2(x: int) -> int:
        return x

    # Execute to populate caches
    func1(1)
    func2(2)

    # Verify caches exist
    assert len(_caches) == 2

    # Close all caches
    close_all_caches()

    # Verify caches are cleared
    assert len(_caches) == 0
