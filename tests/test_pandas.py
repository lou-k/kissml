import tempfile
from pathlib import Path

import pandas as pd
import pytest

from kissml.core import close_all_caches
from kissml.settings import settings
from kissml.step import step
from kissml.types import CacheConfig


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


def test_dataframe_as_cache_key():
    """Test that identical DataFrames produce cache hits."""
    call_count = 0

    @step(cache=CacheConfig(version=0))
    def process_dataframe(df: pd.DataFrame) -> int:
        nonlocal call_count
        call_count += 1
        return len(df)

    # Create first DataFrame
    df1 = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

    # First call - cache miss
    result1 = process_dataframe(df1)
    assert result1 == 3
    assert call_count == 1

    # Create second DataFrame with identical content
    df2 = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

    # Second call with identical DataFrame - should hit cache
    result2 = process_dataframe(df2)
    assert result2 == 3
    assert call_count == 1  # Cache hit - function not executed again

    # Different DataFrame - should miss cache
    df3 = pd.DataFrame({"a": [1, 2, 3, 4], "b": [4, 5, 6, 7]})
    result3 = process_dataframe(df3)
    assert result3 == 4
    assert call_count == 2  # Cache miss - function executed


def test_dataframe_as_return_value():
    """Test that DataFrames round-trip correctly through Parquet serialization."""
    call_count = 0

    @step(cache=CacheConfig(version=0))
    def create_dataframe(rows: int) -> pd.DataFrame:
        nonlocal call_count
        call_count += 1
        return pd.DataFrame(
            {"a": range(rows), "b": [x * 2 for x in range(rows)]}
        )

    # First call - cache miss
    result1 = create_dataframe(5)
    assert call_count == 1
    assert len(result1) == 5
    assert list(result1.columns) == ["a", "b"]
    assert result1["a"].tolist() == [0, 1, 2, 3, 4]
    assert result1["b"].tolist() == [0, 2, 4, 6, 8]

    # Second call - cache hit, should return identical DataFrame
    result2 = create_dataframe(5)
    assert call_count == 1  # Cache hit

    # Verify DataFrames are identical
    pd.testing.assert_frame_equal(result1, result2)


def test_mixed_dataframe_and_primitives():
    """Test caching with DataFrame arguments mixed with primitive types."""
    call_count = 0

    @step(cache=CacheConfig(version=0))
    def transform_dataframe(
        df: pd.DataFrame, multiplier: int, offset: float
    ) -> pd.DataFrame:
        nonlocal call_count
        call_count += 1
        result = df.copy()
        result["transformed"] = df["value"] * multiplier + offset
        return result

    # Create test DataFrame
    df = pd.DataFrame({"value": [1, 2, 3]})

    # First call - cache miss
    result1 = transform_dataframe(df, multiplier=2, offset=10.0)
    assert call_count == 1
    assert result1["transformed"].tolist() == [12.0, 14.0, 16.0]

    # Second call with same args - cache hit
    result2 = transform_dataframe(df, multiplier=2, offset=10.0)
    assert call_count == 1  # Cache hit
    pd.testing.assert_frame_equal(result1, result2)

    # Different multiplier - cache miss
    result3 = transform_dataframe(df, multiplier=3, offset=10.0)
    assert call_count == 2
    assert result3["transformed"].tolist() == [13.0, 16.0, 19.0]

    # Different offset - cache miss
    result4 = transform_dataframe(df, multiplier=2, offset=5.0)
    assert call_count == 3
    assert result4["transformed"].tolist() == [7.0, 9.0, 11.0]

    # Different DataFrame - cache miss
    df2 = pd.DataFrame({"value": [10, 20, 30]})
    result5 = transform_dataframe(df2, multiplier=2, offset=10.0)
    assert call_count == 4
    assert result5["transformed"].tolist() == [30.0, 50.0, 70.0]


def test_tuple_with_dataframes():
    """Test that tuples containing DataFrames serialize elements separately."""
    call_count = 0

    @step(cache=CacheConfig(version=0))
    def return_tuple(rows: int) -> tuple:
        nonlocal call_count
        call_count += 1
        df1 = pd.DataFrame({"a": range(rows)})
        df2 = pd.DataFrame({"b": range(rows, rows * 2)})
        return (df1, rows, df2, "metadata")

    # First call - cache miss
    result1 = return_tuple(3)
    assert call_count == 1
    assert len(result1) == 4
    assert isinstance(result1[0], pd.DataFrame)
    assert result1[1] == 3
    assert isinstance(result1[2], pd.DataFrame)
    assert result1[3] == "metadata"

    # Second call - cache hit
    result2 = return_tuple(3)
    assert call_count == 1  # Cache hit

    # Verify tuple elements match
    pd.testing.assert_frame_equal(result1[0], result2[0])
    assert result1[1] == result2[1]
    pd.testing.assert_frame_equal(result1[2], result2[2])
    assert result1[3] == result2[3]


def test_list_with_dataframes():
    """Test that lists containing DataFrames serialize elements separately."""
    call_count = 0

    @step(cache=CacheConfig(version=0))
    def return_list(rows: int) -> list:
        nonlocal call_count
        call_count += 1
        df1 = pd.DataFrame({"a": range(rows)})
        df2 = pd.DataFrame({"b": range(rows, rows * 2)})
        return [df1, df2]

    # First call - cache miss
    result1 = return_list(3)
    assert call_count == 1
    assert len(result1) == 2
    assert isinstance(result1[0], pd.DataFrame)
    assert isinstance(result1[1], pd.DataFrame)

    # Second call - cache hit
    result2 = return_list(3)
    assert call_count == 1  # Cache hit

    # Verify list elements match
    pd.testing.assert_frame_equal(result1[0], result2[0])
    pd.testing.assert_frame_equal(result1[1], result2[1])

    # Different rows - cache miss
    result3 = return_list(5)
    assert call_count == 2
    assert len(result3[0]) == 5
    assert len(result3[1]) == 5


def test_dict_with_dataframes():
    """Test that dicts containing DataFrames serialize key-value pairs separately."""
    call_count = 0

    @step(cache=CacheConfig(version=0))
    def return_dict(rows: int) -> dict:
        nonlocal call_count
        call_count += 1
        df1 = pd.DataFrame({"a": range(rows)})
        df2 = pd.DataFrame({"b": range(rows, rows * 2)})
        return {"first": df1, "second": df2, "count": rows}

    # First call - cache miss
    result1 = return_dict(3)
    assert call_count == 1
    assert len(result1) == 3
    assert isinstance(result1["first"], pd.DataFrame)
    assert isinstance(result1["second"], pd.DataFrame)
    assert result1["count"] == 3

    # Second call - cache hit
    result2 = return_dict(3)
    assert call_count == 1  # Cache hit

    # Verify dict values match
    pd.testing.assert_frame_equal(result1["first"], result2["first"])
    pd.testing.assert_frame_equal(result1["second"], result2["second"])
    assert result1["count"] == result2["count"]

    # Different rows - cache miss
    result3 = return_dict(5)
    assert call_count == 2
    assert len(result3["first"]) == 5
    assert result3["count"] == 5
