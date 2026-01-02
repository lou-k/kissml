# kissml

Keep It Simple Stupid Tools for Machine Learning

A Python library providing simple, powerful tools for ML workflows with minimal boilerplate.

I made this because:
* Most data science services are notebook based, but notebooks are difficult to debug
* Most frameworks (flyte, metaflow) focus on extending to the cloud. This is great, but for local iteration all we really need is reproducible pipeline steps.

## Installation

```bash
pip install kissml
```

## Steps

The `@step` decorator provides execution tracking and persistent disk-based caching for your functions.

### Basic Usage

```python
from kissml import step, CacheConfig
import logging

# Simple execution time logging
@step(log_level=logging.INFO)
def process_data(data):
    # Your processing logic here
    return result

# With persistent caching
@step(
    log_level=logging.INFO,
    cache=CacheConfig(version=1)
)
def expensive_computation(data):
    # This will only run once per unique input
    # Subsequent calls return cached results
    return result
```

### Key Features

**Execution Time Tracking**: Log how long your functions take to run
```python
@step(log_level=logging.INFO)
def train_model(X, y):
    # Logs: "train_model completed in 45.2341 seconds"
    return model
```

**Persistent Disk Caching**: Cache results to disk and reuse them across runs
```python
@step(cache=CacheConfig(version=1))
def load_and_preprocess(filepath):
    # Expensive preprocessing runs once
    # Subsequent calls load from cache in milliseconds
    return processed_data
```

**Version-Based Invalidation**: Bump the version to invalidate old cache
```python
# Old implementation
@step(cache=CacheConfig(version=1))
def feature_engineering(df):
    return old_features(df)

# Updated implementation - cache automatically invalidated
@step(cache=CacheConfig(version=2))
def feature_engineering(df):
    return new_improved_features(df)
```

**Smart Serialization**: Efficient storage for pandas DataFrames and nested collections
```python
import pandas as pd

@step(cache=CacheConfig(version=1))
def analyze_data(df: pd.DataFrame) -> pd.DataFrame:
    # DataFrames cached as Parquet files (requires pyarrow)
    # Much more efficient than pickle
    return processed_df

@step(cache=CacheConfig(version=1))
def complex_pipeline(data) -> dict:
    # Returns dict with DataFrames, lists, etc.
    # Each type uses optimal serialization
    return {
        "results": some_dataframe,
        "metrics": [metric1, metric2],
        "metadata": {"key": "value"}
    }
```

### Cache Configuration

Control cache behavior with `CacheConfig`:

```python
from kissml import step, CacheConfig, EvictionPolicy

# No eviction (default) - cache grows forever
@step(cache=CacheConfig(version=1, eviction_policy=EvictionPolicy.NONE))
def permanent_cache(x):
    return x

# Least Recently Used - evicts oldest accessed items
@step(cache=CacheConfig(version=1, eviction_policy=EvictionPolicy.LEAST_RECENTLY_USED))
def lru_cache(x):
    return x

# Least Recently Stored - evicts oldest stored items
@step(cache=CacheConfig(version=1, eviction_policy=EvictionPolicy.LEAST_RECENTLY_STORED))
def lrs_cache(x):
    return x

# Least Frequently Used - evicts least accessed items
@step(cache=CacheConfig(version=1, eviction_policy=EvictionPolicy.LEAST_FREQUENTLY_USED))
def lfu_cache(x):
    return x
```

### Configuration

Configure the cache directory via environment variable or settings:

```python
from kissml import settings
from pathlib import Path

# Set cache directory
settings.cache_directory = Path("/path/to/cache")

# Or use environment variable
# export KISSML_CACHE_DIRECTORY=/path/to/cache
```

### Custom Serialization

Register custom serializers for your types:

```python
from kissml.settings import settings
from kissml.types import Serializer
from typing import Any, BinaryIO

class MyCustomSerializer(Serializer):
    def serialize(self, value: Any, out: BinaryIO) -> None:
        # Your serialization logic
        pass

    def deserialize(self, input: BinaryIO) -> Any:
        # Your deserialization logic
        pass

# Register the serializer
settings.serialize_by_type[MyCustomType] = MyCustomSerializer()

# Register a hash function for cache keys
settings.hash_by_type[MyCustomType] = lambda obj: str(hash(obj))
```

## License

Licensed under CC BY-NC-ND 4.0 (Attribution-NonCommercial-NoDerivatives). This is a **non-commercial license** - see the [LICENSE](LICENSE) file for full details.

For commercial use, please contact the author.
