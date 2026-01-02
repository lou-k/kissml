# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

kissml (Keep It Simple Stupid Tools for Machine Learning) is a Python library that provides a decorator-based caching and execution tracking system for ML workflows. The core functionality centers around the `@step` decorator which provides persistent disk-based caching with version control and execution time logging.

## Development Commands

### Environment Setup
This project uses `uv` for dependency management:
```bash
# Install dependencies (includes dev dependencies by default)
uv sync

# Activate virtual environment
source .venv/bin/activate
```

### Testing
```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_step.py

# Run tests with specific marker
pytest -m "not slow"

# Run tests with coverage
coverage run -m pytest
coverage report
coverage html  # Generates htmlcov/ directory
```

### Code Quality
```bash
# Format code (uses ruff)
ruff format kissml/ tests/

# Lint code
ruff check kissml/ tests/ --fix

# Type check (uses ty, not mypy)
ty check kissml --ignore unresolved-import

# Run pre-commit hooks manually
pre-commit run --all-files
```

Pre-commit hooks automatically run ruff-check, ruff-format, isort, and ty on commits.

## Architecture

### Core Components

**[kissml/step.py](kissml/step.py)** - The main `@step` decorator
- Provides execution time tracking with configurable logging
- Disk-based persistent caching with version control
- Normalizes function arguments (positional/keyword) for consistent cache keys
- Uses inspect.signature.bind() to ensure `f(1, 2)` and `f(a=1, b=2)` produce identical cache keys
- Cache keys include version number for easy invalidation

**[kissml/core.py](kissml/core.py)** - Cache management
- `create_cache_key()`: Creates deterministic cache keys from keyword arguments using type-specific hash functions
- `get_cache()`: Returns Cache instances isolated by function name and eviction policy
- `close_all_caches()`: Cleanup function for closing all cache instances
- Global `_caches` dict tracks all Cache instances

**[kissml/disk.py](kissml/disk.py)** - Type-routing disk storage
- `TypeRoutingDisk`: Custom DiskCache Disk implementation that routes to type-specific serializers
- `store()`: Serializes values using registered serializers, stores type info as fully-qualified string in DB
- `fetch()`: Deserializes values by looking up type string and using appropriate deserializer
- Falls back to pickle for types without custom serializers

**[kissml/serializers.py](kissml/serializers.py)** - Custom serializers
- `PandasSerializer`: Uses Parquet format for DataFrames (requires pyarrow)
- `ListSerializer`, `TupleSerializer`, `DictSerializer`: Handle nested collections with type manifests
- All use length-prefixed format with pickled type manifests for heterogeneous containers
- Collection serializers recursively handle elements with custom serializers

**[kissml/settings.py](kissml/settings.py)** - Global configuration
- `settings` singleton (Pydantic BaseSettings) with environment variable support (prefix: `KISSML_`)
- `cache_directory`: Defaults to `~/.kissml`
- `hash_by_type`: Maps types to hash functions (e.g., pandas DataFrames use `pd.util.hash_pandas_object`)
- `serialize_by_type`: Maps types to Serializer instances
- Auto-registers pandas types if available

**[kissml/types.py](kissml/types.py)** - Type definitions
- `EvictionPolicy` enum: NONE, LEAST_RECENTLY_STORED, LEAST_RECENTLY_USED, LEAST_FREQUENTLY_USED
- `CacheConfig` Pydantic model: version + eviction_policy
- `Serializer` ABC: Base class for custom serializers

### Key Design Patterns

**Pluggable Serialization**: The `settings.serialize_by_type` dict allows registering custom serializers for any type. When a value is cached, `TypeRoutingDisk.store()` checks if the type has a registered serializer and uses it; otherwise falls back to pickle.

**Cache Isolation**: Caches are isolated by `(function_name, eviction_policy)` tuples. Each unique combination gets its own Cache instance and directory under `~/.kissml/<function_name>/<eviction_policy>/`.

**Version-Based Invalidation**: Cache keys are `(version, arg_hash)` tuples. Bumping the version number in `CacheConfig` invalidates all cached results for that function without manual cleanup.

**Argument Normalization**: The `@step` decorator uses `inspect.signature.bind()` to normalize all function arguments before hashing. This ensures different calling conventions (positional, keyword, mixed) produce identical cache keys for the same logical arguments.

**Nested Collection Support**: List/tuple/dict serializers use a manifest-based approach:
1. Pickle a manifest of types for each element
2. Serialize each element (using custom serializer if registered, otherwise pickle)
3. Write length-prefixed bytes for each element
4. On deserialization, read manifest to know which deserializer to use per element

## Testing Patterns

- All tests use a `clean_cache` fixture that creates a temporary cache directory per test
- Tests verify both cache hits and cache misses by counting function executions
- Pandas tests use `pd.testing.assert_frame_equal()` to verify DataFrame round-tripping
- Exception tests verify exceptions are NOT cached (call count increases on retry)

## Important Notes

- The type checker is `ty`, NOT `mypy`. Use `ty check kissml --ignore unresolved-import` for type checking.
- Line length is 79 characters (configured in pyproject.toml).
- The project requires Python 3.12+.
- When adding new serializers, register them in `settings._default_serializer_by_type()` and add corresponding hash functions to `settings._default_hash_by_type()` if needed.
- The cache format stores type information in the DiskCache database's value column as fully-qualified strings (e.g., "pandas.core.frame.DataFrame"), not as pickled objects.
