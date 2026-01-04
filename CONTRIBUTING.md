# Contributing to kissml

Thank you for your interest in contributing to kissml! This document provides guidelines and instructions for setting up your development environment and contributing to the project.

## Development Setup

### Prerequisites

- Python 3.12 or higher
- [uv](https://github.com/astral-sh/uv) for dependency management

### Getting Started

1. **Clone the repository**
   ```bash
   git clone https://github.com/lou-k/kissml.git
   cd kissml
   ```

2. **Install dependencies**
   ```bash
   uv sync
   ```

   This installs all dependencies including dev dependencies (pytest, ruff, coverage, etc.) and installs the package in editable mode.

3. **Activate the virtual environment**
   ```bash
   source .venv/bin/activate
   ```

4. **Install pre-commit hooks**
   ```bash
   pre-commit install
   ```

   This sets up automatic code formatting and linting on every commit.

## Development Workflow

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_step.py

# Run specific test
pytest tests/test_step.py::test_caching_enabled

# Run tests with coverage
coverage run -m pytest
coverage report
coverage html  # Generates htmlcov/ directory

# Run tests with specific markers
pytest -m "not slow"
pytest -m network
```

### Code Quality

The project uses several tools to maintain code quality:

**Formatting and Linting**
```bash
# Format code with ruff
ruff format kissml/ tests/

# Lint and auto-fix issues
ruff check kissml/ tests/ --fix

# Sort imports with isort
isort kissml/ tests/
```

**Type Checking**
```bash
# Type check with ty (not mypy)
ty check kissml --ignore unresolved-import
```

**Pre-commit Hooks**

Pre-commit hooks run automatically on `git commit`. To run manually:
```bash
# Run on all files
pre-commit run --all-files

# Run on staged files only
pre-commit run
```

The hooks include:
- `ruff-check`: Linting with auto-fix
- `ruff-format`: Code formatting
- `isort`: Import sorting
- `ty`: Type checking

### Code Style

- **Line length**: 79 characters
- **Python version**: 3.12+
- **Type hints**: Required for all public functions
- **Docstrings**: Use Google-style docstrings for public APIs
- **Imports**: Sorted with isort (multi-line mode 3)

Example:
```python
def my_function(arg1: str, arg2: int) -> bool:
    """
    Brief description of what the function does.

    Longer description if needed, explaining the purpose and behavior
    in more detail.

    Args:
        arg1: Description of arg1
        arg2: Description of arg2

    Returns:
        Description of return value

    Raises:
        ValueError: Description of when this is raised
    """
    pass
```

## Project Structure

```
kissml/
├── kissml/              # Main package]
├── tests/               # Test files
│   ├── test_step.py     # Tests for @step decorator
│   └── test_pandas.py   # Tests for pandas integration
├── pyproject.toml       # Project metadata and tool configs
└── README.md            # User documentation
```

## Adding Features

### Adding a New Serializer

1. **Create the serializer class** in `kissml/serializers.py`:
   ```python
   class MyTypeSerializer(Serializer):
       def serialize(self, value: MyType, out: BinaryIO) -> None:
           # Serialization logic
           pass

       def deserialize(self, input: BinaryIO) -> MyType:
           # Deserialization logic
           pass
   ```

2. **Register the serializer** in `kissml/settings.py`:
   ```python
   def _default_serializer_by_type() -> dict[type, Serializer]:
       rv: dict[type, Serializer] = {
           # ... existing serializers
           MyType: MyTypeSerializer(),
       }
       return rv
   ```

3. **Add hash function** (if needed) in `kissml/settings.py`:
   ```python
   def _default_hash_by_type() -> dict[type, Callable[[Any], str]]:
       rv: dict[type, Callable[[Any], str]] = {
           # ... existing hash functions
           MyType: lambda obj: str(hash(obj)),
       }
       return rv
   ```

4. **Write tests** in a new test file or existing ones:
   ```python
   def test_mytype_serialization():
       @step(cache=CacheConfig(version=0))
       def return_mytype() -> MyType:
           return MyType(...)

       result1 = return_mytype()
       result2 = return_mytype()  # Should hit cache
       assert result1 == result2
   ```

### Testing Guidelines

- **Use fixtures**: The `clean_cache` fixture ensures tests are isolated
- **Test cache hits and misses**: Count function executions to verify caching
- **Test round-tripping**: Ensure serialized data deserializes correctly
- **Test edge cases**: None values, empty collections, exceptions, etc.
- **Use markers**: Add `@pytest.mark.slow` for slow tests, `@pytest.mark.network` for tests requiring network

Example test:
```python
def test_new_feature():
    """Test that new feature works correctly."""
    call_count = 0

    @step(cache=CacheConfig(version=0))
    def my_function(x: int) -> int:
        nonlocal call_count
        call_count += 1
        return x * 2

    # First call - cache miss
    result1 = my_function(5)
    assert result1 == 10
    assert call_count == 1

    # Second call - cache hit
    result2 = my_function(5)
    assert result2 == 10
    assert call_count == 1  # Function not executed again
```

## Pull Request Process

1. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Write code following the style guidelines
   - Add tests for new functionality
   - Update documentation if needed

3. **Run tests and checks**
   ```bash
   # Run all tests
   pytest

   # Check code quality
   pre-commit run --all-files
   ```

4. **Commit your changes**

   **IMPORTANT**: This project uses [Conventional Commits](https://www.conventionalcommits.org/) for automatic semantic versioning. Your commit messages **must** follow this format:

   ```
   <type>: <description>

   [optional body]
   ```

   **Commit types**:
   - `fix:` - Bug fixes (triggers patch version bump: 0.0.x)
   - `feat:` - New features (triggers minor version bump: 0.x.0)
   - `feat!:` or `BREAKING CHANGE:` - Breaking changes (triggers major version bump: x.0.0)
   - `docs:` - Documentation only changes (no version bump)
   - `chore:` - Maintenance tasks (no version bump)
   - `ci:` - CI/CD changes (no version bump)
   - `test:` - Adding or updating tests (no version bump)
   - `refactor:` - Code refactoring (no version bump)

   **Examples**:
   ```bash
   git commit -m "feat: add numpy array serializer"
   git commit -m "fix: handle edge case in cache key generation"
   git commit -m "docs: update README with pandas examples"
   git commit -m "feat!: change step decorator API for better type safety"
   ```

   Pre-commit hooks will run automatically. Fix any issues before proceeding.

5. **Push and create PR**
   ```bash
   git push origin feature/your-feature-name
   ```

   Then create a pull request on GitHub.

## Questions or Issues?

- **Bug reports**: Open an issue on GitHub with a minimal reproduction case
- **Feature requests**: Open an issue describing the use case and proposed solution
- **Questions**: Open a discussion or issue for clarification

## License

See [LICENSE](LICENSE)