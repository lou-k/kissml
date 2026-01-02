import importlib
import os
from pathlib import Path

from diskcache import UNKNOWN, Disk
from diskcache.core import MODE_BINARY

from kissml.settings import settings


def _type_to_str(t: type) -> str:
    """Convert a type to a fully-qualified string representation."""
    return f"{t.__module__}.{t.__qualname__}"


def _str_to_type(type_str: str) -> type | None:
    """
    Convert a fully-qualified type string back to a type object.

    Args:
        type_str: String in format "module.name.ClassName"

    Returns:
        The type object, or None if the type cannot be imported.
    """
    if not isinstance(type_str, str):
        return None

    module_name, _, qualname = type_str.rpartition(".")
    try:
        module = importlib.import_module(module_name)

        obj = module
        for attr in qualname.split("."):
            obj = getattr(obj, attr)

        return obj  # ty:ignore[invalid-return-type]
    except (ValueError, ImportError, AttributeError):
        return None


class TypeRoutingDisk(Disk):
    """
    Custom DiskCache Disk implementation that routes values to type-specific serializers.

    This class extends DiskCache's Disk to support pluggable serialization strategies
    based on value type. Types registered in settings.serialize_by_type use their
    custom serializers (e.g., Parquet for DataFrames), while other types fall back
    to DiskCache's default pickle serialization.

    The type information is stored in the cache database's value column as a
    fully-qualified type string (e.g., "pandas.core.frame.DataFrame"), allowing
    the correct deserializer to be selected during fetch operations.

    Example:
        >>> from kissml.settings import settings
        >>> from kissml.serializers import PandasSerializer
        >>> settings.serialize_by_type[pd.DataFrame] = PandasSerializer()
        >>> # Now all DataFrames will be cached as Parquet files
    """

    def store(self, value, read, key=UNKNOWN):
        value_type = type(value)

        # Use any registered custom serializers
        if value_type in settings.serialize_by_type:
            serializer = settings.serialize_by_type[value_type]

            # Create a filename using diskcache's existing logic
            filename, full_path = self.filename(value=value)

            # Ensure directory exists
            Path(full_path).parent.mkdir(parents=True, exist_ok=True)

            # Save the value
            serializer.serialize(value, path=Path(full_path))

            # Compute the size on disk
            file_size = os.path.getsize(full_path)

            # Return (size, mode, filename, value) tuple for Cache table
            # For `value`, we'll use the type of the value so we can lookup later
            # We'll use `MODE_BINARY` for all serializers
            return (file_size, MODE_BINARY, filename, _type_to_str(value_type))
        else:
            # Fallback to pickle
            return super().store(value, read, key)

    def fetch(self, mode, filename, value, read):
        value_type = _str_to_type(value)
        if (
            mode == MODE_BINARY
            and value_type is not None
            and value_type in settings.serialize_by_type
        ):
            serializer = settings.serialize_by_type[value_type]
            path = Path(self._directory) / filename
            return serializer.deserialize(path)
        else:
            return super().fetch(mode, filename, value, read)
