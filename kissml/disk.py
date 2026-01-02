import importlib
import json
import os
import pickle
from pathlib import Path
from typing import Any

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

    Supports element-wise serialization for composite types (tuples, lists, sets, dicts)
    containing custom-serializable types.

    The type information is stored in the cache database's value column as a
    fully-qualified type string (e.g., "pandas.core.frame.DataFrame"), allowing
    the correct deserializer to be selected during fetch operations.

    Example:
        >>> from kissml.settings import settings
        >>> from kissml.serializers import PandasSerializer
        >>> settings.serialize_by_type[pd.DataFrame] = PandasSerializer()
        >>> # Now all DataFrames will be cached as Parquet files
    """

    def _has_custom_types(self, value) -> bool:
        """Check if a value contains any types registered for custom serialization."""
        if isinstance(value, (tuple, list, set)):
            return any(
                type(elem) in settings.serialize_by_type for elem in value
            )
        elif isinstance(value, dict):
            return any(
                type(k) in settings.serialize_by_type
                or type(v) in settings.serialize_by_type
                for k, v in value.items()
            )
        return False

    def _deserialize_element(self, elem_meta: dict) -> Any:
        """
        Deserialize a single element using custom deserializer or pickle.

        Args:
            elem_meta: Metadata dict containing 'type', 'filename', 'custom' keys

        Returns:
            The deserialized element
        """
        elem_path = Path(self._directory) / elem_meta["filename"]

        if elem_meta["custom"]:
            # Use custom deserializer
            type_obj = _str_to_type(elem_meta["type"])
            if type_obj is not None and type_obj in settings.serialize_by_type:
                serializer = settings.serialize_by_type[type_obj]
                return serializer.deserialize(elem_path)
            else:
                # Fallback to pickle if type can't be found
                with open(elem_path, "rb") as f:
                    return pickle.load(f)
        else:
            # Unpickle
            with open(elem_path, "rb") as f:
                return pickle.load(f)

    def _store_sequence(self, value, read, key, marker: str):
        """
        Store a sequence (tuple, list, or set) with element-wise serialization.

        Args:
            value: The sequence to store
            read: DiskCache read parameter
            key: DiskCache key parameter
            marker: Marker string for this type (e.g., '__tuple__', '__list__', '__set__')

        Returns:
            Tuple of (size, mode, filename, marker)
        """
        # Serialize each element
        element_metadata = []
        total_size = 0

        for i, elem in enumerate(value):
            # Generate a fresh filename for each element
            elem_filename, elem_full_path = self.filename(key=key)
            elem_path = Path(elem_full_path)

            # Ensure parent directory exists
            elem_path.parent.mkdir(parents=True, exist_ok=True)

            elem_type = type(elem)
            type_str = _type_to_str(elem_type)

            if elem_type in settings.serialize_by_type:
                # Use custom serializer
                serializer = settings.serialize_by_type[elem_type]
                serializer.serialize(elem, elem_path)
                elem_size = os.path.getsize(elem_path)
                element_metadata.append(
                    {
                        "type": type_str,
                        "filename": elem_filename,
                        "custom": True,
                    }
                )
            else:
                # Pickle the element
                with open(elem_path, "wb") as f:
                    pickle.dump(elem, f)
                elem_size = os.path.getsize(elem_path)
                element_metadata.append(
                    {
                        "type": type_str,
                        "filename": elem_filename,
                        "custom": False,
                    }
                )

            total_size += elem_size

        # Store metadata manifest as JSON
        manifest = {"elements": element_metadata, "length": len(value)}
        manifest_filename, manifest_full_path = self.filename(key=key)
        manifest_path = Path(manifest_full_path)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)

        with open(manifest_path, "w") as f:
            json.dump(manifest, f)

        total_size += os.path.getsize(manifest_path)

        return (total_size, MODE_BINARY, manifest_filename, marker)

    def _fetch_sequence(self, manifest_filename: str, target_type: type):
        """
        Reconstruct a sequence (tuple, list, or set) from element-wise storage.

        Args:
            manifest_filename: Name of the manifest file
            target_type: Type to reconstruct (tuple, list, or set)

        Returns:
            The reconstructed sequence
        """
        manifest_path = Path(self._directory) / manifest_filename

        with open(manifest_path, "r") as f:
            manifest = json.load(f)

        elements = []
        for elem_meta in manifest["elements"]:
            elem = self._deserialize_element(elem_meta)
            elements.append(elem)

        return target_type(elements)

    def _store_dict(self, value: dict, read, key):
        """
        Store a dict with element-wise serialization for keys and values.

        Args:
            value: The dict to store
            read: DiskCache read parameter
            key: DiskCache key parameter

        Returns:
            Tuple of (size, mode, filename, '__dict__')
        """
        # Serialize each key-value pair
        pairs_metadata = []
        total_size = 0

        for k, v in value.items():
            # Serialize key
            key_filename, key_full_path = self.filename(key=key)
            key_path = Path(key_full_path)
            key_path.parent.mkdir(parents=True, exist_ok=True)

            key_type = type(k)
            key_type_str = _type_to_str(key_type)

            if key_type in settings.serialize_by_type:
                serializer = settings.serialize_by_type[key_type]
                serializer.serialize(k, key_path)
                key_size = os.path.getsize(key_path)
                key_meta = {
                    "type": key_type_str,
                    "filename": key_filename,
                    "custom": True,
                }
            else:
                with open(key_path, "wb") as f:
                    pickle.dump(k, f)
                key_size = os.path.getsize(key_path)
                key_meta = {
                    "type": key_type_str,
                    "filename": key_filename,
                    "custom": False,
                }

            total_size += key_size

            # Serialize value
            val_filename, val_full_path = self.filename(key=key)
            val_path = Path(val_full_path)
            val_path.parent.mkdir(parents=True, exist_ok=True)

            val_type = type(v)
            val_type_str = _type_to_str(val_type)

            if val_type in settings.serialize_by_type:
                serializer = settings.serialize_by_type[val_type]
                serializer.serialize(v, val_path)
                val_size = os.path.getsize(val_path)
                val_meta = {
                    "type": val_type_str,
                    "filename": val_filename,
                    "custom": True,
                }
            else:
                with open(val_path, "wb") as f:
                    pickle.dump(v, f)
                val_size = os.path.getsize(val_path)
                val_meta = {
                    "type": val_type_str,
                    "filename": val_filename,
                    "custom": False,
                }

            total_size += val_size

            pairs_metadata.append({"key": key_meta, "value": val_meta})

        # Store metadata manifest as JSON
        manifest = {"pairs": pairs_metadata, "length": len(value)}
        manifest_filename, manifest_full_path = self.filename(key=key)
        manifest_path = Path(manifest_full_path)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)

        with open(manifest_path, "w") as f:
            json.dump(manifest, f)

        total_size += os.path.getsize(manifest_path)

        return (total_size, MODE_BINARY, manifest_filename, "__dict__")

    def _fetch_dict(self, manifest_filename: str):
        """
        Reconstruct a dict from element-wise storage.

        Args:
            manifest_filename: Name of the manifest file

        Returns:
            The reconstructed dict
        """
        manifest_path = Path(self._directory) / manifest_filename

        with open(manifest_path, "r") as f:
            manifest = json.load(f)

        result = {}
        for pair_meta in manifest["pairs"]:
            key = self._deserialize_element(pair_meta["key"])
            value = self._deserialize_element(pair_meta["value"])
            result[key] = value

        return result

    def store(self, value, read, key=UNKNOWN):
        value_type = type(value)

        # Handle tuples with custom-serializable elements
        if isinstance(value, tuple) and not isinstance(value, type):
            if self._has_custom_types(value):
                return self._store_sequence(value, read, key, "__tuple__")

        # Handle lists with custom-serializable elements
        if isinstance(value, list):
            if self._has_custom_types(value):
                return self._store_sequence(value, read, key, "__list__")

        # Handle sets with custom-serializable elements
        if isinstance(value, set):
            if self._has_custom_types(value):
                return self._store_sequence(value, read, key, "__set__")

        # Handle dicts with custom-serializable keys or values
        if isinstance(value, dict):
            if self._has_custom_types(value):
                return self._store_dict(value, read, key)

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
        # Check for composite type markers
        if value == "__tuple__":
            return self._fetch_sequence(filename, tuple)
        elif value == "__list__":
            return self._fetch_sequence(filename, list)
        elif value == "__set__":
            return self._fetch_sequence(filename, set)
        elif value == "__dict__":
            return self._fetch_dict(filename)

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
