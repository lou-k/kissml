import pickle
from typing import Any, BinaryIO

from kissml.types import Serializer


class PandasSerializer(Serializer):
    """
    Serializer for pandas DataFrames using Parquet format.

    Uses pandas' built-in Parquet support (requires pyarrow or fastparquet).
    Parquet provides efficient columnar storage with compression, making it
    ideal for caching large DataFrames.

    Raises:
        ValueError: If the value is not a pandas DataFrame.
    """

    def serialize(self, value: Any, out: BinaryIO) -> None:
        import pandas as pd

        if not isinstance(value, pd.DataFrame):
            raise ValueError(
                "PandasSerializer can only serialize data frames."
            )
        value.to_parquet(out)

    def deserialize(self, input: BinaryIO) -> Any:
        import pandas as pd

        return pd.read_parquet(input)


class ListSerializer(Serializer):
    """
    Serializer for lists supporting both homogeneous and heterogeneous types.

    Uses a self-contained format:
    1. Pickled type manifest (list of types for each element)
    2. Length-prefixed serialized elements

    For elements with custom serializers, uses BytesIO buffers.
    For elements without custom serializers, falls back to pickle.
    """

    def serialize(self, value: list, out: BinaryIO) -> None:
        from io import BytesIO

        from kissml.settings import settings

        # Write manifest: list of types for each element
        manifest = [type(elem) for elem in value]
        pickle.dump(manifest, out)

        # Serialize each element
        for element in value:
            element_type = type(element)

            if element_type in settings.serialize_by_type:
                # Use custom serializer with BytesIO buffer
                serializer = settings.serialize_by_type[element_type]
                buffer = BytesIO()
                serializer.serialize(element, buffer)
                element_bytes = buffer.getvalue()
            else:
                # Fall back to pickle
                element_bytes = pickle.dumps(element)

            # Write length prefix and bytes
            length = len(element_bytes)
            out.write(length.to_bytes(8, byteorder="big"))
            out.write(element_bytes)

    def deserialize(self, input: BinaryIO) -> list:
        from io import BytesIO

        from kissml.settings import settings

        # Read manifest
        manifest = pickle.load(input)

        # Deserialize each element
        result = []
        for element_type in manifest:
            # Read length prefix
            length_bytes = input.read(8)
            length = int.from_bytes(length_bytes, byteorder="big")

            # Read element bytes
            element_bytes = input.read(length)

            if element_type in settings.serialize_by_type:
                # Use custom serializer
                serializer = settings.serialize_by_type[element_type]
                buffer = BytesIO(element_bytes)
                element = serializer.deserialize(buffer)
            else:
                # Fall back to pickle
                element = pickle.loads(element_bytes)

            result.append(element)

        return result


class TupleSerializer(Serializer):
    """
    Serializer for tuples supporting both homogeneous and heterogeneous types.

    Uses a self-contained format:
    1. Pickled type manifest (list of types for each element)
    2. Length-prefixed serialized elements

    For elements with custom serializers, uses BytesIO buffers.
    For elements without custom serializers, falls back to pickle.
    """

    def serialize(self, value: tuple, out: BinaryIO) -> None:
        from io import BytesIO

        from kissml.settings import settings

        # Write manifest: list of types for each element
        manifest = [type(elem) for elem in value]
        pickle.dump(manifest, out)

        # Serialize each element
        for element in value:
            element_type = type(element)

            if element_type in settings.serialize_by_type:
                # Use custom serializer with BytesIO buffer
                serializer = settings.serialize_by_type[element_type]
                buffer = BytesIO()
                serializer.serialize(element, buffer)
                element_bytes = buffer.getvalue()
            else:
                # Fall back to pickle
                element_bytes = pickle.dumps(element)

            # Write length prefix and bytes
            length = len(element_bytes)
            out.write(length.to_bytes(8, byteorder="big"))
            out.write(element_bytes)

    def deserialize(self, input: BinaryIO) -> tuple:
        from io import BytesIO

        from kissml.settings import settings

        # Read manifest
        manifest = pickle.load(input)

        # Deserialize each element
        result = []
        for element_type in manifest:
            # Read length prefix
            length_bytes = input.read(8)
            length = int.from_bytes(length_bytes, byteorder="big")

            # Read element bytes
            element_bytes = input.read(length)

            if element_type in settings.serialize_by_type:
                # Use custom serializer
                serializer = settings.serialize_by_type[element_type]
                buffer = BytesIO(element_bytes)
                element = serializer.deserialize(buffer)
            else:
                # Fall back to pickle
                element = pickle.loads(element_bytes)

            result.append(element)

        return tuple(result)


class DictSerializer(Serializer):
    """
    Serializer for dicts supporting both homogeneous and heterogeneous types.

    Uses a self-contained format:
    1. Pickled key-value type manifest (list of (key_type, value_type) tuples)
    2. Length-prefixed serialized key-value pairs

    For keys/values with custom serializers, uses BytesIO buffers.
    For keys/values without custom serializers, falls back to pickle.
    """

    def serialize(self, value: dict, out: BinaryIO) -> None:
        from io import BytesIO

        from kissml.settings import settings

        # Write manifest: list of (key_type, value_type) for each pair
        manifest = [(type(k), type(v)) for k, v in value.items()]
        pickle.dump(manifest, out)

        # Serialize each key-value pair
        for key, val in value.items():
            key_type = type(key)
            val_type = type(val)

            # Serialize key
            if key_type in settings.serialize_by_type:
                serializer = settings.serialize_by_type[key_type]
                buffer = BytesIO()
                serializer.serialize(key, buffer)
                key_bytes = buffer.getvalue()
            else:
                key_bytes = pickle.dumps(key)

            # Serialize value
            if val_type in settings.serialize_by_type:
                serializer = settings.serialize_by_type[val_type]
                buffer = BytesIO()
                serializer.serialize(val, buffer)
                val_bytes = buffer.getvalue()
            else:
                val_bytes = pickle.dumps(val)

            # Write length-prefixed key and value
            key_length = len(key_bytes)
            val_length = len(val_bytes)
            out.write(key_length.to_bytes(8, byteorder="big"))
            out.write(key_bytes)
            out.write(val_length.to_bytes(8, byteorder="big"))
            out.write(val_bytes)

    def deserialize(self, input: BinaryIO) -> dict:
        from io import BytesIO

        from kissml.settings import settings

        # Read manifest
        manifest = pickle.load(input)

        # Deserialize each key-value pair
        result = {}
        for key_type, val_type in manifest:
            # Read key
            key_length = int.from_bytes(input.read(8), byteorder="big")
            key_bytes = input.read(key_length)

            if key_type in settings.serialize_by_type:
                serializer = settings.serialize_by_type[key_type]
                buffer = BytesIO(key_bytes)
                key = serializer.deserialize(buffer)
            else:
                key = pickle.loads(key_bytes)

            # Read value
            val_length = int.from_bytes(input.read(8), byteorder="big")
            val_bytes = input.read(val_length)

            if val_type in settings.serialize_by_type:
                serializer = settings.serialize_by_type[val_type]
                buffer = BytesIO(val_bytes)
                val = serializer.deserialize(buffer)
            else:
                val = pickle.loads(val_bytes)

            result[key] = val

        return result
