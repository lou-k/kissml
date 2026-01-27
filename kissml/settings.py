from pathlib import Path
from typing import Any, Callable

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from kissml.serializers import (
    DictSerializer,
    ListSerializer,
    PandasSerializer,
    TupleSerializer,
)
from kissml.types import Serializer


def _default_hash_by_type() -> dict[type, Callable[[Any], str]]:
    rv: dict[type, Callable[[Any], str]] = {}
    try:
        from hashlib import sha256
        from io import BytesIO
        import pandas as pd

        def _hash_dataframe_via_serialization(df: pd.DataFrame) -> str:
            """Hash a DataFrame by serializing it to Parquet and hashing the bytes."""
            serializer = PandasSerializer()
            buffer = BytesIO()
            serializer.serialize(df, buffer)
            return sha256(buffer.getvalue()).hexdigest()

        rv[pd.DataFrame] = _hash_dataframe_via_serialization
        rv[pd.Series] = lambda s: str(pd.util.hash_pandas_object(s))
        rv[pd.Index] = lambda i: str(pd.util.hash_pandas_object(i))
    except ImportError:
        pass
    return rv


def _default_serializer_by_type() -> dict[type, Serializer]:
    rv: dict[type, Serializer] = {
        list: ListSerializer(),
        tuple: TupleSerializer(),
        dict: DictSerializer(),
    }
    try:
        import pandas as pd

        rv[pd.DataFrame] = PandasSerializer()
    except ImportError:
        pass
    return rv


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="KISSML_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    cache_directory: Path = Path.home() / ".kissml"

    hash_by_type: dict[type, Callable[[Any], str]] = Field(
        default_factory=_default_hash_by_type,
        description="A mapping of python type -> custom hash function used to compute cache keys.",
    )

    serialize_by_type: dict[type, Serializer] = Field(
        default_factory=_default_serializer_by_type,
        description="A mapping of python type -> custom serializers to use for disk caching.",
    )


settings = Settings()
