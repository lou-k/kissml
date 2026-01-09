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
        import pandas as pd

        def _stringify_unhashable(obj):
            """Convert DataFrames or Series with dict/list columns to hashable form."""
            if isinstance(obj, pd.DataFrame):
                obj = obj.copy()
                for col in obj.columns:
                    obj[col] = obj[col].apply(
                        lambda x: str(x) if isinstance(x, (dict, list)) else x
                    )
            elif isinstance(obj, pd.Series):
                obj = obj.apply(
                    lambda x: str(x) if isinstance(x, (dict, list)) else x
                )
            return obj

        rv[pd.DataFrame] = lambda df: str(
            pd.util.hash_pandas_object(_stringify_unhashable(df))
        )
        rv[pd.Series] = lambda s: str(
            pd.util.hash_pandas_object(_stringify_unhashable(s))
        )
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
