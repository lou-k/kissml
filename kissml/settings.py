from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from kissml.serializers import PandasSerializer
from kissml.types import Serializer

if TYPE_CHECKING:
    pass


def _default_hash_by_type() -> dict[type, Callable[[Any], str]]:
    rv: dict[type, Callable[[Any], str]] = {}
    try:
        import pandas as pd

        rv[pd.DataFrame] = lambda df: str(pd.util.hash_pandas_object(df))
        rv[pd.Series] = lambda s: str(pd.util.hash_pandas_object(s))
        rv[pd.Index] = lambda i: str(pd.util.hash_pandas_object(i))
    except ImportError:
        pass
    return rv


def _default_serializer_by_type() -> dict[type, Serializer]:
    rv: dict[type, Serializer] = {}
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
