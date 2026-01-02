from pathlib import Path
from typing import Any

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

    def serialize(self, value: Any, path: Path) -> None:
        import pandas as pd

        if not isinstance(value, pd.DataFrame):
            raise ValueError(
                "PandasSerializer can only serialize data frames."
            )
        value.to_parquet(path)

    def deserialize(self, path: Path) -> Any:
        import pandas as pd

        return pd.read_parquet(str(path))
