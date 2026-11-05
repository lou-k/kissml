"""Read a Parquet file into a DataFrame one row group at a time."""

from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from pandas.api.types import is_object_dtype, union_categoricals


def read_dataframe(parquet: pq.ParquetFile) -> pd.DataFrame:
    """Read the file as ``parquet.read().to_pandas()`` would, one row group at a time."""
    df = _stream(parquet) if parquet.metadata.num_rows else None
    if df is None:
        # No rows to stream, or a row group whose pandas dtypes differ from
        # the first's (an object column of ints and None is int64 in a row
        # group without None), so the whole table decides the dtypes.
        df = parquet.read().to_pandas()
    return df


def _stream(parquet: pq.ParquetFile) -> pd.DataFrame | None:
    """Convert each row group with pyarrow and write it into one preallocated frame."""
    num_rows = parquet.metadata.num_rows
    index_columns = (parquet.schema_arrow.pandas_metadata or {}).get(
        "index_columns", []
    )
    # A stored index comes back as each row group's index; a RangeIndex is
    # metadata, which pyarrow applies only when its length is the table's.
    n_levels = len([c for c in index_columns if isinstance(c, str)])
    slots: list[Any] = []
    start = 0
    for i in range(parquet.num_row_groups):
        chunk = parquet.read_row_group(i).to_pandas()
        arrays = _arrays(chunk, n_levels)
        if i == 0:
            columns, names = chunk.columns, chunk.index.names
            slots = [_slot(a.dtype, num_rows) for a in arrays]
        for slot, values in zip(slots, arrays):
            if isinstance(slot, list):
                slot.append(values)
            elif values.dtype == slot.dtype:
                slot[start : start + len(values)] = values
            else:
                return None
        start += len(chunk)
        del chunk, arrays
    arrays = [_concat(s) if isinstance(s, list) else s for s in slots]
    index = _index(arrays[:n_levels], names, index_columns, num_rows)
    df = pd.DataFrame(dict(enumerate(arrays[n_levels:])), index, copy=False)
    df.columns = columns
    return df


def _arrays(chunk: pd.DataFrame, n_levels: int) -> list[Any]:
    """The arrays of a row group's index levels, then of its columns."""
    index = chunk.index
    return [index.get_level_values(k).array for k in range(n_levels)] + [
        chunk.iloc[:, j].array for j in range(chunk.shape[1])
    ]


def _slot(dtype: Any, num_rows: int) -> Any:
    """An uninitialized array of ``dtype`` to fill in place, or a list to collect in."""
    # Filled in place: numpy-backed, nullable numeric, boolean, and tz-aware
    # datetime arrays. Collected and concatenated once: categoricals, since
    # each row group has its own categories; Arrow-backed arrays, since a
    # slice assignment rebuilds them; and the rest (string, period, interval).
    in_place = is_object_dtype(dtype) or (
        dtype.kind in "biufMm" and not isinstance(dtype, pd.ArrowDtype)
    )
    if not in_place:
        return []
    if isinstance(dtype, np.dtype):
        return np.empty(num_rows, dtype)
    return dtype.empty(num_rows)


def _concat(parts: list[Any]) -> Any:
    """Concatenate one column's row-group arrays."""
    if isinstance(parts[0], pd.Categorical):
        # First-seen category order, as pyarrow unifies the dictionaries.
        union = union_categoricals(parts, ignore_order=True)
        return union.set_ordered(parts[0].ordered)
    return pd.concat([pd.Series(p) for p in parts], ignore_index=True).array


def _index(
    levels: list[Any], names: Any, index_columns: list[Any], num_rows: int
) -> pd.Index:
    """The frame's index from its stored levels, or from its RangeIndex metadata."""
    if len(levels) > 1:
        return pd.MultiIndex.from_arrays(levels, names=names)
    if levels:
        return pd.Index(levels[0], name=names[0])
    if index_columns:
        descr = index_columns[0]
        index = pd.RangeIndex(
            descr["start"], descr["stop"], descr["step"], name=descr["name"]
        )
        if len(index) == num_rows:
            return index
    return pd.RangeIndex(num_rows)
