"""Read a Parquet file into a DataFrame one row group at a time."""

from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from pandas.api.types import is_object_dtype, union_categoricals


def read_dataframe(parquet: pq.ParquetFile) -> pd.DataFrame:
    """Read the file as ``parquet.read().to_pandas()`` would, in row groups."""
    num_rows = parquet.metadata.num_rows
    index_columns = (parquet.schema_arrow.pandas_metadata or {}).get(
        "index_columns", []
    )
    # A stored index comes back as each row group's index; a RangeIndex is
    # metadata, which pyarrow applies only when its length is the table's.
    n_levels = sum(isinstance(c, str) for c in index_columns)
    template = parquet.schema_arrow.empty_table().to_pandas()
    slots = [_slot(a, num_rows) for a in _arrays(template, n_levels)]
    start = 0
    for i in range(parquet.num_row_groups):
        chunk = parquet.read_row_group(i).to_pandas()
        _write(slots, _arrays(chunk, n_levels), start)
        start += len(chunk)
        del chunk
    for k, slot in enumerate(slots):
        if isinstance(slot, list):
            slots[k] = _concat(slot)
    index = _index(
        slots[:n_levels], template.index.names, index_columns, num_rows
    )
    # Integer keys: column labels may repeat or form a MultiIndex.
    df = pd.DataFrame(dict(enumerate(slots[n_levels:])), index, copy=False)
    df.columns = template.columns
    return df


def _arrays(frame: pd.DataFrame, n_levels: int) -> list[Any]:
    """The arrays of a frame's index levels, then of its columns."""
    index = frame.index
    return [index.get_level_values(k).array for k in range(n_levels)] + [
        frame.iloc[:, j].array for j in range(frame.shape[1])
    ]


def _slot(empty: Any, num_rows: int) -> Any:
    """An uninitialized array to fill in place, or a list to collect in."""
    # Collected, not filled: categoricals, whose categories differ per row
    # group; Arrow-backed arrays, which a slice assignment rebuilds; the rest.
    dtype = empty.dtype
    in_place = is_object_dtype(dtype) or (
        dtype.kind in "biufMm" and not isinstance(dtype, pd.ArrowDtype)
    )
    if not in_place:
        return [empty]
    if isinstance(dtype, np.dtype):
        return np.empty(num_rows, dtype)
    return dtype.empty(num_rows)


def _write(slots: list[Any], arrays: list[Any], start: int) -> None:
    """Write one row group's arrays into their slots at row ``start``."""
    for k, (slot, values) in enumerate(zip(slots, arrays)):
        if isinstance(slot, list):
            slot.append(values)
            continue
        if values.dtype != slot.dtype:
            # pyarrow types a numpy-backed column by the row group's nulls
            # (int64 without None, float64 with); widen to the table's dtype.
            wide = np.promote_types(
                np.asarray(slot).dtype, np.asarray(values).dtype
            )
            slots[k] = slot = np.asarray(slot).astype(wide, copy=False)
        slot[start : start + len(values)] = values


def _concat(parts: list[Any]) -> Any:
    """Concatenate one column's row-group arrays."""
    if isinstance(parts[0], pd.Categorical):
        # First-seen category order, as pyarrow unifies the dictionaries.
        union = union_categoricals(parts, ignore_order=True)
        return union.set_ordered(parts[0].ordered)
    return type(parts[0])._concat_same_type(parts)


def _index(
    levels: list[Any], names: Any, index_columns: list[Any], num_rows: int
) -> pd.Index:
    """The index from its stored levels, or from the RangeIndex metadata."""
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
