from typing import Any, Dict, List, Optional, Sequence, Union

import pyarrow as pa

import polars as pl

__all__ = [
    "get_dummies",
    "concat",
    "repeat",
    "arg_where",
    "from_rows",
]


def get_dummies(df: "pl.DataFrame") -> "pl.DataFrame":
    """
    Convert categorical variables into dummy/indicator variables.

    Parameters
    ----------
    df
        DataFrame to convert.
    """
    return df.to_dummies()


def arg_where(mask: "pl.Series") -> "pl.Series":
    """
    Get index values where Boolean mask evaluate True.

    Parameters
    ----------
    mask
        Boolean Series.

    Returns
    -------
    UInt32 Series
    """
    return mask.arg_true()


def concat(dfs: Sequence["pl.DataFrame"], rechunk: bool = True) -> "pl.DataFrame":
    """
    Aggregate all the Dataframes in a List of DataFrames to a single DataFrame.

    Parameters
    ----------
    dfs
        DataFrames to concatenate.
    rechunk
        rechunk the final DataFrame.
    """
    assert len(dfs) > 0
    df = dfs[0].clone()
    for i in range(1, len(dfs)):
        try:
            df = df.vstack(dfs[i], in_place=False)  # type: ignore[assignment]
        # could have a double borrow (one mutable one ref)
        except RuntimeError:
            df.vstack(dfs[i].clone(), in_place=True)

    if rechunk:
        return df.rechunk()
    return df


def repeat(
    val: Union[int, float, str], n: int, name: Optional[str] = None
) -> "pl.Series":
    """
    Repeat a single value n times and collect into a Series.

    Parameters
    ----------
    val
        Value to repeat.
    n
        Number of repeats.
    name
        Optional name of the Series.
    """
    if name is None:
        name = ""
    if isinstance(val, str):
        s = pl.Series._repeat(name, val, n)
        s.rename(name)
        return s
    else:
        return pl.Series.from_arrow(name, pa.repeat(val, n))


def from_rows(
    rows: Sequence[Sequence[Any]],
    column_names: Optional[List[str]] = None,
    column_name_mapping: Optional[Dict[int, str]] = None,
) -> "pl.DataFrame":
    """
    Create a DataFrame from rows. This should only be used as a last resort, as this is more expensive than
    creating from columnar data.
    Parameters
    ----------
    rows
        rows.
    column_names
        column names to use for the DataFrame.
    column_name_mapping
        map column index to a new name:
        Example:
        ```python
            column_mapping: {0: "first_column, 3: "fourth column"}
        ```
    """
    return pl.DataFrame.from_rows(rows, column_names, column_name_mapping)
