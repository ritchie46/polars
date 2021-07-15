import typing as tp
from datetime import datetime
from typing import Any, Callable, Optional, Type, Union

import numpy as np

import polars as pl

from ..datatypes import DataType, Date64, Int64

try:
    from ..polars import argsort_by as pyargsort_by
    from ..polars import binary_function as pybinary_function
    from ..polars import col as pycol
    from ..polars import concat_str as _concat_str
    from ..polars import cov as pycov
    from ..polars import except_ as pyexcept
    from ..polars import lit as pylit
    from ..polars import pearson_corr as pypearson_corr
    from ..polars import series_from_range as _series_from_range
except ImportError:
    import warnings

    warnings.warn("Binary files missing.")

__all__ = [
    "col",
    "except_",
    "count",
    "to_list",
    "std",
    "var",
    "max",
    "min",
    "sum",
    "mean",
    "avg",
    "median",
    "n_unique",
    "first",
    "last",
    "head",
    "tail",
    "lit_date",
    "lit",
    "pearson_corr",
    "cov",
    "map_binary",
    "fold",
    "any",
    "all",
    "groups",
    "quantile",
    "arange",
    "argsort_by",
    "concat_str",
    "UDF",  # deprecated
    "udf",  # deprecated
]


def col(name: str) -> "pl.Expr":
    """
    A column in a DataFrame.
    """
    return pl.lazy.expr.wrap_expr(pycol(name))


def except_(name: str) -> "pl.Expr":
    """
    Exclude a column from a selection.

    # Example
    ```python
    df = pl.DataFrame({
        "ham": [1, 1, 2, 2, 3],
        "foo": [1, 1, 2, 2, 3],
        "bar": [1, 1, 2, 2, 3],
    })

    df.lazy()
        .select(["*", except_("foo")])
        .collect()
    ```
    Outputs:

    ```text
    ╭─────┬─────╮
    │ ham ┆ bar │
    │ --- ┆ --- │
    │ f64 ┆ f64 │
    ╞═════╪═════╡
    │ 1   ┆ 1   │
    ├╌╌╌╌╌┼╌╌╌╌╌┤
    │ 1   ┆ 1   │
    ├╌╌╌╌╌┼╌╌╌╌╌┤
    │ 2   ┆ 2   │
    ├╌╌╌╌╌┼╌╌╌╌╌┤
    │ 2   ┆ 2   │
    ├╌╌╌╌╌┼╌╌╌╌╌┤
    │ 3   ┆ 3   │
    ╰─────┴─────╯
    ```
    """
    return pl.lazy.expr.wrap_expr(pyexcept(name))


def count(column: Union[str, "pl.Series"] = "") -> Union["pl.Expr", int]:
    """
    Count the number of values in this column.
    """
    if isinstance(column, pl.Series):
        return column.len()
    return col(column).count()


def to_list(name: str) -> "pl.Expr":
    """
    Aggregate to list.
    """
    return col(name).list()


def std(column: Union[str, "pl.Series"]) -> Union["pl.Expr", float]:
    """
    Get the standard deviation.
    """
    if isinstance(column, pl.Series):
        return column.std()
    return col(column).std()


def var(column: Union[str, "pl.Series"]) -> Union["pl.Expr", float]:
    """
    Get the variance.
    """
    if isinstance(column, pl.Series):
        return column.var()
    return col(column).var()


def max(column: Union[str, tp.List["pl.Expr"], "pl.Series"]) -> Union["pl.Expr", Any]:
    """
    Get the maximum value. Can be used horizontally or vertically.

    Parameters
    ----------
    column
        Column(s) to be used in aggregation. Will lead to different behavior based on the input.
        input:
            - Union[str, Series] -> aggregate the maximum value of that column.
            - tp.List[Expr] -> aggregate the maximum value horizontally.
    """
    if isinstance(column, pl.Series):
        return column.max()
    elif isinstance(column, list):

        def max_(acc: "pl.Series", val: "pl.Series") -> "pl.Series":
            mask = acc < val
            return acc.zip_with(mask, val)

        return fold(lit(0), max_, column).alias("max")
    else:
        return col(column).max()


def min(column: Union[str, tp.List["pl.Expr"], "pl.Series"]) -> Union["pl.Expr", Any]:
    """
    Get the minimum value.

    column
        Column(s) to be used in aggregation. Will lead to different behavior based on the input.
        input:
            - Union[str, Series] -> aggregate the sum value of that column.
            - tp.List[Expr] -> aggregate the sum value horizontally.
    """
    if isinstance(column, pl.Series):
        return column.min()
    elif isinstance(column, list):

        def min_(acc: "pl.Series", val: "pl.Series") -> "pl.Series":
            mask = acc > val
            return acc.zip_with(mask, val)

        return fold(lit(0), min_, column).alias("min")
    else:
        return col(column).min()


def sum(column: Union[str, tp.List["pl.Expr"], "pl.Series"]) -> Union["pl.Expr", Any]:
    """
    Get the sum value.

    column
        Column(s) to be used in aggregation. Will lead to different behavior based on the input.
        input:
            - Union[str, Series] -> aggregate the sum value of that column.
            - tp.List[Expr] -> aggregate the sum value horizontally.
    """
    if isinstance(column, pl.Series):
        return column.sum()
    elif isinstance(column, list):
        return fold(lit(0), lambda a, b: a + b, column).alias("sum")
    else:
        return col(column).sum()


def mean(column: Union[str, "pl.Series"]) -> Union["pl.Expr", float]:
    """
    Get the mean value.
    """
    if isinstance(column, pl.Series):
        return column.mean()
    return col(column).mean()


def avg(column: Union[str, "pl.Series"]) -> Union["pl.Expr", float]:
    """
    Alias for mean.
    """
    return mean(column)


def median(column: Union[str, "pl.Series"]) -> Union["pl.Expr", float, int]:
    """
    Get the median value.
    """
    if isinstance(column, pl.Series):
        return column.median()
    return col(column).median()


def n_unique(column: Union[str, "pl.Series"]) -> Union["pl.Expr", int]:
    """Count unique values."""
    if isinstance(column, pl.Series):
        return column.n_unique()
    return col(column).n_unique()


def first(column: Union[str, "pl.Series"]) -> Union["pl.Expr", Any]:
    """
    Get the first value.
    """
    if isinstance(column, pl.Series):
        if column.len() > 0:
            return column[0]
        else:
            raise IndexError("The series is empty, so no first value can be returned.")
    return col(column).first()


def last(column: Union[str, "pl.Series"]) -> "pl.Expr":
    """
    Get the last value.
    """
    if isinstance(column, pl.Series):
        if column.len() > 0:
            return column[-1]
        else:
            raise IndexError("The series is empty, so no last value can be returned,")
    return col(column).last()


def head(
    column: Union[str, "pl.Series"], n: Optional[int] = None
) -> Union["pl.Expr", "pl.Series"]:
    """
    Get the first n rows of an Expression.

    Parameters
    ----------
    column
        Column name or Series.
    n
        Number of rows to take.
    """
    if isinstance(column, pl.Series):
        return column.head(n)
    return col(column).head(n)


def tail(
    column: Union[str, "pl.Series"], n: Optional[int] = None
) -> Union["pl.Expr", "pl.Series"]:
    """
    Get the last n rows of an Expression.

    Parameters
    ----------
    column
        Column name or Series.
    n
        Number of rows to take.
    """
    if isinstance(column, pl.Series):
        return column.tail(n)
    return col(column).tail(n)


def lit_date(dt: datetime) -> "pl.Expr":
    """
    Converts a Python DateTime to a literal Expression.

    Parameters
    ----------
    dt
        datetime.datetime
    """
    return lit(int(dt.timestamp() * 1e3))


def lit(
    value: Optional[Union[float, int, str, datetime, "pl.Series"]],
    dtype: Optional[Type[DataType]] = None,
) -> "pl.Expr":
    """
    A literal value.

    Parameters
    ----------
    value
        Value that should be used as a `literal`.
    dtype
        Optionally define a dtype.

    # Example

    ```python
    # literal integer
    lit(1)

    # literal str.
    lit("foo")

    # literal date64
    lit(datetime(2021, 1, 20))

    # literal Null
    lit(None)

    # literal eager Series
    lit(Series("a", [1, 2, 3])
    ```
    """
    if isinstance(value, datetime):
        return lit(int(value.timestamp() * 1e3)).cast(Date64)

    if isinstance(value, pl.Series):
        name = value.name
        value = value._s
        return pl.lazy.expr.wrap_expr(pylit(value)).alias(name)

    if isinstance(value, np.ndarray):
        return lit(pl.Series("", value))

    if dtype:
        return pl.lazy.expr.wrap_expr(pylit(value)).cast(dtype)
    return pl.lazy.expr.wrap_expr(pylit(value))


def pearson_corr(
    a: Union[str, "pl.Expr"],
    b: Union[str, "pl.Expr"],
) -> "pl.Expr":
    """
    Compute the pearson's correlation between two columns.

    Parameters
    ----------
    a
        Column name or Expression.
    b
        Column name or Expression.
    """
    if isinstance(a, str):
        a = col(a)
    if isinstance(b, str):
        b = col(b)
    return pl.lazy.expr.wrap_expr(pypearson_corr(a._pyexpr, b._pyexpr))


def cov(
    a: Union[str, "pl.Expr"],
    b: Union[str, "pl.Expr"],
) -> "pl.Expr":
    """
    Compute the covariance between two columns/ expressions.

    Parameters
    ----------
    a
        Column name or Expression.
    b
        Column name or Expression.
    """
    if isinstance(a, str):
        a = col(a)
    if isinstance(b, str):
        b = col(b)
    return pl.lazy.expr.wrap_expr(pycov(a._pyexpr, b._pyexpr))


def map_binary(
    a: Union[str, "pl.Expr"],
    b: Union[str, "pl.Expr"],
    f: Callable[["pl.Series", "pl.Series"], "pl.Series"],
    return_dtype: Optional[Type[DataType]] = None,
) -> "pl.Expr":
    """
    Map a custom function over two columns and produce a single Series result.

    Parameters
    ----------
    a
        Input Series a.
    b
        Input Series b.
    f
        Function to apply.
    return_dtype
        Output type of the udf.
    """
    if isinstance(a, str):
        a = col(a)
    if isinstance(b, str):
        b = col(b)
    return pl.lazy.expr.wrap_expr(
        pybinary_function(a._pyexpr, b._pyexpr, f, return_dtype)
    )


def fold(
    acc: "pl.Expr",
    f: Callable[["pl.Series", "pl.Series"], "pl.Series"],
    exprs: tp.List["pl.Expr"],
) -> "pl.Expr":
    """
    Accumulate over multiple columns horizontally/ row wise with a left fold.

    Parameters
    ----------
    acc
     Accumulator Expression. This is the value that will be initialized when the fold starts.
     For a sum this could for instance be lit(0).

    f
        Function to apply over the accumulator and the value.
        Fn(acc, value) -> new_value
    exprs
        Expressions to aggregate over.
    """
    for e in exprs:
        acc = map_binary(acc, e, f, None)
    return acc


def any(name: Union[str, tp.List["pl.Expr"]]) -> "pl.Expr":
    """
    Evaluate columnwise or elementwise with a bitwise OR operation.
    """
    if isinstance(name, list):
        return fold(lit(0), lambda a, b: a | b, name).alias("any")
    return col(name).sum() > 0


def all(name: Union[str, tp.List["pl.Expr"]]) -> "pl.Expr":
    """
    Evaluate columnwise or elementwise with a bitwise AND operation.
    """
    if isinstance(name, list):
        return fold(lit(0), lambda a, b: a & b, name).alias("all")
    return col(name).cast(bool).sum() == col(name).count()


def groups(column: str) -> "pl.Expr":
    """
    Syntactic sugar for `column("foo").agg_groups()`.
    """
    return col(column).agg_groups()


def quantile(column: str, quantile: float) -> "pl.Expr":
    """
    Syntactic sugar for `column("foo").quantile(..)`.
    """
    return col(column).quantile(quantile)


class UDF:
    """
    Deprecated: don't use me
    """

    def __init__(
        self, f: Callable[["pl.Series"], "pl.Series"], return_dtype: Type[DataType]
    ):
        self.f = f
        self.return_dtype = return_dtype


def udf(f: Callable[["pl.Series"], "pl.Series"], return_dtype: Type[DataType]) -> UDF:
    """
    Deprecated: don't use me
    """
    return UDF(f, return_dtype)


def arange(
    low: Union[int, "pl.Expr"],
    high: Union[int, "pl.Expr"],
    step: int = 1,
    dtype: Optional[Type[DataType]] = None,
    eager: bool = False,
) -> Union["pl.Expr", "pl.Series"]:
    """
    Create a range expression. This can be used in a `select`, `with_column` etc.
    Be sure that the range size is equal to the DataFrame you are collecting.

    # Example

    ```python
    (df.lazy()
        .filter(pl.col("foo") < pl.arange(0, 100))
        .collect())
    ```

    Parameters
    ----------
    low
        Lower bound of range.
    high
        Upper bound of range.
    step
        Step size of the range
    dtype
        DataType of the range. Valid dtypes:
            * Int32
            * Int64
            * UInt32
    eager
        If eager evaluation is `True`, a Series is returned instead of an Expr
    """
    if dtype is None:
        dtype = Int64

    def create_range(s1: "pl.Series", s2: "pl.Series") -> "pl.Series":
        assert s1.len() == 1
        assert s2.len() == 1
        return pl.Series._from_pyseries(_series_from_range(s1[0], s2[0], step, dtype))

    # eager execution can only work if low and high are literals
    if eager:
        if not (isinstance(low, int) and isinstance(high, int)):
            raise ValueError(
                "arguments low and high must be integers in eager execution"
            )
        return pl.Series._from_pyseries(_series_from_range(low, high, step, dtype))

    if isinstance(low, int):
        low = lit(low)
    if isinstance(high, int):
        high = lit(high)

    return map_binary(low, high, create_range, return_dtype=dtype)


def argsort_by(
    exprs: tp.List["pl.Expr"], reverse: Union[tp.List[bool], bool] = False
) -> "pl.Expr":
    """
    Find the indexes that would sort the columns.

    Argsort by multiple columns. The first column will be used for the ordering.
    If there are duplicates in the first column, the second column will be used to determine the ordering
    and so on.

    Parameters
    ----------
    exprs
        Columns use to determine the ordering.
    reverse
        Default is ascending.
    """
    if not isinstance(reverse, list):
        reverse = [reverse]
    exprs = pl.lazy.expr._selection_to_pyexpr_list(exprs)
    return pl.lazy.expr.wrap_expr(pyargsort_by(exprs, reverse))


def concat_str(exprs: tp.List["pl.Expr"], delimiter: str = "") -> "pl.Expr":
    """
    Concat Utf8 Series in linear time. Non utf8 columns are cast to utf8.

    Parameters
    ----------
    exprs
        Columns to concat into a Utf8 Series
    delimiter
        String value that will be used to separate the values.
    """
    exprs = pl.lazy.expr._selection_to_pyexpr_list(exprs)
    return pl.lazy.expr.wrap_expr(_concat_str(exprs, delimiter))
