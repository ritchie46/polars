"""
This module contains all expressions and classes needed for lazy computation/ query execution.
"""
import os
import shutil
import subprocess
import tempfile
import typing as tp
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Type, Union

import polars as pl

from ..datatypes import DataType, pytype_to_polars_type
from ..utils import _process_null_values
from .expr import UDF, Expr, _selection_to_pyexpr_list, col, expr_to_lit_or_expr, lit

try:
    from ..polars import PyExpr, PyLazyFrame, PyLazyGroupBy
except ImportError:
    import warnings

    warnings.warn("Binary files missing.")
    __pdoc__ = {"wrap_ldf": False}

__all__ = [
    "LazyFrame",
]


def wrap_ldf(ldf: "PyLazyFrame") -> "LazyFrame":
    return LazyFrame._from_pyldf(ldf)


class LazyFrame:
    """
    Representation of a Lazy computation graph/ query.
    """

    def __init__(self) -> None:
        self._ldf: PyLazyFrame

    @staticmethod
    def _from_pyldf(ldf: "PyLazyFrame") -> "LazyFrame":
        self = LazyFrame.__new__(LazyFrame)
        self._ldf = ldf
        return self

    @staticmethod
    def scan_csv(
        file: str,
        has_headers: bool = True,
        ignore_errors: bool = False,
        sep: str = ",",
        skip_rows: int = 0,
        stop_after_n_rows: Optional[int] = None,
        cache: bool = True,
        dtype: Optional[Dict[str, Type[DataType]]] = None,
        low_memory: bool = False,
        comment_char: Optional[str] = None,
        null_values: Optional[Union[str, tp.List[str], Dict[str, str]]] = None,
    ) -> "LazyFrame":
        """
        See Also: `pl.scan_csv`
        """
        dtype_list: Optional[tp.List[Tuple[str, Type[DataType]]]] = None
        if dtype is not None:
            dtype_list = []
            for k, v in dtype.items():
                dtype_list.append((k, pytype_to_polars_type(v)))
        processed_null_values = _process_null_values(null_values)

        self = LazyFrame.__new__(LazyFrame)
        self._ldf = PyLazyFrame.new_from_csv(
            file,
            sep,
            has_headers,
            ignore_errors,
            skip_rows,
            stop_after_n_rows,
            cache,
            dtype_list,
            low_memory,
            comment_char,
            processed_null_values,
        )
        return self

    @staticmethod
    def scan_parquet(
        file: str, stop_after_n_rows: Optional[int] = None, cache: bool = True
    ) -> "LazyFrame":
        """
        See Also: `pl.scan_parquet`
        """

        self = LazyFrame.__new__(LazyFrame)
        self._ldf = PyLazyFrame.new_from_parquet(file, stop_after_n_rows, cache)
        return self

    def pipe(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """
        Apply a function on Self.

        Parameters
        ----------
        func
            Callable.
        args
            Arguments.
        kwargs
            Keyword arguments.
        """
        return func(self, *args, **kwargs)

    def describe_plan(self) -> str:
        """
        A string representation of the unoptimized query plan.
        """
        return self._ldf.describe_plan()

    def describe_optimized_plan(
        self,
        type_coercion: bool = True,
        predicate_pushdown: bool = True,
        projection_pushdown: bool = True,
        simplify_expression: bool = True,
    ) -> str:
        """
        A string representation of the optimized query plan.
        """

        ldf = self._ldf.optimization_toggle(
            type_coercion,
            predicate_pushdown,
            projection_pushdown,
            simplify_expression,
            string_cache=False,
        )

        return ldf.describe_optimized_plan()

    def show_graph(
        self,
        optimized: bool = True,
        show: bool = True,
        output_path: Optional[str] = None,
        raw_output: bool = False,
        figsize: Tuple[float, float] = (16.0, 12.0),
    ) -> Optional[str]:
        """
        Show a plot of the query plan. Note that you should have graphviz installed.

        Parameters
        ----------
        optimized
            Optimize the query plan.
        show
            Show the figure.
        output_path
            Write the figure to disk.
        raw_output
            Return dot syntax.
        figsize
            Passed to matlotlib if `show` == True.
        """
        try:
            import matplotlib.image as mpimg
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError(
                "Graphviz dot binary should be on your PATH and matplotlib should be installed to show graph."
            )
        dot = self._ldf.to_dot(optimized)
        if raw_output:
            return dot
        with tempfile.TemporaryDirectory() as tmpdir_name:
            dot_path = os.path.join(tmpdir_name, "dot")
            with open(dot_path, "w") as f:
                f.write(dot)

            subprocess.run(["dot", "-Nshape=box", "-Tpng", "-O", dot_path])
            out_path = os.path.join(tmpdir_name, "dot.png")

            if output_path is not None:
                shutil.copy(out_path, output_path)

            if show:
                plt.figure(figsize=figsize)
                img = mpimg.imread(out_path)
                plt.imshow(img)
                plt.show()
        return None

    def sort(
        self,
        by_columns: Union[str, "Expr", tp.List["Expr"]],
        reverse: Union[bool, tp.List[bool]] = False,
    ) -> "LazyFrame":
        """
        Sort the DataFrame by:

            - A single column name
            - An expression
            - Multiple expressions

        Parameters
        ----------
        by_columns
            Column (expressions) to sort by.
        reverse
            Whether or not to sort in reverse order.
        """
        if type(by_columns) is str:
            return wrap_ldf(self._ldf.sort(by_columns, reverse))
        if type(reverse) is bool:
            reverse = [reverse]

        by_columns = expr_to_lit_or_expr(by_columns, str_to_lit=False)
        by_columns = _selection_to_pyexpr_list(by_columns)
        return wrap_ldf(self._ldf.sort_by_exprs(by_columns, reverse))

    def collect(
        self,
        type_coercion: bool = True,
        predicate_pushdown: bool = True,
        projection_pushdown: bool = True,
        simplify_expression: bool = True,
        string_cache: bool = False,
        no_optimization: bool = False,
    ) -> "pl.DataFrame":
        """
        Collect into a DataFrame.

        Parameters
        ----------
        type_coercion
            Do type coercion optimization.
        predicate_pushdown
            Do predicate pushdown optimization.
        projection_pushdown
            Do projection pushdown optimization.
        simplify_expression
            Run simplify expressions optimization.
        string_cache
            Use a global string cache in this query.
            This is needed if you want to join on categorical columns.

            Caution!
                If you already have set a global string cache, set this to `False` as this will reset the
                global cache when the query is finished.
        no_optimization
            Turn off optimizations.

        Returns
        -------
        DataFrame
        """
        if no_optimization:
            predicate_pushdown = False
            projection_pushdown = False

        ldf = self._ldf.optimization_toggle(
            type_coercion,
            predicate_pushdown,
            projection_pushdown,
            simplify_expression,
            string_cache,
        )
        return pl.eager.frame.wrap_df(ldf.collect())

    def fetch(
        self,
        n_rows: int = 500,
        type_coercion: bool = True,
        predicate_pushdown: bool = True,
        projection_pushdown: bool = True,
        simplify_expression: bool = True,
        string_cache: bool = True,
        no_optimization: bool = False,
    ) -> "pl.DataFrame":
        """
        Fetch is like a collect operation, but it overwrites the number of rows read by every scan
        operation. This is a utility that helps debug a query on a smaller number of rows.

        Note that the fetch does not guarantee the final number of rows in the DataFrame.
        Filter, join operations and a lower number of rows available in the scanned file influence
        the final number of rows.

        Parameters
        ----------
        n_rows
            Collect n_rows from the data sources.
        type_coercion
            Run type coercion optimization.
        predicate_pushdown
            Run predicate pushdown optimization.
        projection_pushdown
            Run projection pushdown optimization.
        simplify_expression
            Run simplify expressions optimization.
        string_cache
            Use a global string cache in this query.
            This is needed if you want to join on categorical columns.
        no_optimization
            Turn off optimizations.

        Returns
        -------
        DataFrame
        """
        if no_optimization:
            predicate_pushdown = False
            projection_pushdown = False

        ldf = self._ldf.optimization_toggle(
            type_coercion,
            predicate_pushdown,
            projection_pushdown,
            simplify_expression,
            string_cache,
        )
        return pl.eager.frame.wrap_df(ldf.fetch(n_rows))

    def cache(
        self,
    ) -> "LazyFrame":
        """
        Cache the result once the execution of the physical plan hits this node.
        """
        return wrap_ldf(self._ldf.cache())

    def filter(self, predicate: "Expr") -> "LazyFrame":
        """
        Filter the rows in the DataFrame based on a predicate expression.

        Parameters
        ----------
        predicate
            Expression that evaluates to a boolean Series.
        """
        if isinstance(predicate, str):
            predicate = col(predicate)
        return wrap_ldf(self._ldf.filter(predicate._pyexpr))

    def select(
        self, exprs: Union[str, "Expr", Sequence[str], Sequence["Expr"]]
    ) -> "LazyFrame":
        """
        Select columns from this DataFrame.

        Parameters
        ----------
        exprs
            Column or columns to select.
        """
        exprs = _selection_to_pyexpr_list(exprs)
        return wrap_ldf(self._ldf.select(exprs))

    def groupby(
        self, by: Union[str, tp.List[str], "Expr", tp.List["Expr"]]
    ) -> "LazyGroupBy":
        """
        Start a groupby operation.

        Parameters
        ----------
        by
            Column(s) to group by.
        """
        new_by: tp.List[PyExpr]
        if isinstance(by, list):
            new_by = []
            for e in by:
                if isinstance(e, str):
                    e = col(e)
                new_by.append(e._pyexpr)
        elif isinstance(by, str):
            new_by = [col(by)._pyexpr]
        elif isinstance(by, Expr):
            new_by = [by._pyexpr]
        lgb = self._ldf.groupby(new_by)
        return LazyGroupBy(lgb)

    def join(
        self,
        ldf: "LazyFrame",
        left_on: Optional[Union[str, "Expr", tp.List[str], tp.List["Expr"]]] = None,
        right_on: Optional[Union[str, "Expr", tp.List[str], tp.List["Expr"]]] = None,
        on: Optional[Union[str, "Expr", tp.List[str], tp.List["Expr"]]] = None,
        how: str = "inner",
        allow_parallel: bool = True,
        force_parallel: bool = False,
    ) -> "LazyFrame":
        """
        Add a join operation to the Logical Plan.

        Parameters
        ----------
        ldf
            Lazy DataFrame to join with.
        left_on
            Join column of the left DataFrame.
        right_on
            Join column of the right DataFrame.
        on
            Join column of both DataFrames. If set, `left_on` and `right_on` should be None.
        how
            one of:
                "inner"
                "left"
                "outer"
                "asof",
                "cross"

        allow_parallel
            Allow the physical plan to optionally evaluate the computation of both DataFrames up to the join in parallel.
        force_parallel
            Force the physical plan to evaluate the computation of both DataFrames up to the join in parallel.

        # Asof joins
        This is similar to a left-join except that we match on nearest key rather than equal keys.
        The keys must be sorted to perform an asof join

        """
        if how == "cross":
            return wrap_ldf(
                self._ldf.join(ldf._ldf, [], [], allow_parallel, force_parallel, how)
            )

        left_on_: Union[tp.List[str], tp.List[Expr], None]
        if isinstance(left_on, (str, Expr)):
            left_on_ = [left_on]  # type: ignore[assignment]
        else:
            left_on_ = left_on

        right_on_: Union[tp.List[str], tp.List[Expr], None]
        if isinstance(right_on, (str, Expr)):
            right_on_ = [right_on]  # type: ignore[assignment]
        else:
            right_on_ = right_on

        if isinstance(on, str):
            left_on_ = [on]
            right_on_ = [on]
        elif isinstance(on, list):
            left_on_ = on
            right_on_ = on

        if left_on_ is None or right_on_ is None:
            raise ValueError("You should pass the column to join on as an argument.")

        new_left_on = []
        for column in left_on_:
            if isinstance(column, str):
                column = col(column)
            new_left_on.append(column._pyexpr)
        new_right_on = []
        for column in right_on_:
            if isinstance(column, str):
                column = col(column)
            new_right_on.append(column._pyexpr)

        return wrap_ldf(
            self._ldf.join(
                ldf._ldf, new_left_on, new_right_on, allow_parallel, force_parallel, how
            )
        )

    def with_columns(self, exprs: Union[tp.List["Expr"], "Expr"]) -> "LazyFrame":
        """
        Add or overwrite multiple columns in a DataFrame.

        Parameters
        ----------
        exprs
            List of Expressions that evaluate to columns.
        """
        if isinstance(exprs, Expr):
            return self.with_column(exprs)

        pyexprs = []

        for e in exprs:
            if isinstance(e, Expr):
                pyexprs.append(e._pyexpr)
            elif isinstance(e, pl.Series):
                pyexprs.append(lit(e)._pyexpr)

        return wrap_ldf(self._ldf.with_columns(pyexprs))

    def with_column(self, expr: "Expr") -> "LazyFrame":
        """
        Add or overwrite column in a DataFrame.

        Parameters
        ----------
        expr
            Expression that evaluates to column.
        """
        return self.with_columns([expr])

    def drop_columns(self, columns: tp.List[str]) -> "LazyFrame":
        """
        Remove multiple columns from a DataFrame.

        Parameters
        ----------
        columns
            List of column names.
        """
        return wrap_ldf(self._ldf.drop_columns(columns))

    def drop_column(self, column: str) -> "LazyFrame":
        """
        Remove a column from the DataFrame.

        Parameters
        ----------
        column
            Name of the column that should be removed.
        """
        return self.drop_columns([column])

    def with_column_renamed(self, existing_name: str, new_name: str) -> "LazyFrame":
        """
        Rename a column in the DataFrame
        """
        return wrap_ldf(self._ldf.with_column_renamed(existing_name, new_name))

    def reverse(self) -> "LazyFrame":
        """
        Reverse the DataFrame.
        """
        return wrap_ldf(self._ldf.reverse())

    def shift(self, periods: int) -> "LazyFrame":
        """
        Shift the values by a given period and fill the parts that will be empty due to this operation
        with `Nones`.

        Parameters
        ----------
        periods
            Number of places to shift (may be negative).
        """
        return wrap_ldf(self._ldf.shift(periods))

    def shift_and_fill(
        self, periods: int, fill_value: Union["Expr", int, str, float]
    ) -> "LazyFrame":
        """
        Shift the values by a given period and fill the parts that will be empty due to this operation
        with the result of the `fill_value` expression.

        Parameters
        ----------
        periods
            Number of places to shift (may be negative).
        fill_value
            fill None values with the result of this expression.
        """
        if not isinstance(fill_value, Expr):
            fill_value = lit(fill_value)
        return wrap_ldf(self._ldf.shift_and_fill(periods, fill_value._pyexpr))

    def slice(self, offset: int, length: int) -> "LazyFrame":
        """
        Slice the DataFrame.

        Parameters
        ----------
        offset
            Start index.
        length
            Length of the slice.
        """
        return wrap_ldf(self._ldf.slice(offset, length))

    def limit(self, n: int) -> "LazyFrame":
        """
        Limit the DataFrame to the first `n` rows. Note if you don't want the rows to be scanned,
        use the `fetch` operation.

        Parameters
        ----------
        n
            Number of rows.
        """
        return self.slice(0, n)

    def head(self, n: int) -> "LazyFrame":
        """
        Get the first `n` rows of the DataFrame
        Note if you don't want the rows to be scanned,
        use the `fetch` operation.

        Parameters
        ----------
        n
            Number of rows.
        """
        return self.limit(n)

    def tail(self, n: int) -> "LazyFrame":
        """
        Get the last `n` rows of the DataFrame.

        Parameters
        ----------
        n
            Number of rows.
        """
        return wrap_ldf(self._ldf.tail(n))

    def last(self) -> "LazyFrame":
        """
        Get the last row of the DataFrame.
        """
        return self.tail(1)

    def first(self) -> "LazyFrame":
        """
        Get the first row of the DataFrame.
        """
        return self.slice(0, 1)

    def fill_none(self, fill_value: Union[int, str, "Expr"]) -> "LazyFrame":
        if not isinstance(fill_value, Expr):
            fill_value = lit(fill_value)
        return wrap_ldf(self._ldf.fill_none(fill_value._pyexpr))

    def std(self) -> "LazyFrame":
        """
        Aggregate the columns in the DataFrame to their standard deviation value.
        """
        return wrap_ldf(self._ldf.std())

    def var(self) -> "LazyFrame":
        """
        Aggregate the columns in the DataFrame to their variance value.
        """
        return wrap_ldf(self._ldf.var())

    def max(self) -> "LazyFrame":
        """
        Aggregate the columns in the DataFrame to their maximum value.
        """
        return wrap_ldf(self._ldf.max())

    def min(self) -> "LazyFrame":
        """
        Aggregate the columns in the DataFrame to their minimum value.
        """
        return wrap_ldf(self._ldf.min())

    def sum(self) -> "LazyFrame":
        """
        Aggregate the columns in the DataFrame to their sum value.
        """
        return wrap_ldf(self._ldf.sum())

    def mean(self) -> "LazyFrame":
        """
        Aggregate the columns in the DataFrame to their mean value.
        """
        return wrap_ldf(self._ldf.mean())

    def median(self) -> "LazyFrame":
        """
        Aggregate the columns in the DataFrame to their median value.
        """
        return wrap_ldf(self._ldf.median())

    def quantile(self, quantile: float) -> "LazyFrame":
        """
        Aggregate the columns in the DataFrame to their quantile value.
        """
        return wrap_ldf(self._ldf.quantile(quantile))

    def explode(self, columns: Union[str, tp.List[str]]) -> "LazyFrame":
        """
        Explode lists to long format.
        """
        if isinstance(columns, str):
            columns = [columns]
        return wrap_ldf(self._ldf.explode(columns))

    def drop_duplicates(
        self,
        maintain_order: bool = False,
        subset: Optional[Union[tp.List[str], str]] = None,
    ) -> "LazyFrame":
        """
        Drop duplicate rows from this DataFrame.
        Note that this fails if there is a column of type `List` in the DataFrame.
        """
        if subset is not None and not isinstance(subset, list):
            subset = [subset]
        return wrap_ldf(self._ldf.drop_duplicates(maintain_order, subset))

    def drop_nulls(
        self, subset: Optional[Union[tp.List[str], str]] = None
    ) -> "LazyFrame":
        """
        Drop rows with null values from this DataFrame.
        """
        if subset is not None and not isinstance(subset, list):
            subset = [subset]
        return wrap_ldf(self._ldf.drop_nulls(subset))

    def melt(
        self, id_vars: Union[str, tp.List[str]], value_vars: Union[str, tp.List[str]]
    ) -> "LazyFrame":
        """
        Unpivot DataFrame to long format.

        Parameters
        ----------
        id_vars
            Columns to use as identifier variables.
        value_vars
            Values to use as identifier variables.
        """
        if isinstance(value_vars, str):
            value_vars = [value_vars]
        if isinstance(id_vars, str):
            id_vars = [id_vars]
        return wrap_ldf(self._ldf.melt(id_vars, value_vars))

    def map(
        self,
        f: Union["UDF", Callable[["pl.DataFrame"], "pl.DataFrame"]],
        predicate_pushdown: bool = True,
        projection_pushdown: bool = True,
        no_optimizations: bool = False,
    ) -> "LazyFrame":
        """
        Apply a custom UDF. It is important that the UDF returns a Polars DataFrame.

        Parameters
        ----------
        f
            Lambda/ function to apply.
        predicate_pushdown
            Allow predicate pushdown optimization to pass this node.
        projection_pushdown
            Allow projection pushdown optimization to pass this node.
        no_optimizations
            Turn off all optimizations past this point.
        """
        if not no_optimizations:
            predicate_pushdown = False
            projection_pushdown = False
        return wrap_ldf(self._ldf.map(f, predicate_pushdown, projection_pushdown))


class LazyGroupBy:
    """
    Created by `df.lazy().groupby("foo)"`
    """

    def __init__(self, lgb: "PyLazyGroupBy"):
        self.lgb = lgb

    def agg(self, aggs: Union[tp.List["Expr"], "Expr"]) -> "LazyFrame":
        """
        Describe the aggregation that need to be done on a group.

        Parameters
        ----------
        aggs
            Single/ Multiple aggregation expression(s).

        # Example

        ```python
        (pl.scan_csv("data.csv")
            .groupby("groups")
            .agg([
                    pl.col("name").n_unique().alias("unique_names"),
                    pl.max("values")
                ])
        )
        ```
        """
        aggs = _selection_to_pyexpr_list(aggs)
        return wrap_ldf(self.lgb.agg(aggs))

    def apply(self, f: Callable[["pl.DataFrame"], "pl.DataFrame"]) -> "LazyFrame":
        """
        Apply a function over the groups as a new `DataFrame`. It is not recommended that you use
        this as materializing the `DataFrame` is quite expensive.

        Parameters
        ----------
        f
            Function to apply over the `DataFrame`.
        """
        return wrap_ldf(self.lgb.apply(f))
