from typing import Any, Union

import polars as pl

from .expr import expr_to_lit_or_expr

try:
    from ..polars import when as pywhen
except ImportError:
    import warnings

    warnings.warn("Binary files missing.")

__all__ = ["when"]


class WhenThenThen:
    """
    Utility class. See the `when` function.
    """

    def __init__(self, pywhenthenthen: Any):
        self.pywenthenthen = pywhenthenthen

    def when(self, predicate: "pl.Expr") -> "WhenThenThen":
        """
        Start another when, then, otherwise layer.
        """
        return WhenThenThen(self.pywenthenthen.when(predicate._pyexpr))

    def then(self, expr: Union["pl.Expr", int, float, str]) -> "WhenThenThen":
        """
        Values to return in case of the predicate being `True`.

        See Also: the `when` function.
        """
        expr_ = expr_to_lit_or_expr(expr)
        return WhenThenThen(self.pywenthenthen.then(expr_._pyexpr))

    def otherwise(self, expr: Union["pl.Expr", int, float, str]) -> "pl.Expr":
        """
        Values to return in case of the predicate being `False`.

        See Also: the `when` function.
        """
        expr = expr_to_lit_or_expr(expr)
        return pl.lazy.expr.wrap_expr(self.pywenthenthen.otherwise(expr._pyexpr))


class WhenThen:
    """
    Utility class. See the `when` function.
    """

    def __init__(self, pywhenthen: Any):
        self._pywhenthen = pywhenthen

    def when(self, predicate: "pl.Expr") -> WhenThenThen:
        """
        Start another when, then, otherwise layer.
        """
        return WhenThenThen(self._pywhenthen.when(predicate._pyexpr))

    def otherwise(self, expr: Union["pl.Expr", int, float, str]) -> "pl.Expr":
        """
        Values to return in case of the predicate being `False`.

        See Also: the `when` function.
        """
        expr = expr_to_lit_or_expr(expr)
        return pl.lazy.expr.wrap_expr(self._pywhenthen.otherwise(expr._pyexpr))


class When:
    """
    Utility class. See the `when` function.
    """

    def __init__(self, pywhen: "pywhen"):
        self._pywhen = pywhen

    def then(self, expr: Union["pl.Expr", int, float, str]) -> WhenThen:
        """
        Values to return in case of the predicate being `True`.

        See Also: the `when` function.
        """
        expr = expr_to_lit_or_expr(expr)
        pywhenthen = self._pywhen.then(expr._pyexpr)
        return WhenThen(pywhenthen)


def when(expr: "pl.Expr") -> When:
    """
    Start a when, then, otherwise expression.

    # Example

    Below we add a column with the value 1, where column "foo" > 2 and the value -1 where it isn't.

    ```python
    lf.with_column(
        when(col("foo") > 2)
        .then(lit(1))
        .otherwise(lit(-1))
    )
    ```

    Or with multiple `when, thens` chained:


    ```python
    lf.with_column(
        when(col("foo") > 2).then(1)
        when(col("bar") > 2).then(4)
        .otherwise(-1)
    )
    ```
    """
    expr = expr_to_lit_or_expr(expr)
    pw = pywhen(expr._pyexpr)
    return When(pw)
