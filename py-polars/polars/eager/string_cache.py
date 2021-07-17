from types import TracebackType
from typing import Optional, Type

try:
    from ..polars import toggle_string_cache as pytoggle_string_cache
except ImportError:
    import warnings

    warnings.warn("binary files missing")


__all__ = [
    "StringCache",
    "toggle_string_cache",
]


class StringCache:
    """
    Context manager that allows data sources to share the same categorical features.
    This will temporarily cache the string categories until the context manager is finished.
    """

    def __init__(self) -> None:
        pass

    def __enter__(self) -> "StringCache":
        pytoggle_string_cache(True)
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        pytoggle_string_cache(False)


def toggle_string_cache(toggle: bool) -> None:
    """
    Turn on/off the global string cache. This ensures that casts to Categorical types have the categories when string
    values are equal.
    """
    pytoggle_string_cache(toggle)
