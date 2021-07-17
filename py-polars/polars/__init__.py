# mypy needs these imported explicitly
from polars.eager.frame import DataFrame, wrap_df  # noqa: F401
from polars.eager.series import Series, wrap_s  # noqa: F401
from polars.lazy.expr import Expr, wrap_expr  # noqa: F401
from polars.lazy.frame import LazyFrame, wrap_ldf  # noqa: F401

from . import datatypes, eager, io, lazy
from .datatypes import *  # noqa: F401, F403
from .eager import *  # noqa: F401, F403
from .io import *  # noqa: F401, F403
from .lazy import *  # noqa: F401, F403

# during docs building the binary code is not yet available
try:
    from .polars import version

    __version__ = version()
except ImportError:
    pass

__all__ = datatypes.__all__ + eager.__all__ + io.__all__ + lazy.__all__
