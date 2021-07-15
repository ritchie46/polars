from . import datatypes, eager, functions, lazy
from .datatypes import *  # noqa: F401, F403
from .eager import *  # noqa: F401, F403
from .functions import *  # noqa: F401, F403
from .lazy import *  # noqa: F401, F403

# during docs building the binary code is not yet available
try:
    from .polars import version

    __version__ = version()
except ImportError:
    pass

__all__ = datatypes.__all__ + eager.__all__ + functions.__all__ + lazy.__all__
