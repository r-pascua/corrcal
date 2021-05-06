from pathlib import Path
import logging
LOGGER = logging.Logger(name="corrcal_logger", level=logging.WARN)

# Ripped from hera_sim
try:
    from importlib.metadata import version, PackageNotFoundError
except ImportError:
    from importlib_metadata import version, PackageNotFoundError

try:
    DATA_PATH = Path(__file__).parent / "data"
    SRC_PATH = Path(__file__).parent / "src"
    __version__ = version(__name__)
except PackageNotFoundError:
    print("Package not found.")
    pass


from . import cfuncs
from . import corrcal
from . import io
from . import optimize
from . import sparse
from . import utils
from .sparse import Sparse2Level
