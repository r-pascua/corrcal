from pathlib import Path
import logging

from . import calibrate
from . import cfuncs
from . import gridding
from . import io
from . import linalg
from . import noise
from . import optimize
from . import parsers
from . import pipeline
from . import sparse
from . import utils

from .calibrate import CorrCal
from .sparse import Sparse2Level

LOGGER = logging.Logger(name="corrcal_logger", level=logging.WARN)
NOISE_MODELS = {}

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
    print("corrcal is not installed.")
    pass
