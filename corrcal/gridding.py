"""
Module containing various gridding functions.
"""
import numpy as np

try:
    import hera_cal

    HERA_CAL = True
except (ImportError, FileNotFoundError) as err:
    if issubclass(err, ImportError):
        missing = "hera_cal"
    else:
        missing = "git"
    print(f"{missing} is not installed. Some gridding features unavailable.")
    HERA_CAL = False

try:
    import pyfof

    PYFOF = True
except ImportError:
    print("pyfof is not installed. Some gridding features unavailable.")
    PYFOF = False
