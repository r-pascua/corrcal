import numpy as np

from pyuvdata import UVData, UVCal

from . import io
from .sparse import Sparse2Level


class CorrCal:
    """Class for managing correlation calibration execution.

    Parameters
    ----------

    Attributes
    ----------

    """

    def __init__(
        self,
        data=None,
        cov=None,
        gains=None,
        ant_1_array=None,
        ant_2_array=None,
    ):
        # Initialize the various important attributes.
        self.data = None
        self.cov = None
        self.gains = None
        self.ant_1_array = None
        self.ant_2_array = None

        if data:
            self.load_data(data)
        if cov:
            self.load_cov(cov)
        if gains:
            self.load_gains(gains)
        if ant_1_array:
            self.load_ants(ant_1_array, 1)
        if ant_2_array:
            self.load_ants(ant_2_array, 2)

    def load_data(self, data):
        """Load the provided data into the ``data`` attribute."""
        if isinstance(data, np.ndarray):
            self.data = data
        else:
            self._read_data(data)

    def load_cov(self, cov):
        """Load the provided sparse covariance into the ``cov`` attribute."""
        if isinstance(cov, Sparse2Level):
            self.cov = cov
        else:
            self.cov = io.read_sparse(cov)

    def load_ants(self, ants, array_num):
        if isinstance(ants, np.ndarray):
            setattr(self, f"ant_{array_num}_array", ants)
        else:
            # Assume it's a binary for back-compat
            ants = io.read_binary(ants, "int64")
            self.load_ants(ants, array_num)

    def load_gains(self, gains):
        if isinstance(gains, np.ndarray):
            self.gains = gains
        else:
            self._read_gains(gains)

    @property
    def is_calibratable(self):
        return all(
            getattr(self, attr) is not None
            for attr in ("data", "cov", "gains", "ant_1_array", "ant_2_array",)
        )

    def _read_data(self, data):
        try:
            uvd = UVData()
            uvd.read(data)
            self._unpack_uvdata(uvd)
        except ValueError:
            # Assume it's a binary file, just for back-compat
            self.data = io.read_binary(data, "float64")

    def _read_gains(self, gains):
        try:
            uvc = UVCal()
            uvc.read_calfits(gains)
            self._unpack_uvcal(uvc)
        except OSError:
            # Not a calfits file, assume it's binary
            self.gains = io.read_binary(gains, "float64")

    def _unpack_uvdata(self, uvd):
        # First, assume single frequency/time, then extend to higher dims
        raise NotImplementedError

    def _unpack_uvcal(self, uvc):
        # Same note as _unpack_uvdata
        raise NotImplementedError
