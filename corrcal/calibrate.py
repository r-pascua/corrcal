import numpy as np
import warnings

from pyuvdata import UVData, UVCal

from . import io
from .sparse import Sparse2Level


class CorrCal:
    """Class for managing correlation calibration execution.

    Parameters
    ----------
    data
        The data to be calibrated. This can be provided either as a
        :class:`pyuvdata.UVData` object or path to a compatible file, or an
        array with shape (Ntimes, Nfreqs, Nbls, Npols). If an array is provided,
        then you must also provide the extra metadata.
    cov
        :class:`~.Sparse2Level` object or a path to a compatible file. This
        should contain the noise and model covariance.
    cal
        The calibration solutions to be updated in the calibration routine. This
        can be provided either as a :class:`pyuvdata.UVCal` object or path to a
        compatible file, or as an array with shape (Ntimes, Nfreqs, Nants, Npols).
    antpos
        Dictionary mapping antenna numbers to local ENU positions, or an 


    Attributes
    ----------

    """

    def __init__(
        self,
        *,
        data,
        cov,
        cal,
        antpos=None,
        ant_1_array=None,
        ant_2_array=None,
        redundant_groups=None,
        group_edges=None,
        redundancy_tol=1.0,
        frequency_array=None,
        time_array=None,
        polarization_array=None,
        source_model=None,
        diffuse_model=None,
        is_complex=False,
    ):
        # Initialize the various important attributes.
        self.data = None
        self.cov = None
        self.cal = None
        self.gain_mat = None
        self.antpos = antpos
        self.ant_1_array = None
        self.ant_2_array = None
        self.redundant_groups = redundant_groups
        self.group_edges = group_edges
        self.redundancy_tol = 1.0
        self.is_complex = is_complex
        self.is_grouped = False

        self.load_data(data)
        if ant_1_array:
            if self.ant_1_array is not None:
                warnings.warn("Antenna 1 array set from data. Ignoring input.")
            else:
                self.load_ants(ant_1_array, 1)
        if ant_2_array:
            if self.ant_2_array is not None:
                warnings.warn("Antenna 2 array set from data. Ignoring input.")
            else:
                self.load_ants(ant_2_array, 2)
        
        if not self.is_grouped:
            self._group_by_redundancy()

        # Load these after so that things are grouped by redundancy.
        if cov is not None:
            self.load_cov(cov)
        else:
            self._build_cov(source_model=source_model, diffuse_model=diffuse_model)

        self._sort_cov()
        self.load_cal(cal)
        
        # Use alternating real/imaginary if desired.
        if not self.is_complex:
            self._split_complex()

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

    def load_cal(self, cal):
        if isinstance(cal, np.ndarray):
            self.cal = cal
        else:
            self._read_cal(cal)

    @property
    def is_calibratable(self):
        return all(
            getattr(self, attr) is not None
            for attr in (
                "data",
                "cov",
                "cal",
                "ant_1_array",
                "ant_2_array",
                "redundancies",
                "group_edges",
            )
        )

    def _group_by_redundancy(self):
        pass

    def _read_data(self, data):
        try:
            uvd = UVData()
            uvd.read(data)
            self._unpack_uvdata(uvd)
        except ValueError:
            # Assume it's a binary file, just for back-compat
            self.data = io.read_binary(data, "float64")

    def _read_cal(self, cal):
        try:
            uvc = UVCal()
            uvc.read_calfits(cal)
            self._unpack_uvcal(uvc)
        except OSError:
            # Not a calfits file, assume it's binary
            self.gains = io.read_binary(gains, "float64")

    def _sort_cov(self):
        # TODO: docs
        # Idea here is to sort the covariance so that it matches the way the
        # data is sorted.
        #
        # I'll do the actual implementation later, but here's the outline
        data_bl_array = self.data.antnums_to_baseline(
            self.ant_1_array, self.ant_2_array
        )
        cov_bl_array = self.data.antnums_to_baseline(
            self.cov.ant_1_array, self.cov.ant_2_array
        )
        cov_to_data_key = np.array(
            cov_bl_array.tolist().index(bl) for bl in data_bl_array
        )
        apply_sort(sort=cov_to_data_key, self.cov.noise) # and so on

    def _unpack_uvdata(self, uvd):
        # TODO: extend to multiple freq/time/pol
        if uvd.Ntimes > 1:
            raise NotImplementedError
        if uvd.Nfreqs > 1:
            raise NotImplementedError
        if uvd.Npols > 1:
            raise NotImplementedError

        # Setup information to extract.
        Nbls = uvd.Nants_data * (uvd.Nants_data - 1) / 2
        ant_1_array = np.zeros(Nbls, dtype=int)
        ant_2_array = np.zeros(Nbls, dtype=int)
        data = np.zeros(Nbls, dtype=complex)
        redundancies = []
        group_edges = [0]

        # Fill out arrays, grouping by redundancy.
        groups, _, lengths = uvd.get_redundancies(
            tol=self.redundancy_tol, conjugate_bls=True
        )
        i = 0
        for j, (group, length) in enumerate(zip(groups, lengths)):
            if np.isclose(length, 0, rtol=0, atol=self.redundancy_tol):
                continue
            redundancies.append([])
            for baseline in group:
                ant1, ant2 = uvd.baseline_to_antnums(baseline)
                ant_1_array[i] = ant1
                ant_2_array[i] = ant2
                data[i] = uvd.get_data(ant1, ant2)[0,0]  # Single freq/time
                redundancies[j].append((ant1, ant2))
                i += 1
            group_edges.append(i)

        # Pull the antenna positions for good measure.
        pos, ants = uvd.get_ENU_antpos()
        self.antpos = dict(zip(ants, pos))

        # Set the attribute values.
        self.data = data
        self.ant_1_array = ant_1_array
        self.ant_2_array = ant_2_array
        self.redundant_groups = redundancies
        self.group_edges = group_edges

    def _unpack_uvcal(self, uvc):
        # Same note as _unpack_uvdata
        raise NotImplementedError
