import numpy as np
import warnings

from pyuvdata import UVData, UVCal
from scipy.optimize import fmin_cg, minimize

from . import io, optimize, utils
from .sparse import Sparse2Level


class CorrCal:
    """Interface for running Correlation Calibration.

    Some things to think about...
    1) The proposed build is *not* BDA compliant
        - Time/baseline axis not separable for BDA in a single file/object
    2) Current implementation is per-time, per-freq, per-pol
        - Can treat as embarrassingly parallelizable task, but polarized
          calibration is planned for the near future
        - Do we parallelize here, or let the dispatcher parallelize?
    3) CHORD will likely inherit CHIME data format
    4) Will also want to deploy on HIRAX; need to know their conventions
    5) Want to eventually add GPU acceleration
    6) Tracking errors/uncertainties would be a nice feature

    Attributes
    ----------

    Data-like
    =========
    data_array[time,freq,pol,bl]  # complex
    flag_array[time,freq,pol]  # bool
    gain_array[time,freq,jpol,ant]  # complex
    src_mat[time,freq,pol,bl,src]  # complex; only if using sparse repr
    red_mat[time,freq,pol,bl,red]  # complex; only if using sparse repr
    covariance[time,freq,pol,bl,bl]  # complex; only if using dense repr
    noise[time,freq,pol,bl]  # complex
    n_samples[time,freq,pol,bl]  # tentative, uint
    vis_weights[time,freq,pol,bl]  # tentative, float
    gain_weights[time,freq,pol,bl]  # tentative, float

    Metadata
    ========
    freq_array[freq]  # float, Hz
    time_array[time]  # float, JD
    vispol_array[pol]  # int, polarization numbers
    jones_array[jpol]  # int, jones polarization numbers
    ant_1_array[bl]  # uint
    ant_2_array[bl]  # uint
    antenna_numbers[ant]  # uint
    antenna_names[ant]  # str
    antenna_positions[ant,3]  # float
    ant_1_inds[bl]  # uint
    ant_2_inds[bl]  # uint
    edges[red+1]  # uint
    src_ids[src]  # tentative, would help for selecting sources
    src_names[src]  # same note as above
    integration_time  # not BDA compliant
    channel_width  # could make an array
    Nwhatever  # antennas, freqs, times, etc...
    data_file
    cal_file
    cov_file
    history
    minimizer  # e.g. scipy.optimize.fmin_cg
    normalization  # e.g. determinant
    redundancy_tol  # for grouping
    dense  # as opposed to sparse
    dof  # for calculating reduced chi-squared if desired

    Methods
    -------
    from_file  # classmethod
    select
    write_cal  # for writing calibration solutions (i.e. gains)
    write_data  # for writing calibrated visibilities
    read_data  # need support for UVData and CHIME data products
    read_cal  # need support for UVCal and CHIME data products
    read_cov  # in-house
    read  # tentative; just a convenient interface
    comply_arrays  # make indexing/shapes agree
    sort_by_redundancy
    build_gain_matrix
    apply_gains
    invert
    chisq
    grad_chisq
    nll
    run_corrcal
    """

    def __init__(
        self,
        *
        data=None,
        cal=None,
        cov=None,
        file_format="hera",
        minimizer="CG",
        normalization="det",
        redundancy_tol=1.0,
    ):
        # Assign the basic things.
        self.minimizer = minimizer
        self.normalization = normalization
        self.redundancy_tol = redundancy_tol
        
        # Initialize all the attributes to be filled in later.
        self.Ntimes = None
        self.Nfreqs = None
        self.Npols = None
        self.Njones = None
        self.Nbls = None
        self.Nants = None
        self.Nsrc = None
        self.Neig = None
        self.Ngrp = None
        self.dense = None
        self.data = None
        self.gains = None
        self.cov = None
        self.freq_array = None
        self.time_array = None
        self.lst_array = None
        self.polarization_array = None
        self.jones_array = None
        self.ant_array = None
        self.ant_1_array = None
        self.ant_2_array = None
        self.ant_1_inds = None
        self.ant_2_inds = None
        self.edges = None

        cov = self._load_uvcov(cov)
        self.populate_attributes(
            data=data, cal=cal, cov=cov, file_format=file_format
        )
        
    def populate_attributes(self, data, cal, cov, file_format):
        if file_format.lower() == "hera":
            self._populate_from_hera_data(uvdata=data, uvcal=cal, uvcov=cov)
        elif file_format.lower() in ("chime", "chord"):
            self._populate_from_chime_data(data=data, cal=cal, cov=cov)
        elif file_format.lower() == "hirax":
            self._populate_from_hirax_data(data=data, cal=cal, cov=cov)
        else:
            raise NotImplementedError("File format not supported.")

        # The UVCov stuff should be contained within this package, so we
        # don't need special handling for the different file formats. That
        # said, we do need to provide it to the above _populate_x methods
        # to ensure compliance between the data sets.
        self.Nsrc = uvcov.Nsrc
        self.Neig = uvcov.Neig
        self.Ngrp = uvcov.Ngrp
        self.ant_1_inds = uvcov.ant_1_inds
        self.ant_2_inds = uvcov.ant_2_inds
        self.edges = uvcov.edges
        self.cov = uvcov.cov  # TODO: decide on a convention for this


    def _populate_from_hera_data(self, uvdata, uvcal, uvcov):
        # Assume we're using UVData/UVCal objects.
        uvdata = self._load_uvdata(uvdata)
        uvcal = self._load_uvcal(uvcal)
        uvcov = self._load_uvcov(uvcov)
        uvdata, uvcal, uvcov = self._comply_hera_data(
            uvdata=uvdata, uvcal=uvcal, uvcov=uvcov
        )  # Ensure times, freqs, antennas/baselines match.

        # Extract the "counting" metadata.
        self.Ntimes = uvdata.Ntimes
        self.Nfreqs = uvdata.Nfreqs
        self.Npols = uvdata.Npols
        self.Njones = uvcal.Njones
        self.Nbls = uvdata.Nbls
        self.Nants_data = uvdata.Nants_data
        self.Nants_telescope = uvdata.Nants_telescope

        # Extract the array-like metadata.
        self.freq_array = uvdata.freq_array.squeeze()
        times, inds = np.unique(uvdata.time_array, return_index=True)
        self.time_array = times
        self.lst_array = uvdata.lst_array[inds]
        self.ant_1_array = uvdata.ant_1_array
        self.ant_2_array = uvdata.ant_2_array
        self.ant_array = uvcal.ant_array

        # Extract the relevant data.
        self.data = uvdata.data_array.reshape(
            self.Ntimes, self.Nfreqs, self.Npols, self.Nbls
        )  # We want the baseline axis to be the fast axis.
        complex_gains = uvcal.gain_array.reshape(
            self.Ntimes, self.Nfreqs, self.Njones, self.Nants_data
        )  # This makes it easy to make the gain matrix.
        self.gains = np.zeros(
            (self.Ntimes, self.Nfreqs, self.Njones, 2*self.Nants_data),
            dtype=float
        )  # Fitting routine must act on real/imag parts independently.
        self.gains[...,:self.Nants_data] = complex_gains.real
        self.gains[...,self.Nants_data:] = complex_gains.imag
        self.flags = uvdata.flag_array.reshape(self.data.shape)
        self.n_samples = uvdata.n_samples.reshape(self.data.shape)
        self.data[self.flags] = 0  # Ignore flagged data in calibration.
        self.n_samples[self.flags] = 0
        

    def _populate_from_chime_data(self, data, cal, cov):
        raise NotImplementedError


    def _populate_from_hirax_data(self, data, cal, cov):
        raise NotImplementedError

    @classmethod
    def from_file(cls, config):
        """Create a CorrCal instance from a YAML configuration file.

        Parameters
        ----------
        config
            Path to the configuration file. See the example configuration
            file or the ReadTheDocs for details about what the configuration
            file must contain.

        Returns
        -------
        calibrator
            Instance of the CorrCal class, complete with all the attributes
            necessary for running Correlation Calibration.
        """
        with open(config) as cfg:
            contents = yaml.load(cfg.read(), Loader=yaml.SafeLoader)
        return CorrCal(**contents)


    def calibrate(
        self,
        minimizer=None,
        normalization=None,
        polarized=False,
        refant=None,
    ):
        minimizer = minimizer or self.minimizer
        normalization = normalization or self.normalization
        if refant is not None:
            callback = lambda gains: self.rephase_to_ant(gains, refant)
        else:
            callback = None

        # TODO: figure out how to parallelize efficiently
        if polarized:
            raise NotImplementedError
        else:
            for ti in range(self.Ntimes):
                for fi in range(self.Nfreqs):
                    for pi, pol in enumerate(self.polarization_array):
                        ji = [-5, -6, -7, -8].index(pol)
                        args = (
                            self.data[ti,fi,pi],
                        )
                        calibration = optimize(
                            self.nll,
                            self.gains[ti,fi,ji],
                            args=args,
                            method=minimizer,
                            jac=self.grad_nll,
                            hess=self.curv_nll,
                            callback=callback,
                        )
                        self.gains[ti,fi,pi,:] = calibration.x
                        if not calibration.success:
                            warnings.warn(calibration.message)


    def nll(self):
        return optimize.dense_nll if self.dense else optimize.nll


    def grad_nll(self):
        return optimize.dense_grad_nll if self.dense else optimize.grad_nll


    def curv_nll(self):
        return None  # Not yet supported.


    def rephase_to_ant(self, gains, refant):
        index = np.argwhere(self.ant_array == refant)
        return utils.rephase_to_ant(gains, index)


def corrcal_run(*args, **kwargs):
    """Run the CorrCal pipeline given the appropriate parameters."""
    raise NotImplementedError
