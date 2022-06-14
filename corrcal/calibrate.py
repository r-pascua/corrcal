import numpy as np
import warnings

from pyuvdata import UVData, UVCal

from . import io
from .sparse import Sparse2Level


class CorrCal:
    """Interface for running Correlation Calibration."""

    def __init__(self, *args, **kwargs):
        """
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
        from_yaml  # classmethod
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
        pass

def corrcal_run(*args, **kwargs):
    """Run the CorrCal pipeline given the appropriate parameters."""
    raise NotImplementedError
