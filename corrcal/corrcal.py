"""
Module for managing correlation calibration.
"""
from pathlib import Path
import numpy as np
from . import gridding
from . import io
from . import linalg
from . import noise
from . import optimize
from . import sparse
from . import utils
from . import NOISE_MODELS


def corrcal_pipe(
    input_data,
    initial_gains=None,
    noise_model=None,
    sky_model=None,
    cov_model=None,
    algorithm="conjugate gradient",
    outdir=None,
    clobber=False,
    parallel=False,
    nproc=1,
    gpu=False,
):
    """
    Run correlation calibration on data given noise and sky models.

    Parameters
    ----------
    input_data: list of path-like objects
        Path(s) to input data. If it is a single file, then it is
        assumed that the data is intended to be loaded into a
        ``pyuvdata.UVData`` object or is a ``.npz`` archive. Otherwise,
        it should be a list of five binary files.
        See :func:`io.load_data` for details.
    initial_gains: path-like object, optional
        Path to either a binary file containing the gains or a
        :class:`pyuvdata.UVCal`-compatible file. If not provided, then
        an initial guess of unity gains is used.
    noise_model: str or path-like object, optional
        Either a path to a :class:`pyuvdata.UVData`-compatible file, a
        path to a `.npy` file, or a keyword for a built-in noise model.
        For details on file configuration, see :func:`io.load_noise`.
        For details on built-in noise models, see :mod:`noise`.
        Default is to estimate the noise from the provided data.
    sky_model: path-like object, optional
        A path to either a :class:`pyuvdata.UVData`-compatible file, a
        ``.npz`` archive containing the point source and redundant
        vectors that can be used to generate the sky covariance.
        Default is to not insert any point-source information,
        and to assume that every redundant group exhibits perfect
        redundancy (both positional and beam redundancy).
    cov_model: path-like object, optional
        A path to a binary or h5 file that can be loaded into a
        :class:`Sparse2Level` object. See :func:`io.read_sparse` for
        details on how the file should be formatted. Ignored if either
        a sky model or a noise model has been provided.
    algorithm: str, optional
        Name of the optimization algorithm to use. Default is to use
        a conjugate gradient solver.
    outdir: path-like object, optional
        Where to save the calibration products. Default is the same
        directory as the input data.
    clobber: bool, optional
        Whether to overwrite files that have name conflicts. Default is
        to not overwrite files.
    parallel: bool, optional
        Whether to perform calculations in parallel wherever possible.
        Default is to perform calculations in serial.
    nproc: int, optional
        Number of processors to use if calculations are performed in
        parallel. Ignored if ``parallel`` is set to ``False``.
    gpu: bool, optional
        Whether to use GPU acceleration wherever possible. GPU
        acceleration is not currently supported.
    """
    # Check the arguments.
    if not isinstance(input_data, (list, tuple)):
        raise TypeError("Input data should be a list of path-like objects.")
    if len(input_data) == 0:
        raise ValueError("No data provided.")
    else:
        data = io.load_data(input_data)  # Extra checks done in loading func.
    if noise_model:
        if Path(noise_model).exists():
            noise_model = io.load_noise(noise_model)
        elif noise_model in NOISE_MODELS:
            noise_model = NOISE_MODELS[noise_model]
        else:
            raise ValueError("Noise model not found.")
    if initial_gains:
        gains = io.load_gains(initial_gains)
        data.gains = gains
    else:
        data.gains = utils.basic_gain_model(data)
    if sky_model:
        sky_model = io.load_sky(sky_model)  # Handle bad files in load func.
    if cov_model and not (noise_model or sky_model):
        cov_model = io.read_sparse(cov_model)
        data.cov_model = cov_model
    if not cov_model:
        if not noise_model:
            noise_model = noise.estimate_noise(data)
        if not sky_model:
            sky_model = utils.basic_sky_model(data)
        data.noise_model = noise_model
        data.sky_model = sky_model
        data.build_covariance_model()
    calibration = corrcal_run(
        data, algorithm=algorithm, parallel=parallel, nproc=nproc, gpu=gpu,
    )
    io.write_cal(calibration, outdir=outdir)


def corrcal_run(
    data, algorithm="conjugate gradient", parallel=False, nproc=1, gpu=False,
):
    """
    Run correlation calibration.

    Parameters
    ----------
    data: <make a special calibration object>
        <write docstring after doing ^>
    algorithm: str, optional
        Name of an optimization algorithm. See :mod:`optimize` for
        supported optimization routines. Default is to use a conjugate
        gradient solver.
    parallel: bool, optional
        Whether to perform calculations in parallel wherever possible.
        Default is to perform calculations in serial.
    nproc: int, optional
        Number of processors to use when performing parallelized
        calculations. Ignored if parallelization is not used.
    gpu: bool, optional
        Whether to use GPU acceleration wherever possible. This feature
        is not currently supported.

    Returns
    -------
    calibration: <make a calibration object?>
        <write docstring after answering ^>
    """
    pass
