import ctypes
import numpy as np
from pathlib import Path
from numpy.typing import NDArray
from typing import Sequence, NoReturn, Optional
from . import linalg
from . import _cfuncs


def apply_gains_to_mat(
    gains: NDArray[float],
    mat: NDArray[float],
    ant_1_array: NDArray[int],
    ant_2_array: NDArray[int],
) -> NDArray[float]:
    """Apply per-antenna gains to a per-baseline matrix.

    Parameters
    ----------
    gains
        Per-antenna gains, arranged into alternating real/imag parts.
    mat
        Matrix to apply the gains to, with alternating real/imag parts
        along the baseline axis. The matrix is assumed to index over
        baselines along the zeroth axis.
    ant_1_array, ant_2_array
        Index arrays indicating which antennas are used in each baseline.

    Returns
    -------
    out
        Input matrix with the provided gains applied.
    """
    complex_gains = gains[::2] + 1j*gains[1::2]
    gain_mat = (
        complex_gains[ant_1_array] * complex_gains[ant_2_array].conj()
    )[:,None]
    out = np.zeros_like(mat)
    out[::2] = gain_mat.real * mat[::2] - gain_mat.imag * mat[1::2]
    out[1::2] = gain_mat.imag * mat[::2] + gain_mat.real * mat[1::2]
    return out
    

def check_parallel(parallel: bool, gpu: bool) -> NoReturn:
    """
    Ensure that only parallelization or GPU acceleration is requested.

    Parameters
    ----------
    parallel: bool
        Whether to perform the operation in parallel.
    gpu: bool
        Whether to use GPU acceleration.
    """
    if parallel and gpu:
        raise ValueError(
            "CPU parallelization and GPU acceleration cannot be "
            "performed simultaneously."
        )


def build_baseline_array(
    ant_1_array: NDArray[int],
    ant_2_array: NDArray[int],
    antpos: NDArray[float],
    antnums: Sequence,
):
    """Calculate all the baseline vectors for the provided parameters.

    Parameters
    ----------
    ant_1_array, ant_2_array
        Index arrays indicating which antennas are used in each baseline.
    antpos
        Array with shape ``(Nants, 3)`` giving the position, in meters,
        of each antenna in the array in a local ENU frame.
    antnums
        Iterable indicating the label of each antenna whose position is
        provided in the ``antpos`` array.

    Returns
    -------
    baselines
        Array with shape ``(N_baseline, 3)`` containing the baseline
        vectors for each pair of antennas. Baselines are calculated using
        the "j-i" convention, where :math:`b_{ij} = x_j - x_i`.
    """
    ant_1_inds = np.zeros_like(ant_1_array)
    ant_2_inds = np.zeros_like(ant_2_array)
    for i, ant in enumerate(antnums):
        ant_1_inds[ant_1_array == ant] = i
        ant_2_inds[ant_2_array == ant] = i
    return antpos[ant_2_inds] - antpos[ant_1_inds]


def rephase_to_ant(
    gains: NDArray[float] | NDArray[complex], ant: Optional[int] = 0
) -> NDArray[float] | NDArray[complex]:
    """Rephase gains to a reference antenna."""
    if np.iscomplexobj(gains):
        complex_gains = gains.copy()
    else:
        complex_gains = build_complex_gains(gains)
    rephased_complex_gains = complex_gains * np.exp(
        -1j * np.angle(complex_gains[ant])
    )
    if np.iscomplexobj(gains):
        # If input is complex, output should be as well
        return rephased_complex_gains
    else:
        rephased_gains = np.zeros(2*complex_gains.size)
        rephased_gains[::2] = rephased_complex_gains.real
        rephased_gains[1::2] = rephased_complex_gains.imag
        return rephased_gains


def comply_shape(mat: NDArray[float]) -> NoReturn:
    """Check that the provided array has the right shape."""
    if mat.shape[-1] != mat.shape[-2]:
        raise ValueError("Array is not square!")


def fetch_models(
	data_lsts: NDArray[float], model_cov_dir: Path, file_prototype: str
) -> list[Path]:
	"""
	Retrieve relevant model files provided observed LSTs.

	Parameters
	----------
	data_lsts
		Observed Local Sidereal Times, in radians.
	model_cov_dir
		Where the model covariance files are located on the filesystem.
	file_prototype
		Glob-parsable string that may be used to fetch model files.

	Returns
	-------
	cov_files
		List of files containing relevant model covariance.

	Notes
	-----
	This function assumes that the model covariance files indicate the
	first LST, in radians, in the file in the file names themselves, with
	the phase wrap occurring at 2pi. More precisely, this function looks
	for the substring "\d.\d+" in the model covariance file name and assumes
	that the first instance of this substring indicates the file start LST.
	"""
	all_model_files = sorted(model_cov_dir.glob(file_prototype))
	start_lsts = np.array(
		[float(re.findall("\d.\d+", fn.name)[0]) for fn in all_model_files]
	)

	# Find the nearest file preceding the first LST in the data.
	model_phasors = np.exp(1j * start_lsts)
	start_dlst = np.angle(model_phasors * np.exp(-1j*data_lsts[0]))
	start = np.argmin(np.abs(start_dlst))
	if start_dlst[start] > 0:
		start -= 1

	# Now find the nearest file following the last LST in the data.
	end_dlst = np.angle(model_phasors * np.exp(-1j*data_lsts[-1]))
	end = np.argmin(np.abs(end_dlst))
	if end_dlst[end] < 0:
		end += 1

	# Retrieve the model files.
	if start == -1:
		return all_model_files[-1:] + all_model_files[:end+1]
	elif end == len(all_model_files):
		return all_model_files[start:] + all_model_files[:2]
	elif stop < start:
		return all_model_files[start:] + all_model_files[:end+1]
	else:
		return all_model_files[start:end+1]
