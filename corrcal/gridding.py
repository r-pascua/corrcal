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


def make_redundant_groups(u, v, tol=0.01, do_fof=True):
    """
    Find redundant groups given a uv-array and a specified tolerance.

    Parameters
    ----------
    u: array-like of float
        Eastward component of each baseline in units of wavelengths.
    v: array-like of float
        Northward component of each baseline in units of wavelengths.
    tol: float, optional
        Maximum difference between baselines within a redundant group.
        The exact type of distance specified here depends on the
        algorithm used. If friends-of-friends is used, then this
        specifies the Euclidean distance; otherwise, the uv-plane is
        gridded by the tolerance and this gridding is used to group the
        baselines. Default tolerance is 0.1 wavelengths.
    do_fof: bool, optional
        Whether to do the calculation using the friends-of-friends
        algorithm. Default is to use friends-of-friends.

    Returns
    -------
    sorting_key: np.ndarray of int
        Array with the same length as u and v that can be used to sort
        the uv arrays by redundant groups.
    edges: np.ndarray of int
        Array marking the edges of each redundant group when sorted
        according to ``sorting_key``. In order to retrieve the u-modes
        for the i-th redundant group, you would do:
            ``u[sorting_key][edges[i]:edges[i+1]]``
    is_conj: np.ndarray of bool
        Array with the same length as u and v, denoting which baselines
        need to be conjugated to abide by the v >= 0 convention. It is
        sorted according to the redundant grouping, so to conjugate
        e.g. the (unsorted) u-modes, you would do:
            ``u[sorting_key][is_conj] *= -1``
    """
    is_conj = v < 0
    uv = np.vstack([u, v]).T
    uv[is_conj] *= -1
    if PYFOF and do_fof:
        groups = pyfof.friends_of_friends(uv, tol)
        grouping_key = np.zeros_like(u).astype(int)
        for i, group in enumerate(groups):
            grouping_key[group] = i
        sorting_key = np.argsort(grouping_key)
        edges = np.where(np.diff(grouping_key[sorting_key]))[0].astype(int) + 1
    else:
        raise NotImplementedError("In development.")

