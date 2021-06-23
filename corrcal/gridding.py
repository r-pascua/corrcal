"""
Module containing various gridding functions.
"""
import numpy as np

try:
    import hera_cal

    HERA_CAL = True
except (ImportError, FileNotFoundError) as err:  # pragma: no cover
    if issubclass(err, ImportError):
        missing = "hera_cal"
    else:
        missing = "git"
    print(f"{missing} is not installed. Some gridding features unavailable.")
    HERA_CAL = False

try:
    import pyfof

    PYFOF = True
except ImportError:  # pragma: no cover
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
        Maximum deviation of any baseline from the mean of all
        baselines within each redundant group. In other words, this
        specifies the size of the neighborhood used for calculating
        redundant groups. Default is one per-cent of a wavelength.
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
        for the i-th redundant group, you would do::
            ``u[sorting_key][edges[i]:edges[i+1]]``
    is_conj: np.ndarray of bool
        Array with the same length as u and v, denoting which baselines
        need to be conjugated to abide by the v >= 0 convention. It is
        sorted according to the redundant grouping, so to conjugate
        e.g. the (unsorted) u-modes, you would do::
            ``u[sorting_key][is_conj] *= -1``
    """
    is_conj = v < 0
    uv = np.vstack([u, v]).T
    uv[is_conj] *= -1
    if PYFOF and do_fof:
        reds = pyfof.friends_of_friends(uv, tol)
    else:
        reds, centers = [], []
        for i, uv_ in enumerate(uv):
            new_group = True
            for j, (red, center) in enumerate(zip(reds, centers)):
                if np.linalg.norm(uv_ - center) <= tol:
                    # Update the center of the redundant group.
                    N = len(red)
                    centers[j] = (N * center + uv_) / (N + 1)
                    # Add the baseline's index to the list of reds.
                    red.append(i)
                    new_group = False
                    break
            if new_group:
                reds.append([i])
                centers.append(uv_)

        # Recursively trim the set to ensure convergence.
        converged = False
        while not converged:
            new_reds, new_centers = [], []
            skip = []
            for i, (red, center) in enumerate(zip(reds, centers)):
                # Skip groups that have already been merged.
                if i in skip:
                    continue
                merge_groups = False
                for j, (other_red, other_center) in enumerate(
                    zip(reds, centers)
                ):
                    if i == j:
                        continue
                    if np.linalg.norm(center - other_center) <= tol:
                        merge_groups = True
                        skip.append(j)
                        break
                # Update the redundant group and its center.
                if merge_groups:
                    N1, N2 = len(red), len(other_red)
                    red += other_red
                    center = np.average(
                        [center, other_center], axis=0, weights=[N1, N2]
                    )
                new_reds.append(red)
                new_centers.append(center)
            if len(reds) == len(new_reds):
                converged = True
            reds = new_reds
            centers = new_centers
    grouping_key = np.zeros_like(u).astype(int)
    for i, group in enumerate(reds):
        grouping_key[group] = i
    sorting_key = np.argsort(grouping_key)
    edges = np.where(np.diff(grouping_key[sorting_key]))[0].astype(int) + 1
    is_conj = is_conj[sorting_key]
    return sorting_key, edges, is_conj
