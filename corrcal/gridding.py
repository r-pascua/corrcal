"""
Module containing various gridding functions.
"""
import numpy as np

try:
    import hera_cal

    HERA_CAL = True
except (ImportError, FileNotFoundError) as err:  # pragma: no cover
    if isinstance(err, ImportError):
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
    # TODO: this will need to be expanded if we allow w =/= 0.
    # TODO: write unit tests for edge cases
    is_conj = v < 0
    is_conj |= np.isclose(v, 0) & (u < 0)
    uv = np.vstack([u, v]).T
    uv[is_conj] *= -1
    if PYFOF and do_fof:
        reds = pyfof.friends_of_friends(uv, tol)
    else:
        reds = calc_reds_from_uvs(uv, tol)
    grouping_key = np.zeros_like(u).astype(int)
    for i, group in enumerate(reds):
        grouping_key[group] = i
    sorting_key = np.argsort(grouping_key)
    edges = np.where(np.diff(grouping_key[sorting_key]))[0].astype(int) + 1
    edges = np.concatenate([[0], edges, [len(u) + 1]])
    is_conj = is_conj[sorting_key]
    return sorting_key, edges, is_conj


def calc_reds_from_uvs(uv, tol=0.01):
    """Calculate redundant groups from an array of uv-coordinates.

    Parameters
    ----------
    uv
        Array containing the uv-coordinates for each baseline. Different
        baselines should be accessed along the zero-th axis, so that the
        array has shape (Nbls, 2).
    tol
        Radius of each quasi-redundant group (measured relative to the
        mean of all baselines within the group), in units of wavelengths.
        Default is 1/100 of a wavelength.

    Returns
    -------
    reds
        List of the quasi-redundant groups, with each entry being a list
        of the baselines within the redundant group.
    """
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
            for j, (other_red, other_center) in enumerate(zip(reds, centers)):
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
    return reds


def parse_reds(
    reds,
    bl_lens,
    min_bl_len=0,
    max_bl_len=np.inf,
    min_group_size=1,
    tol=1.0,
):
    """TODO: write doc"""
    ant_1_array = []
    ant_2_array = []
    edges = [0,]
    idx = 0
    for group, bl_len in zip(reds, bl_lens):
        bl_len_not_ok = (bl_len < min_bl_len) or (bl_len > max_bl_len)
        if bl_len_not_ok or (len(group) < min_group_size):
            continue

        for (ai, aj) in group:
            ant_1_array.append(ai)
            ant_2_array.append(aj)
            idx += 1
        edges.append(idx)

    # Convert to numpy arrays and return the sorted arrays.
    ant_1_array = np.array(ant_1_array)
    ant_2_array = np.array(ant_2_array)
    edges = 2 * np.array(edges)  # Since the data is split into re/im components
    return ant_1_array, ant_2_array, edges


def make_groups_from_uvdata(
    uvdata, min_bl_len=0, max_bl_len=np.inf, min_group_size=1, tol=1.0
):
    """TODO: write doc"""
    _reds, bl_vecs, bl_lens, conj = uvdata.get_redundancies(
        include_conjugates=True, tol=tol
    )
    conj = set(conj)
    reds = []
    for group, vec in zip(_reds, bl_vecs):
        grp = []
        for bl in group:
            ai, aj = uvdata.baseline_to_antnums(bl)
            if bl in conj:
                ai, aj = aj, ai
            grp.append((ai,aj))
        if vec[0] < 0:  # Assert u > 0
            grp = [(aj, ai) for ai, aj in grp]
        reds.append(grp)

    return parse_reds(
        reds=reds,
        bl_lens=bl_lens,
        min_bl_len=min_bl_len,
        max_bl_len=max_bl_len,
        min_group_size=min_group_size,
        tol=tol,
    )


def make_groups_from_antpos(
    antpos, min_bl_len=0, max_bl_len=np.inf, min_group_size=1, tol=1.0
):
    """TODO: write doc"""
    antenna_numbers = np.array(list(antpos.keys()))
    antenna_positions = np.array(list(antpos.values()))
    try:
        from pyuvdata.utils.redundancy import get_antenna_redundancies
        from pyuvdata.utils import baseline_to_antnums
        HAVE_PYUVDATA = True
    except ImportError:
        HAVE_PYUVDATA = False

    if HAVE_PYUVDATA:
        reds, _, bl_lens = get_antenna_redundancies(
            antenna_numbers=antenna_numbers,
            antenna_positions=antenna_positions,
            tol=tol,
        )  # Baselines in u > 0 convention with b_ij = x_j - x_i
        reds = [
            [baseline_to_antnums(bl, Nants_telescope=1000) for bl in grp]
            for grp in reds
        ]
    else:
        raise NotImplementedError
        
    return parse_reds(
        reds=reds,
        bl_lens=bl_lens,
        min_bl_len=min_bl_len,
        max_bl_len=max_bl_len,
        min_group_size=min_group_size,
        tol=tol,
    )
