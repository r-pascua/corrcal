import numpy as np
from numpy.typing import NDArray
from typing import Sequence, Optional

def parse_reds(
    reds: Sequence[Sequence[tuple[int,int]]],
    bl_lens: Sequence[float],
    min_bl_len: Optional[float] = 0,
    max_bl_len: Optional[float] = np.inf,
    min_group_size: Optional[int] = 1,
) -> tuple[NDArray[int], NDArray[int], NDArray[int]]:
    """
    Select out redundant groups that conform to certain properties.

    This function will take a list of redundant groups and the lengths
    of their corresponding representative baseline, then reduce the list
    to only include groups fitting within a specified baseline length
    range (so one can choose to exclude short and/or long baselines
    beyond some bound) or groups with a minimum number of members.

    Parameters
    ----------
    reds
        List of baseline groups; each group should be a list of tuples
        indicating the first and second antenna participating in each
        baseline.
    bl_lens
        The length of the average baseline for each redundant group.
    min_bl_len
        The minimum baseline length to use for calibration; all groups
        with a baseline length less than this are discarded.
    max_bl_len
        The maximum baseline length to use for calibration; all groups
        with a baseline length greater than this are discarded.
    min_group_size
        The minimum number of baselines required to include a redundant
        group in calibration.

    Returns
    -------
    ant_1_array, ant_2_array
        Index arrays mapping baseline indices to antenna numbers, sorted
        into redundant groups.
    edges
        Array indicating the edges of each redundant group for an
        array that has been split into alternating real/imag components
        and sorted by redundancy.
    """
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
    edges = 2 * np.array(edges)  # Since the data is split into re/im
    return ant_1_array, ant_2_array, edges


def make_groups_from_uvdata(
    uvdata: "UVData",
	min_bl_len: Optional[float] = 0,
	max_bl_len: Optional[float] = np.inf,
	min_group_size: Optional[int] = 1,
	tol: Optional[float] = 1.0,
):
    """
    Construct redundant groups from a ``pyuvdata.UVData`` object.

	This function uses ``uvdata.get_redundancies`` to compute redundant
	groups, then trims these down to obey the provided constraints via
	:func:`parse_reds`.

	Parameters
	----------
	uvdata
		:class:`pyuvdata.UVData` instance containing the relevant array
		metadata for sorting baselines into redundant groups.
    min_bl_len
        The minimum baseline length to use for calibration; all groups
        with a baseline length less than this are discarded.
    max_bl_len
        The maximum baseline length to use for calibration; all groups
        with a baseline length greater than this are discarded.
    min_group_size
        The minimum number of baselines required to include a redundant
        group in calibration.
	tol
		Maximum allowed discrepancy between baselines within a redundant
		group, in meters.

    Returns
    -------
    ant_1_array, ant_2_array
        Index arrays mapping baseline indices to antenna numbers, sorted
        into redundant groups.
    edges
        Array indicating the edges of each redundant group for an
        array that has been split into alternating real/imag components
        and sorted by redundancy.
    """
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
    )


def make_groups_from_antpos(
    antpos: dict[int,NDArray[float]],
	min_bl_len: Optional[float] = 0,
	max_bl_len: Optional[float] = np.inf,
	min_group_size: Optional[int] = 1,
	tol: Optional[float] = 1.0,
):
    """
	Construct redundant groups from a provided array layout.

	This function computes all of the baselines for the provided array
	layout, then determines redundant groupings (within the provided
	constraints) for the full set of baselines.
	
	Parameters
	----------
	antpos
		Mapping from antenna numbers to antenna positions.
    min_bl_len
        The minimum baseline length to use for calibration; all groups
        with a baseline length less than this are discarded.
    max_bl_len
        The maximum baseline length to use for calibration; all groups
        with a baseline length greater than this are discarded.
    min_group_size
        The minimum number of baselines required to include a redundant
        group in calibration.
	tol
		Maximum allowed discrepancy between baselines within a redundant
		group, in meters.

    Returns
    -------
    ant_1_array, ant_2_array
        Index arrays mapping baseline indices to antenna numbers, sorted
        into redundant groups.
    edges
        Array indicating the edges of each redundant group for an
        array that has been split into alternating real/imag components
        and sorted by redundancy.
	"""
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
    )
