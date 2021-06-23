import numpy as np
import pytest

from corrcal import gridding

def square_array(order=5, scale=1, jitter=0):
    antpos = {
        i + order * j: scale * np.array([i,j], dtype=float)
        for j in range(order) for i in range(order)
    }
    Nants = len(antpos)
    offsets = scale * np.random.normal(size=(Nants,2), scale=jitter)
    inds = np.arange(Nants)
    np.random.shuffle(inds)  # To randomize conjugation
    return {i: antpos[j] + offsets[j] for i, j in enumerate(inds)}


@pytest.fixture
def small_array():
    return square_array(3)


def get_baselines(antpos):
    Nants = len(antpos)
    return {
        (ai,aj): antpos[aj] - antpos[ai]
        for ai in range(Nants)
        for aj in range(ai, Nants)
    }


@pytest.mark.parametrize("do_fof", [False, True])
def test_redundancy_grouping(small_array, do_fof):
    # TODO: update to use non-perfect redundancy.
    bls = get_baselines(small_array)
    uv = np.array(list(bls.values())).T
    sorting_key, edges, is_conj = gridding.make_redundant_groups(
        *uv, do_fof=do_fof
    )
    groups_accurate = True
    sorted_uv = uv[:,sorting_key]
    sorted_uv[:,is_conj] *= -1
    for low, high in zip(edges[:-1], edges[1:]):
        this_group = sorted_uv[:,low:high]
        if this_group.shape[1] == 1:
            continue  # Only one baseline in this group.
        groups_accurate &= all(
            np.allclose(this_group[:,0], baseline)
            for baseline in this_group[:,1:].T
        )
    assert groups_accurate
