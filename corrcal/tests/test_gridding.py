import numpy as np
import pytest

from corrcal import gridding

def test_bl_grouping(bls, edges):
    groups_ok = []
    for start, stop in zip(edges//2, edges[1:]//2):
        avg_bl = bls[start:stop].mean(axis=0)
        proj = bls[start:stop] @ avg_bl / np.linalg.norm(avg_bl)**2
        groups_ok.append(np.allclose(proj, 1, atol=0.001))
    assert all(groups_ok)
