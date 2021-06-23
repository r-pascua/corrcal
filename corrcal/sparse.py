from copy import deepcopy
import numpy as np

from . import cfuncs


class Sparse2Level:
    """
    A class for managing the covariance matrices used for calibration.

    Attributes
    ----------
    noise_variance: np.ndarray of float
        Per-baseline noise variance.
    diffuse_vectors: np.ndarray of float
        Vectors describing the diffuse emission on the sky.
    source_vectors: np.ndarray of float
        Vectors describing the point-source emission.
    group_edges: np.ndarray of int
        Array denoting the edges of the redundant groups. The slice
        ``redundant_group_edges[i]:redundant_group_edges[i+1]`` gives
        the data (or uv-modes) for the i-th redundant group.
    is_inverse: bool
        Whether the matrix has already been inverted (this has to do
        with the Woodbury decomposition and dictates the sign with
        which the noise enters the covariance). (Explain this further.)
    """

    def __init__(
        self,
        noise_variance=None,
        diffuse_vectors=None,
        source_vectors=None,
        group_edges=None,
        is_inverse=False,
    ):
        self.noise_variance = deepcopy(noise_variance)
        self.diffuse_vectors = deepcopy(diffuse_vectors)
        self.source_vectors = deepcopy(source_vectors)
        # There is a note in the public version of corrcal that states
        # that not handling this carefully gives segmentation faults.
        # So keep an eye out...
        self.group_edges = group_edges.astype(int)
        self.is_inverse = is_inverse
        self.Ngroups = len(group_edges) - 1

    def copy(self):
        """Convenience method for creating a copy."""
        return Sparse2Level(
            self.noise_variance,
            self.diffuse_vectors,
            self.source_vectors,
            self.group_edges,
            self.is_inverse,
        )

    def __mul__(self, vec):
        """
        Multiply a vector by the covariance from the left.


        Parameters
        ----------
        vec: np.ndarray
            Vector to multiply by the covariance.

        Notes
        -----
        I believe this is used in calculating chi-squared, though this
        needs to be investigated a bit further.
        """
        pass
        # Setup to use the C implementation.
        # out = np.zeros_like(vec)
        # Nobs = self.noise_variance.size
        # Neig = self.diffuse_vectors.shape[0]
