import ctypes
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
    isinv: bool
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
        isinv=False,
    ):
        self.noise_variance = deepcopy(noise_variance)
        self.diffuse_vectors = deepcopy(diffuse_vectors)
        self.source_vectors = deepcopy(source_vectors)
        # There is a note in the public version of corrcal that states
        # that not handling this carefully gives segmentation faults.
        # So keep an eye out...
        self.group_edges = group_edges.astype(int)
        self.isinv = isinv
        self.Ngroups = len(group_edges) - 1

    def copy(self):
        """Convenience method for creating a copy."""
        return Sparse2Level(
            self.noise_variance,
            self.diffuse_vectors,
            self.source_vectors,
            self.group_edges,
            self.isinv,
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
        product = np.zeros_like(vec)
        # TODO: Update this when extending to >1 freq channel and time
        Nobs = self.noise_variance.size
        Neig = self.diffuse_vectors.shape[0]
        Nsrc = self.source_vectors.shape[0]
        cfuncs.sparse_matrix_vector_multiplication(
            self.noise_variance.ctypes.data,
            self.diffuse_vectors.ctypes.data,
            self.source_vectors.ctypes.data,
            Nobs,
            Neig,
            Nsrc,
            self.Ngroups,
            self.group_edges.ctypes.data,
            self.isinv,
            vec.ctypes.data,
            product.ctypes.data,
        )
        return product

    def expand(self):
        """Generate the full covariance matrix.
        
        Returns
        -------
        cov: np.ndarray of float
            Full covariance matrix (sky covariance + noise variance).
        """
        # The source vectors have shape (Nsrc, 2 * Nbls). We initialize
        # the sky covariance with the contributions from point sources.
        sky_cov = self.source_vectors.T @ self.source_vectors
        edges = zip(self.group_edges[:-1], self.group_edges[1:])
        # Next, we add the contribution from the diffuse emission.
        for start, stop in edges:
            this_slice = slice(start, stop)
            this_block = (this_slice, this_slice)
            diffuse_signal = self.diffuse_vectors[:,this_slice]
            sky_cov[this_block] = diffuse_signal.T @ diffuse_signal
        if self.isinv:
            return np.diag(self.noise_variance) - sky_cov
        else:
            return np.diag(self.noise_variance) + sky_cov

    def invert(self):
        """Invert the sparse covariance and return the result."""
        inverse = self.copy()
        inverse.isinv = not self.isinv
        inverse.noise_variance = 1 / self.noise_variance
        Neig = self.diffuse_vectors.shape[0]
        Nsrc = self.source_vectors.shape[0]
        Nbls = self.noise_variance.size
        
        # The following is basically just copied straight from Jon's
        # implementation.
        # TODO: unpack this, make it make sense.
        tmp = np.zeros((self.Ngroups, Neig, Neig), dtype=float)
        cfuncs.make_all_small_blocks_c(
            self.noise_variance.ctypes.data,
            self.diffuse_vectors.ctypes.data,
            self.group_edges.ctypes.data,
            self.Ngroups,
            Nbls,
            Neig,
            tmp.ctypes.data,
        )

        block_identity = np.repeat(
            [np.eye(Neig)], self.Ngroups, axis=0
        )
        if self.isinv:
            tmp2 = block_identity - tmp
        else:
            tmp2 = block_identity + tmp

        cfuncs.many_chol_c(tmp2.ctypes.data, Neig, self.Ngroups)
        tmp3 = cfuncs.many_tri_inv(tmp2)
        tmp4 = cfuncs.mult_vecs_by_blocks(
            self.diffuse_vectors, tmp3, self.group_edges
        )

        for block in range(tmp4.shape[0]):
            tmp4[block] = tmp4[block] * inverse.noise_variance

        inverse.diffuse_vectors = tmp4
        for i in range(Nsrc):
            cfuncs.sparse_matrix_vector_multiplication(
                inverse.noise_variance.ctypes.data,
                inverse.diffuse_vectors.ctypes.data,
                inverse.source_vectors.ctypes.data,
                Nbls,
                Neig,
                0,  # Somehow this makes the block mult easy
                self.Ngroups,
                self.group_edges.ctypes.data,
                inverse.isinv,
                self.source_vectors[i].ctypes.data,
                tmp[i].ctypes.data,
            )

        small_mat = tmp * self.source_vectors.T  # This will need checking
        if self.isinv:
            small_mat = np.eye(Nsrc) - small_mat
        else:
            small_mat = np.eye(Nsrc) + small_mat
        small_mat = np.linalg.inv(np.linalg.cholesky(small_mat))
        inverse.source_vectors = small_mat * tmp  # This as well

        return inverse


    def apply_gains(self, gains, ant1, ant2):
        """
        Apply complex gains to the covariance.

        Parameters
        ----------
        gains: np.ndarray of float
            Figure out the proper format.
        ant1: np.ndarray of int
            Figure out the proper format.
        ant2: np.ndarray of int
            Same.
        """
        # Apply it to the diffuse emission
        cfuncs.apply_gains_to_matrix(
            self.diffuse_vectors.ctypes.data,
            gains.ctypes.data,
            ant1.ctypes.data,
            ant2.ctypes.data,
            self.diffuse_vectors.shape[1] // 2,
            self.diffuse_vectors.shape[0],
        )
        # Then to the point sources
        cfuncs.apply_gains_to_matrix(
            self.source_vectors.ctypes.data,
            gains.ctypes.data,
            ant1.ctypes.data,
            ant2.ctypes.data,
            self.source_vectors.shape[1] // 2,
            self.source_vectors.shape[0],
        )
