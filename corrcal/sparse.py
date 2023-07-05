import ctypes
from copy import deepcopy

import numpy as np

from . import linalg
from . import utils


class SparseCov:
    r"""
    Fill this out in full later. Rough notes for now:

    Attributes
    ----------
    noise
        The diagonal of the noise variance matrix. Expected to be
        a 1-d array of complex numbers.
    src_mat
        The :math:`$\sigma$` matrix containing information about point
        sources. See Eq. ?? of Pascua+ 23. Expected shape ``(n_bls, n_src)``.
    diff_mat
        The :math:`$\Delta$` matrix containing information about the
        sky angular power spectrum and array redundancy. See Eq. ?? of
        Pascua+ 23. If not including information about correlations between
        distinct quasi-redundant groups, then the expected shape is
        ``(n_bls, n_eig)``, and the matrix is interpreted as the
        block-diagonal entries of the diffuse matrix; otherwise, the shape
        should be ``(n_bls, n_grp*n_eig)``, and the matrix is interpreted
        as the full diffuse matrix.
    edges
        Array of integers denoting the edges of each quasi-redundant
        group.
    n_grp
        The number of quasi-redundant groups in the array.
    n_src
        The number of sources used in the model covariance.
    n_eig
        The number of eigenmodes used to represent each quasi-redundant group.
    n_bls
        The total number of baselines in the array.
    isinv
        Whether the matrix is the inverse of the covariance or not.
    """

    def __init__(self, noise, src_mat, diff_mat, edges, n_eig, isinv=False):
        """
        Decide how to split docs between class and constructor.
        """
        self.noise = noise.astype(complex)
        self.src_mat = src_mat.astype(complex)
        self.diff_mat = diff_mat.astype(complex)
        self.edges = np.array(edges, dtype=int)
        self.n_grp = edges.size - 1
        self.n_bls = src_mat.shape[0]
        self.n_src = src_mat.shape[1]
        self.n_eig = n_eig
        self.diff_is_diag = diff_mat.shape == (self.n_bls, n_eig)
        self.isinv = isinv
        if not self.diff_is_diag:
            if diff_mat.shape != (self.n_bls, self.n_grp*n_eig):
                raise ValueError(
                    "Diffuse matrix shape is not understood. See class "
                    "docstring for information on expected shape."
                )


    def __matmul__(self, other):
        """Multiply by a vector on the right."""
        return linalg.sparse_cov_times_vec(self, other)


    def copy(self):
        """Return a copy of the class instance."""
        return SparseCov(
            noise=self.noise,
            src_mat=self.src_mat,
            diff_mat=self.diff_mat,
            edges=self.edges,
            n_eig=self.n_eig,
            isinv=self.isinv,
        )


    def expand(self):
        """Return the dense covariance."""
        if self.diff_is_diag:
            diff_mat = np.zeros(
                (self.n_bls, self.n_grp*self.n_eig), dtype=complex
            )
            for grp in range(self.n_grp):
                start, stop = self.edges[grp:grp+2]
                left, right = np.arange(grp, grp+2) * self.n_eig
                diff_mat[start:stop,left:right] = self.diff_mat[start:stop]
        else:
            diff_mat = self.diff_mat

        cov = (
            self.src_mat @ self.src_mat.T.conj() + diff_mat @ diff_mat.T.conj()
        )
        if self.isinv:
            return np.diag(self.noise) - cov
        return np.diag(self.noise) + cov


    def inv(self, return_det=False):
        """Invert the covariance with the Woodbury identity.

        Parameters
        ----------
        return_det
            Whether to accumulate the determinant during the inversion.

        Returns
        -------
        Cinv
            Inverse of the covariance matrix stored in a sparse matrix.
        logdet
            Logarithm of the determinant. Returned if ``return_det == True``.
        """
        if return_det and self.isinv:
            raise NotImplementedError(
                "The determinant should only be needed when inverting the "
                "covariance, not when recovering the covariance from the "
                "inverse."
            )

        if self.diff_is_diag:
            return self._diag_inv(return_det=return_det)
        else:
            raise NotImplementedError("Work in progress.")
            return self._full_inv(return_det=return_det)

    def _diag_inv(self, return_det=False):
        """Inversion routine for when the diffuse matrix is block-diagonal."""
        # The noise is independent of the gains, so we can ignore it here.
        if return_det:
            logdet = 0

        # Initialize a new SparseCov
        Cinv = self.copy()
        Cinv.isinv = not self.isinv
        Cinv.noise = 1 / self.noise
        
        # Calculate 1 + D^\dag G^\dag Ninv DG.
        small_blocks = utils.make_small_blocks(
            noise_diag=self.noise,
            diff_mat=self.diff_mat,
            edges=self.edges,
        )
        if Cinv.isinv:
            small_blocks = np.eye(self.n_eig)[None,...] + small_blocks
        else:
            small_blocks = np.eye(self.n_eig)[None,...] - small_blocks

        small_blocks = np.linalg.cholesky(small_blocks)
        if return_det:
            logdet += sum(
                np.log(np.diag(block)).sum() for block in small_blocks
            )

        # This is faster than using np.linalg.inv
        small_inv = linalg.tril_inv(small_blocks).transpose(0,2,1).conj()
        
        # Calculate Ninv D L_D^{-1\dag}
        tmp = Cinv.noise[:,None] * self.diff_mat
        Cinv.diff_mat = linalg.block_multiply(tmp, small_inv, self.edges)

        # Now invert the source matrix.
        tmp = linalg.mult_src_by_blocks(
            Cinv.diff_mat.T.conj(), self.src_mat, self.edges
        )

        tmp = Cinv.diff_mat @ tmp
        if Cinv.isinv:
            tmp = Cinv.noise[:,None] * self.src_mat - tmp
            small_inv = np.eye(self.n_src).astype(complex) + (
                self.src_mat.T.conj() @ tmp
            )
        else:
            tmp = Cinv.noise[:,None] * self.src_mat + tmp
            small_inv = np.eye(self.n_src).astype(complex) - (
                self.src_mat.T.conj() @ tmp
            )

        small_inv = np.linalg.cholesky(small_inv)
        if return_det:
            logdet += np.log(np.diag(small_inv)).sum()

        small_inv = np.linalg.inv(small_inv)
        Cinv.src_mat = tmp @ self.src_mat @ small_inv.T.conj()
        if return_det:
            return Cinv, logdet
        return Cinv
        

    def _full_inv(self, return_det=False):
        """Inversion routine for non-block-diagonal diffuse matrix."""
        raise NotImplementedError("Work in progress.")

    def old_inv(self, return_det=False):
        """
        Efficiently invert the covariance with the Woodbury identity.

        This implementation of the inversion routine assumes that each quasi-
        redundant group is statistically independent of every other group (i.e.
        it is assumed that the phased beam kernel only negligibly overlaps for
        any pair of quasi-redundant groups).

        Parameters
        ----------
        return_det
            Whether to accumulate the determinant while doing the inversion.

        Returns
        -------
        Cinv
            Shape (n_bls,n_bls) inverse covariance matrix.
        det
            Logarithm of the determinant of the covariance matrix. Only
            returned if ``return_det`` is set to True.
        """
        if return_det:
            logdet = 0

        # Initialize the inverse covariance matrix.
        Cinv = np.zeros((self.n_bls, self.n_bls), dtype=complex)

        # Invert the quasi-redundant blocks.
        # NOTE: this will need to be updated if we remove the assumption that
        # quasi-redundant groups are mutually independent (i.e. we'll have to
        # actually do two rounds of Woodbury inversion).
        # TODO: parallelize this
        for grp, (start, stop) in enumerate(zip(self.edges, self.edges[1:])):
            # Figure out which section of the Delta matrix to use.
            left = grp * self.n_eig
            right = left + self.n_eig

            # Calculate the small block for this quasi-redundant group.
            block = self.diff_mat[start:stop, left:right].copy()
            block = np.diag(self.noise[start:stop]) + block @ block.T.conj()
            # See Eq. ?? of Pascua+ 22 for details on determinant calculation.
            if return_det:
                logdet += 2 * np.log(np.diag(np.linalg.cholesky(block))).sum()
            Cinv[start:stop, start:stop] = np.linalg.inv(block)

        # Finish the inversion with these next few lines.
        CGS = Cinv @ self.src_mat
        tmp = np.eye(self.n_src) + self.src_mat.T.conj() @ CGS
        Cinv -= CGS @ np.linalg.inv(tmp) @ CGS.T.conj()

        # Finish calculating the determinant if requested.
        if return_det:
            logdet += 2 * np.log(np.diag(np.linalg.cholesky(tmp))).sum()
            return Cinv, np.real(logdet)
        return Cinv


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
        *,
        noise_variance=None,
        diffuse_vectors=None,
        source_vectors=None,
        ant_1_array=None,
        ant_2_array=None,
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
            noise_variance=self.noise_variance,
            diffuse_vectors=self.diffuse_vectors,
            source_vectors=self.source_vectors,
            group_edges=self.group_edges,
            isinv=self.isinv,
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
            diffuse_signal = self.diffuse_vectors[:, this_slice]
            sky_cov[this_block] = diffuse_signal.T @ diffuse_signal
        if self.isinv:
            return np.diag(self.noise_variance) - sky_cov
        else:
            return np.diag(self.noise_variance) + sky_cov

    def inverse(self):
        """Invert the sparse covariance and return the result."""
        # TODO: rewrite this from scratch, do the math.
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

        block_identity = np.repeat([np.eye(Neig)], self.Ngroups, axis=0)
        if self.isinv:
            tmp2 = block_identity - tmp
        else:
            tmp2 = block_identity + tmp

        cfuncs.cholesky_factorization_parallel(
            tmp2.ctypes.data, Neig, self.Ngroups
        )
        tmp3 = linalg.many_tri_inv(tmp2)
        tmp4 = linalg.block_multiply(
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
