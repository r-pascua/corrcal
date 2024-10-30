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
        group, accounting for the real/imaginary split.
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
        self.noise = np.ascontiguousarray(noise, dtype=float)
        self.src_mat = np.ascontiguousarray(src_mat, dtype=float)
        self.diff_mat = np.ascontiguousarray(diff_mat, dtype=float)
        self.edges = np.array(edges, dtype=int)
        self.n_grp = edges.size - 1
        self.n_bls = src_mat.shape[0] // 2  # Alternating real/imag
        self.n_src = src_mat.shape[1]
        self.n_eig = n_eig
        self.diff_is_diag = diff_mat.shape == (2*self.n_bls, n_eig)
        self.isinv = isinv
        if not self.diff_is_diag:
            if diff_mat.shape != (2*self.n_bls, self.n_grp*n_eig):
                raise ValueError(
                    "Diffuse matrix shape is not understood. See class "
                    "docstring for information on expected shape."
                )
        if np.any(self.edges % 2):
            raise ValueError(
                "The `edges` array appears to be formatted incorrectly. "
                "Please ensure the `edges` array also accounts for the "
                "real/imaginary split along the baseline axis."
            )


    def __matmul__(self, other):
        """Multiply by a vector on the right."""
        return linalg.sparse_cov_times_vec(self, other)


    def apply_gains(self, gains, ant_1_array, ant_2_array):
        """Apply complex gains to source and diffuse matrices."""
        self.diff_mat = utils.apply_gains_to_mat(
            gains, self.diff_mat, ant_1_array, ant_2_array
        )
        self.src_mat = utils.apply_gains_to_mat(
            gains, self.src_mat, ant_1_array, ant_2_array
        )


    def copy(self):
        """Return a copy of the class instance."""
        return SparseCov(
            noise=self.noise.copy(),
            src_mat=self.src_mat.copy(),
            diff_mat=self.diff_mat.copy(),
            edges=self.edges.copy(),
            n_eig=self.n_eig,
            isinv=self.isinv,
        )


    def expand(self):
        """Return the dense covariance (i.e., multiply and add terms)."""
        if self.diff_is_diag:
            diff_mat = np.zeros(
                (2*self.n_bls, self.n_grp*self.n_eig), dtype=float
            )
            for grp in range(self.n_grp):
                start, stop = self.edges[grp:grp+2]
                left, right = np.arange(grp, grp+2) * self.n_eig
                diff_mat[start:stop,left:right] = self.diff_mat[start:stop]
        else:
            diff_mat = self.diff_mat

        cov = self.src_mat @ self.src_mat.T + diff_mat @ diff_mat.T

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

        tmp = linalg.mult_src_blocks_by_diffuse(
            Cinv.diff_mat, tmp, self.edges
        )
        if Cinv.isinv:
            tmp = Cinv.noise[:,None]*self.src_mat - tmp
            small_inv = np.eye(self.n_src) + self.src_mat.T @ tmp
        else:
            tmp = Cinv.noise[:,None]*self.src_mat + tmp
            small_inv = np.eye(self.n_src) - self.src_mat.T @ tmp

        small_inv = np.linalg.cholesky(small_inv)
        if return_det:
            logdet += np.log(np.diag(small_inv)).sum()

        small_inv = np.linalg.inv(small_inv)
        Cinv.src_mat = tmp @ small_inv.T
        if return_det:
            return Cinv, np.real(logdet)
        return Cinv
        

    def _full_inv(self, return_det=False):
        """Inversion routine for non-block-diagonal diffuse matrix."""
        raise NotImplementedError("Work in progress.")
