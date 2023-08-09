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
        self.noise = np.ascontiguousarray(noise, dtype=complex)
        self.src_mat = np.ascontiguousarray(src_mat, dtype=complex)
        self.diff_mat = np.ascontiguousarray(diff_mat, dtype=complex)
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


    def apply_gains(self, gains, ant_1_array, ant_2_array):
        """Apply complex gains to source and diffuse matrices."""
        gains = gains[::2] + 1j*gains[1::2]
        gains = gains[ant_1_array] * gains[ant_2_array].conj()
        self.diff_mat = gains[:,None] * self.diff_mat
        self.src_mat = gains[:,None] * self.src_mat


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

        tmp = linalg.mult_src_blocks_by_diffuse(
            Cinv.diff_mat, tmp, self.edges
        )
        if Cinv.isinv:
            tmp = Cinv.noise[:,None]*self.src_mat - tmp
            small_inv = np.eye(self.n_src).astype(complex) + (
                self.src_mat.T.conj() @ tmp
            )
        else:
            tmp = Cinv.noise[:,None]*self.src_mat + tmp
            small_inv = np.eye(self.n_src).astype(complex) - (
                self.src_mat.T.conj() @ tmp
            )

        small_inv = np.linalg.cholesky(small_inv)
        if return_det:
            logdet += np.log(np.diag(small_inv)).sum()

        small_inv = np.linalg.inv(small_inv)
        Cinv.src_mat = tmp @ small_inv.T.conj()
        if return_det:
            return Cinv, np.real(logdet)
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
