import ctypes
from copy import deepcopy

import numpy as np
from numpy.typing import NDArray
from typing import Optional, Sequence

from . import _cfuncs
from . import linalg
from . import utils


class SparseCov:
    r"""
    Class for the doubly sparse representation of the CorrCal covariance.

    Attributes
    ----------
    noise
        The diagonal of the noise variance matrix. Expected to be
        a 1-d array of complex numbers.
    src_mat
        The :math:`\Sigma` matrix containing the source vectors (see
        Equations 49 and 50 in Pascua+ 2025). Expected shape `(n_bls, n_src)`.
    diff_mat
        The :math:`$\Delta$` matrix containing information about the
        sky angular power spectrum and array redundancy. Currently, the
        diffuse matrix is assumed to be block diagonal, as in Equation 45
        of Pascua+ 2025; there is not yet support for including off-diagonal
        blocks. The expected shape is `(n_bls, n_eig)`, where `n_eig` should
        be a multiple of two (i.e., the eigenmodes for both the real-real and
        imaginary-imaginary covariance must be provided).
    edges
        Array of integers denoting the edges of each quasi-redundant
        group, accounting for the real/imaginary split.
    n_grp
        The number of redundant groups in the array.
    n_src
        The number of sources used in the model covariance.
    n_eig
        The number of eigenmodes used to represent each redundant group.
    n_bls
        The total number of baselines in the array.
    isinv
        Whether the matrix is the inverse of the covariance or not.

    Methods
    -------
    __matmul__
        Left multiplication acting on a vector (i.e., C @ v).
    apply_gains
        Scale diffuse/source matrices by per-antenna gains.
    copy
        Return a copy of self.
    expand
        Return the dense covariance as a numpy array.
    inv
        Compute sparse inverse representation and return as SparseCov.
    """

    def __init__(
            self,
            noise: NDArray[float],
            src_mat: NDArray[float],
            diff_mat: NDArray[float],
            edges: NDArray[int],
            n_eig: int,
            isinv: bool = False
        ) -> None:
        """
        Create a SparseCov object.

        Parameters
        ----------
        noise
            Diagonal of the thermal noise variance matrix.
        src_mat
            Matrix of source vectors encoding array response to point sources.
        diff_mat
            Block-diagonal entries of diffuse matrix.
        edges
            Array indicating the start and end of each redundant group.
        n_eig
            Number of eigenmodes used for each redundant group. This is 
            currently fixed to be uniform across groups.
        isinv
            Whether this object represents the inverse representation.
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


    def __matmul__(self, other: NDArray[float]) -> NDArray[float]:
        """Multiply by a vector on the right."""
        return linalg.sparse_cov_times_vec(self, other)


    def apply_gains(
            self,
            gains: NDArray[float],
            ant_1_array: NDArray[int],
            ant_2_array: NDArray[int],
        ) -> None:
        """Apply complex gains to source and diffuse matrices.

        Parameters
        ----------
        gains
            Per-antenna gains in alternating real/imaginary format.
        ant_1_array, ant_2_array
            Index arrays indicating which pair of antennas are used for
            each baseline.
        """
        self.diff_mat = utils.apply_gains_to_mat(
            gains, self.diff_mat, ant_1_array, ant_2_array
        )
        self.src_mat = utils.apply_gains_to_mat(
            gains, self.src_mat, ant_1_array, ant_2_array
        )


    def copy(self) -> "SparseCov":
        """Return a copy of the class instance."""
        return SparseCov(
            noise=self.noise.copy(),
            src_mat=self.src_mat.copy(),
            diff_mat=self.diff_mat.copy(),
            edges=self.edges.copy(),
            n_eig=self.n_eig,
            isinv=self.isinv,
        )


    def expand(self) -> NDArray[float]:
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


    def inv(self, return_det: bool = False) -> "SparseCov":
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
            raise NotImplementedError(
                "Inverting with a non-block-diagonal diffuse matrix is"
                "not yet supported."
            )
            return self._full_inv(return_det=return_det)


    def _diag_inv(self, return_det: bool = False) -> NDArray[float]:
        """Inversion routine for when the diffuse matrix is block-diagonal."""
        # The noise is independent of the gains, so we can ignore it here.
        if return_det:
            logdet = 0

        # Initialize a new SparseCov
        Cinv = self.copy()
        Cinv.isinv = not self.isinv
        Cinv.noise = 1 / self.noise
        
        # Calculate 1 + D.T @ Ninv @ D; first the small blocks.
        small_blocks = linalg.make_small_blocks(
            noise_diag=self.noise,
            diff_mat=self.diff_mat,
            edges=self.edges,
        )
        
        # Then add the identity and take the Cholesky decomposition.
        if Cinv.isinv:
            small_blocks = np.eye(self.n_eig)[None,...] + small_blocks
        else:
            small_blocks = np.eye(self.n_eig)[None,...] - small_blocks

        small_blocks = np.linalg.cholesky(small_blocks)
        if return_det:
            logdet += linalg.sum_diags(small_blocks)

        # This is faster than using np.linalg.inv
        small_inv = linalg.tril_inv(small_blocks).transpose(0,2,1).copy()
        
        # Calculate Ninv D L_D^{-1\dag}
        tmp = Cinv.noise[:,None] * self.diff_mat
        Cinv.diff_mat = linalg.block_multiply(tmp, small_inv, self.edges)

        # Now invert the source matrix.
        # First, compute Delta'.T @ Sigma.
        tmp = linalg.mult_src_by_blocks(
            Cinv.diff_mat.T.copy(), self.src_mat, self.edges
        )

        # Next, compute Delta' @ Delta'.T @ Sigma.
        tmp = linalg.mult_src_blocks_by_diffuse(
            Cinv.diff_mat, tmp, self.edges
        )

        # Then include the noise to construct \tilde{C}^{-1} @ Sigma,
        # and finish the small matrix 1 \pm Sigma.T @ \tilde{C}^{-1} @ Sigma.
        if Cinv.isinv:
            tmp = Cinv.noise[:,None]*self.src_mat - tmp
            small_inv = np.eye(self.n_src) + self.src_mat.T @ tmp
        else:
            tmp = Cinv.noise[:,None]*self.src_mat + tmp
            small_inv = np.eye(self.n_src) - self.src_mat.T @ tmp

        # Now Cholesky and accumulate determinant if requested.
        small_inv = np.linalg.cholesky(small_inv)
        if return_det:
            logdet += 2 * np.log(np.diag(small_inv)).sum()

        # Finally, invert then compute Sigma'.
        # At this point, tmp = \tilde{C}^{-1} @ Sigma.
        small_inv = np.linalg.inv(small_inv)
        Cinv.src_mat = tmp @ small_inv.T
        if return_det:
            return Cinv, logdet
        return Cinv
        

    def _full_inv(self, return_det=False):
        """Inversion routine for non-block-diagonal diffuse matrix."""
        raise NotImplementedError("Work in progress.")


class ModelCov:
	"""Class for managing models at many times/frequencies.

	This class provides a helpful interface for various operations one
	may want to apply to model covariance matrices in practical applications
	of CorrCal, including array downselection and file I/O.
	
	Attributes
	----------
	freq_array
		Frequency at which each model covariance is evaluated, in Hz.
	lst_array
		Local Sidereal Time at which each model covariance is evaluated,
		in radians.
	pol_array
		Visibility polarizations included in the model, provided either
		as strings or AIPS polarization integers.
	array_layout
		Mapping from antenna numbers to antenna positions, in meters.
	ant_1_array, ant_2_array
		Index arrays mapping baseline indices to antenna pairs.
	edges
		Array indicating the start and end of each redundant group.
	src_mats
		Source matrix for each LST, frequency, and polarization, with
		shape ``(n_lst, n_freq, n_pol, n_bls, n_src)``.
	diff_mats
		Diffuse matrix for each LST, frequency, and polarization, with
		shape ``(n_lst, n_freq, n_pol, n_bls, n_eig)``.
	noise_diags
		Diagonal of noise covariance for each LST, frequency, and
		polarization, with shape ``(n_lst, n_freq, n_pol, n_bls)``.
	"""

	def __init__(
		self,
		freq_array: Optional[NDArray[float]] = None,
		lst_array: Optional[NDArray[float]] = None,
		pol_array: Optional[NDArray[str] | NDArray[int]] = None,
		array_layout: Optional[dict[int,NDArray[float]]] = None,
		ant_1_array: Optional[NDArray[int]] = None,
		ant_2_array: Optional[NDArray[int]] = None,
		edges: Optional[NDArray[int]] = None,
		src_mats: Optional[NDArray[float]] = None,
		diff_mats: Optional[NDArray[float]] = None,
		noise_diags: Optional[NDArray[float]] = None,
	):
		self.freq_array = freq_array
		self.lst_array = lst_array
		self.pol_array = pol_array
		self.array_layout = array_layout
		self.ant_1_array = ant_1_array
		self.ant_2_array = ant_2_array
		self.edges = edges
		self.src_mats = src_mats
		self.diff_mats = diff_mats
		self.noise_diags = noise_diags

	@property
	def n_bls(self):
		return self.ant_1_array.size

	def select(
		self,
		antennas: Optional[NDArray[int]] = None,
		baselines: Sequence[tuple[int,int]] = None,
		polarizations: Optional[NDArray[str] | NDArray[int]] = None,
		frequencies: Optional[NDArray[float]] = None,
		freq_chans: Optional[NDArray[int]] = None,
		lsts: Optional[NDArray[float]] = None,
		min_group_size: Optional[int] = 1,
	):
		if antennas is not None:
			self._antenna_select(antennas, min_group_size)
		if baselines is not None:
			self._baseline_select(baselines, min_group_size)
		if polarizations is not None:
			self._polarization_select(polarizations)
		if (frequencies is not None) or (freq_chans is not None):
			self._freq_select(
				frequencies=frequencies, freq_chans=freq_chans
			)
		if lsts is not None:
			self._lst_select(lsts)
		if min_group_size > 1:
			self._remove_small_groups(min_group_size)


	def _baseline_select(
		self,
		baselines: Sequence[tuple[int,int]],
		min_group_size: Optional[int] = 1,
	):
		"""Remove attribute components not in the provided baselines."""
		baselines = set(baselines)  # For faster lookup
		select = []
		new_edges = [0,]
		edge_iter = zip(self.edges//2, self.edges[1:]//2)
		for grp, (start, stop) in enumerate(edge_iter):
			count = 0
			for bl in zip(ant_1_array[start:stop], ant_2_array[start:stop]):
				keep = (bl in baselines) or (bl[::-1] in baselines)
				count += 2 * int(keep)
				select += 2 * [keep,]

			if count:
				new_edges.append(new_edges[-1] + count)
			
		self._apply_baseline_select(np.asarray(select))
		self.edges = np.array(new_edges)

		if min_group_size > 1:
			self._remove_small_groups(min_group_size)

	def _antenna_select(
		self, antennas: Sequence[int], min_group_size: int = 1
	):
		"""
		Remove all antennas not contained in the provided set of antennas.
		"""
		antennas = set(antennas)  # For faster lookup
		select = []
		new_edges = [0,]
		edge_iter = zip(self.edges//2, self.edges[1:]//2)
		for grp, (start, stop) in enumerate(edge_iter):
			ai_here = ant_1_array[start:stop]
			aj_here = ant_2_array[start:stop]

			count = 0
			for ai, aj in zip(ai_here, aj_here):
				keep = (ai in antennas) and (aj in antennas)
				count += 2 * int(keep)
				select += 2 * [keep,]

			if count:
				new_edges.append(new_edges[-1] + count)

		self._apply_baseline_select(np.asarray(select))
		self.edges = np.array(new_edges)

		if min_group_size > 1:
			self._remove_small_groups(min_group_size)

	def _apply_baseline_select(self, select: NDArray[bool]):
		"""Apply baseline downselection to all relevant arrays."""
		self.diff_mats = self.diff_mats[...,select,:].copy()
		self.src_mats = self.src_mats[...,select,:].copy()
		self.ant_1_array = self.ant_1_array[select[::2]].copy()
		self.ant_2_array = self.ant_2_array[select[::2]].copy()
		if self.noise_diags is not None:
			self.noise_diags = self.noise_diags[...,select].copy()

	def _remove_small_groups(self, min_group_size: int):
		"""Remove any baseline group smaller than requested size."""
		group_sizes = np.diff(self.edges) // 2
		keep_group = group_sizes >= min_group_size
		if keep_group.all():
			return  # Nothing to do, since every group is big enough

		select = np.zeros(2*self.ant_1_array.size)
		for grp, (start, stop) in enumerate(zip(edges, edges[1:])):
			select[start:stop] = keep_group[grp]

		self._apply_baseline_select(select)
		self.edges = np.cumsum(self.edges[keep_group])

	def _freq_select(
		self,
		frequencies: Optional[NDArray[float]] = None,
		freq_chans: Optional[NDArray[int]] = None,
		tol: Optional[float] = 100,
	):
		if (frequencies is None) and (freq_chans is None):
			return  # Nothing to do

		if frequencies is not None:
			freq_chans = []
			for freq in frequencies:
				is_close = np.abs(np.diff(self.freq_array - freq)) < tol
				if is_close.any():
					freq_chans.append(np.argwhere(is_close).flatten()[0])
			freq_chans = np.array(freq_chans)

		self.freq_array = self.freq_array[freq_chans].copy()
		self.diff_mats = self.diff_mats[:,freq_chans].copy()
		self.src_mats = self.src_mats[:,freq_chans].copy()
		if self.noise_diags is not None:
			self.noise_diags = self.noise_diags[:,freq_chans].copy()

	def _lst_select(
		self, lsts: NDArray[float], tol: Optional[float] = 1e-5
	):
		select = []
		for lst in lsts:
			is_close = np.abs(np.diff(self.lst_array - lst)) < tol
			if is_close.any():
				select.append(np.argwhere(is_close).flatten()[0])

		select = np.asarray(select)
		self.lst_array = self.lst_array[select].copy()
		self.diff_mats = self.diff_mats[select].copy()
		self.src_mats = self.src_mats[select].copy()
		if self.noise_diags is not None:
			self.noise_diags = self.noise_diags[select].copy()

	def interp(
		self,
		freq_array: Optional[NDArray[float]] = None,
		lst_array: Optional[NDArray[float]] = None,
		inplace: bool = False,
	):  # TODO: figure out how to make the typing work here
		pass

	def write(
		self,
		filename: str,
		clobber: bool = False,
	):
		pass

	def read(
		self,
		filename: str,
		**select_kwds
	):
		pass

	@classmethod
	def from_file(
		filename: str,
		**select_kwds
	) -> "ModelCov":
		pass
