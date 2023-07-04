"""
Module providing interfaces to various C-functions.

All C-functions are loaded in as private functions, and are not expected
to be interfaced with by the end-user. Wrapper functions for the C-functions
are provided in other modules (e.g.:mod:`~.linalg`).
"""
import ctypes as ct
import site
from pathlib import Path

lib_dir = Path(site.getsitepackages()[0]) / "corrcal"
if not lib_dir.exists():
    # This is in case a development installation is used.
    # Development installs do not create a directory in the site
    # packages, but rather put this directory in the python path.
    # So the shared object library is written to the same directory
    # as this file.
    lib_dir = Path(__file__).parent

lib_path = list(lib_dir.glob("c_corrcal*.so"))[0]
lib = ct.cdll.LoadLibrary(lib_path)

# Lower-triangular matrix inversion
tril_inv = lib.tril_inv
tril_inv.argtypes = [
    ct.c_void_p,  # Matrix to invert
    ct.c_void_p,  # Where to write output
    ct.c_int,  # Matrix size
]

# Parallelized lower-triangular matrix inversion
many_tril_inv = lib.many_tril_inv
many_tril_inv.argtypes = [
    ct.c_void_p,  # Matrices to invert
    ct.c_void_p,  # Where to write output
    ct.c_int,  # Size of each block
    ct.c_int,  # Number of blocks to invert
]

# Cholesky decomposition LL^\dag
cholesky = lib.cholesky
cholesky.argtypes = [
    ct.c_void_p,  # Matrix to decompose
    ct.c_void_p,  # Where to write output
    ct.c_int,  # Matrix size
]

# In-place Cholesky decomposition LL^\dag
cholesky_inplace = lib.cholesky_inplace
cholesky_inplace.argtypes = [
    ct.c_void_p,  # Matrix to decompose
    ct.c_int,  # Matrix size
]

# Parallelized Cholesky decomposition LL^\dag
many_chol = lib.many_chol
many_chol.argtypes = [
    ct.c_void_p,  # Matrices to decompose
    ct.c_void_p,  # Where to write output
    ct.c_int,  # Size of each block
    ct.c_int,  # Number of blocks to decompose
]

# Parallelized in-place Cholesky decomposition LL^\dag
many_chol_inplace = lib.many_chol_inplace
many_chol_inplace.argtypes = [
    ct.c_void_p,  # Matrices to decompose
    ct.c_int,  # Size of each block
    ct.c_int,  # Number of blocks to decompose
]

# Parallelized matrix multiplication
matmul = lib.matmul
matmul.argtypes = [
    ct.c_void_p,  # Matrix on left side
    ct.c_void_p,  # Matrix on right side
    ct.c_void_p,  # Where to write the output
    ct.c_int,  # Number of columns in left-hand matrix
    ct.c_int,  # Number of rows in right-hand matrix
    ct.c_int,  # Number of columns in output
]

# Single block matrix multiplication
mymatmul = lib.mymatmul
mymatmul.argtypes = [
    ct.c_void_p,  # Matrix on left side
    ct.c_void_p,  # Matrix on right side
    ct.c_void_p,  # Where to write the output
    ct.c_int,  # Number of columns in left-hand matrix
    ct.c_int,  # Number of rows in right-hand matrix
    ct.c_int,  # Number of columns in output
    ct.c_int,  # Number of rows in this block of left
    ct.c_int,  # Number of columns in this block of right
    ct.c_int,  # Number of columns in this block of left
]

# Block multiplication \Delta @ L_\Delta^{-1}^\dag
block_multiply = lib.block_multiply
block_multiply.argtypes = [
    ct.c_void_p,  # Small blocks
    ct.c_void_p,  # Diffuse matrix
    ct.c_void_p,  # Where to write output
    ct.c_void_p,  # Redundant group edges
    ct.c_int,  # Number of eigenmodes
    ct.c_int,  # Number of redundant groups
]

# Make one small block of \Delta^\dag @ Ninv @ \Delta
make_small_block = lib.make_small_block
make_small_block.argtypes = [
    ct.c_void_p,  # Noise variance diagonal
    ct.c_void_p,  # Diffuse matrix
    ct.c_void_p,  # Where to write output
    ct.c_int,  # Number of eigenmodes
    ct.c_int,  # Starting index of this group
    ct.c_int,  # Ending index of this group
]

# Make all small blocks of \Delta^\dag @ Ninv @ \Delta
make_all_small_blocks = lib.make_all_small_blocks
make_all_small_blocks.argtypes = [
    ct.c_void_p,  # Noise variance diagonal
    ct.c_void_p,  # Diffuse matrix
    ct.c_void_p,  # Where to write output
    ct.c_void_p,  # Redundant group edges
    ct.c_int,  # Number of eigenmodes
    ct.c_int,  # Number of redundant groups
]

# Calculate \Delta'.H @ \Sigma
mult_src_by_blocks = lib.mult_src_by_blocks
mult_src_by_blocks.argtypes = [
    ct.c_void_p,  # Hermitian conjugate of "inverse" diffuse matrix
    ct.c_void_p,  # Source matrix
    ct.c_void_p,  # Where to write output
    ct.c_void_p,  # Redundant group edges
    ct.c_int,  # Number of baselines
    ct.c_int,  # Number of sources
    ct.c_int,  # Number of eigenmodes
    ct.c_int,  # Number of redundant groups
]

# Make a sparse covariance structure
init_cov = lib.init_cov
init_cov.argtypes = [
    ct.c_void_p,  # Noise variance diagonal
    ct.c_void_p,  # Diffuse matrix
    ct.c_void_p,  # Source matrix
    ct.c_int,  # Number of baselines
    ct.c_int,  # Number of eigenmodes
    ct.c_int,  # Number of sources
    ct.c_int,  # Number of redundant groups
    ct.c_void_p,  # Redundant group edges
    ct.c_int,  # Whether the covariance is inverted
]

# Sparse matrix-vector multiplication
sparse_cov_times_vec = lib.sparse_cov_times_vec_wrapper
sparse_cov_times_vec.argtypes = [
    ct.c_void_p,  # Noise variance diagonal
    ct.c_void_p,  # Diffuse matrix
    ct.c_void_p,  # Source matrix
    ct.c_int,  # Number of baselines
    ct.c_int,  # Number of eigenmodes
    ct.c_int,  # Number of sources
    ct.c_int,  # Number of redundant groups
    ct.c_void_p,  # Redundant group edges
    ct.c_int,  # Whether the covariance is inverted
    ct.c_void_p,  # Vector to multiply by the covariance
    ct.c_void_p,  # Where to write the output
]
