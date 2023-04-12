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
lib = ctypes.cdll.LoadLibrary(lib_path)

cholesky = lib.cholesky
cholesky.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_int]

cholesky_inplace = lib.cholesky_inplace
cholesky_inplace.argtypes = [ct.c_void_p, ct.c_int]

many_chol = lib.many_chol
many_chol.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_void_p, ct.c_int]

tril_inv = lib.tril_inv
tril_inv.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_int]

many_tril_inv = lib.many_tril_inv
many_tril_inv.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_void_p, ct.c_int]

make_small_block = lib.make_small_block
make_small_block.argtypes = [
    ct.c_void_p, ct.c_void_p, ct.c_void_p, ct.c_int, ct.c_int, ct.c_int
]

make_all_small_blocks = lib.make_all_small_blocks
make_all_small_blocks.argtypes = [
    ct.c_void_p, ct.c_void_p, ct.c_void_p, ct.c_void_p, ct.c_int, ct.c_int
]
