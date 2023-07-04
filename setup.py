from setuptools import setup, Extension

ext = Extension(
    "corrcal.c_corrcal",
    sources=["corrcal/src/c_funcs.c"],
    include_dirs=["corrcal/src"],
    extra_compile_args=["-fopenmp", "-O3", "-lgomp", "-fPIC", "-lm"],
    extra_link_args=["-shared"],
)

setup(
    use_scm_version=True, setup_requires=["setuptools_scm"], ext_modules=[ext],
)
