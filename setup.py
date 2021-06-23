from setuptools import setup, Extension

ext = Extension(
    "corrcal.c_corrcal",
    sources=["corrcal/src/corrcal_c_funcs.c"],
    extra_compile_args=["-fopenmp"],
    extra_link_args=["-lgomp"],
)

setup(
    use_scm_version=True, setup_requires=["setuptools_scm"], ext_modules=[ext],
)
