from setuptools import setup, Extension

ext = Extension(
    "corrcal.src.c_corrcal",
    language="C",
    sources=["corrcal/src/c_funcs.c"],
    include_dirs=["corrcal/src"],
    extra_compile_args=[
        "-fopenmp", "-O3", "-fPIC", "-std=c99"
    ],
    extra_link_args=["-shared", "-lgomp", "-lm"],
)

setup(
    use_scm_version=True, setup_requires=["setuptools_scm"], ext_modules=[ext],
)
