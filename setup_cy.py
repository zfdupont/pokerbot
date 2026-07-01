"""Build the Cython extension in-place: python setup_cy.py build_ext --inplace"""
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

ext = Extension(
    "cfr.mccfr_cy",
    sources=["cfr/mccfr_cy.pyx"],
    include_dirs=[np.get_include()],
    extra_compile_args=["-O3", "-ffast-math"],
)

setup(
    name="pokerbot_cy",
    packages=[],
    ext_modules=cythonize(
        [ext],
        compiler_directives={"language_level": 3},
        annotate=False,
    ),
)
