from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import build_ext
import numpy as np

ext_modules = [
    Extension("carmine.algorithms", ["carmine/algorithms.pyx"])
]

setup(
    name="carmine",
    cmdclass={"build_ext": build_ext},
    include_dirs=[np.get_include()],
    ext_modules=ext_modules
)
