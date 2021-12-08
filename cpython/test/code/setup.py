# import Cython path

# sys.path.insert(0, cython_path)

from distutils.core import setup

from Cython.Build import cythonize

# setup(ext_modules=cythonize("lml_cpython_test.py"))

setup(
    ext_modules=cythonize("lml_hello.pyx")
)
#   python setup.py build_ext --inplace
