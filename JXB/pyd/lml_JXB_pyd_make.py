# import Cython path

# sys.path.insert(0, cython_path)

from distutils.core import setup

from Cython.Build import cythonize

# setup(ext_modules=cythonize("lml_cpython_test.py"))

setup(
    ext_modules=cythonize("lml_JXB_pyd.py")
)
setup(
    ext_modules=cythonize("lml_JXB_lib.py")
)
setup(
    ext_modules=cythonize("lml_JXB_TCP_Server.py")
)
#   cd D:\WORK\lml-python\JXB\pyd\
#   python lml_JXB_pyd_make.py build_ext
#   python lml_JXB_pyd_make.py build_ext --inplace
