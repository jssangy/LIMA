from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension(
        "cpp_sch",
        ["cpp/cpp_sch.cpp"],
        cxx_std=17,
        extra_compile_args=["-O3", "-DNDEBUG", "-march=native"],
        extra_link_args=["-flto"],
    ),
]

setup(
    name="cpp_sch",
    version="0.0.1",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)

"""

python setup.py build_ext --inplace

"""