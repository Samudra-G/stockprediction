from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy


# Define the Extension object
extensions = [
    Extension(
        name="financial_functions", 
        sources=["financial_functions.pyx", "trading.cpp"],  # Include both .pyx and .cpp files
        include_dirs=[numpy.get_include(), "."],  # Include numpy headers and the current directory for header files
        language="c++",  # Ensure that C++ compiler is used
    )
]

# Setup function
setup(
    name="financial_functions",
    ext_modules=cythonize(extensions),
    zip_safe=False,
)
