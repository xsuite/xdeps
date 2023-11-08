# copyright ############################### #
# This file is part of the Xdeps Package.   #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

from Cython.Build import cythonize
from setuptools import setup, find_packages
from pathlib import Path

version_file = Path(__file__).parent / 'xdeps/_version.py'
dd = {}
with open(version_file.absolute(), 'r') as fp:
    exec(fp.read(), dd)
__version__ = dd['__version__']

setup(
    name='xdeps',
    version=__version__,
    description='Data dependency manager',
    long_description=("Data dependency manager\n"
                     "\nThis package is part of the Xsuite collection."),
    author='Riccardo De Maria',
    author_email='riccardo.de.maria@cern.ch',
    packages=find_packages(),
    install_requires=['lark', 'numpy', 'scipy', 'cython'],
    url='https://xsuite.readthedocs.io/',
    license='Apache 2.0',
    download_url="https://pypi.python.org/pypi/xdeps",
    project_urls={
        "Bug Tracker": "https://github.com/xsuite/xsuite/issues",
        "Documentation": 'https://xsuite.readthedocs.io/',
        "Source Code": "https://github.com/xsuite/xdeps",
    },
    extras_require={
        'tests': ['pytest'],
    },
    ext_modules=cythonize("xdeps/refs.py"),
    extra_compile_args=["-O3"],
)
