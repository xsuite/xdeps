# copyright ############################### #
# This file is part of the Xdeps Package.   #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

from setuptools import setup, find_packages

setup(
        name='xdeps',
        version='0.0.6',
        description='Data dependency manager',
        long_description=("Data dependency manager\n"
                         "\nThis package is part of the Xsuite collection."),
        author='Riccardo De Maria',
        author_email='riccardo.de.maria@cern.ch',
        packages=find_packages(),
        install_requires=['lark'],
        url='https://xsuite.readthedocs.io/',
        license='Apache 2.0',
        download_url="https://pypi.python.org/pypi/xdeps",
        project_urls={
            "Bug Tracker": "https://github.com/xsuite/xsuite/issues",
            "Documentation": 'https://xsuite.readthedocs.io/',
            "Source Code": "https://github.com/xsuite/xdeps",
        },
)
