from setuptools import setup, find_packages

setup(
        name='xdeps',
        version='0.0.5',
        description='Data dependency manager',
        author='Riccardo De Maria',
        author_email='riccardo.de.maria@cern.ch',
        url='https://github.com/xsuite/xdeps',
        packages=find_packages(),
        install_requires=['lark']
)
