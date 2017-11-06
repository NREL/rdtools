#!/usr/bin/env python

try:
    from setuptools import setup
except ImportError:
    raise RuntimeError('setuptools is required')


import versioneer


DESCRIPTION = 'Functions for analyzing the degradation of photovoltaic systems.'

LONG_DESCRIPTION = """
Rdtools is a collection of tools for the analysis of photovoltaic degradation.

Source code: https://github.com/NREL/rdtools
"""

DISTNAME = 'rdtools'
LICENSE = 'MIT'
AUTHOR = 'Rdtools Python Developers'
MAINTAINER_EMAIL = 'RdTools@nrel.gov'

URL = 'https://github.com/NREL/rdtools'

INSTALL_REQUIRES = [
    'numpy >= 1.11.2',
    'pandas >= 0.20.3',
    'pvlib >= 0.5.0',
    'statsmodels >= 0.8.0',
    'scipy >= 0.19.1',
    'patsy >= 0.4.1',
    'h5py >= 2.7.1',
    'pytz',
    'six',
]

CLASSIFIERS = [
    'Development Status :: 4 - Beta',
    'License :: OSI Approved :: MIT License',
    'Operating System :: OS Independent',
    'Intended Audience :: Science/Research',
    'Programming Language :: Python',
    'Programming Language :: Python :: 2',
    'Programming Language :: Python :: 2.7',
    'Topic :: Scientific/Engineering',
]

KEYWORDS = [
    'photovoltaic',
    'solar',
    'analytics',
    'analysis',
    'performance',
    'degradation',
    'PV'
]

setuptools_kwargs = {
    'zip_safe': False,
    'scripts': [],
    'include_package_data': True
}

# set up packages to be installed and extensions to be compiled
PACKAGES = ['rdtools']

PACKAGE_DATA = {
    'rdtools': 'rdtools/data/temperature.hdf5'
}

setup(name=DISTNAME,
      version=versioneer.get_version(),
      cmdclass=versioneer.get_cmdclass(),
      packages=PACKAGES,
      keywords=KEYWORDS,
      install_requires=INSTALL_REQUIRES,
      package_data=PACKAGE_DATA,
      description=DESCRIPTION,
      long_description=LONG_DESCRIPTION,
      author=AUTHOR,
      author_email=AUTHOR,
      maintainer_email=MAINTAINER_EMAIL,
      license=LICENSE,
      url=URL,
      classifiers=CLASSIFIERS,
      **setuptools_kwargs)
