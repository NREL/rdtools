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
AUTHOR_EMAIL = 'RdTools@nrel.gov'
MAINTAINER_EMAIL = 'RdTools@nrel.gov'

URL = 'https://github.com/NREL/rdtools'

SETUP_REQUIRES = [
    'pytest-runner',
]

TESTS_REQUIRE = [
    'pytest >= 3.6.3',
]

INSTALL_REQUIRES = [
    'h5py >= 2.7.1',
    'matplotlib >= 2.2.2',
    'numpy >= 1.12',
    'pandas >= 0.23.0',
    'pvlib >= 0.6.0, <0.7.0',
    'statsmodels >= 0.8.0',
    'scipy >= 0.19.1',
]

CLASSIFIERS = [
    'Development Status :: 4 - Beta',
    'License :: OSI Approved :: MIT License',
    'Operating System :: OS Independent',
    'Intended Audience :: Science/Research',
    'Programming Language :: Python',
    'Programming Language :: Python :: 2',
    'Programming Language :: Python :: 2.7',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.6',
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


setup(name=DISTNAME,
      version=versioneer.get_version(),
      cmdclass=versioneer.get_cmdclass(),
      packages=PACKAGES,
      keywords=KEYWORDS,
      setup_requires=SETUP_REQUIRES,
      tests_require=TESTS_REQUIRE,
      install_requires=INSTALL_REQUIRES,
      description=DESCRIPTION,
      long_description=LONG_DESCRIPTION,
      author=AUTHOR,
      author_email=AUTHOR_EMAIL,
      maintainer_email=MAINTAINER_EMAIL,
      license=LICENSE,
      url=URL,
      classifiers=CLASSIFIERS,
      **setuptools_kwargs)
