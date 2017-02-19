#!/usr/bin/env python

try:
    from setuptools import setup
except ImportError:
    raise RuntimeError('setuptools is required')


import versioneer


DESCRIPTION = 'Functions for analytics the degradation of photovoltaic systems.'

LONG_DESCRIPTION = """
Rdtools is a collection of tools for the analysis of photovoltaic degradation.

Source code: https://github.com/kwhanalytics/rdtools
"""

DISTNAME = 'rdtools'
LICENSE = 'MIT'
AUTHOR = 'Rdtools Python Developers'
MAINTAINER_EMAIL = 'adam.b.shinn@gmail.com'
URL = 'https://github.com/kwhanalytics/rdtools'

INSTALL_REQUIRES = [
    'numpy >= 1.11.0',
    'pandas >= 0.19.0',
    'pvlib >= 0.4.1',
    'statsmodels >= 0.6.1',
    'patsy >= 0.4.1',
    'nose >= 1.3.7',
    'pytz',
    'six',
]

TESTS_REQUIRE = ['nose']

CLASSIFIERS = [
    'Development Status :: 1 - Beta',
    'License :: MIT License',
    'Operating System :: OS Independent',
    'Intended Audience :: Science/Research',
    'Programming Language :: Python',
    'Programming Language :: Python :: 2',
    'Programming Language :: Python :: 2.7',
    'Topic :: Scientific/Engineering',
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
      install_requires=INSTALL_REQUIRES,
      tests_require=TESTS_REQUIRE,
      description=DESCRIPTION,
      long_description=LONG_DESCRIPTION,
      author=AUTHOR,
      maintainer_email=MAINTAINER_EMAIL,
      license=LICENSE,
      url=URL,
      classifiers=CLASSIFIERS,
      **setuptools_kwargs)
