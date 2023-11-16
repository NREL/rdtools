#!/usr/bin/env python

try:
    from setuptools import setup
except ImportError:
    raise RuntimeError('setuptools is required')


import versioneer


DESCRIPTION = 'Functions for reproducible timeseries analysis of photovoltaic systems.'

LONG_DESCRIPTION = """
RdTools is an open-source library to support reproducible technical analysis of
PV time series data. The library aims to provide best practice analysis
routines along with the building blocks for users to tailor their own analyses.

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
    'coverage',
    'flake8',
    'nbval==0.9.6',  # https://github.com/computationalmodelling/nbval/issues/194
    'pytest-mock',
]

INSTALL_REQUIRES = [
    'matplotlib >= 3.0.0',
    'numpy >= 1.17.3',
    # pandas restricted to <2.1 until
    # https://github.com/pandas-dev/pandas/issues/55794
    # is resolved
    'pandas >= 1.3.0, <2.1',
    'statsmodels >= 0.11.0',
    'scipy >= 1.2.0',
    'h5py >= 2.8.0',
    'plotly>=4.0.0',
    'xgboost >= 1.3.3',
    'pvlib >= 0.7.0, <0.11.0',
    'scikit-learn >= 0.22.0',
]

EXTRAS_REQUIRE = {
    'doc': [
        'sphinx==4.5.0',
        'nbsphinx==0.8.8',
        'nbsphinx-link==1.3.0',
        'sphinx_rtd_theme==0.5.2',
        'ipython',
        # sphinx-gallery used indirectly for nbsphinx thumbnail galleries; see:
        # https://nbsphinx.readthedocs.io/en/0.6.0/subdir/gallery.html#Creating-Thumbnail-Galleries
        'sphinx-gallery==0.8.1',
    ],
    'test': TESTS_REQUIRE,
}
EXTRAS_REQUIRE['all'] = sorted(set(sum(EXTRAS_REQUIRE.values(), [])))


CLASSIFIERS = [
    'Development Status :: 5 - Production/Stable',
    'License :: OSI Approved :: MIT License',
    'Operating System :: OS Independent',
    'Intended Audience :: Science/Research',
    'Programming Language :: Python',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3.9',
    'Programming Language :: Python :: 3.10',
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

PROJECT_URLS = {
    "Bug Tracker": "https://github.com/NREL/rdtools/issues",
    "Documentation": "https://rdtools.readthedocs.io/",
    "Source Code": "https://github.com/NREL/rdtools",
}

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
      extras_require=EXTRAS_REQUIRE,
      description=DESCRIPTION,
      long_description=LONG_DESCRIPTION,
      author=AUTHOR,
      author_email=AUTHOR_EMAIL,
      maintainer_email=MAINTAINER_EMAIL,
      license=LICENSE,
      url=URL,
      project_urls=PROJECT_URLS,
      classifiers=CLASSIFIERS,
      **setuptools_kwargs)
