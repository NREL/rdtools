# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#


# prefer local rdtools folder to one installed in a venv or site-packages:
import os
import sys
sys.path.insert(0, os.path.abspath('../../..'))


# -- Project information -----------------------------------------------------

project = 'RdTools'
copyright = '2016–2019 kwhanalytics, Alliance for Sustainable Energy, LLC, and SunPower'
author = 'kwhanalytics, Alliance for Sustainable Energy, LLC, and SunPower'

# The full version, including alpha/beta/rc tags
import rdtools
release = version = rdtools.__version__


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.napoleon',
    'sphinx.ext.extlinks',
    'sphinx_rtd_theme',
    'sphinx.ext.autosummary',
    'nbsphinx',
    'nbsphinx_link',
]

autosummary_generate = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.

exclude_patterns = []

source_suffix = ['.rst', '.md']

# List of external link aliases.  Allows use of :pull:`123` to autolink that PR
extlinks = {
    'issue': ('https://github.com/NREL/rdtools/issues/%s', 'GH #'),
    'pull': ('https://github.com/NREL/rdtools/pull/%s', 'GH #'),
    'ghuser': ('https://github.com/%s', '@')
}

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# toctree sidebar depth
html_theme_options = {
    'navigation_depth': 4,
    'collapse_navigation': False
}
# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static', '_images']

master_doc = 'index'
# A workaround for the responsive tables always having annoying scrollbars.
def setup(app):
    app.add_stylesheet("no_scrollbars.css")