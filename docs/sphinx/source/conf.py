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
import inspect
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
    'm2r',
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
html_static_path = ['_static', '../../../_static']

master_doc = 'index'
# A workaround for the responsive tables always having annoying scrollbars.
def setup(app):
    app.add_stylesheet("no_scrollbars.css")


# %% helper functions for intelligent "View on Github" linking
# based on
# https://gist.github.com/flying-sheep/b65875c0ce965fbdd1d9e5d0b9851ef1

def get_obj_module(qualname):
    """
    Get a module/class/attribute and its original module by qualname.
    Useful for looking up the original location when a function is imported
    into an __init__.py

    Examples
    --------
    >>> func, mod = get_obj_module("rdtools.degradation_ols")
    >>> mod.__name__
    'rdtools.degradation'
    """
    modname = qualname
    classname = None
    attrname = None
    while modname not in sys.modules:
        attrname = classname
        modname, classname = modname.rsplit('.', 1)

    # retrieve object and find original module name
    if classname:
        cls = getattr(sys.modules[modname], classname)
        modname = cls.__module__
        obj = getattr(cls, attrname) if attrname else cls
    else:
        obj = None

    return obj, sys.modules[modname]


def get_linenos(obj):
    """Get an object’s line numbers in its source code file"""
    try:
        lines, start = inspect.getsourcelines(obj)
    except TypeError:  # obj is an attribute or None
        return None, None
    else:
        return start, start + len(lines) - 1


def make_github_url(pagename):
    """
    Generate the appropriate GH link for a given docs page.  This function
    is intended for use in sphinx template files.
    The target URL is built differently based on the type of page.  Sphinx
    provides templates with a built-in `pagename` variable that is the path
    at the end of the URL, without the extension.  For instance,
    https://rdtools.rtfd.io/en/latest/generated/rdtools.soiling.soiling_srr.html
    will have pagename = "generated/rdtools.soiling.soiling_srr".

    Returns None if not building development or master.
    """

    # RTD automatically sets READTHEDOCS_VERSION to the version being built.
    rtd_version = os.environ.get('READTHEDOCS_VERSION', None)
    version_map = {
        'stable': 'master',
        'latest': 'development',        
    }
    try:
        branch = version_map[rtd_version]
    except KeyError:
        # for other builds (PRs, local builds etc), it's unclear where to link
        return None

    URL_BASE = "https://github.com/nrel/rdtools/blob/{}/".format(branch)

    # is it an API autogen page?
    if pagename.startswith("generated/"):
        # pagename looks like "generated/rdtools.degradation.degradation_ols"
        qualname = pagename.split("/")[-1]
        obj, module = get_obj_module(qualname)
        path = module.__name__.replace(".", "/") + ".py"
        target_url = URL_BASE + path
        # add line numbers if possible:
        start, end = get_linenos(obj)
        if start and end:
            target_url += '#L{}-L{}'.format(start, end)

    # is it the example notebook?
    elif pagename == "example":
        target_url = URL_BASE + "docs/degradation_and_soiling_example.ipynb"

    # is the the changelog page?
    elif pagename == "changelog":
        target_url = URL_BASE + "docs/sphinx/source/changelog"
        
    # Just a normal source RST page
    else:
        target_url = URL_BASE + "docs/sphinx/source/" + pagename + ".rst"

    display_text = "View on github/" + branch
    link_info = {
        'url': target_url,
        'text': display_text        
    }
    return link_info


# variables to pass into the HTML templating engine; these are accessible from
# _templates/breadcrumbs.html
html_context = {
    'make_github_url': make_github_url,
}