.. _developer_notes:

Developer Notes
===============

This page documents some of the workflows specific to RdTools development.

Installing RdTools source code
------------------------------

To make changes to RdTools, run the test suite, or build the documentation
locally, you'll need to have a local copy of the git repository.
Installing RdTools using pip will install a condensed version that
doesn't include the full source code.  To get the full source code,
you'll need to clone the RdTools source repository from Github with e.g.

::

    git clone https://github.com/NREL/rdtools.git

from the command line, or using a GUI git client like Github Desktop.  This
will clone the entire git repository onto your computer.  

Installing RdTools dependencies
-------------------------------

The packages necessary to run RdTools itself can be installed with ``pip``.
You can install the dependencies along with RdTools itself from 
`PyPI <https://pypi.org/project/rdtools/>`_:

::

    pip install rdtools

This will install the latest official release of RdTools.  If you want to work
with a development version and you have cloned the Github repository to your
computer, you can also install RdTools and dependencies by navigating to the
repository root and running:

::

    pip install .

This will install the development version of RdTools along with the current
set of requirements. 

.. _installing-optional-dependencies:

Installing optional dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

RdTools has extra dependencies for running its test suite and building its
documentation.  These packages aren't necessary for running RdTools itself and
are only needed if you want to contribute source code to RdTools.  

.. note::
    These will install RdTools along with other packages necessary to build its
    documentation and run its test suite.  We recommend doing this in a virtual
    environment to keep package installations between projects separate!

Optional dependencies can be installed with the special 
`syntax <https://setuptools.readthedocs.io/en/latest/setuptools.html#declaring-extras-optional-features-with-their-own-dependencies>`_:

::

    pip install rdtools[test]  # test suite dependencies
    pip install rdtools[doc]   # documentation dependecies

Or, if your local repository has an updated dependencies list:

::

    pip install .[test]  # test suite dependencies
    pip install .[doc]   # documentation dependecies


Running the test suite
----------------------

RdTools uses `pytest <https://docs.pytest.org/en/latest/>`_ to run its test
suite.  If you haven't already, install the testing depencencies
(:ref:`installing-optional-dependencies`).

To run the entire test suite, navigate to the git repo folder and run

::

    pytest

For convenience, pytest lets you run tests for a single module if you don't
want to wait around for the entire suite to finish:

::

    pytest rdtools/test/soiling_test.py

And even a single test function:

::

    pytest rdtools/test/soiling_test.py::test_soiling_srr

You can also evaluate code coverage when running the test suite using the 
`coverage <https://coverage.readthedocs.io>`_ package:

::

    coverage run -m pytest
    coverage report

The first line runs the test suite and keeps track of exactly what lines of
code were run during test execution.  The second line then prints out a
summary report showing how much much of each source file was
executed in the test suite.  If a percentage is below 100, that means a
function isn't tested or a branch inside a function isn't tested.  To get
specific details, you can run ``coverage html`` to generate a detailed HTML
report at ``htmlcov/index.html`` to view in a browser.  

Building documentation locally
------------------------------

RdTools uses `Sphinx <https://www.sphinx-doc.org/>`_ to build its documentation.
If you haven't already, install the documentation depencencies
(:ref:`installing-optional-dependencies`).

Once the required packages are installed, change your console's working
directory to ``rdtools/docs/sphinx`` and run

::

    make html

Note that on Windows, you don't actually need the ``make`` utility installed for
this to work because there is a ``make.bat`` in this directory.  Building the
docs should result in output like this:

::

    (venv)$ make html
    Running Sphinx v1.8.5
    making output directory...
    [autosummary] generating autosummary for: api.rst, example.nblink, index.rst, readme_link.rst
    [autosummary] generating autosummary for: C:\Users\KANDERSO\projects\rdtools\docs\sphinx\source\generated\rdtools.aggregation.aggregation_insol.rst, C:\Users\KANDERSO\projects\rdtools\docs\sphinx\source\generated\rdtools.aggregation.rst, C:\Users\KANDERSO\projects\rdtools\docs\sphinx\source\generated\rdtools.clearsky_temperature.get_clearsky_tamb.rst, C:\Users\KANDERSO\projects\rdtools\docs\sphinx\source\generated\rdtools.clearsky_temperature.rst, C:\Users\KANDERSO\projects\rdtools\docs\sphinx\source\generated\rdtools.degradation.degradation_classical_decomposition.rst, C:\Users\KANDERSO\projects\rdtools\docs\sphinx\source\generated\rdtools.degradation.degradation_ols.rst, C:\Users\KANDERSO\projects\rdtools\docs\sphinx\source\generated\rdtools.degradation.degradation_year_on_year.rst, C:\Users\KANDERSO\projects\rdtools\docs\sphinx\source\generated\rdtools.degradation.rst, C:\Users\KANDERSO\projects\rdtools\docs\sphinx\source\generated\rdtools.filtering.clip_filter.rst, C:\Users\KANDERSO\projects\rdtools\docs\sphinx\source\generated\rdtools.filtering.csi_filter.rst, ..., C:\Users\KANDERSO\projects\rdtools\docs\sphinx\source\generated\rdtools.normalization.normalize_with_pvwatts.rst, C:\Users\KANDERSO\projects\rdtools\docs\sphinx\source\generated\rdtools.normalization.normalize_with_sapm.rst, C:\Users\KANDERSO\projects\rdtools\docs\sphinx\source\generated\rdtools.normalization.pvwatts_dc_power.rst, C:\Users\KANDERSO\projects\rdtools\docs\sphinx\source\generated\rdtools.normalization.rst, C:\Users\KANDERSO\projects\rdtools\docs\sphinx\source\generated\rdtools.normalization.sapm_dc_power.rst, C:\Users\KANDERSO\projects\rdtools\docs\sphinx\source\generated\rdtools.normalization.t_step_nanoseconds.rst, C:\Users\KANDERSO\projects\rdtools\docs\sphinx\source\generated\rdtools.normalization.trapz_aggregate.rst, C:\Users\KANDERSO\projects\rdtools\docs\sphinx\source\generated\rdtools.soiling.rst, C:\Users\KANDERSO\projects\rdtools\docs\sphinx\source\generated\rdtools.soiling.soiling_srr.rst, C:\Users\KANDERSO\projects\rdtools\docs\sphinx\source\generated\rdtools.soiling.srr_analysis.rst
    building [mo]: targets for 0 po files that are out of date
    building [html]: targets for 4 source files that are out of date
    updating environment: 33 added, 0 changed, 0 removed
    reading sources... [100%] readme_link
    looking for now-outdated files... none found
    pickling environment... done
    checking consistency... done
    preparing documents... done
    writing output... [100%] readme_link
    generating indices... genindex py-modindex
    writing additional pages... search
    copying images... [100%] ../build/doctrees/nbsphinx/example_33_2.png
    copying static files... done
    copying extra files... done
    dumping search index in English (code: en) ... done
    dumping object inventory... done
    build succeeded.
    
    The HTML pages are in build\html.

If you get an error like ``Pandoc wasn't found``, you can install it with conda:

::

    conda install -c conda-forge pandoc

The built documentation should be in ``rdtools/docs/sphinx/build`` and opening
``index.html`` with a web browser will display it.

Code requirements
-----------------

RdTools follows the `PEP 8 <https://www.python.org/dev/peps/pep-0008/>`_ style guide.
We recommend setting up your text editor to automatically highlight style
violations because it's easy to miss some isses (trailing whitespace, etc) otherwise.

Additionally, our documentation is built in part from docstrings in the source
code.  These docstrings must be in `NumpyDoc format <https://numpydoc.readthedocs.io/en/latest/format.html>`_
to be rendered correctly in the documentation.  

Finally, all code should be tested.  Some older tests in RdTools use the unittest
module, but new tests should all use pytest. 