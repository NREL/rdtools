.. _developer_notes:

Developer Notes
===============

This page documents some of the workflows specific to RdTools development.

Installing RdTools source code
------------------------------

Installing RdTools using pip will install a condensed version that
doesn't include the full source code.  If you want to make changes to RdTools,
you'll need to clone the RdTools source repository from Github with e.g.

::

    git clone https://github.com/NREL/rdtools.git

from the command line, or using a GUI git client like Github Desktop.  This
will clone the entire git repository onto your computer.  

Running the test suite
----------------------

RdTools uses `PyTest <https://docs.pytest.org/en/latest/>`_ to run its test
suite.  If you don't already have it installed:

::

    pip install pytest

To run the entire test suite, navigate to the git repo folder and run

::

    python -m pytest rdtools

For convenience, pytest lets you run tests for a single module if you don't
want to wait around for the entire suite to finish:

::

    python -m pytest rdtools\test\soiling_test.py

And even a single test function:

::

    python -m pytest rdtools\test\soiling_test.py::test_soiling_srr


Building Documentation
----------------------

RdTools uses `Sphinx <https://www.sphinx-doc.org/>`_ to build its documentation.
To build the documentation locally, you'll need to have a local copy of the git
repository.  

Sphinx and the other required libraries can be installed with pip by
installing the `doc` extras (see `here <https://setuptools.readthedocs.io/en/latest/setuptools.html#declaring-extras-optional-features-with-their-own-dependencies>`_
for more info): 

::

    pip install rdtools[doc]

or you can install from your own local git repo (run from the repo folder):

::

    pip install .[doc]

This will install rdtools along with all the packages necessary to build its
documentation.  We recommend doing this in a virtual environment to keep
package installations between projects separate!

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

If you get a warning like ``Pygments lexer name 'ipython3' is not known``, your
environment doesn't have ``ipython``.  You can install it from conda or pypi:

::

    pip install ipython

The built documentation should be in ``rdtools/docs/sphinx/build`` and opening
``index.html`` with a web browser will display it.
