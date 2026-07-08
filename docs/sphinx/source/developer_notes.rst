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

    git clone https://github.com/NatLabRockies/rdtools.git

from the command line, or using a GUI git client like Github Desktop.  This
will clone the entire git repository onto your computer.

Installing RdTools dependencies
-------------------------------

RdTools uses `pixi <https://pixi.sh>`_ for reproducible environment management.
Pixi creates isolated environments from the lockfile (``pixi.lock``), ensuring
every contributor gets the exact same package versions.

To install pixi, follow the instructions at https://pixi.sh.

Once pixi is installed, navigate to the repository root and run:

::

    pixi install

This creates the default environment with RdTools and its core dependencies.
Pixi environments are stored locally in ``.pixi/`` (gitignored) and do not
interfere with other Python environments on your system.

To run a command inside a pixi environment, use ``pixi run``:

::

    pixi run python -c "import rdtools; print(rdtools.__version__)"

.. _pixi-environments:

Available pixi environments
~~~~~~~~~~~~~~~~~~~~~~~~~~~

RdTools defines several pixi environments in ``pyproject.toml``:

- **core** — bare RdTools with core dependencies only (Python 3.13)
- **default** — RdTools + notebook extras for regular users (Python 3.13)
- **dev** — notebooks + test extras combined (Python 3.13); recommended for day-to-day development (alias for **dev-py313**)
- **dev-py310** through **dev-py314** — full dev environment pinned to a specific Python version
- **dev-min** — test-only environment with minimum supported dependency versions (Python 3.10)

For most contributors, the **dev** environment is the best starting point
because it includes everything needed to run the test suite, launch Jupyter
notebooks, and check code style:

::

    pixi install -e dev
    pixi run -e dev test   # run the test suite
    pixi run -e dev lab    # launch Jupyter Lab

To test against a specific Python version:

::

    pixi run -e dev-py310 test

.. _updating-pixi-environments:

Updating pixi environments
~~~~~~~~~~~~~~~~~~~~~~~~~~~

When ``pyproject.toml`` changes (e.g. a dependency is added, removed, or its
version constraint is updated), the pixi lockfile must be regenerated:

1. **Re-solve dependencies** — run ``pixi update`` from the repository root.
   This re-solves all environments against the current ``pyproject.toml``
   constraints and writes a new ``pixi.lock``:

   ::

       pixi update

2. **Verify environments install correctly** — run ``pixi install`` to
   re-create the ``default`` environment from the updated lockfile:

   ::

       pixi install

   ``pixi install`` on its own only refreshes the ``default`` environment.
   To refresh a specific environment (for example ``dev``), pass ``-e``:

   ::

       pixi install -e dev

   To refresh every environment defined in ``pyproject.toml`` in one go
   (useful after a lockfile format bump or a broad dependency change),
   use ``--all``:

   ::

       pixi install --all

3. **Run tests** to make sure nothing is broken:

   ::

       pixi run -e dev test

4. **Commit both files** — always commit ``pyproject.toml`` and ``pixi.lock``
   together so that CI and other contributors stay in sync:

   ::

       git add pyproject.toml pixi.lock
       git commit -m "Update dependencies"

.. note::
    If you only want to update a single package, you can target it:
    ``pixi update numpy``. To update packages only within a specific
    environment, use ``pixi update -e dev numpy``.

.. note::
    Contributors who pull changes that include an updated ``pixi.lock``
    typically only need to run ``pixi install`` (which refreshes
    ``default``) or ``pixi install -e dev`` (if they work in the ``dev``
    environment) to pick up the new packages.

Installing without pixi
~~~~~~~~~~~~~~~~~~~~~~~~

If you prefer not to use pixi, you can still install RdTools and its
dependencies with pip. The packages necessary to run RdTools itself can be
installed from `PyPI <https://pypi.org/project/rdtools/>`_:

::

    pip install rdtools

This will install the latest official release of RdTools.  If you want to work
with a development version and you have cloned the Github repository to your
computer, you can also install RdTools and dependencies by navigating to the
repository root, switching to the branch you're interested in, for instance:

::

    git checkout development

and running:

::

    pip install .

This will install based on whatever RdTools branch you have checked out.  You
can check what version is currently installed by inspecting
``rdtools.__version__``:

::

    >>> rdtools.__version__
    '1.2.0+188.g5a96bb2'

The hex string at the end represents the hash of the git commit for your
installed version.

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

With pixi (recommended):

::

    pixi install -e dev          # all development dependencies (recommended)
    pixi install -e dev-py310    # dev environment with Python 3.10
    pixi install -e dev-min      # minimum supported dependency versions

With pip:

::

    pip install rdtools[dev]   # notebooks + test dependencies
    pip install rdtools[test]  # test suite dependencies only
    pip install rdtools[doc]   # documentation dependencies

Or, if your local repository has an updated dependencies list:

::

    pip install .[dev]   # notebooks + test dependencies
    pip install .[test]  # test suite dependencies only
    pip install .[doc]   # documentation dependencies


Running the test suite
----------------------

RdTools uses `pytest <https://docs.pytest.org/en/latest/>`_ to run its test
suite.  If you haven't already, install the testing dependencies
(:ref:`installing-optional-dependencies`).

With pixi
~~~~~~~~~

Run the full test suite in the dev environment:

::

    pixi run -e dev test

Run a single test module:

::

    pixi run -e dev pytest rdtools/test/soiling_test.py

Run a single test function:

::

    pixi run -e dev pytest rdtools/test/soiling_test.py::test_soiling_srr

Without pixi
~~~~~~~~~~~~~

If you installed RdTools and its test dependencies with pip, you can invoke
pytest directly:

::

    pytest rdtools/test/
    pytest rdtools/test/soiling_test.py
    pytest rdtools/test/soiling_test.py::test_soiling_srr

Measuring code coverage
~~~~~~~~~~~~~~~~~~~~~~~

You can evaluate code coverage when running the test suite using the
`coverage <https://coverage.readthedocs.io>`_ package:

::

    coverage run -m pytest
    coverage report

The first line runs the test suite and keeps track of exactly what lines of
code were run during test execution.  The second line then prints out a
summary report showing how much of each source file was executed in the
test suite.  If a percentage is below 100, that means a function isn't tested
or a branch inside a function isn't tested.  To get specific details, you can
run ``coverage html`` to generate a detailed HTML report at
``htmlcov/index.html`` to view in a browser.

Note that the pixi ``test`` task already includes ``--cov`` flags, so coverage
data is collected automatically when running ``pixi run -e dev test``.

RdTools also uses `Codecov <https://codecov.io/gh/NatLabRockies/rdtools>`_ to
track coverage over time.  Coverage reports are uploaded automatically by CI
after each test run.  Pull requests will show a Codecov status check indicating
whether the change increases or decreases overall coverage.


Running the notebooks as tests
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Re-running the example notebooks to see if their outputs have changed is
another good check. Besides re-running and examining the notebook outputs
manually, this command will automatically re-run the specified notebook
and compare outputs for you:

::

    pixi run -e dev pytest --nbval docs/system_availability_example.ipynb

A helpful option is ``--sanitize-with docs/nbval_sanitization_rules.cfg`` to
ignore nuisance differences like the username shown in warning messages:

::

    pixi run -e dev pytest --nbval --sanitize-with docs/nbval_sanitization_rules.cfg docs/system_availability_example.ipynb

Or to run all notebooks at once using the pixi task:

::

    pixi run -e dev nbval


Re-running and saving notebook outputs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If notebook outputs need to be refreshed (e.g. after a code change that
affects plots or printed results), you can re-execute all notebooks in place
using ``jupyter nbconvert``:

::

    pixi run -e dev jupyter nbconvert --to notebook --execute --inplace docs/*.ipynb

Or a single notebook:

::

    pixi run -e dev jupyter nbconvert --to notebook --execute --inplace docs/TrendAnalysis_example.ipynb

This overwrites each notebook file with freshly executed outputs.  Review the
diffs before committing to make sure the changes look correct.


Checking for code style
-----------------------

RdTools uses `flake8 <https://flake8.pycqa.org/en/latest/>`_ to validate
code style. To run this check locally you'll need to have flake8 installed
(see :ref:`installing-optional-dependencies`). Then navigate to the git repo
folder and run

::

    pixi run -e dev flake8

Or, for a more detailed report:

::

    pixi run -e dev flake8 --count --statistics --show-source


Building documentation locally
------------------------------

RdTools uses `Sphinx <https://www.sphinx-doc.org/>`_ to build its documentation.
If you haven't already, install the documentation dependencies
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

Contributing
------------

Community participation is welcome!  New contributions should be based on the
``development`` branch as the ``master`` branch is used only for releases.

RdTools follows the `PEP 8 <https://www.python.org/dev/peps/pep-0008/>`_ style guide.
We recommend setting up your text editor to automatically highlight style
violations because it's easy to miss some issues (trailing whitespace, etc) otherwise.

Additionally, our documentation is built in part from docstrings in the source
code.  These docstrings must be in `NumpyDoc format <https://numpydoc.readthedocs.io/en/latest/format.html>`_
to be rendered correctly in the documentation.

Finally, all code should be tested.  Some older tests in RdTools use the unittest
module, but new tests should all use pytest.