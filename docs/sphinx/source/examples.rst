.. _examples:

Examples
========

This page shows example usage of the RdTools analysis functions.


.. Note that the entries in the list below are nblink filenames, not notebook filenames!

.. There is a limitation in sphinx that I don't understand, but it means that
   you cannot directly access files outside the source directory unless you use
   something like nbsphinx_link, which is what we do here.
   To add a notebook to the gallery, create a .nblink file and add it to the list below.
   Note: the make_github_url() function in conf.py assumes that the name of the .nblink file
   is the same as the notebook it points to!

.. To select a thumbnail image, you need to edit the metadata of the cell with the
   desired image to include a special tags value:
        "metadata": {"tags": ["nbsphinx-thumbnail"]},


.. nbgallery::

    examples/degradation_and_soiling_example_pvdaq_4
    examples/system_availability_example
