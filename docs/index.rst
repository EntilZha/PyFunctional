.. ScalaFunctional documentation master file, created by
   sphinx-quickstart on Wed Mar 11 23:00:20 2015.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to ScalaFunctional's documentation!
===========================================

The `functional` package has a single entrypoint, the function `seq`. This function
receives an iterable value and wraps it with a custom class which supports the large
set of functionality documented below. In the near future other data streams will
be implemented such as natively reading from csv, json, files, and functions like range.

Below are some samples of using `functional`, documentation for `seq`, and documentation
for all the supported operations.

.. autofunction:: functional.seq

.. autoclass:: functional.pipeline.Sequence
    :members:
    :undoc-members:
    :show-inheritance:


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

