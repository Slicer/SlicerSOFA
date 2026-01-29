Tests
=====

Unit and integration tests
--------------------------

SlicerSOFA includes a set of tests to validate basic loading, stepping and data mapping functionality. Tests are implemented as C++ tests (ctest) and Python-based tests that exercise the generated Python wrappers inside Slicer.

Running tests
-------------

If you built with CTest support:

.. code-block:: bash

   cd build
   ctest -j2 --output-on-failure

Python tests from inside Slicer
------------------------------

Some tests rely on Slicer runtime and GUI integration. To run Python tests from inside Slicer:

.. code-block:: python

   tests = slicer.util.getModuleLogic('SlicerSofaTests')
   tests.run()

Notes
-----

- Tests that require GPU or specialized SOFA plugins may be skipped in CI unless configured.
- Tests that interact with the Slicer GUI are best run manually or in a controlled environment.
