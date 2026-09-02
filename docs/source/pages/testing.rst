Testing
=======

The test suites
---------------

Tests are Python unit tests registered with CTest via
``slicer_add_python_unittest`` and run inside Slicer's Python environment:

- ``SlicerSofa/Testing/Python/SlicerSofaModuleTest.py`` — exercises the
  ``SlicerSofaUtils`` mapping functions (MRML↔SOFA data transfer).
- ``SparseGridSimulation/Testing/Python/SparseGridSimulationGeometryTest.py``
  — geometry checks for the sparse-grid pipeline.
- Each example module also carries a self-test class
  (``ScriptedLoadableModuleTest``) covering an end-to-end simulation run —
  including, for Sparse Grid Simulation, tests that the displacement grid
  transform is stored as *TransformToParent*, reproduces the mesh
  deformation, and that sequence recording is off by default and
  step-aligned.

Running with CTest
------------------

From the inner build directory of a source build (see :doc:`building`):

.. code-block:: bash

   cd SlicerSOFA-build/inner-build
   ctest --output-on-failure

Use ``ctest -R <pattern>`` to select individual tests.

Running inside Slicer
---------------------

With *Developer mode* enabled (**Edit → Application Settings → Developer**),
open an example module and use the **Reload and Test** button to run that
module's self-test against the live application.
