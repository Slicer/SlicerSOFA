SparseGridSimulation
=====================

Overview
--------

The SparseGridSimulation module demonstrates simulations using sparse-grid data structures. It is meant as a research-oriented example showing how to couple specialized simulation backends to Slicer visualization and UI.

What it provides
---------------

- Example scenes using sparse-grid solvers and custom SOFA components.
- GUI controls to change grid resolution and solver settings.
- Export of simulation results to VTK/Model nodes for visualization and downstream processing.

Usage
-----

.. code-block:: python

   logic = slicer.util.getModuleLogic('SlicerSofa')
   scene = logic.LoadSofaScene('/path/to/SparseGridSimulation/Resources/Scenes/sparse_grid.scn')
   scene.SetFixedFrame('world')
   scene.Play()

Notes
-----

This module highlights how to connect custom SOFA plugins into the Slicer environment. If your project has non-standard SOFA components, add them to the SuperBuild or point the build to your SOFA installation with the custom plugins.
