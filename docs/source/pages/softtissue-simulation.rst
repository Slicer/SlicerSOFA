SoftTissueSimulation
====================

Overview
--------

The SoftTissueSimulation module is an example application built on top of SlicerSofa showcasing common soft-tissue workflows: loading a deformable organ, running dynamic simulation, and visualizing the results. It demonstrates:

- Loading SOFA scenes (.scn or .py scenes).
- Linking SOFA meshes to Slicer models for rendering and measurement.
- User controls for solver parameters, collision toggles and time stepping.

Resources
---------

The module includes resource files (icons and a UI .ui file) located under SoftTissueSimulation/Resources.

Usage example (Python)
----------------------

.. code-block:: python

   # Open module UI or use logic programmatically
   logic = slicer.util.getModuleLogic('SlicerSofa')
   sceneNode = logic.LoadSofaScene('/path/to/SoftTissueSimulation/Resources/Scenes/liver.scn')
   sceneNode.Play()

Tips
----

- Use smaller time steps for more accurate simulations (but higher CPU cost).
- Visualize intermediate meshes via Export or GetVTKMesh methods if you need to save frames.

Extending scenes
----------------

Create or edit SOFA scene files to customize objects, materials, collisions and solvers. The module ships with example scenes that are good starting points.
