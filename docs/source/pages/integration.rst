Integration design
==================

Goals
-----

SlicerSOFA aims to tightly integrate SOFA's simulation capabilities with 3D Slicer's visualization and scripting environment while keeping the two systems loosely coupled.

Key ideas
---------

- Each loaded SOFA scene is represented by an MRML node (vtkMRMLSofaSceneNode). This enables:
  - Observers and callback mechanisms available in Slicer to respond to simulation changes.
  - Export and persistence with MRML scenes.
  - Easy retrieval and manipulation via Slicer's Python API.

- Simulation loop integration:
  - The main Slicer GUI thread triggers periodic simulation steps via a Qt timer. This avoids complex multithreading issues with VTK and MRML scene updates.
  - The default step frequency is configurable (for example 60 Hz). Users can Play, Pause or Step manually.

- Data mapping:
  - A set of utility functions map SOFA data (mesh vertices, transforms, pointclouds) to VTK/ MRML data structures (vtkPolyData, vtkMatrix4x4, vtkPoints).
  - The SlicerSofaUtils package contains the mapping logic and convenience helpers.

C++ vs Python
-------------

SlicerSOFA is implemented in C++ for robust linking with SOFA. The build generates Python bindings for the MRML/C++ classes (via VTK/Slicer wrappers and/or pybind11 where appropriate). This provides a full Python API inside Slicer without requiring to import SOFA directly from Python.

Typical workflow
----------------

1. Build and load module in Slicer.
2. In Slicer, create or load a SOFA scene (from a .scn or .py scene file).
3. The scene MRML node can expose:
   * Play / Pause / Stop / Step operations.
   * Export of current mesh as a VTK node.
   * Parameters to adjust solver and visual properties.
4. Observers can be attached to the scene node to react to simulation updates (for example update other Slicer modules when tissue moves).

Example Python usage
--------------------

.. code-block:: python

   sofaLogic = slicer.util.getModuleLogic('SlicerSofa')
   # load and add a sofa scene
   sceneNode = sofaLogic.LoadSofaScene('/home/user/models/soft_tissue.scn')

   # add an observer to update something when the scene is stepped
   def onSimulationTick(caller, event):
       # get a VTK mesh from sofa scene and add to Slicer view
       polydata = sceneNode.GetVTKMesh('liverMesh')
       # update/attach polydata to a MRML model node...
   sceneNode.AddObserver(sceneNode.SimulationTickEvent, onSimulationTick)

Threading and execution model
-----------------------------

To avoid race conditions with VTK and the MRML scene, simulation stepping is invoked on the main thread via a Qt timer. If you need a separate simulation thread for heavy computation, you must ensure MRML/VTK updates are marshalled back to the main thread using Slicer/Qt mechanisms.

Extending mappings
------------------

SlicerSofaUtils contains Mappings.py (Python) and mapping C++ helpers. If you need additional conversions (e.g. custom SOFA components), implement the conversion in the Mappings module and expose accessors on your scene node.
