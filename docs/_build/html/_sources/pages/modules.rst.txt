Slicer modules and MRML nodes
============================

Module layout
-------------

This repository provides several Slicer modules:

- SlicerSofa: the core integration logic, MRML node definitions and utility classes.
- SoftTissueSimulation: example module and UI that demonstrates typical soft-tissue simulations and parameters.
- SparseGridSimulation: demonstration of sparse-grid approaches and example scenes.

MRML nodes
----------

The core MRML/C++ classes exposed by SlicerSofa typically include:

- vtkMRMLSofaSceneNode
  - Represents a loaded SOFA scene.
  - Methods: LoadScene(path), Play(), Pause(), Step(), Stop()
  - Accessors to map named SOFA objects (meshes, transforms) to VTK/MRML.
- vtkMRMLSofaModelNode
  - Represents a single object in a SOFA scene (mesh, collision model, etc.).
- vtkMRMLSofaParameterNode
  - Exposes runtime parameters (solver tolerances, time step) and allows them to be adjusted from Slicer.

Python API examples
-------------------

Loading and controlling a scene:

.. code-block:: python

   sofaLogic = slicer.util.getModuleLogic('SlicerSofa')
   sceneNode = sofaLogic.LoadSofaScene('/path/to/soft_tissue.scn')
   sceneNode.SetTimeStep(0.005)
   sceneNode.Play()

Accessing mesh output:

.. code-block:: python

   modelNode = sceneNode.GetModelNodeByName('Liver')
   vtkPolyData = modelNode.GetPolyData()
   # Add to a Slicer model node:
   model = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLModelNode', 'LiverModel')
   model.SetAndObservePolyData(vtkPolyData)

C++ usage
---------

If you are developing inside C++ modules you can access the logic via:

.. code-block:: c++

   vtkMRMLAbstractLogic * logic = this->GetModuleLogic("SlicerSofa");
   vtkSlicerSofaLogic * sofaLogic = vtkSlicerSofaLogic::SafeDownCast(logic);
   vtkMRMLSofaSceneNode * scene = sofaLogic->LoadSofaScene("/path/to/scene.scn");

See the SlicerSofa code for the exact class names and API.
