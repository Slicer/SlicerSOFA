Writing a simulation module
===========================

This tutorial walks through the anatomy of a SlicerSOFA-based scripted module.
It is a distilled version of the ``SoftTissueSimulation`` example module,
which is the best reference for a complete implementation.

A simulation module consists of four parts:

1. a **parameter node** class decorated with ``SofaParameterNodeWrapper``,
2. a **scene factory** that builds the SOFA scene from the parameter node,
3. a **logic** class deriving from ``SlicerSofaLogic`` that registers
   mappings,
4. a **widget** class deriving from ``SlicerSofaWidget`` that owns the
   stepping timer.

Imports
-------

.. code-block:: python

   import qt
   import slicer
   from slicer import vtkMRMLModelNode, vtkMRMLMarkupsROINode

   from SlicerSofa import (
       SlicerSofaLogic,
       SlicerSofaWidget,
       SofaParameterNodeWrapper,
   )
   from SofaEnvironment import Sofa
   from SlicerSofaUtils.Mappings import (
       mrmlModelGridToSofaTetrahedronTopologyContainer,
       mrmlMarkupsROIToSofaBoxROI,
       sofaMechanicalObjectToMRMLModelGrid,
   )

The parameter node
------------------

Declare the MRML nodes and scalar parameters your simulation needs. The
decorator adds the standard control fields (``dt``, ``totalSteps``,
``currentStep``, ``isSimulationRunning``, ``simulationProgress``) and wraps
the class with Slicer's ``parameterNodeWrapper``, so fields can be bound
directly to Qt widgets in a ``.ui`` file via the ``SlicerParameterName``
dynamic property.

.. code-block:: python

   @SofaParameterNodeWrapper
   class MySimulationParameterNode:
       modelNode: vtkMRMLModelNode          # simulated mesh
       boundaryROI: vtkMRMLMarkupsROINode   # fixed region
       recordSequence: bool = False

The scene factory
-----------------

``createScene`` receives the parameter node and must return a
``Sofa.Core.Node`` root. It can read MRML data directly for values that are
fixed at start time; anything that must stay synchronized during the
simulation is handled by mappings instead.

.. code-block:: python

   def CreateScene(parameterNode) -> Sofa.Core.Node:
       root = Sofa.Core.Node("Root")
       for plugin in ["Sofa.Component.ODESolver.Backward",
                      "Sofa.Component.LinearSolver.Iterative",
                      "Sofa.Component.SolidMechanics.FEM.Elastic",
                      "Sofa.Component.StateContainer",
                      "Sofa.Component.Topology.Container.Dynamic",
                      "Sofa.Component.Mass",
                      "Sofa.Component.Engine.Select",
                      "Sofa.Component.Constraint.Projective"]:
           root.addObject("RequiredPlugin", name=plugin)

       fem = root.addChild("FEM")
       fem.addObject("EulerImplicitSolver")
       fem.addObject("CGLinearSolver", iterations=25,
                     tolerance=1e-9, threshold=1e-9)
       fem.addObject("TetrahedronSetTopologyContainer", name="Container")
       fem.addObject("MechanicalObject", name="mstate", template="Vec3d")
       fem.addObject("TetrahedronFEMForceField", youngModulus=1.5,
                     poissonRatio=0.45, method="large")
       fem.addObject("MeshMatrixMass", totalMass=1)

       fixed = fem.addChild("FixedROI")
       fixed.addObject("BoxROI", template="Vec3", name="BoxROI",
                       position="@../mstate.rest_position",
                       computeTriangles=False, computeTetrahedra=False,
                       computeEdges=False)
       fixed.addObject("FixedConstraint", indices="@BoxROI.indices")

       return root

The logic
---------

The logic implements ``createScene`` and ``setupMappings``, and typically
provides ``getParameterNode`` returning the wrapped parameter node:

.. code-block:: python

   class MySimulationLogic(SlicerSofaLogic):

       def __init__(self):
           super().__init__()
           self._parameterNode = None

       def getParameterNode(self):
           if self._parameterNode is None:
               self._parameterNode = MySimulationParameterNode(
                   super().getParameterNode())
           return self._parameterNode

       def createScene(self, parameterNode) -> Sofa.Core.Node:
           return CreateScene(parameterNode)

       def setupMappings(self):
           pn = self.getParameterNode()

           # MRML → SOFA: pushed before every animate() step
           self.registerMRMLToSOFAMapping(
               'modelNode', 'FEM.Container',
               mrmlModelGridToSofaTetrahedronTopologyContainer,
               runOnce=True)  # topology only needs to be transferred once
           self.registerMRMLToSOFAMapping(
               'boundaryROI', 'FEM.FixedROI.BoxROI',
               mrmlMarkupsROIToSofaBoxROI)

           # SOFA → MRML: pulled after every animate() step
           self.registerSOFAToMRMLMapping(
               'modelNode', 'FEM.mstate',
               sofaMechanicalObjectToMRMLModelGrid)

           # Opt-in sequence recording
           self.setRecordSequenceFlag('modelNode',
                                      lambda: pn.recordSequence)

       def startSimulation(self):
           self.setupMappings()
           super().startSimulation()

A custom mapping is just a method or function with the signature
``fn(fieldValue, sofaObject)`` — see
``SoftTissueSimulationLogic.mrmlMarkupsLineToGravityVector`` for an example
that maps a line markup to the scene's gravity vector.

To support **Reset simulation**, override the state hooks:

.. code-block:: python

   def _saveState(self):
       import vtk
       self._originalGrid = vtk.vtkUnstructuredGrid()
       self._originalGrid.DeepCopy(
           self._parameterNode.modelNode.GetUnstructuredGrid())

   def _restoreState(self):
       self._parameterNode.modelNode.SetAndObserveMesh(self._originalGrid)

The widget
----------

The widget owns a zero-interval timer that calls ``simulationStep()`` once
per event-loop iteration while the simulation runs:

.. code-block:: python

   class MySimulationWidget(SlicerSofaWidget):

       def __init__(self, parent=None):
           super().__init__(parent)
           self.logic = None
           self.timer = qt.QTimer()
           self.timer.timeout.connect(self.simulationStep)

       def setup(self):
           super().setup()
           uiWidget = slicer.util.loadUI(
               self.resourcePath("UI/MySimulation.ui"))
           self.layout.addWidget(uiWidget)
           self.ui = slicer.util.childWidgetVariables(uiWidget)
           self.logic = MySimulationLogic()
           uiWidget.setMRMLScene(slicer.mrmlScene)

           self.ui.startSimulationPushButton.connect(
               "clicked()", self.startSimulation)
           self.ui.stopSimulationPushButton.connect(
               "clicked()", self.stopSimulation)

           self.setParameterNode(self.logic.getParameterNode())
           self.logic.setUi(self)

       def startSimulation(self):
           self.logic.startSimulation()
           self.timer.start(0)

       def stopSimulation(self):
           self.timer.stop()
           self.logic.stopSimulation()

       def simulationStep(self):
           self.logic.simulationStep()

       def cleanup(self):
           self.timer.stop()
           self.logic.stopSimulation()
           self.logic.clean()
           self.removeObservers()

UI conveniences
---------------

- Bind ``.ui`` widgets to parameter-node fields with the
  ``SlicerParameterName`` dynamic property (e.g. ``dt``, ``totalSteps``,
  ``simulationProgress``); ``SlicerSofaWidget`` connects and disconnects the
  GUI automatically.
- Set the boolean dynamic property ``SlicerDisableOnSimulation`` on widgets
  that should be disabled while the simulation runs (``True``) or only
  enabled while it runs (``False``); ``updateWidgetOnSimulation()`` applies
  it when the simulation starts and stops.
