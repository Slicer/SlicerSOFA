###################################################################################
# MIT License
#
# Copyright (c) 2024 Oslo University Hospital, Norway. All Rights Reserved.
# Copyright (c) 2024 NTNU, Norway. All Rights Reserved.
# Copyright (c) 2024 INRIA, France. All Rights Reserved.
# Copyright (c) 2024 Harvard Medical School. All Rights Reserved.
# Copyright (c) 2004 Brigham and Women's Hospital (BWH). All Rights Reserved.
# Copyright (c) 2024 Isomics, Inc., USA. All Rights Reserved.
# Copyright (c) 2024 Queen's University, Canada. All Rights Reserved.
# Copyright (c) 2024 Kitware Inc. All Rights Reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
###################################################################################

import logging
import os
import qt
import vtk
import random
import time
import uuid
import numpy as np

import slicer
from slicer.i18n import tr as _
from slicer.i18n import translate
from slicer.ScriptedLoadableModule import *
from slicer import (
    vtkMRMLGridTransformNode,
    vtkMRMLMarkupsFiducialNode,
    vtkMRMLMarkupsLineNode,
    vtkMRMLMarkupsNode,
    vtkMRMLMarkupsROINode,
    vtkMRMLModelNode,
)

from SofaEnvironment import Sofa
from SlicerSofa import (
    SlicerSofaWidget,
    SlicerSofaLogic,
    SofaParameterNodeWrapper,
    NodeMapper,
)

from slicer.parameterNodeWrapper import parameterPack

from SlicerSofaUtils.Mappings import (
    mrmlModelPolyToSofaTriangleTopologyContainer,
    mrmlMarkupsFiducialToSofaPointer,
    mrmlMarkupsROIToSofaBoxROI,
    sofaMechanicalObjectToMRMLModelPoly,
    sofaMechanicalObjectToMRMLModelGrid,
    sofaSparseGridTopologyToMRMLModelGrid,
    sofaVonMisesStressToMRMLModelGrid,
    arrayFromMarkupsROIPoints,
    arrayVectorFromMarkupsLinePoints,
    RunOnce
)

# -----------------------------------------------------------------------------
# Class: SparseGridSimulation
# -----------------------------------------------------------------------------

class SparseGridSimulation(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = _("Sparse Grid Simulation")
        self.parent.categories = [translate("qSlicerAbstractCoreModule", "Examples")]
        self.parent.dependencies = []
        self.parent.contributors = [
            "Rafael Palomar (Oslo University Hospital)",
            "Paul Baksic (INRIA)",
            "Steve Pieper (Isomics, Inc.)",
            "Andras Lasso (Queen's University)",
            "Sam Horvath (Kitware, Inc.)",
            "Jean-Christophe Fillion-Robin (Kitware, Inc.)",
            "Nazim Haouchine (Harvard Medical School / Brigham and Women's Hospital)"
        ]
        self.parent.helpText = _("""
        This is a Slicer-SOFA example module. The module allows to create a simulation based on sparse grid topology (which does not require tetrahedral meshes). In addition the module allows the application of grid transforms derived from the simulation, to medical images (e.g., segmentations and volumes).
        """)
        self.parent.acknowledgementText = _("""This project was funded by Oslo University Hospital.""")

        # Additional initialization step after application startup is complete
        slicer.app.connect("startupCompleted()", registerSampleData)


#
# Register sample data sets in Sample Data module
#

SOFA_DATA_URL = 'https://github.com/rafaelpalomar/SlicerSofaTestingData/releases/download/'

def registerSampleData():
    """Register sample data sets in the Sample Data module."""

    import SampleData

    iconsPath = os.path.join(os.path.dirname(__file__), "Resources/Icons")

    sofaDataURL= 'https://github.com/rafaelpalomar/SlicerSofaTestingData/releases/download/'

    # To ensure that the source code repository remains small (can be downloaded and installed quickly)
    # it is recommended to store data sets that are larger than a few MB in a Github release.

    # Right lung low poly tetrahedral mesh dataset
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        category='SOFA',
        sampleName='LiverSimulationScene',
        thumbnailFileName=os.path.join(iconsPath, 'LiverSimulationScene.png'),
        uris=sofaDataURL+ 'SHA256/19b38403d6ef301f2b049e3f32134962461b74dbccf31278938c2f7986371e89',
        fileNames='LiverSimulationScene.mrb',
        checksums='SHA256:19b38403d6ef301f2b049e3f32134962461b74dbccf31278938c2f7986371e89',
        nodeNames='LiverSimulationScene',
        loadFileType='SceneFile',
        loadFiles=True
    )


# -----------------------------------------------------------------------------
# Function: CreateScene
# -----------------------------------------------------------------------------
def CreateScene() -> Sofa.Core.Node:
    """
    Creates the main SOFA scene with required components for simulation.

    Returns:
        Sofa.Core.Node: The root node of the SOFA simulation scene.
    """
    from stlib3.scene import MainHeader, ContactHeader
    from stlib3.solver import DefaultSolver
    from stlib3.physics.deformable import ElasticMaterialObject
    from stlib3.physics.rigid import Floor
    from splib3.numerics import Vec3

    rootNode = Sofa.Core.Node()
    MainHeader(rootNode, plugins=[
        "Sofa.Component.IO.Mesh",
        "Sofa.Component.LinearSolver.Direct",
        "Sofa.Component.LinearSolver.Iterative",
        "Sofa.Component.Mapping.Linear",
        "Sofa.Component.Mass",
        "Sofa.Component.ODESolver.Backward",
        "Sofa.Component.Setting",
        "Sofa.Component.SolidMechanics.FEM.Elastic",
        "Sofa.Component.StateContainer",
        "Sofa.Component.Topology.Container.Dynamic",
        "Sofa.Component.Visual",
        "Sofa.GL.Component.Rendering3D",
        "Sofa.Component.AnimationLoop",
        "Sofa.Component.Collision.Detection.Algorithm",
        "Sofa.Component.Collision.Detection.Intersection",
        "Sofa.Component.Collision.Geometry",
        "Sofa.Component.Collision.Response.Contact",
        "Sofa.Component.Constraint.Lagrangian.Solver",
        "Sofa.Component.Constraint.Lagrangian.Correction",
        "Sofa.Component.LinearSystem",
        "Sofa.Component.MechanicalLoad",
        "MultiThreading",
        "Sofa.Component.SolidMechanics.Spring",
        "Sofa.Component.Constraint.Lagrangian.Model",
        "Sofa.Component.Mapping.NonLinear",
        "Sofa.Component.Topology.Container.Constant",
        "Sofa.Component.Topology.Mapping",
        "Sofa.Component.Engine.Select",
        "Sofa.Component.Constraint.Projective",
        "Sofa.Component.Topology.Container.Grid",
    ])

    rootNode.addObject('DefaultAnimationLoop', parallelODESolving=True)
    rootNode.addObject('DefaultPipeline', depth=6, verbose=0, draw=0)
    rootNode.addObject('ParallelBruteForceBroadPhase')
    rootNode.addObject('BVHNarrowPhase')
    rootNode.addObject('ParallelBVHNarrowPhase')
    rootNode.addObject('MinProximityIntersection', name="Proximity", alarmDistance=0.005, contactDistance=0.003)
    rootNode.addObject('DefaultContactManager', name="Response", response="PenalityContactForceField")

    inputNode = rootNode.addChild('InputSurfaceNode', name='InputSurfaceNode')
    container = inputNode.addObject('TriangleSetTopologyContainer', name='Container')

    fem = rootNode.addChild('FEM', name='FEM')
    fem.addObject('SparseGridTopology', name='SparseGridTopology', n=[20, 20, 20], position="@../InputSurfaceNode/Container.position")
    fem.addObject('EulerImplicitSolver', rayleighStiffness=0.1, rayleighMass=0.1)
    fem.addObject('CGLinearSolver', iterations=100, tolerance=1e-5, threshold=1e-5)
    fem.addObject('MechanicalObject', name='MO')
    fem.addObject('UniformMass', totalMass=0.5)
    fem.addObject('ParallelHexahedronFEMForceField', name="FEMForce", youngModulus=5, poissonRatio=0.40, method="large")

    surf = fem.addChild('Surf', name='Surf')
    surf.addObject('MeshTopology', position="@../../InputSurfaceNode/Container.position")
    surf.addObject('MechanicalObject', name='MechanicalObject', position="@../../InputSurfaceNode/Container.position")
    surf.addObject('TriangleCollisionModel', selfCollision=True)
    surf.addObject('LineCollisionModel')
    surf.addObject('PointCollisionModel')
    surf.addObject('BarycentricMapping')

    fem.addObject('BoxROI', name="FixedROI",
                  template="Vec3", box=[0.0]*6, drawBoxes=False,
                  position="@../MO.rest_position",
                  computeTriangles=False, computeTetrahedra=False, computeEdges=False)
    fem.addObject('FixedConstraint', indices="@FixedROI.indices")

    return rootNode

# -----------------------------------------------------------------------------
# ParameterPack Class: Grid dimensions
# -----------------------------------------------------------------------------
@parameterPack
class GridDimensions:
  x: int
  y: int
  z: int

# -----------------------------------------------------------------------------
# Class: SparseGridSimulationParameterNode
# -----------------------------------------------------------------------------
@SofaParameterNodeWrapper
class SparseGridSimulationParameterNode:
    """
    Parameter class for the sparse grid simulation.
    Defines nodes to map between SOFA and MRML scenes with recording options.
    """

    # Model node with SOFA mapping and sequence recording
    modelNode: vtkMRMLModelNode = \
        NodeMapper(
            sofaMapping = lambda self: RunOnce(mrmlModelPolyToSofaTriangleTopologyContainer)(self, "InputSurfaceNode.Container"),
            mrmlMapping = lambda self: sofaMechanicalObjectToMRMLModelPoly(self, "FEM.Surf.MechanicalObject"),
            recordSequence=lambda self: self.recordSequence
        )

    # SparseGrid Model node with SOFA mapping and sequence recording
    sparseGridModelNode: vtkMRMLModelNode = \
        NodeMapper(
            mrmlMapping = ( lambda self: sofaMechanicalObjectToMRMLModelGrid(self, "FEM.MO"),
                            lambda self: RunOnce(sofaSparseGridTopologyToMRMLModelGrid)(self, "FEM.SparseGridTopology"),
                            lambda self: self.sofaDisplacementToModelGridArray("FEM.MO")),
            recordSequence=lambda self: self.recordSequence
        )

    # Boundary ROI node with sequence recording
    boundaryROI: vtkMRMLMarkupsROINode = \
        NodeMapper(
            sofaMapping=lambda self: mrmlMarkupsROIToSofaBoxROI(self,"FEM.FixedROI"),
            recordSequence=lambda self: self.recordSequence
        )

    # Gravity vector node with sequence recording
    gravityVector: vtkMRMLMarkupsLineNode = \
        NodeMapper(
            sofaMapping=lambda self: self.mrmlMarkupsLineToSofaGravityVector(""),
            recordSequence=lambda self: self.recordSequence
        )

    sparseGridDimensions: GridDimensions = \
        NodeMapper(
            sofaMapping=lambda self: self.gridDimensionsToSofaSparseGridTopology("FEM.SparseGridTopology"),
            recordSequence=lambda self: self.recordSequence
        )

    gridTransformNode: vtkMRMLGridTransformNode

    gravityMagnitude: int = 1    # Additional parameter for gravity strength
    recordSequence: bool = False # Record sequence?

    def mrmlMarkupsLineToSofaGravityVector(self, nodePath):
        """
        Maps a line node as a gravity vector in the SOFA node.

        Args:
            sofaNode: The corresponding SOFA node to update.
        """
        if self.gravityVector is None:
            return

        gravityVector = arrayVectorFromMarkupsLinePoints(self.gravityVector)
        magnitude = np.linalg.norm(np.array(gravityVector))
        normalizedGravityVector = gravityVector / magnitude if magnitude != 0 else gravityVector
        self._rootNode.gravity = normalizedGravityVector * self.gravityMagnitude

    def gridDimensionsToSofaSparseGridTopology(self, nodePath):
        """
        Maps thee individual components packed in GridDimensions to an array
        of dimensions

        Args:
            sofaNode: The corresponding SOFA node to update.
        """
        self._rootNode[nodePath].n = [self.sparseGridDimensions.x, self.sparseGridDimensions.y, self.sparseGridDimensions.z]

    def sofaDisplacementToModelGridArray(self, nodePath) -> None:
        self._currentMappingObject.GetUnstructuredGrid().GetPointData().GetArray("Displacement").SetNumberOfTuples(int(self._rootNode[nodePath].position.size/3))
        displacementArray = slicer.util.arrayFromModelPointData(self._currentMappingObject, "Displacement")
        displacementArray[:] = (self._rootNode[nodePath].position - self._rootNode[nodePath].rest_position)
        slicer.util.arrayFromModelPointsModified(self._currentMappingObject)

# -----------------------------------------------------------------------------
# Class: SparseGridSimulationWidget
# -----------------------------------------------------------------------------
class SparseGridSimulationWidget(SlicerSofaWidget):
    """
    UI widget for the Soft Tissue Simulation module.
    Manages user interactions and GUI elements.
    """
    def __init__(self, parent=None) -> None:
        """
        Initialize the widget and set up observation mixin.

        Args:
            parent: The parent widget.
        """
        SlicerSofaWidget.__init__(self, parent)
        self.logic = None
        self.timer = qt.QTimer(parent)
        self.timer.timeout.connect(self.simulationStep)

    def setup(self) -> None:
        """
        Sets up the user interface, logic, and connections.
        """
        ScriptedLoadableModuleWidget.setup(self)

        # Load the widget interface from a .ui file
        uiWidget = slicer.util.loadUI(self.resourcePath("UI/SparseGridSimulation.ui"))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)

        # Initialize logic for simulation computations
        self.logic = SparseGridSimulationLogic()
        uiWidget.setMRMLScene(slicer.mrmlScene)

        # Setup event connections for scene close events
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)

        # Connect UI buttons to their respective methods
        self.ui.startSimulationPushButton.connect("clicked()", self.startSimulation)
        self.ui.stopSimulationPushButton.connect("clicked()", self.stopSimulation)
        self.ui.addBoundaryROIPushButton.connect("clicked()", self.logic.addBoundaryROI)
        self.ui.addGravityVectorPushButton.connect("clicked()", self.logic.addGravityVector)
        self.ui.addSparseGridModelNodePushButton.connect("clicked()", self.logic.addSparseGridModelNode)
        self.ui.addGridTransformNodePushButton.connect("clicked()", self.logic.addGridTransformNode)

        # Initialize parameter node and GUI bindings
        self.setParameterNode(self.logic.getParameterNode())
        self.initializeParameterNode()
        self.logic.getParameterNode().AddObserver(vtk.vtkCommand.ModifiedEvent, self.updateSimulationGUI)
        self.logic.setUi(self)

    def cleanup(self) -> None:
        """
        Cleanup when the module widget is destroyed.
        Stops timers, simulation, and removes observers.
        """
        self.timer.stop()
        self.logic.stopSimulation()
        self.logic.clean()
        self.removeObservers()

    def initializeParameterNode(self) -> None:
        """
        Initializes and sets the parameter node in logic.
        """
        if self.logic:
            self.setParameterNode(self.logic.getParameterNode())
            self.logic.resetParameterNode()
        else:
            logging.debug("Could not initialize the parameter node. No logic found")

    def updateSimulationGUI(self, caller, event):
        """
        Updates the GUI based on the simulation state.

        Args:
            caller: The caller object.
            event: The event triggered.
        """
        parameterNode = self.logic.getParameterNode()
        self.ui.startSimulationPushButton.setEnabled(not parameterNode.isSimulationRunning and parameterNode.modelNode is not None)
        self.ui.stopSimulationPushButton.setEnabled(parameterNode.isSimulationRunning)

    def onSceneStartClose(self, caller, event) -> None:
        """
        Handles the event when the scene starts to close.

        Args:
            caller: The caller object.
            event: The event triggered.
        """
        self.setParameterNode(None)

    def onSceneEndClose(self, caller, event) -> None:
        """
        Handles the event when the scene has closed.

        Args:
            caller: The caller object.
            event: The event triggered.
        """
        if self.parent.isEntered:
            self.initializeParameterNode()

    def startSimulation(self):
        """
        Starts the simulation and begins the timer for simulation steps.
        """
        self.logic.startSimulation()
        self.timer.start(0)

    def stopSimulation(self):
        """
        Stops the simulation and the timer.
        """
        self.timer.stop()
        self.logic.stopSimulation()

    def simulationStep(self):
        """
        Executes a single simulation step.
        """
        self.logic.simulationStep()

# -----------------------------------------------------------------------------
# Class: SparseGridSimulationLogic
# -----------------------------------------------------------------------------
class SparseGridSimulationLogic(SlicerSofaLogic):
    """
    Logic class for the Soft Tissue Simulation.
    Handles scene setup, parameter node management, and simulation steps.
    """
    def __init__(self) -> None:
        """
        Initialize the logic with the SOFA scene.
        """
        super().__init__()
        self._rootNode = CreateScene()
        self._parameterNode = None

    def CreateScene(self):
        return CreateScene()

    def getParameterNode(self):
        """
        Retrieves or creates a wrapped parameter node.

        Returns:
            SparseGridSimulationParameterNode: The parameter node for the simulation.
        """
        if self._parameterNode is None:
            self._parameterNode = SparseGridSimulationParameterNode(super().getParameterNode())
        return self._parameterNode

    def resetParameterNode(self):
        """
        Resets simulation parameters in the parameter node to default values.
        """
        if self.getParameterNode() is not None:
            self.getParameterNode().modelNode = None
            self.getParameterNode().boundaryROI = None
            self.getParameterNode().gravityVector = None
            self.getParameterNode().movingPointNode = None
            self.getParameterNode().dt = 0.01
            self.getParameterNode().currentStep = 0
            self.getParameterNode().totalSteps = -1

    def startSimulation(self) -> None:
        """
        Sets up the scene and starts the simulation.
        """
        self.setupScene(self.getParameterNode())
        super().startSimulation()
        self._simulationRunning = True
        self.getParameterNode().Modified()
        self._createGridTransformPipeline()

    def simulationStep(self) -> None:
        super().simulationStep()
        self._updateProbingImage()

    def stopSimulation(self) -> None:
        """
        Stops the simulation.
        """
        super().stopSimulation()
        self._simulationRunning = False
        self.getParameterNode().Modified()

    def onModelNodeModified(self, caller, event) -> None:
        """
        Updates the model node from SOFA to MRML when modified.

        Args:
            caller: The caller object.
            event: The event triggered.
        """
        if self.getParameterNode().modelNode.GetUnstructuredGrid() is not None:
            self.getParameterNode().modelNode.GetUnstructuredGrid().SetPoints(caller.GetPolyData().GetPoints())
        elif self.getParameterNode().modelNode.GetPolyData() is not None:
            self.getParameterNode().modelNode.GetPolyData().SetPoints(caller.GetPolyData().GetPoints())

    def addBoundaryROI(self) -> None:
        """
        Adds a boundary Region of Interest (ROI) based on the model's bounds.
        """
        roiNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsROINode")
        mesh = None
        bounds = None

        # Determine the mesh type and retrieve bounds
        if self.getParameterNode().modelNode is not None:
            if self.getParameterNode().modelNode.GetUnstructuredGrid() is not None:
                mesh = self.getParameterNode().modelNode.GetUnstructuredGrid()
            elif self.getParameterNode().modelNode.GetPolyData() is not None:
                mesh = self.getParameterNode().modelNode.GetPolyData()

        # If mesh is available, calculate the bounds and set ROI
        if mesh is not None:
            bounds = mesh.GetBounds()
            center = [
                (bounds[0] + bounds[1]) / 2.0,
                (bounds[2] + bounds[3]) / 2.0,
                (bounds[4] + bounds[5]) / 2.0
            ]
            size = [
                abs(bounds[1] - bounds[0]) / 2.0,
                abs(bounds[3] - bounds[2]) / 2.0,
                abs(bounds[5] - bounds[4]) / 2.0
            ]
            roiNode.SetXYZ(center)
            roiNode.SetRadiusXYZ(size[0], size[1], size[2])

        # Assign the ROI node to the parameter node
        self.getParameterNode().boundaryROI = roiNode

    def addSparseGridModelNode(self) -> None:
        """
        Adds a model node to hold the sparse grid from SOFA
        """
        sparseGridModelNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLModelNode')
        sparseGridModelNode.SetName("Sparse Grid Model")
        sparseGridModelNode.CreateDefaultDisplayNodes()
        unstructuredGrid = vtk.vtkUnstructuredGrid()
        points = vtk.vtkPoints()
        unstructuredGrid.SetPoints(points)
        displacementVTKArray = vtk.vtkFloatArray()
        displacementVTKArray.SetNumberOfComponents(3)
        displacementVTKArray.SetName('Displacement')
        unstructuredGrid.GetPointData().AddArray(displacementVTKArray)
        sparseGridModelNode.SetAndObserveMesh(unstructuredGrid)
        self._parameterNode.sparseGridModelNode = sparseGridModelNode

    def addGravityVector(self) -> None:
        """
        Adds a gravity vector as a line in the scene.
        """
        gravityVector = slicer.vtkMRMLMarkupsLineNode()
        gravityVector.SetName("Gravity")
        mesh = None

        # Determine the mesh type and retrieve bounds
        if self.getParameterNode().modelNode is not None:
            if self.getParameterNode().modelNode.GetUnstructuredGrid() is not None:
                mesh = self.getParameterNode().modelNode.GetUnstructuredGrid()
            elif self.getParameterNode().modelNode.GetPolyData() is not None:
                mesh = self.getParameterNode().modelNode.GetPolyData()

        # If mesh is available, calculate the gravity vector based on bounds
        if mesh is not None:
            bounds = mesh.GetBounds()
            center = [
                (bounds[0] + bounds[1]) / 2.0,
                (bounds[2] + bounds[3]) / 2.0,
                (bounds[4] + bounds[5]) / 2.0
            ]
            startPoint = [center[0], bounds[2], center[2]]
            endPoint = [center[0], bounds[3], center[2]]
            gravityVector.AddControlPointWorld(vtk.vtkVector3d(startPoint))
            gravityVector.AddControlPointWorld(vtk.vtkVector3d(endPoint))

        # Add the gravity vector node to the scene and create display nodes
        gravityVector = slicer.mrmlScene.AddNode(gravityVector)
        if gravityVector is not None:
            gravityVector.CreateDefaultDisplayNodes()

        # Assign the gravity vector node to the parameter node
        self.getParameterNode().gravityVector = gravityVector

    def _createGridTransformPipeline(self) -> None:
        self.probeGrid = vtk.vtkImageData()
        # NOTE: For now, the probe dimension is equal to the sparse grid dimension
        self.probeGrid.SetDimensions(self._parameterNode.sparseGridDimensions.x,
                                     self._parameterNode.sparseGridDimensions.y,
                                     self._parameterNode.sparseGridDimensions.z)
        self.probeGrid.AllocateScalars(vtk.VTK_DOUBLE, 1)
        self.probeGrid.SetOrigin(0, 0, 0)
        self.probeGrid.SetSpacing(1, 1, 1)
        self.probeFilter = vtk.vtkProbeFilter()
        self.probeFilter.SetInputData(self.probeGrid)
        self.probeFilter.SetSourceData(self._parameterNode.sparseGridModelNode.GetUnstructuredGrid())
        self.probeFilter.SetPassPointArrays(True)

        displacementGrid = self._parameterNode.gridTransformNode.GetTransformFromParent().GetDisplacementGrid()
        # NOTE: The order is slices, rows columns
        displacementGrid.SetDimensions(self._parameterNode.sparseGridDimensions.z,
                                       self._parameterNode.sparseGridDimensions.y,
                                       self._parameterNode.sparseGridDimensions.x)

        narray = np.zeros((self._parameterNode.sparseGridDimensions.x,
                           self._parameterNode.sparseGridDimensions.y,
                           self._parameterNode.sparseGridDimensions.z, 3))

        scalarType = vtk.util.numpy_support.get_vtk_array_type(narray.dtype)
        displacementGrid.AllocateScalars(scalarType, 3)
        displacementArray = slicer.util.arrayFromGridTransform(self._parameterNode.gridTransformNode)
        displacementArray[:] = narray
        slicer.util.arrayFromGridTransformModified(self._parameterNode.gridTransformNode)

    def _updateProbingImage(self) -> None:
        # Update the geometry of the probing image, which need to match the sparse grid created by SOFA
        femGridBounds = [0] * 6
        self._parameterNode.modelNode.GetRASBounds(femGridBounds)

        self.probeGrid.SetOrigin(femGridBounds[0], femGridBounds[2], femGridBounds[4])
        probeSize = (abs(femGridBounds[1] - femGridBounds[0]),
                     abs(femGridBounds[3] - femGridBounds[2]),
                     abs(femGridBounds[5] - femGridBounds[4]))
        self.probeGrid.SetSpacing(probeSize[0] / self._parameterNode.sparseGridDimensions.x,
                                  probeSize[1] / self._parameterNode.sparseGridDimensions.y,
                                  probeSize[2] / self._parameterNode.sparseGridDimensions.z)
        self.probeGrid.AllocateScalars(vtk.VTK_DOUBLE, 1)
        self.probeGrid.Modified()
        self.probeFilter.Update()

        probeGrid = self.probeFilter.GetOutputDataObject(0)
        probeVTKArray = probeGrid.GetPointData().GetArray("Displacement")
        probeArray = vtk.util.numpy_support.vtk_to_numpy(probeVTKArray)
        probeArrayShape = (self._parameterNode.sparseGridDimensions.x,
                           self._parameterNode.sparseGridDimensions.y,
                           self._parameterNode.sparseGridDimensions.z,
                           3)
        probeArray = probeArray.reshape(probeArrayShape)
        gridArray = slicer.util.arrayFromGridTransform(self._parameterNode.gridTransformNode)
        gridArray[:] = -1. * probeArray
        slicer.util.arrayFromGridTransformModified(self._parameterNode.gridTransformNode)
        self.displacementGrid = self._parameterNode.gridTransformNode.GetTransformFromParent().GetDisplacementGrid()
        self.displacementGrid.SetOrigin(probeGrid.GetOrigin())
        self.displacementGrid.SetSpacing(probeGrid.GetSpacing())

    def addGridTransformNode(self) -> None:
        """
        Adds a grid transform node to the scene
        """
        gridTransformNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLGridTransformNode')
        gridTransformNode.CreateDefaultDisplayNodes()
        self._parameterNode.gridTransformNode = gridTransformNode

#
# SparseGridSimulationTest
#


class SparseGridSimulationTest(ScriptedLoadableModuleTest):
    """
    This is the test case for your scripted module.
    Uses ScriptedLoadableModuleTest base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def setUp(self):
        """Do whatever is needed to reset the state - typically a scene clear will be enough."""
        slicer.mrmlScene.Clear()

    def runTest(self):
        """Run as few or as many tests as needed here."""
        self.setUp()
        self.test_SparseGridSimulation1()

    def test_SparseGridSimulation1(self):
        """Ideally you should have several levels of tests.  At the lowest level
        tests should exercise the functionality of the logic with different inputs
        (both valid and invalid).  At higher levels your tests should emulate the
        way the user would interact with your code and confirm that it still works
        the way you intended.
        One of the most important features of the tests is that it should alert other
        developers when their changes will have an impact on the behavior of your
        module.  For example, if a developer removes a feature that you depend on,
        your test should break so they know that the feature is needed.
        """

        self.delayDisplay("Starting the test")

        # Get/create input data

        import SampleData

        registerSampleData()
        inputVolume = SampleData.downloadSample("SparseGridSimulation1")
        self.delayDisplay("Loaded test data set")

        inputScalarRange = inputVolume.GetImageData().GetScalarRange()
        self.assertEqual(inputScalarRange[0], 0)
        self.assertEqual(inputScalarRange[1], 695)

        outputVolume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")
        threshold = 100

        # Test the module logic

        logic = SparseGridSimulationLogic()

        # Test algorithm with non-inverted threshold
        logic.process(inputVolume, outputVolume, threshold, True)
        outputScalarRange = outputVolume.GetImageData().GetScalarRange()
        self.assertEqual(outputScalarRange[0], inputScalarRange[0])
        self.assertEqual(outputScalarRange[1], threshold)

        # Test algorithm with inverted threshold
        logic.process(inputVolume, outputVolume, threshold, False)
        outputScalarRange = outputVolume.GetImageData().GetScalarRange()
        self.assertEqual(outputScalarRange[0], inputScalarRange[0])
        self.assertEqual(outputScalarRange[1], inputScalarRange[1])

        self.delayDisplay("Test passed")
