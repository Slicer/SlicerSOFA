import logging
import os
from typing import Annotated, Optional
import qt
import vtk
from vtk.util.numpy_support import numpy_to_vtk
import random
import time
import uuid
import numpy as np

import slicer
from slicer.i18n import tr as _
from slicer.i18n import translate
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin

from slicer import vtkMRMLIGTLConnectorNode
from slicer import vtkMRMLMarkupsFiducialNode
from slicer import vtkMRMLMarkupsLineNode
from slicer import vtkMRMLMarkupsNode
from slicer import vtkMRMLMarkupsROINode
from slicer import vtkMRMLModelNode

from SofaEnvironment import Sofa
from SlicerSofa import (
    SlicerSofaLogic,
    SofaParameterNodeWrapper,
    NodeMapper,
    RunOnce
)

def CreateScene() -> Sofa.Core.Node:
    from stlib3.scene import MainHeader, ContactHeader
    from stlib3.solver import DefaultSolver
    from stlib3.physics.deformable import ElasticMaterialObject
    from stlib3.physics.rigid import Floor
    from splib3.numerics import Vec3

    rootNode = Sofa.Core.Node("Root")

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
        "Sofa.Component.Topology.Container.Dynamic",
        "Sofa.Component.Engine.Select",
        "Sofa.Component.Constraint.Projective",
        "SofaIGTLink"
    ])

    rootNode.gravity = [0,0,0]

    rootNode.addObject('FreeMotionAnimationLoop', parallelODESolving=True, parallelCollisionDetectionAndFreeMotion=True)
    rootNode.addObject('GenericConstraintSolver', maxIterations=10, multithreading=True, tolerance=1.0e-3)

    femNode = rootNode.addChild('FEM')
    femNode.addObject('EulerImplicitSolver', firstOrder=False, rayleighMass=0.1, rayleighStiffness=0.1)
    femNode.addObject('SparseLDLSolver', name="precond", template="CompressedRowSparseMatrixd", parallelInverseProduct=True)
    femNode.addObject('TetrahedronSetTopologyContainer', name="Container", position=None, tetrahedra=None)
    femNode.addObject('TetrahedronSetTopologyModifier', name="Modifier")
    femNode.addObject('MechanicalObject', name="mstate", template="Vec3d")
    femNode.addObject('TetrahedronFEMForceField', name="FEM", youngModulus=1.5, poissonRatio=0.45, method="large")
    femNode.addObject('MeshMatrixMass', totalMass=1)

    fixedROI = femNode.addChild('FixedROI')
    fixedROI.addObject('BoxROI', template="Vec3", box=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], drawBoxes=False,
                       position="@../mstate.rest_position", name="BoxROI",
                       computeTriangles=False, computeTetrahedra=False, computeEdges=False)
    fixedROI.addObject('FixedConstraint', indices="@BoxROI.indices")

    collisionNode = femNode.addChild('Collision')
    collisionNode.addObject('TriangleSetTopologyContainer', name="Container")
    collisionNode.addObject('TriangleSetTopologyModifier', name="Modifier")
    collisionNode.addObject('Tetra2TriangleTopologicalMapping', input="@../Container", output="@Container")
    collisionNode.addObject('TriangleCollisionModel', name="collisionModel", proximity=0.001, contactStiffness=20)
    collisionNode.addObject('MechanicalObject', name='dofs', rest_position="@../mstate.rest_position")
    collisionNode.addObject('IdentityMapping', name='visualMapping')

    femNode.addObject('LinearSolverConstraintCorrection', linearSolver="@precond")

    attachPointNode = rootNode.addChild('AttachPoint')
    attachPointNode.addObject('PointSetTopologyContainer', name="Container")
    attachPointNode.addObject('PointSetTopologyModifier', name="Modifier")
    attachPointNode.addObject('MechanicalObject', name="mstate", template="Vec3d", drawMode=2, showObjectScale=0.01, showObject=False)
    attachPointNode.addObject('iGTLinkMouseInteractor', name="mouseInteractor", pickingType="constraint", reactionTime=20, destCollisionModel="@../FEM/Collision/collisionModel")

    return rootNode





@SofaParameterNodeWrapper
class SoftTissueSimulationParameterNode:
    """
    The parameters needed by the module.
    """

    # Simulation data with recording enabled
    modelNode: vtkMRMLModelNode = \
        NodeMapper(
            nodeName="FEM",
            sofaMapping=lambda self, sofaNode: self.modelNodetoSofaNode(sofaNode),
            mrmlMapping=lambda self, sofaNode: self.sofaNodeToModelNode(sofaNode),
            recordSequence=True
        )

    movingPointNode: vtkMRMLMarkupsFiducialNode = \
        NodeMapper(
            nodeName="AttachPoint.mouseInteractor",
            sofaMapping=lambda self, sofaNode: self.markupsFiducialNodeToSofaPoint(sofaNode),
            recordSequence=True
        )

    # Nodes without recording
    boundaryROI: vtkMRMLMarkupsROINode = \
        NodeMapper(
            nodeName="FEM.FixedROI.BoxROI",
            sofaMapping=lambda self, sofaNode: self.markupsROIToSofaROI(sofaNode),
            recordSequence=True
        )

    gravityVector: vtkMRMLMarkupsLineNode = \
        NodeMapper(
            nodeName="",
            sofaMapping=lambda self, sofaNode: self.markupsLineToGravityVector(sofaNode),
            recordSequence=True
        )

    # Additional parameters
    gravityMagnitude: int = 1

    def markupsROIToSofaROI(self, sofaNode):

        if self.boundaryROI is None:
            return [0.0]*6

        center = [0]*3
        self.boundaryROI.GetCenter(center)
        size = self.boundaryROI.GetSize()

        # Calculate min and max RAS bounds from center and size
        R_min = center[0] - size[0] / 2
        R_max = center[0] + size[0] / 2
        A_min = center[1] - size[1] / 2
        A_max = center[1] + size[1] / 2
        S_min = center[2] - size[2] / 2
        S_max = center[2] + size[2] / 2

        # Return the two opposing bounds corners
        # First corner: (minL, minP, minS), Second corner: (maxL, maxP, maxS)
        sofaNode.box=[[R_min, A_min, S_min, R_max, A_max, S_max]]

    def markupsLineToGravityVector(self, sofaNode):

        if self.gravityVector is None:
            return [0.0]*3

        p1 = self.gravityVector.GetNthControlPointPosition(0)
        p2 = self.gravityVector.GetNthControlPointPosition(1)
        gravity_vector = np.array([p2[0]-p1[0], p2[1]-p1[1], p2[2]-p1[2]])
        magnitude = np.linalg.norm(gravity_vector)
        normalized_gravity_vector = gravity_vector / magnitude if magnitude != 0 else gravity_vector

        sofaNode.gravity = normalized_gravity_vector*self.gravityMagnitude

    # Maps VTK model node to SOFA node data
    @RunOnce
    def modelNodetoSofaNode(self, sofaNode):
        unstructuredGrid = self.modelNode.GetUnstructuredGrid()
        points = unstructuredGrid.GetPoints()
        numPoints = points.GetNumberOfPoints()

        # Convert VTK points to a list for SOFA node
        pointCoords = [points.GetPoint(i) for i in range(numPoints)]

        cells = unstructuredGrid.GetCells()
        cellArray = vtk.util.numpy_support.vtk_to_numpy(cells.GetData())

        # Parse cell data (tetrahedra connectivity)
        cellConnectivity = []
        idx = 0
        for i in range(unstructuredGrid.GetNumberOfCells()):
            numPoints = cellArray[idx]
            cellConnectivity.append(cellArray[idx+1:idx+1+numPoints].tolist())
            idx += numPoints + 1

        sofaNode["Container"].tetrahedra = cellConnectivity
        sofaNode["Container"].position = pointCoords

    # Maps SOFA node to VTK model node data
    def sofaNodeToModelNode(self, sofaNode):
        convertedPoints = numpy_to_vtk(num_array=sofaNode["Collision.dofs"].position.array(), deep=True, array_type=vtk.VTK_FLOAT)
        points = vtk.vtkPoints()
        points.SetData(convertedPoints)
        self.modelNode.GetUnstructuredGrid().SetPoints(points)
        self.modelNode.Modified()

    # Maps MRML fiducial node to SOFA node point position
    def markupsFiducialNodeToSofaPoint(self, sofaNode):
        sofaNode.position = [list(self.movingPointNode.GetNthControlPointPosition(0))*3]

# Main module definition for Slicer UI and metadata setup
class SoftTissueSimulation(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = _("Soft Tissue Simulation")
        self.parent.categories = [translate("qSlicerAbstractCoreModule", "Examples")]
        self.parent.dependencies = []
        self.parent.contributors = ["Rafael Palomar (Oslo University Hospital), Paul Baksic (INRIA), Steve Pieper (Isomics, Inc.), Andras Lasso (Queen's University), Sam Horvath (Kitware, Inc.)"]
        self.parent.helpText = _("""This module uses SOFA framework to simulate soft tissue""")
        self.parent.acknowledgementText = _("""This project was funded by Oslo University Hospital""")

        # Additional initialization after startup
        slicer.app.connect("startupCompleted()", registerSampleData)

# Registers sample data for module
def registerSampleData():
    import SampleData

    iconsPath = os.path.join(os.path.dirname(__file__), "Resources/Icons")
    sofaDataURL = 'https://github.com/rafaelpalomar/SlicerSofaTestingData/releases/download/'

    # Registers a sample lung model data set for the module
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        category='SOFA',
        sampleName='RightLungLowTetra',
        thumbnailFileName=os.path.join(iconsPath, 'RightLungLowTetra.png'),
        uris=sofaDataURL + 'SHA256/a35ce6ca2ae565fe039010eca3bb23f5ef5f5de518b1c10257f12cb7ead05c5d',
        fileNames='RightLungLowTetra.vtk',
        checksums='SHA256:a35ce6ca2ae565fe039010eca3bb23f5ef5f5de518b1c10257f12cb7ead05c5d',
        nodeNames='RightLung',
        loadFileType='ModelFile'
    )

#
# SoftTissueSimulationWidget
#

class SoftTissueSimulationWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    """Uses ScriptedLoadableModuleWidget base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent=None) -> None:
        """Called when the user opens the module the first time and the widget is initialized."""
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)  # needed for parameter node observation
        self.logic = None
        self.parameterNode = None
        self.parameterNodeGuiTag = None
        self.timer = qt.QTimer(parent)
        self.timer.timeout.connect(self.simulationStep)

    def setup(self) -> None:
        """Called when the user opens the module the first time and the widget is initialized."""
        ScriptedLoadableModuleWidget.setup(self)

        # Load widget from .ui file (created by Qt Designer).
        # Additional widgets can be instantiated manually and added to self.layout.
        uiWidget = slicer.util.loadUI(self.resourcePath("UI/SoftTissueSimulation.ui"))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)

        # Create logic class. Logic implements all computations that should be possible to run
        # in batch mode, without a graphical user interface.
        self.logic = SoftTissueSimulationLogic()

        # Set scene in MRML widgets.
        uiWidget.setMRMLScene(slicer.mrmlScene)

        # Connections


        # These connections ensure that we update parameter node when scene is closed
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)

        self.ui.startSimulationPushButton.connect("clicked()", self.startSimulation)
        self.ui.stopSimulationPushButton.connect("clicked()", self.stopSimulation)
        self.ui.addBoundaryROIPushButton.connect("clicked()", self.logic.addBoundaryROI)
        self.ui.addGravityVectorPushButton.connect("clicked()", self.logic.addGravityVector)
        self.ui.addMovingPointPushButton.connect("clicked()", self.logic.addMovingPoint)

        self.initializeParameterNode()
        self.logic.getParameterNode().AddObserver(vtk.vtkCommand.ModifiedEvent, self.updateSimulationGUI)

    def cleanup(self) -> None:
        """Called when the application closes and the module widget is destroyed."""
        self.timer.stop()
        self.logic.stopSimulation()
        self.logic.clean()
        self.removeObservers()

    def enter(self) -> None:
        """Called each time the user opens this module."""
        self.initializeParameterNode()

    def exit(self) -> None:
        """Called each time the user opens a different module."""
        if self.parameterNode:
            self.parameterNode.disconnectGui(self.parameterNodeGuiTag)
            self.parameterNodeGuiTag = None

    def initializeParameterNode(self) -> None:
        if self.logic:
            self.setParameterNode(self.logic.getParameterNode())
            self.logic.resetParameterNode()

    def setParameterNode(self, inputParameterNode: Optional[SoftTissueSimulationParameterNode]) -> None:
        if self.parameterNode:
            self.parameterNode.disconnectGui(self.parameterNodeGuiTag)
        self.parameterNode = inputParameterNode
        if self.parameterNode:
            self.parameterNodeGuiTag = self.parameterNode.connectGui(self.ui)

    def updateSimulationGUI(self, caller, event):
        parameterNode = self.logic.getParameterNode()
        self.ui.startSimulationPushButton.setEnabled(not parameterNode.isSimulationRunning and
                                                     parameterNode.modelNode is not None)
        self.ui.stopSimulationPushButton.setEnabled(parameterNode.isSimulationRunning)

    def onSceneStartClose(self, caller, event) -> None:
        """Called just before the scene is closed."""
        # Parameter node will be reset, do not use it anymore
        self.setParameterNode(None)

    def onSceneEndClose(self, caller, event) -> None:
        """Called just after the scene is closed."""
        # If this module is shown while the scene is closed then recreate a new parameter node immediately
        if self.parent.isEntered:
            self.initializeParameterNode()

    def startSimulation(self):
        self.logic.startSimulation()
        self.timer.start(0)

    def stopSimulation(self):
        self.timer.stop()
        self.logic.stopSimulation()

    def simulationStep(self):
       self.logic.simulationStep()

#
# SoftTissueSimulationLogic
#

class SoftTissueSimulationLogic(SlicerSofaLogic):
    def __init__(self) -> None:
        super().__init__()
        self._rootNode = CreateScene()

    def getParameterNode(self):
        return SoftTissueSimulationParameterNode(super().getParameterNode())

    def resetParameterNode(self):
        if self.getParameterNode():
            self.getParameterNode().modelNode = None
            self.getParameterNode().boundaryROI = None
            self.getParameterNode().gravityVector = None
            self.getParameterNode().movingPointNode = None
            self.getParameterNode().dt = 0.01
            self.getParameterNode().currentStep = 0
            self.getParameterNode().totalSteps = -1

    def startSimulation(self) -> None:
        self.setupScene(self.getParameterNode(), self._rootNode)
        super().startSimulation()
        self._simulationRunning = True
        self.getParameterNode().Modified()

    def stopSimulation(self) -> None:
        super().stopSimulation()
        self._simulationRunning = False
        self.getParameterNode().Modified()

    def onModelNodeModified(self, caller, event) -> None:
        if self.getParameterNode().modelNode.GetUnstructuredGrid() is not None:
            self.getParameterNode().modelNode.GetUnstructuredGrid().SetPoints(caller.GetPolyData().GetPoints())
        elif self.getParameterNode().modelNode.GetPolyData() is not None:
            self.getParameterNode().modelNode.GetPolyData().SetPoints(caller.GetPolyData().GetPoints())

    def addBoundaryROI(self) -> None:
        roiNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsROINode")
        mesh = None
        bounds = None

        if self.getParameterNode().modelNode is not None:
            if self.getParameterNode().modelNode.GetUnstructuredGrid() is not None:
                mesh = self.getParameterNode().modelNode.GetUnstructuredGrid()
            elif self.getParameterNode().modelNode.GetPolyData() is not None:
                mesh = self.getParameterNode().modelNode.GetPolyData()

        if mesh is not None:
            bounds = mesh.GetBounds()
            center = [(bounds[0] + bounds[1])/2.0, (bounds[2] + bounds[3])/2.0, (bounds[4] + bounds[5])/2.0]
            size = [abs(bounds[1] - bounds[0])/2.0, abs(bounds[3] - bounds[2])/2.0, abs(bounds[5] - bounds[4])/2.0]
            roiNode.SetXYZ(center)
            roiNode.SetRadiusXYZ(size[0], size[1], size[2])

        self.getParameterNode().boundaryROI = roiNode

    def addGravityVector(self) -> None:
        # Create a new line node for the gravity vector
        gravityVector = slicer.vtkMRMLMarkupsLineNode()
        gravityVector.SetName("Gravity")
        mesh = None

        # Check if there is a model node set in the parameter node and get its mesh
        if self.getParameterNode().modelNode is not None:
            if self.getParameterNode().modelNode.GetUnstructuredGrid() is not None:
                mesh = self.getParameterNode().modelNode.GetUnstructuredGrid()
            elif self.getParameterNode().modelNode.GetPolyData() is not None:
                mesh = self.getParameterNode().modelNode.GetPolyData()

        # If a mesh is found, compute its bounding box and center
        if mesh is not None:
            bounds = mesh.GetBounds()

            # Calculate the center of the bounding box
            center = [(bounds[0] + bounds[1])/2.0, (bounds[2] + bounds[3])/2.0, (bounds[4] + bounds[5])/2.0]

            # Calculate the vector's start and end points along the Y-axis, centered on the bounding box
            startPoint = [center[0], bounds[2], center[2]]  # Start at the bottom of the bounding box
            endPoint = [center[0], bounds[3], center[2]]  # End at the top of the bounding box

            # Adjust the start and end points to center the vector in the bounding box
            vectorLength = endPoint[1] - startPoint[1]
            midPoint = startPoint[1] + vectorLength / 2.0
            startPoint[1] = midPoint - vectorLength / 2.0
            endPoint[1] = midPoint + vectorLength / 2.0

            # Add control points to define the line
            gravityVector.AddControlPoint(vtk.vtkVector3d(startPoint))
            gravityVector.AddControlPoint(vtk.vtkVector3d(endPoint))

        # Add the gravity vector line node to the scene
        gravityVector = slicer.mrmlScene.AddNode(gravityVector)
        if gravityVector is not None:
            gravityVector.CreateDefaultDisplayNodes()

        self.getParameterNode().gravityVector = gravityVector

    def addMovingPoint(self) -> None:
        cameraNode = slicer.util.getNode('Camera')
        if None not in [self.getParameterNode().modelNode, cameraNode]:
            fiducialNode = self.addFiducialToClosestPoint(self.getParameterNode().modelNode, cameraNode)
            self.getParameterNode().movingPointNode = fiducialNode

    def addFiducialToClosestPoint(self, modelNode, cameraNode) -> vtkMRMLMarkupsFiducialNode:
        # Obtain the camera's position
        camera = cameraNode.GetCamera()
        camPosition = camera.GetPosition()

        # Get the polydata from the model node
        modelData = None

        if self.getParameterNode().modelNode.GetUnstructuredGrid() is not None:
            modelData = self.getParameterNode().modelNode.GetUnstructuredGrid()
        elif self.getParameterNode().modelNode.GetPolyData() is not None:
            modelData = self.getParameterNode().modelNode.GetPolyData()

        # Set up the point locator
        pointLocator = vtk.vtkPointLocator()
        pointLocator.SetDataSet(modelData)
        pointLocator.BuildLocator()

        # Find the closest point on the model to the camera
        closestPointId = pointLocator.FindClosestPoint(camPosition)
        closestPoint = modelData.GetPoint(closestPointId)

        # Create a new fiducial node
        fiducialNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode")
        fiducialNode.AddControlPointWorld(vtk.vtkVector3d(closestPoint))

        # Optionally, set the name and display properties
        fiducialNode.SetName("Closest Fiducial")
        displayNode = fiducialNode.GetDisplayNode()
        if displayNode:
            displayNode.SetSelectedColor(1, 0, 0)  # Red color for the selected fiducial

        return fiducialNode
