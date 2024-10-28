###################################################################################
# MIT License
#
# Copyright (c) 2024 Oslo University Hospital, Norway. All Rights Reserved.
# Copyright (c) 2024 NTNU, Norway. All Rights Reserved.
# Copyright (c) 2024 INRIA, France. All Rights Reserved.
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

# Creates the main SOFA scene with required components for simulation
def CreateScene() -> Sofa.Core.Node:
    from stlib3.scene import MainHeader, ContactHeader
    from stlib3.solver import DefaultSolver
    from stlib3.physics.deformable import ElasticMaterialObject
    from stlib3.physics.rigid import Floor
    from splib3.numerics import Vec3

    rootNode = Sofa.Core.Node("Root")

    # Initialize main scene headers with necessary plugins for SOFA components
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

    # Adds animation and constraint solver objects to root node
    rootNode.addObject('FreeMotionAnimationLoop', parallelODESolving=True, parallelCollisionDetectionAndFreeMotion=True)
    rootNode.addObject('GenericConstraintSolver', maxIterations=10, multithreading=True, tolerance=1.0e-3)

    # Defines a deformable FEM object
    femNode = rootNode.addChild('FEM')
    femNode.addObject('EulerImplicitSolver', firstOrder=False, rayleighMass=0.1, rayleighStiffness=0.1)
    femNode.addObject('SparseLDLSolver', name="precond", template="CompressedRowSparseMatrixd", parallelInverseProduct=True)
    femNode.addObject('TetrahedronSetTopologyContainer', name="Container", position=None, tetrahedra=None)
    femNode.addObject('TetrahedronSetTopologyModifier', name="Modifier")
    femNode.addObject('MechanicalObject', name="mstate", template="Vec3d")
    femNode.addObject('TetrahedronFEMForceField', name="FEM", youngModulus=1.5, poissonRatio=0.45, method="large")
    femNode.addObject('MeshMatrixMass', totalMass=1)

    # Adds a region of interest (ROI) with fixed constraints in the FEM node
    fixedROI = femNode.addChild('FixedROI')
    fixedROI.addObject('BoxROI', template="Vec3", box=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], drawBoxes=False,
                       position="@../mstate.rest_position", name="BoxROI",
                       computeTriangles=False, computeTetrahedra=False, computeEdges=False)
    fixedROI.addObject('FixedConstraint', indices="@BoxROI.indices")

    # Collision setup in FEM node
    collisionNode = femNode.addChild('Collision')
    collisionNode.addObject('TriangleSetTopologyContainer', name="Container")
    collisionNode.addObject('TriangleSetTopologyModifier', name="Modifier")
    collisionNode.addObject('Tetra2TriangleTopologicalMapping', input="@../Container", output="@Container")
    collisionNode.addObject('TriangleCollisionModel', name="collisionModel", proximity=0.001, contactStiffness=20)
    collisionNode.addObject('MechanicalObject', name='dofs', rest_position="@../mstate.rest_position")
    collisionNode.addObject('IdentityMapping', name='visualMapping')

    # Applies a linear solver constraint correction in FEM node
    femNode.addObject('LinearSolverConstraintCorrection', linearSolver="@precond")

    # Adds a node for attaching points to mouse interactor
    attachPointNode = rootNode.addChild('AttachPoint')
    attachPointNode.addObject('PointSetTopologyContainer', name="Container")
    attachPointNode.addObject('PointSetTopologyModifier', name="Modifier")
    attachPointNode.addObject('MechanicalObject', name="mstate", template="Vec3d", drawMode=2, showObjectScale=0.01, showObject=False)
    attachPointNode.addObject('iGTLinkMouseInteractor', name="mouseInteractor", pickingType="constraint", reactionTime=20, destCollisionModel="@../FEM/Collision/collisionModel")

    return rootNode

# Parameter node for soft tissue simulation with mappers to sync between SOFA and MRML scenes
@SofaParameterNodeWrapper
class SoftTissueSimulationParameterNode:
    """
    Parameter class for simulation. Defines nodes to map between SOFA and MRML with recording options.
    """

    # Model node with SOFA mapping and sequence recording
    modelNode: vtkMRMLModelNode = \
        NodeMapper(
            nodeName="FEM",
            sofaMapping=lambda self, sofaNode: self.modelNodetoSofaNode(sofaNode),
            mrmlMapping=lambda self, sofaNode: self.sofaNodeToModelNode(sofaNode),
            recordSequence=True
        )

    # Fiducial node for tracking a moving point, with sequence recording
    movingPointNode: vtkMRMLMarkupsFiducialNode = \
        NodeMapper(
            nodeName="AttachPoint.mouseInteractor",
            sofaMapping=lambda self, sofaNode: self.markupsFiducialNodeToSofaPoint(sofaNode),
            recordSequence=True
        )

    # Boundary ROI node with sequence recording
    boundaryROI: vtkMRMLMarkupsROINode = \
        NodeMapper(
            nodeName="FEM.FixedROI.BoxROI",
            sofaMapping=lambda self, sofaNode: self.markupsROIToSofaROI(sofaNode),
            recordSequence=True
        )

    # Gravity vector node with sequence recording
    gravityVector: vtkMRMLMarkupsLineNode = \
        NodeMapper(
            nodeName="",
            sofaMapping=lambda self, sofaNode: self.markupsLineToGravityVector(sofaNode),
            recordSequence=True
        )

    gravityMagnitude: int = 1  # Additional parameter for gravity strength

    # Mapping ROI bounds from MRML node to SOFA node box
    def markupsROIToSofaROI(self, sofaNode):
        if self.boundaryROI is None:
            return [0.0]*6

        center = [0]*3
        self.boundaryROI.GetCenter(center)
        size = self.boundaryROI.GetSize()

        # Calculate min and max RAS bounds
        R_min = center[0] - size[0] / 2
        R_max = center[0] + size[0] / 2
        A_min = center[1] - size[1] / 2
        A_max = center[1] + size[1] / 2
        S_min = center[2] - size[2] / 2
        S_max = center[2] + size[2] / 2

        # Define SOFA node box with calculated bounds
        sofaNode.box=[[R_min, A_min, S_min, R_max, A_max, S_max]]

    # Mapping a line node as a gravity vector in the SOFA node
    def markupsLineToGravityVector(self, sofaNode):
        if self.gravityVector is None:
            return [0.0]*3

        # Calculate vector direction and normalize
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

# UI widget for soft tissue simulation
class SoftTissueSimulationWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    def __init__(self, parent=None) -> None:
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)  # Needed for parameter node observation
        self.logic = None
        self.parameterNode = None
        self.parameterNodeGuiTag = None
        self.timer = qt.QTimer(parent)
        self.timer.timeout.connect(self.simulationStep)

    def setup(self) -> None:
        """Sets up the user interface, logic, and connections."""
        ScriptedLoadableModuleWidget.setup(self)

        # Load the widget interface from a .ui file
        uiWidget = slicer.util.loadUI(self.resourcePath("UI/SoftTissueSimulation.ui"))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)

        # Initialize logic for simulation computations
        self.logic = SoftTissueSimulationLogic()
        uiWidget.setMRMLScene(slicer.mrmlScene)

        # Setup event connections
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)

        self.ui.startSimulationPushButton.connect("clicked()", self.startSimulation)
        self.ui.stopSimulationPushButton.connect("clicked()", self.stopSimulation)
        self.ui.addBoundaryROIPushButton.connect("clicked()", self.logic.addBoundaryROI)
        self.ui.addGravityVectorPushButton.connect("clicked()", self.logic.addGravityVector)
        self.ui.addMovingPointPushButton.connect("clicked()", self.logic.addMovingPoint)

        # Initialize parameter node and GUI bindings
        self.initializeParameterNode()
        self.logic.getParameterNode().AddObserver(vtk.vtkCommand.ModifiedEvent, self.updateSimulationGUI)

    def cleanup(self) -> None:
        """Cleanup when the module widget is destroyed."""
        self.timer.stop()
        self.logic.stopSimulation()
        self.logic.clean()
        self.removeObservers()

    def enter(self) -> None:
        """Initialize parameter node on module entry."""
        self.initializeParameterNode()

    def exit(self) -> None:
        """Cleanup GUI connections on module exit."""
        if self.parameterNode:
            self.parameterNode.disconnectGui(self.parameterNodeGuiTag)
            self.parameterNodeGuiTag = None

    # Initializes and sets the parameter node in logic
    def initializeParameterNode(self) -> None:
        if self.logic:
            self.setParameterNode(self.logic.getParameterNode())
            self.logic.resetParameterNode()

    # Set the parameter node and GUI bindings
    def setParameterNode(self, inputParameterNode: Optional[SoftTissueSimulationParameterNode]) -> None:
        if self.parameterNode:
            self.parameterNode.disconnectGui(self.parameterNodeGuiTag)
        self.parameterNode = inputParameterNode
        if self.parameterNode:
            self.parameterNodeGuiTag = self.parameterNode.connectGui(self.ui)

    # Update GUI based on simulation state
    def updateSimulationGUI(self, caller, event):
        parameterNode = self.logic.getParameterNode()
        self.ui.startSimulationPushButton.setEnabled(not parameterNode.isSimulationRunning and parameterNode.modelNode is not None)
        self.ui.stopSimulationPushButton.setEnabled(parameterNode.isSimulationRunning)

    # Scene close event handling
    def onSceneStartClose(self, caller, event) -> None:
        self.setParameterNode(None)

    def onSceneEndClose(self, caller, event) -> None:
        if self.parent.isEntered:
            self.initializeParameterNode()

    # Start the simulation and timer
    def startSimulation(self):
        self.logic.startSimulation()
        self.timer.start(0)

    # Stop the simulation and timer
    def stopSimulation(self):
        self.timer.stop()
        self.logic.stopSimulation()

    def simulationStep(self):
        self.logic.simulationStep()

# Logic for the simulation, including scene setup and parameter node management
class SoftTissueSimulationLogic(SlicerSofaLogic):
    def __init__(self) -> None:
        super().__init__()
        self._rootNode = CreateScene()

    # Retrieve or create a wrapped parameter node
    def getParameterNode(self):
        return SoftTissueSimulationParameterNode(super().getParameterNode())

    # Reset simulation parameters in the parameter node
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

    # Update model node from SOFA to MRML when modified
    def onModelNodeModified(self, caller, event) -> None:
        if self.getParameterNode().modelNode.GetUnstructuredGrid() is not None:
            self.getParameterNode().modelNode.GetUnstructuredGrid().SetPoints(caller.GetPolyData().GetPoints())
        elif self.getParameterNode().modelNode.GetPolyData() is not None:
            self.getParameterNode().modelNode.GetPolyData().SetPoints(caller.GetPolyData().GetPoints())

    # Add boundary ROI based on model bounds
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

    # Adds a gravity vector as a line in the scene
    def addGravityVector(self) -> None:
        gravityVector = slicer.vtkMRMLMarkupsLineNode()
        gravityVector.SetName("Gravity")
        mesh = None

        if self.getParameterNode().modelNode is not None:
            if self.getParameterNode().modelNode.GetUnstructuredGrid() is not None:
                mesh = self.getParameterNode().modelNode.GetUnstructuredGrid()
            elif self.getParameterNode().modelNode.GetPolyData() is not None:
                mesh = self.getParameterNode().modelNode.GetPolyData()

        if mesh is not None:
            bounds = mesh.GetBounds()
            center = [(bounds[0] + bounds[1])/2.0, (bounds[2] + bounds[3])/2.0, (bounds[4] + bounds[5])/2.0]
            startPoint = [center[0], bounds[2], center[2]]
            endPoint = [center[0], bounds[3], center[2]]
            gravityVector.AddControlPoint(vtk.vtkVector3d(startPoint))
            gravityVector.AddControlPoint(vtk.vtkVector3d(endPoint))

        gravityVector = slicer.mrmlScene.AddNode(gravityVector)
        if gravityVector is not None:
            gravityVector.CreateDefaultDisplayNodes()

        self.getParameterNode().gravityVector = gravityVector

    # Adds a moving point based on the closest point to the camera
    def addMovingPoint(self) -> None:
        cameraNode = slicer.util.getNode('Camera')
        if None not in [self.getParameterNode().modelNode, cameraNode]:
            fiducialNode = self.addFiducialToClosestPoint(self.getParameterNode().modelNode, cameraNode)
            self.getParameterNode().movingPointNode = fiducialNode

    # Adds a fiducial at the closest point on the model to the camera position
    def addFiducialToClosestPoint(self, modelNode, cameraNode) -> vtkMRMLMarkupsFiducialNode:
        camera = cameraNode.GetCamera()
        camPosition = camera.GetPosition()

        modelData = None
        if self.getParameterNode().modelNode.GetUnstructuredGrid() is not None:
            modelData = self.getParameterNode().modelNode.GetUnstructuredGrid()
        elif self.getParameterNode().modelNode.GetPolyData() is not None:
            modelData = self.getParameterNode().modelNode.GetPolyData()

        pointLocator = vtk.vtkPointLocator()
        pointLocator.SetDataSet(modelData)
        pointLocator.BuildLocator()

        closestPointId = pointLocator.FindClosestPoint(camPosition)
        closestPoint = modelData.GetPoint(closestPointId)

        fiducialNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode")
        fiducialNode.AddControlPointWorld(vtk.vtkVector3d(closestPoint))

        fiducialNode.SetName("Closest Fiducial")
        displayNode = fiducialNode.GetDisplayNode()
        if displayNode:
            displayNode.SetSelectedColor(1, 0, 0)

        return fiducialNode
