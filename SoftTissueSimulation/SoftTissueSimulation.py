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
from slicer import vtkMRMLMarkupsFiducialNode
from slicer import vtkMRMLMarkupsLineNode
from slicer import vtkMRMLMarkupsNode
from slicer import vtkMRMLMarkupsROINode
from slicer import vtkMRMLModelNode

from SofaEnvironment import Sofa
from SlicerSofa import (
    SlicerSofaWidget,
    SlicerSofaLogic,
    SofaParameterNodeWrapper,
)

from SlicerSofaUtils.Mappings import (
    mrmlModelGridToSofaTetrahedronTopologyContainer,
    mrmlMarkupsFiducialToSofaPointer,
    mrmlMarkupsROIToSofaBoxROI,
    sofaMechanicalObjectToMRMLModelGrid,
    sofaVonMisesStressToMRMLModelGrid,
    arrayFromMarkupsROIPoints,
    arrayVectorFromMarkupsLinePoints,
    arrayFromModelGridCells,
    arrayFromMarkupsROIPoints,
)

# -----------------------------------------------------------------------------
# Function: CreateScene
# -----------------------------------------------------------------------------
def CreateScene(parameterNode) -> Sofa.Core.Node:
    """
    Creates the main SOFA scene with required components for simulation.

    Returns:
        Sofa.Core.Node: The root node of the SOFA simulation scene.
    """
    from stlib3.solver import DefaultSolver
    from stlib3.physics.deformable import ElasticMaterialObject
    from stlib3.physics.rigid import Floor
    from splib3.numerics import Vec3

    vonMisesMode = {
        "none": 0,
        "corotational": 1,
        "fullGreen": 2
    }

    # Initialize the root node of the SOFA scene
    rootNode = Sofa.Core.Node("Root")

    # Initialize main scene headers with necessary plugins for SOFA components
    plugins=["Sofa.Component.IO.Mesh",
             "Sofa.Component.LinearSolver.Direct",
             "Sofa.Component.LinearSolver.Iterative",
             "Sofa.Component.Mapping.Linear",
             "Sofa.Component.Mass",
             "Sofa.Component.ODESolver.Backward",
             "Sofa.Component.Setting",
             "Sofa.Component.SolidMechanics.FEM.Elastic",
             "Sofa.Component.StateContainer",
             "Sofa.Component.Topology.Container.Dynamic",
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
             "Sofa.Component.SolidMechanics.Spring",
             "Sofa.Component.Constraint.Lagrangian.Model",
             "Sofa.Component.Mapping.NonLinear",
             "Sofa.Component.Topology.Container.Constant",
             "Sofa.Component.Topology.Mapping",
             "Sofa.Component.Topology.Container.Dynamic",
             "Sofa.Component.Engine.Select",
             "Sofa.Component.Constraint.Projective",
             "MultiThreading",]
             #"Sofa.IGTLink"]

    for plugin_name in plugins:
         rootNode.addObject("RequiredPlugin", name=plugin_name)

    # Set gravity vector for the simulation (no gravity in this case)
    rootNode.gravity = [0, 0, 0]

    # Add animation and constraint solver objects to the root node
    rootNode.addObject('FreeMotionAnimationLoop', parallelODESolving=True, parallelCollisionDetectionAndFreeMotion=True)
    rootNode.addObject('GenericConstraintSolver', maxIterations=10, multithreading=True, tolerance=1.0e-3)

    # Define a deformable Finite Element Method (FEM) object
    femNode = rootNode.addChild('FEM')
    femNode.addObject('EulerImplicitSolver', firstOrder=False, rayleighMass=0.1, rayleighStiffness=0.1)
    femNode.addObject('SparseLDLSolver', name="precond", template="CompressedRowSparseMatrixd", parallelInverseProduct=True)
    femNode.addObject('TetrahedronSetTopologyContainer', name="Container", position=slicer.util.arrayFromModelPoints(parameterNode.modelNode), tetrahedra=arrayFromModelGridCells(parameterNode.modelNode))
    femNode.addObject('TetrahedronSetTopologyModifier', name="Modifier")
    femNode.addObject('MechanicalObject', name="mstate", template="Vec3d")
    femNode.addObject('TetrahedronFEMForceField', name="FEM", youngModulus=1.5, poissonRatio=0.45, method="large", computeVonMisesStress=vonMisesMode['fullGreen'])
    femNode.addObject('MeshMatrixMass', totalMass=1)

    # Add a region of interest (ROI) with fixed constraints in the FEM node
    fixedROI = femNode.addChild('FixedROI')
    fixedROI.addObject('BoxROI', template="Vec3", box=arrayFromMarkupsROIPoints(parameterNode.boundaryROI), drawBoxes=False,
                       position="@../mstate.rest_position", name="BoxROI",
                       computeTriangles=False, computeTetrahedra=False, computeEdges=False)
    fixedROI.addObject('FixedConstraint', indices="@BoxROI.indices")

    # # Set up collision detection within the FEM node
    # collisionNode = femNode.addChild('Collision')
    # collisionNode.addObject('TriangleSetTopologyContainer', name="Container")
    # collisionNode.addObject('TriangleSetTopologyModifier', name="Modifier")
    # collisionNode.addObject('Tetra2TriangleTopologicalMapping', input="@../Container", output="@Container")
    # collisionNode.addObject('TriangleCollisionModel', name="collisionModel", proximity=0.001, contactStiffness=20)
    # collisionNode.addObject('MechanicalObject', name='dofs', rest_position="@../mstate.rest_position")
    # collisionNode.addObject('IdentityMapping', name='visualMapping')

    # Apply a linear solver constraint correction in the FEM node
    femNode.addObject('LinearSolverConstraintCorrection', linearSolver="@precond")

    # # Add a node for attaching points to the mouse interactor
    # attachPointNode = rootNode.addChild('AttachPoint')
    # attachPointNode.addObject('PointSetTopologyContainer', name="Container")
    # attachPointNode.addObject('PointSetTopologyModifier', name="Modifier")
    # attachPointNode.addObject('MechanicalObject', name="mstate", template="Vec3d", drawMode=2, showObjectScale=0.01, showObject=False)
    # attachPointNode.addObject('iGTLinkMouseInteractor', name="mouseInteractor", pickingType="constraint", reactionTime=20, destCollisionModel="@../FEM/Collision/collisionModel")

    return rootNode

# -----------------------------------------------------------------------------
# Class: SoftTissueSimulationParameterNode
# -----------------------------------------------------------------------------
@SofaParameterNodeWrapper
class SoftTissueSimulationParameterNode:
    """
    Parameter class for the soft tissue simulation.
    Defines nodes to map between SOFA and MRML scenes with recording options.
    """
    modelNode: vtkMRMLModelNode                    # Model node with SOFA mapping and sequence recording
    #movingPointNode: vtkMRMLMarkupsFiducialNode    # Fiducial node for tracking a moving point, with sequence recording
    boundaryROI: vtkMRMLMarkupsROINode             # Boundary ROI node with sequence recording
    gravityVector: vtkMRMLMarkupsLineNode          # Gravity vector node with sequence recording
    gravityMagnitude: int = 1                      # Additional parameter for gravity strength
    recordSequence: bool = False                   # Record sequence?

# -----------------------------------------------------------------------------
# Class: SoftTissueSimulation
# -----------------------------------------------------------------------------
class SoftTissueSimulation(ScriptedLoadableModule):
    """
    Main module definition for the Soft Tissue Simulation.
    Sets up UI and metadata.
    """
    def __init__(self, parent):
        """
        Initialize the module with metadata.

        Args:
            parent: The parent object.
        """
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = _("Soft Tissue Simulation")
        self.parent.categories = [translate("qSlicerAbstractCoreModule", "Examples")]
        self.parent.dependencies = []
        self.parent.contributors = [
            "Rafael Palomar (Oslo University Hospital)",
            "Paul Baksic (INRIA)",
            "Steve Pieper (Isomics, Inc.)",
            "Andras Lasso (Queen's University)",
            "Sam Horvath (Kitware, Inc.)",
            "Jean-Christophe Fillion-Robin (Kitware, Inc.)"
        ]
        self.parent.helpText = _("""
        This is a Slicer-SOFA example module. The module uses the SOFA framework to simulate soft tissue.
        """)
        self.parent.acknowledgementText = _("""This project was funded by Oslo University Hospital.""")

        # Connect additional initialization after application startup
        slicer.app.connect("startupCompleted()", self.registerSampleData)

    # -----------------------------------------------------------------------------
    # Function: registerSampleData
    # -----------------------------------------------------------------------------
    def registerSampleData(self):
        """
        Registers sample data for the module, allowing users to load predefined datasets.
        """
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

# -----------------------------------------------------------------------------
# Class: SoftTissueSimulationWidget
# -----------------------------------------------------------------------------
class SoftTissueSimulationWidget(SlicerSofaWidget):
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

        super().setup()

        # Load the widget interface from a .ui file
        uiWidget = slicer.util.loadUI(self.resourcePath("UI/SoftTissueSimulation.ui"))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)

        # Initialize logic for simulation computations
        self.logic = SoftTissueSimulationLogic()
        uiWidget.setMRMLScene(slicer.mrmlScene)

        # Connect UI buttons to their respective methods
        self.ui.startSimulationPushButton.connect("clicked()", self.startSimulation)
        self.ui.stopSimulationPushButton.connect("clicked()", self.stopSimulation)
        self.ui.addBoundaryROIPushButton.connect("clicked()", self.logic.addBoundaryROI)
        self.ui.addGravityVectorPushButton.connect("clicked()", self.logic.addGravityVector)
        #self.ui.addMovingPointPushButton.connect("clicked()", self.logic.addMovingPoint)
        self.ui.resetSimulationPushButton.connect("clicked()", self.logic.resetSimulation)

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

    def startSimulation(self) -> None:
        """
        Starts the simulation and begins the timer for simulation steps.
        """
        self.logic.startSimulation()
        self.timer.start(0)

    def stopSimulation(self) -> None:
        """
        Stops the simulation and the timer.
        """
        self.timer.stop()
        self.logic.stopSimulation()

    def simulationStep(self) -> None:
        """
        Executes a single simulation step.
        """
        self.logic.simulationStep()

# -----------------------------------------------------------------------------
# Class: SoftTissueSimulationLogic
# -----------------------------------------------------------------------------
class SoftTissueSimulationLogic(SlicerSofaLogic):
    """
    Logic class for the Soft Tissue Simulation.
    Handles scene setup, parameter node management, and simulation steps.
    """
    def __init__(self) -> None:
        """
        Initialize the logic with the SOFA scene.
        """
        super().__init__()
        self._parameterNode = None

    def createScene(self, parameterNode) -> Sofa.Core.Node:
        return CreateScene(parameterNode)

    def getParameterNode(self):
        """
        Retrieves or creates a wrapped parameter node.

        Returns:
            SoftTissueSimulationParameterNode: The parameter node for the simulation.
        """
        if self._parameterNode is None:
            self._parameterNode = SoftTissueSimulationParameterNode(super().getParameterNode())
        return self._parameterNode

    def resetParameterNode(self):
        """
        Resets simulation parameters in the parameter node to default values.
        """
        if self.getParameterNode() is not None:
            self.getParameterNode().modelNode = None
            self.getParameterNode().boundaryROI = None
            self.getParameterNode().gravityVector = None
            #self.getParameterNode().movingPointNode = None
            self.getParameterNode().dt = 0.01
            self.getParameterNode().currentStep = 0
            self.getParameterNode().totalSteps = -1

    def startSimulation(self) -> None:
        """
        Sets up the scene and starts the simulation.
        """
        # TODO: The order here is important. Maybe move part to SlicerSOFA to enforce correct order
        self.setupMappings()
        self.setupScene(self.getParameterNode())
        super().startSimulation()
        self._simulationRunning = True
        self.getParameterNode().Modified()

    def stopSimulation(self) -> None:
        """
        Stops the simulation.
        """
        super().stopSimulation()
        self._simulationRunning = False
        self.getParameterNode().Modified()

    def setupMappings(self):
        """
        Registers mappings between MRML and SOFA nodes.
        """
        pn = self.getParameterNode()

        if pn is not None:
            # Register MRML-to-SOFA mappings
            self.registerMRMLToSOFAMapping('modelNode', 'FEM.Container', mrmlModelGridToSofaTetrahedronTopologyContainer, runOnce=True)
            #self.registerMRMLToSOFAMapping('movingPointNode', 'AttachPoint.mouseInteractor', mrmlMarkupsFiducialToSofaPointer)
            self.registerMRMLToSOFAMapping('boundaryROI', 'FEM.FixedROI.BoxROI', mrmlMarkupsROIToSofaBoxROI)
            self.registerMRMLToSOFAMapping('gravityVector', '', self.mrmlMarkupsLineToGravityVector)

            # Register SOFA-to-MRML mappings
            self.registerSOFAToMRMLMapping('modelNode', 'FEM.Collision.dofs', sofaMechanicalObjectToMRMLModelGrid)
            self.registerSOFAToMRMLMapping('modelNode', 'FEM.FEM', sofaVonMisesStressToMRMLModelGrid)

            # Set sequence recording flags
            self.setRecordSequenceFlag('modelNode', lambda: pn.recordSequence)
            #self.setRecordSequenceFlag('movingPointNode', lambda: pn.recordSequence)
            self.setRecordSequenceFlag('boundaryROI', lambda: pn.recordSequence)
            self.setRecordSequenceFlag('gravityVector', lambda: pn.recordSequence)

    def mrmlMarkupsLineToGravityVector(self, gravityVectorNode, sofaRootNode):
        """
        Maps the gravity vector from MRML to the SOFA root node.
        """
        if gravityVectorNode is None:
            return
        gravityVector = arrayVectorFromMarkupsLinePoints(gravityVectorNode)
        magnitude = np.linalg.norm(np.array(gravityVector))
        normalizedGravityVector = gravityVector / magnitude if magnitude != 0 else gravityVector
        sofaRootNode.gravity = normalizedGravityVector * self.getParameterNode().gravityMagnitude



    def _saveState(self) -> None:
        self._originalModelGrid = vtk.vtkUnstructuredGrid()
        self._originalModelGrid.DeepCopy(self._parameterNode.modelNode.GetUnstructuredGrid())

    def _restoreState(self) -> None:
        self._parameterNode.modelNode.SetAndObserveMesh(self._originalModelGrid)

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

    # def addMovingPoint(self) -> None:
    #     """
    #     Adds a moving point based on the closest point to the camera.
    #     """
    #     cameraNode = slicer.util.getNode('Camera')
    #     if None not in [self.getParameterNode().modelNode, cameraNode]:
    #         fiducialNode = self.addFiducialToClosestPoint(self.getParameterNode().modelNode, cameraNode)
    #         self.getParameterNode().movingPointNode = fiducialNode

    def addFiducialToClosestPoint(self, modelNode, cameraNode) -> vtkMRMLMarkupsFiducialNode:
        """
        Adds a fiducial at the closest point on the model to the camera position.

        Args:
            modelNode: The model node in the scene.
            cameraNode: The camera node in the scene.

        Returns:
            vtkMRMLMarkupsFiducialNode: The created fiducial node.
        """
        camera = cameraNode.GetCamera()
        camPosition = camera.GetPosition()

        # Determine the mesh type and retrieve data
        modelData = None
        if self.getParameterNode().modelNode.GetUnstructuredGrid() is not None:
            modelData = self.getParameterNode().modelNode.GetUnstructuredGrid()
        elif self.getParameterNode().modelNode.GetPolyData() is not None:
            modelData = self.getParameterNode().modelNode.GetPolyData()

        # Use a point locator to find the closest point to the camera
        pointLocator = vtk.vtkPointLocator()
        pointLocator.SetDataSet(modelData)
        pointLocator.BuildLocator()

        closestPointId = pointLocator.FindClosestPoint(camPosition)
        closestPoint = modelData.GetPoint(closestPointId)

        # Create and configure the fiducial node
        fiducialNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode")
        fiducialNode.AddControlPointWorld(vtk.vtkVector3d(closestPoint))
        fiducialNode.SetName("Closest Fiducial")

        # Set display properties for the fiducial node
        displayNode = fiducialNode.GetDisplayNode()
        if displayNode:
            displayNode.SetSelectedColor(1, 0, 0)  # Red color for visibility

        return fiducialNode

# -----------------------------------------------------------------------------
# Class: SoftTissueSimulationTest
# -----------------------------------------------------------------------------
class SoftTissueSimulationTest(ScriptedLoadableModuleTest):
    """
    Test case for the SoftTissueSimulation module.
    Verifies the functionality of gravity and moving point simulations.
    """
    def setUp(self):
        """
        Reset the state by clearing the MRML scene.
        """
        slicer.mrmlScene.Clear()

    def runTest(self):
        """
        Run the tests for the SoftTissueSimulation module.
        """
        self.delayDisplay("Starting SoftTissueSimulation test")
        self.testGravitySimulation()
        #self.testMovingPointSimulation()
        self.delayDisplay("SoftTissueSimulation tests passed")

    def testGravitySimulation(self):
        """
        Test the soft tissue simulation with gravity only.
        """
        import SampleData

        self.setUp()
        logic = SoftTissueSimulationLogic()

        self.delayDisplay("Loading registered sample data")
        sampleDataLogic = SampleData.SampleDataLogic()

        # Define the data source for the deformed lung model
        deformedModelDataSource = SampleData.SampleDataSource(
            sampleName='RightLungLowTetra_deformed',
            uris='https://github.com/rafaelpalomar/SlicerSofaTestingData/releases/download/SHA256/a35ce6ca2ae565fe039010eca3bb23f5ef5f5de518b1c10257f12cb7ead05c5d',
            fileNames='RightLungLowTetra_deformed.vtk',
            checksums='SHA256:a35ce6ca2ae565fe039010eca3bb23f5ef5f5de518b1c10257f12cb7ead05c5d',
            nodeNames='RightLungLowTetra_deformed',
            loadFileType='ModelFile'
        )

        # Download and load the deformed model
        simulationModelNode = sampleDataLogic.downloadFromSource(deformedModelDataSource)[0]


        # Set the layout to 3D view for visualization
        layoutManager = slicer.app.layoutManager()
        layoutManager.setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutOneUp3DView)

        self.delayDisplay("Creating ROI box for lower third of the model")
        modelBounds = [0.0] * 6
        simulationModelNode.GetBounds(modelBounds)

        # Define the size and center of the ROI (lower third of the model)
        lowerThirdSize = [
            (modelBounds[1] - modelBounds[0]) * 0.5,
            (modelBounds[3] - modelBounds[2]) * 0.5,
            (modelBounds[5] - modelBounds[4]) / 3
        ]
        lowerThirdCenter = [
            (modelBounds[1] + modelBounds[0]) / 2,
            (modelBounds[3] + modelBounds[2]) / 2,
            modelBounds[4] + lowerThirdSize[2] / 2
        ]

        # Create and configure the ROI node
        fixedROINode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsROINode", "FixedROI")
        fixedROINode.SetXYZ(lowerThirdCenter)
        fixedROINode.SetRadiusXYZ(*lowerThirdSize)

        self.delayDisplay("Creating gravity vector")
        # Create and configure the gravity vector node
        gravityVectorNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsLineNode", "Gravity")
        gravityVectorNode.AddControlPoint([0, modelBounds[2], 0])
        gravityVectorNode.AddControlPoint([0, modelBounds[3], 0])

        self.delayDisplay("Setting up simulation parameters")
        # Assign simulation parameters to the parameter node
        parameterNode = logic.getParameterNode()
        parameterNode.modelNode = simulationModelNode
        parameterNode.boundaryROI = fixedROINode
        parameterNode.gravityVector = gravityVectorNode
        parameterNode.gravityMagnitude = 10000
        parameterNode.dt = 0.01
        parameterNode.currentStep = 0
        parameterNode.totalSteps = 100

        self.delayDisplay("Starting gravity-only simulation")
        # Start the simulation and render view
        logic.startSimulation()
        view = slicer.app.layoutManager().threeDWidget(0).threeDView()

        # Run simulation steps
        for _ in range(parameterNode.totalSteps):
            logic.simulationStep()
            view.forceRender()

        # Stop the simulation and clean up
        logic.stopSimulation()
        logic.clean()

    # def testMovingPointSimulation(self):
    #     """
    #     Test the soft tissue simulation with a moving point and no gravity.
    #     """
    #     import SampleData

    #     self.setUp()
    #     logic = SoftTissueSimulationLogic()

    #     self.delayDisplay("Loading registered sample data")
    #     sampleDataLogic = SampleData.SampleDataLogic()

    #     # Define the data source for the deformed lung model
    #     deformedModelDataSource = SampleData.SampleDataSource(
    #         sampleName='RightLungLowTetra_deformed',
    #         uris='https://github.com/rafaelpalomar/SlicerSofaTestingData/releases/download/SHA256/a35ce6ca2ae565fe039010eca3bb23f5ef5f5de518b1c10257f12cb7ead05c5d',
    #         fileNames='RightLungLowTetra_deformed.vtk',
    #         checksums='SHA256:a35ce6ca2ae565fe039010eca3bb23f5ef5f5de518b1c10257f12cb7ead05c5d',
    #         nodeNames='RightLungLowTetra_deformed',
    #         loadFileType='ModelFile'
    #     )

    #     # Download and load the deformed model
    #     simulationModelNode = sampleDataLogic.downloadFromSource(deformedModelDataSource)[0]

    #     # Set the layout to 3D view for visualization
    #     layoutManager = slicer.app.layoutManager()
    #     layoutManager.setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutOneUp3DView)

    #     self.delayDisplay("Creating ROI box for lower tenth of the model")
    #     modelBounds = [0.0] * 6
    #     simulationModelNode.GetBounds(modelBounds)

    #     # Define the size and center of the ROI (lower tenth of the model)
    #     lowerTenthSize = [
    #         (modelBounds[1] - modelBounds[0]) / 2,
    #         (modelBounds[3] - modelBounds[2]) / 2,
    #         (modelBounds[5] - modelBounds[4]) / 10
    #     ]
    #     lowerTenthCenter = [
    #         (modelBounds[1] + modelBounds[0]) / 2,
    #         (modelBounds[3] + modelBounds[2]) / 2,
    #         modelBounds[4] + lowerTenthSize[2] / 2
    #     ]

    #     # Create and configure the ROI node
    #     fixedROINode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsROINode", "FixedROI")
    #     fixedROINode.SetXYZ(lowerTenthCenter)
    #     fixedROINode.SetRadiusXYZ(*lowerTenthSize)

    #     self.delayDisplay("Creating initial moving point")
    #     # Set initial position of the moving point using logic
    #     parameterNode = logic.getParameterNode()
    #     parameterNode.modelNode = simulationModelNode
    #     logic.addMovingPoint()
    #     movingPointNode = logic.getParameterNode().movingPointNode

    #     # Define start and end positions for the moving point
    #     startPosition = list(movingPointNode.GetNthControlPointPosition(0))
    #     endPosition = [
    #         lowerTenthCenter[0] - lowerTenthSize[0],  # X-axis boundary
    #         lowerTenthCenter[1] + lowerTenthSize[1] / 2,
    #         lowerTenthCenter[2] + lowerTenthSize[2] * 2
    #     ]

    #     # Calculate step size for linear interpolation
    #     totalSteps = 100
    #     interpolationStep = [(end - start) / totalSteps for start, end in zip(startPosition, endPosition)]

    #     self.delayDisplay("Setting up simulation parameters")
    #     # Assign simulation parameters to the parameter node
    #     parameterNode.boundaryROI = fixedROINode
    #     parameterNode.movingPointNode = movingPointNode
    #     parameterNode.gravityMagnitude = 0  # Disabling gravity
    #     parameterNode.dt = 0.01
    #     parameterNode.currentStep = 0
    #     parameterNode.totalSteps = totalSteps

    #     self.delayDisplay("Starting moving point-only simulation")
    #     # Start the simulation and render view
    #     logic.startSimulation()
    #     view = slicer.app.layoutManager().threeDWidget(0).threeDView()

    #     # Run simulation steps with gradual movement of the point
    #     for step in range(parameterNode.totalSteps):
    #         # Update the point's position by adding the interpolation step to the current position
    #         new_position = [
    #             startPosition[0] + step * interpolationStep[0],
    #             startPosition[1] + step * interpolationStep[1],
    #             startPosition[2] + step * interpolationStep[2]
    #         ]
    #         movingPointNode.SetNthControlPointPosition(0, *new_position)

    #         # Advance simulation and render
    #         logic.simulationStep()
    #         view.forceRender()

    #     # Stop the simulation and clean up
    #     logic.stopSimulation()
    #     logic.clean()
