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

# import Simulations.SOFASimulationMulti as multi

import slicer
from slicer.i18n import tr as _
from slicer.i18n import translate
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin
from slicer.parameterNodeWrapper import (
    parameterNodeWrapper,
    WithinRange,
)

from slicer import vtkMRMLMarkupsLineNode
from slicer import vtkMRMLMarkupsNode
from slicer import vtkMRMLMarkupsROINode
from slicer import vtkMRMLModelNode
from slicer import vtkMRMLSequenceBrowserNode
from slicer import vtkMRMLSequenceNode
from slicer import vtkMRMLTransformNode

from SofaEnvironment import Sofa
from SlicerSofa import SlicerSofaLogic

#
# MultiMaterialSimulation
#

sofaDataURL= 'https://github.com/rafaelpalomar/SlicerSofaTestingData/releases/download/'

class MultiMaterialSimulation(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = _("Multi-Material Simulation")
        self.parent.categories = [translate("qSlicerAbstractCoreModule", "Examples")]
        self.parent.dependencies = []
        self.parent.contributors = ["Rafael Palomar (Oslo University Hospital/NTNU, Norway), Nazim Haouchine (Harvard/BWH, USA), Paul Baksic (INRIA, France), Steve Pieper (Isomics, Inc., USA), Andras Lasso (Queen's University, Canada)"]
        self.parent.helpText = _("""This is an example module to use the SOFA framework to do simple multi-material simulation based on surface meshes, ROI selections and a single force vector""")
        self.parent.acknowledgementText = _("""""")

        # Additional initialization step after application startup is complete
        slicer.app.connect("startupCompleted()", registerSampleData)

#
# Register sample data sets in Sample Data module
#


def registerSampleData():
    """Add data sets to Sample Data module."""

    import SampleData

    iconsPath = os.path.join(os.path.dirname(__file__), "Resources/Icons")

    # To ensure that the source code repository remains small (can be downloaded and installed quickly)
    # it is recommended to store data sets that are larger than a few MB in a Github release.

    # Right lung low poly tetrahedral mesh dataset
    SampleData.SampleDataLogic.registerCustomSampleDataSource( category='SOFA',
                                                               sampleName='HeartDeviceJoint',
                                                               thumbnailFileName=os.path.join(iconsPath, 'HeartDeviceJoint.png'),
                                                               uris=sofaDataURL+ 'SHA256/6d0cecb9f1e8dd48c6bd458ae2ca29aa592c219ff9190220dc80adb507e33676',
                                                               fileNames='HeartDeviceJoint.vtk',
                                                               checksums='SHA256:6d0cecb9f1e8dd48c6bd458ae2ca29aa592c219ff9190220dc80adb507e33676',
                                                               nodeNames='HeartDeviceJoint',
                                                               loadFileType='ModelFile'
                                                              )

#
# MultiMaterialSimulationParameterNode
#

@parameterNodeWrapper
class MultiMaterialSimulationParameterNode:
    """
    The parameters needed by module.
    """
    #Simulation data
    simulationModelNode: vtkMRMLModelNode
    fixedROI: vtkMRMLMarkupsROINode
    movingROI: vtkMRMLMarkupsROINode
    sequenceNode: vtkMRMLSequenceNode
    sequenceBrowserNode: vtkMRMLSequenceBrowserNode
    forceVector: vtkMRMLMarkupsLineNode
    forceMagnitude: float
    #Simulation control
    dt: float
    totalSteps: int
    currentStep: int
    simulationRunning: bool

    def getROI(self, ROIType):

        roi = None
        if ROIType == 'Fixed':
            roi = self.fixedROI
        elif ROIType == 'Moving':
            roi = self.movingROI
        else:
            raise ValueError('ROIType must be either \'Fixed\' or \'Moving\'')

        if roi is None:
            return [0.0]*6

        center = [0]*3
        roi.GetCenter(center)
        size = roi.GetSize()

        # Calculate min and max RAS bounds from center and size
        R_min = center[0] - size[0] / 2
        R_max = center[0] + size[0] / 2
        A_min = center[1] - size[1] / 2
        A_max = center[1] + size[1] / 2
        S_min = center[2] - size[2] / 2
        S_max = center[2] + size[2] / 2

        # Return the two opposing bounds corners
        # First corner: (minL, minP, minS), Second corner: (maxL, maxP, maxS)
        return np.array([R_min, A_min, S_min, R_max, A_max, S_max])*0.001

    def getForceVector(self):

        if self.forceVector is None:
            return [0.0]*3

        p1 = self.forceVector.GetNthControlPointPosition(0)
        p2 = self.forceVector.GetNthControlPointPosition(1)
        force_vector = np.array([p2[0]-p1[0], p2[1]-p1[1], p2[2]-p1[2]])
        magnitude = np.linalg.norm(force_vector)
        normalized_force_vector = force_vector / magnitude if magnitude != 0 else force_vector

        return normalized_force_vector*self.forceMagnitude

#
# MultiMaterialSimulationWidget
#

class MultiMaterialSimulationWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
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
        uiWidget = slicer.util.loadUI(self.resourcePath("UI/MultiMaterialSimulation.ui"))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)

        # Create logic class. Logic implements all computations that should be possible to run
        # in batch mode, without a graphical user interface.
        self.logic = MultiMaterialSimulationLogic()

        # Set scene in MRML widgets. Make sure that in Qt designer the top-level qMRMLWidget's
        # "mrmlSceneChanged(vtkMRMLScene*)" signal in is connected to each MRML widget's.
        # "setMRMLScene(vtkMRMLScene*)" slot.
        uiWidget.setMRMLScene(slicer.mrmlScene)

        # Connections

        # These connections ensure that we update parameter node when scene is closed
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)

        self.ui.startSimulationPushButton.connect("clicked()", self.startSimulation)
        self.ui.stopSimulationPushButton.connect("clicked()", self.stopSimulation)
        self.ui.addFixedROIPushButton.connect("clicked()", self.logic.addFixedROI)
        self.ui.addMovingROIPushButton.connect("clicked()", self.logic.addMovingROI)
        self.ui.addForceVectorPushButton.connect("clicked()", self.logic.addForceVector)
        self.ui.addRecordingSequencePushButton.connect("clicked()", self.logic.addRecordingSequence)

        self.initializeParameterNode()
        self.logic.getParameterNode().AddObserver(vtk.vtkCommand.ModifiedEvent, self.updateSimulationGUI)

    def cleanup(self) -> None:
        """Called when the application closes and the module widget is destroyed."""
        self.timer.stop()
        self.logic.stopSimulation()
        self.logic.clean()
        self.removeObservers()

    def enter(self) -> None:
        # """Called each time the user opens this module."""
        # # Make sure parameter node exists and observed
        self.initializeParameterNode()

    def exit(self) -> None:
        """Called each time the user opens a different module."""
        # Do not react to parameter node changes (GUI will be updated when the user enters into the module)
        if self.parameterNode:
            self.parameterNode.disconnectGui(self.parameterNodeGuiTag)
            self.parameterNodeGuiTag = None

    def onSceneStartClose(self, caller, event) -> None:
        """Called just before the scene is closed."""
        # Parameter node will be reset, do not use it anymore
        self.setParameterNode(None)

    def onSceneEndClose(self, caller, event) -> None:
        """Called just after the scene is closed."""
        # If this module is shown while the scene is closed then recreate a new parameter node immediately
        if self.parent.isEntered:
            self.initializeParameterNode()

    def initializeParameterNode(self) -> None:
        # Parameter node stores all user choices in parameter values, node selections, etc.
        # so that when the scene is saved and reloaded, these settings are restored.

        if self.logic:
            self.setParameterNode(self.logic.getParameterNode())
            self.logic.resetParameterNode()

    def setParameterNode(self, inputParameterNode: Optional[MultiMaterialSimulationParameterNode]) -> None:
        """
        Set and observe parameter node.
        Observation is needed because when the parameter node is changed then the GUI must be updated immediately.
        """
        if self.parameterNode:
            self.parameterNode.disconnectGui(self.parameterNodeGuiTag)
        self.parameterNode = inputParameterNode
        if self.parameterNode:
            # Note: in the .ui file, a Qt dynamic property called "SlicerParameterName" is set on each
            # ui element that needs connection.
            self.parameterNodeGuiTag = self.parameterNode.connectGui(self.ui)

    def updateSimulationGUI(self, caller, event):
        """This enables/disables the simulation buttons according to the state of the parameter node"""
        self.ui.startSimulationPushButton.setEnabled(not self.logic.isSimulationRunning and
                                                     self.logic.getParameterNode().simulationModelNode is not None)
        self.ui.stopSimulationPushButton.setEnabled(self.logic.isSimulationRunning)

        self.ui.addFixedROIPushButton.setEnabled(self.parameterNode is not None)
        self.ui.addMovingROIPushButton.setEnabled(self.parameterNode is not None)
        self.ui.addForceVectorPushButton.setEnabled(self.parameterNode is not None)

    def startSimulation(self):
        self.logic.dt = self.ui.dtSpinBox.value
        self.logic.totalSteps = self.ui.totalStepsSpinBox.value
        self.logic.currentStep = self.ui.currentStepSpinBox.value
        self.logic.startSimulation()
        self.timer.start(0) #This timer drives the simulation updates

    def stopSimulation(self):
        self.timer.stop()
        self.logic.stopSimulation()

    def simulationStep(self):
       self.logic.simulationStep(self.parameterNode)

#
# MultiMaterialSimulationLogic
#

class MultiMaterialSimulationLogic(SlicerSofaLogic):
    """This class should implement all the actual
    computation done by your module.  The interface
    should be such that other python code can import
    this class and make use of the functionality without
    requiring an instance of the Widget.
    Uses ScriptedLoadableModuleLogic base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self) -> None:
        """Called when the logic class is instantiated. Can be used for initializing member variables."""
        super().__init__()
        self.connectionStatus = 0
        self.fixedBoxROI = None
        self.movingBoxROI = None
        self.mouseInteractor = None

    def updateMRML(self, parameterNode) -> None:
        meshPointsArray = self.mechanicalObject.position.array()*1000
        modelPointsArray = slicer.util.arrayFromModelPoints(parameterNode.simulationModelNode)
        modelPointsArrayNew = meshPointsArray
        modelPointsArray[:] = modelPointsArrayNew
        slicer.util.arrayFromModelPointsModified(parameterNode.simulationModelNode)

    def getParameterNode(self):
        return MultiMaterialSimulationParameterNode(super().getParameterNode())

    def resetParameterNode(self):
        if self.getParameterNode():
            self.getParameterNode().simulationModelNode = None
            self.getParameterNode().boundaryROI = None
            self.getParameterNode().sequenceNode = None
            self.getParameterNode().sequenceBrowserNode = None
            self.getParameterNode().dt = 0.001
            self.getParameterNode().currentStep = 0
            self.getParameterNode().totalSteps = -1

    def startSimulation(self) -> None:
        sequenceNode = self.getParameterNode().sequenceNode
        browserNode = self.getParameterNode().sequenceBrowserNode
        simulationModelNode = self.getParameterNode().simulationModelNode

        # Synchronize and set up the sequence browser node
        if None not in [sequenceNode, browserNode, simulationModelNode]:
            browserNode.AddSynchronizedSequenceNodeID(sequenceNode.GetID())
            browserNode.AddProxyNode(simulationModelNode, sequenceNode, False)
            browserNode.SetRecording(sequenceNode, True)
            browserNode.SetRecordingActive(True)

        super().startSimulation(self.getParameterNode())
        self._simulationRunning = True
        self.getParameterNode().Modified()

    def stopSimulation(self) -> None:
        super().stopSimulation()
        self._simulationRunning = False
        browserNode = self.getParameterNode().sequenceBrowserNode
        if browserNode is not None:
            browserNode.SetRecordingActive(False)
        self.getParameterNode().Modified()

    def onSimulationModelNodeModified(self, caller, event) -> None:
        if self.getParameterNode().simulationModelNode.GetUnstructuredGrid() is not None:
            self.getParameterNode().simulationModelNode.GetUnstructuredGrid().SetPoints(caller.GetPolyData().GetPoints())
        elif self.getParameterNode().simulationModelNode.GetPolyData() is not None:
            self.getParameterNode().simulationModelNode.GetPolyData().SetPoints(caller.GetPolyData().GetPoints())

    def addFixedROI(self) -> None:
        roiNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsROINode")
        mesh = None
        bounds = None

        if self.getParameterNode().simulationModelNode is not None:
            if self.getParameterNode().simulationModelNode.GetUnstructuredGrid() is not None:
                mesh = self.getParameterNode().simulationModelNode.GetUnstructuredGrid()
            elif self.getParameterNode().simulationModelNode.GetPolyData() is not None:
                mesh = self.getParameterNode().simulationModelNode.GetPolyData()

        if mesh is not None:
            bounds = mesh.GetBounds()
            center = [(bounds[0] + bounds[1])/2.0, (bounds[2] + bounds[3])/2.0, (bounds[4] + bounds[5])/2.0]
            size = [abs(bounds[1] - bounds[0])/2.0, abs(bounds[3] - bounds[2])/2.0, abs(bounds[5] - bounds[4])/2.0]
            roiNode.SetXYZ(center)
            roiNode.SetRadiusXYZ(size[0], size[1], size[2])

        self.getParameterNode().fixedROI= roiNode


    def addMovingROI(self) -> None:
        roiNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsROINode")
        mesh = None
        bounds = None

        if self.getParameterNode().simulationModelNode is not None:
            if self.getParameterNode().simulationModelNode.GetUnstructuredGrid() is not None:
                mesh = self.getParameterNode().simulationModelNode.GetUnstructuredGrid()
            elif self.getParameterNode().simulationModelNode.GetPolyData() is not None:
                mesh = self.getParameterNode().simulationModelNode.GetPolyData()

        if mesh is not None:
            bounds = mesh.GetBounds()
            center = [(bounds[0] + bounds[1])/2.0, (bounds[2] + bounds[3])/2.0, (bounds[4] + bounds[5])/2.0]
            size = [abs(bounds[1] - bounds[0])/2.0, abs(bounds[3] - bounds[2])/2.0, abs(bounds[5] - bounds[4])/2.0]
            roiNode.SetXYZ(center)
            roiNode.SetRadiusXYZ(size[0], size[1], size[2])

        self.getParameterNode().movingROI= roiNode

    def addForceVector(self) -> None:
        # Create a new line node for the force vector
        #forceVector = slicer.vtkMRMLMarkupsLineNode()
        forceVector = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLMarkupsLineNode', 'Force')
        if forceVector is not None:
            forceVector.CreateDefaultDisplayNodes()
        measurement = forceVector.GetMeasurement('length')
        measurement.EnabledOff()
        mesh = None

        # Check if there is a model node set in the parameter node and get its mesh
        if self.getParameterNode().simulationModelNode is not None:
            if self.getParameterNode().simulationModelNode.GetUnstructuredGrid() is not None:
                mesh = self.getParameterNode().simulationModelNode.GetUnstructuredGrid()
            elif self.getParameterNode().simulationModelNode.GetPolyData() is not None:
                mesh = self.getParameterNode().simulationModelNode.GetPolyData()

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
            forceVector.AddControlPoint(vtk.vtkVector3d(startPoint))
            forceVector.AddControlPoint(vtk.vtkVector3d(endPoint))

        self.getParameterNode().forceVector = forceVector


    def addDeviceTransform(self) -> None:
        transformNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLTransformNode", "Device Transform")
        transformNode.CreateDefaultDisplayNodes()
        self.getParameterNode().DeviceTransformNode = transformNode

    def addRecordingSequence(self) -> None:

        browserNode = self.getParameterNode().sequenceBrowserNode
        modelNode = self.getParameterNode().simulationModelNode

        # Ensure there is a sequence browser node; create if not present
        if browserNode is None:
            browserNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLSequenceBrowserNode', "SOFA Simulation")
            browserNode.SetPlaybackActive(False)
            browserNode.SetRecordingActive(False)
            self.getParameterNode().sequenceBrowserNode = browserNode  # Update the parameter node reference

        sequenceNode = slicer.vtkMRMLSequenceNode()

        # Configure the sequence node based on the proxy model node
        if modelNode is not None:
            sequenceNodeName = modelNode.GetName() + "-Sequence"
            sequenceNode.SetName(sequenceNodeName)

        # Now add the configured sequence node to the scene
        slicer.mrmlScene.AddNode(sequenceNode)

        self.getParameterNode().sequenceNode = sequenceNode  # Update the parameter node reference

        # Configure index name and unit based on the master sequence node, if present
        masterSequenceNode = browserNode.GetMasterSequenceNode()
        if masterSequenceNode:
            sequenceNode.SetIndexName(masterSequenceNode.GetIndexName())
            sequenceNode.SetIndexUnit(masterSequenceNode.GetIndexUnit())

    def createScene(self, parameterNode) -> Sofa.Core.Node:
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
            "Sofa.Component.Topology.Container.Grid"
        ])

        rootNode.dt = parameterNode.dt
        rootNode.gravity = [0, 0, 0]

        rootNode.addObject('DefaultAnimationLoop', parallelODESolving=True)
        rootNode.addObject('VisualStyle', displayFlags="showBehaviorModels showForceFields")
        rootNode.addObject('DefaultPipeline', depth=6, verbose=0, draw=0)
        rootNode.addObject('ParallelBruteForceBroadPhase')
        rootNode.addObject('BVHNarrowPhase')
        rootNode.addObject('ParallelBVHNarrowPhase')
        rootNode.addObject('MinProximityIntersection', name="Proximity", alarmDistance=0.005, contactDistance=0.003)
        rootNode.addObject('DefaultContactManager', name="Response", response="PenalityContactForceField")

        inputNode = rootNode.addChild('InputSurfaceNode')
        inputNode.addObject('TriangleSetTopologyContainer', name='Container')

        fem = rootNode.addChild('FEM')
        fem.addObject('SparseGridTopology', n=[10, 10, 10], position="@../InputSurfaceNode/Container.position")
        fem.addObject('EulerImplicitSolver', rayleighStiffness=0.1, rayleighMass=0.1)
        fem.addObject('CGLinearSolver', iterations=100, tolerance=1e-5, threshold=1e-5)
        fem.addObject('MechanicalObject', name='MO')
        fem.addObject('UniformMass', totalMass=0.5)
        fem.addObject('ParallelHexahedronFEMForceField', name="FEMForce", youngModulus=50000000, poissonRatio=0.40, method="large")

        surf = fem.addChild('Surf')
        surf.addObject('MeshTopology', position="@../../InputSurfaceNode/Container.position")
        self.mechanicalObject = surf.addObject('MechanicalObject', position="@../../InputSurfaceNode/Container.position")
        surf.addObject('TriangleCollisionModel', selfCollision=True)
        surf.addObject('LineCollisionModel')
        surf.addObject('PointCollisionModel')
        surf.addObject('BarycentricMapping')

        self.fixedBoxROI = fem.addObject('BoxROI', name="FixedROI",
                                    template="Vec3", box=[parameterNode.getROI('Fixed')], drawBoxes=False,
                                    position="@../MO.rest_position",
                                    computeTriangles=False, computeTetrahedra=False, computeEdges=False)
        fem.addObject('FixedConstraint', indices="@FixedROI.indices")

        self.movingBoxROI = fem.addObject('BoxROI', name="boxForce", box=[parameterNode.getROI('Moving')], drawBoxes=True)
        self.forceVector = fem.addObject('AffineMovementConstraint', name="bilinearConstraint", template="Vec3d", indices="@boxForce.indices", meshIndices="@boxForce.indices",
                     translation=[0.05, 0, 0], rotation=[[1, 0, 0], [0, 1, 0], [0, 0, 1]], drawConstrainedPoints=1, beginConstraintTime=0, endConstraintTime=1)
        self.forceVector.translation = parameterNode.getForceVector()


        return rootNode

#
# MultiMaterialSimulationTest
#


class MultiMaterialSimulationTest(ScriptedLoadableModuleTest):
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

        self.delayDisplay("Starting test_multi_material_simulation")
        self.test_multi_material_simulation()
        self.delayDisplay('Test test_multi_material_simulation passed')

    def compareModels(self, referenceModelNode, testModelNode):

        # Compute distance map
        distance_filter = vtk.vtkDistancePolyDataFilter()
        distance_filter.SetInputData(0, referenceModelNode.GetPolyData())
        distance_filter.SetInputData(1, testModelNode.GetPolyData())
        distance_filter.Update()

        # Calculate summary statistics
        distance_poly_data = distance_filter.GetOutput()
        distance_array = vtk.util.numpy_support.vtk_to_numpy(distance_poly_data.GetPointData().GetScalars('Distance'))
        mean_distance = distance_array.mean()
        max_distance = distance_array.max()
        std_distance = distance_array.std()

        # Define thresholds
        mean_threshold_pass = 0.5
        mean_threshold_fail = 1.0
        max_threshold_pass = 2.0
        max_threshold_fail = 3.0
        std_threshold_pass = 0.2
        std_threshold_fail = 0.5

        # Evaluate pass/fail
        mean_status = "Pass" if mean_distance < mean_threshold_pass else "Warning" if mean_distance < mean_threshold_fail else "Fail"
        max_status = "Pass" if max_distance < max_threshold_pass else "Warning" if max_distance < max_threshold_fail else "Fail"
        std_status = "Pass" if std_distance < std_threshold_pass else "Warning" if std_distance < std_threshold_fail else "Fail"

        # Overall status
        overall_status = "Pass" if mean_status == "Pass" and max_status == "Pass" and std_status == "Pass" else "Warning" if "Warning" in [mean_status, max_status, std_status] else "Fail"

        # Print results
        print(f'Mean Distance: {mean_distance:.3f} mm - Status: {mean_status}')
        print(f'Max Distance: {max_distance:.3f} mm - Status: {max_status}')
        print(f'Standard Deviation: {std_distance:.3f} mm - Status: {std_status}')
        print(f'Overall Status: {overall_status}')

        return True if overall_status == "Pass" else False

    def test_multi_material_simulation(self):
        import SampleData

        # Get the layout manager
        layoutManager = slicer.app.layoutManager()

        # Set layout to 3D-only view
        layoutManager.setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutOneUp3DView)

        self.setUp()
        logic = MultiMaterialSimulationLogic()

        self.delayDisplay('Loading Testing Data')
        simulationModelNode = SampleData.downloadSample("HeartDeviceJoint")
        deformedModelDataSource = SampleData.SampleDataSource(sampleName='HeartDeviceJointDeformed',
                                                               uris=sofaDataURL+ 'SHA256/17cfdce795b0df95049f8fe4f5c6923fdaa3db304e1dfd7e6276e5e7c6a2497e',
                                                               fileNames='HeartDeviceJointDeformed.vtk',
                                                               checksums='SHA256:17cfdce795b0df95049f8fe4f5c6923fdaa3db304e1dfd7e6276e5e7c6a2497e',
                                                               nodeNames='HeartDeviceJointDeformed',
                                                               loadFileType='ModelFile')
        sampleDataLogic = SampleData.SampleDataLogic()
        deformedModelNode = sampleDataLogic.downloadFromSource(deformedModelDataSource)[0]

        self.delayDisplay('Creating fixed ROI selection')
        fixedROINode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLMarkupsROINode', 'FixedROI')
        fixedROINode.SetSize([39.54909142993799, 143.343979, 128.243714])
        fixedROINode.SetCenter([19.346689224243164, -106.59119415283203, -218.28884887695312])

        self.delayDisplay('Creating moving ROI selection')
        movingROINode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLMarkupsROINode', 'MovingROI')
        movingROINode.SetSize([57.7977844590271, 55.72434095648424, 60.922046056195654])
        movingROINode.SetCenter([-140.12936401367188, -99.91403198242188, -234.55825805664062])

        self.delayDisplay('Creating force vector')
        forceLineNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLMarkupsLineNode', 'Force')
        forceLineNode.AddControlPoint([-183.86535407878318, -94.46268756919284, -234.9103926334161])
        forceLineNode.AddControlPoint([44.804605575202274, -109.81929381733754, -180.68919726749039])

        self.delayDisplay('Setting simulation parameters')
        logic.getParameterNode().simulationModelNode = simulationModelNode
        logic.getParameterNode().fixedROI = fixedROINode
        logic.getParameterNode().movingROI = movingROINode
        logic.getParameterNode().forceVector = forceLineNode
        logic.getParameterNode().forceMagnitude = 0.5
        logic.getParameterNode().dt = 0.001
        logic.getParameterNode().currentStep = 0
        logic.getParameterNode().totalSteps = 100
        logic.totalSteps = logic.getParameterNode().totalSteps
        logic.currentStep = logic.getParameterNode().currentStep

        self.delayDisplay('Starting simulation')
        view=slicer.app.layoutManager().threeDWidget(0).threeDView()
        logic.startSimulation()
        for i in range(logic.getParameterNode().totalSteps):
            logic.simulationStep(logic.getParameterNode())
            view.forceRender()
        logic.stopSimulation()
        logic.clean()

        if not self.compareModels(deformedModelNode, simulationModelNode):
            raise(Exception("Model comparison failed"))
