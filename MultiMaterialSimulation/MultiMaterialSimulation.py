import logging
import os
from typing import Optional
import random
import time
import uuid
import numpy as np

import qt
import vtk
from vtk.util.numpy_support import numpy_to_vtk

import slicer
from slicer.i18n import tr as _
from slicer.i18n import translate
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin
from slicer.parameterNodeWrapper import parameterNodeWrapper, WithinRange
from SofaEnvironment import Sofa
from SlicerSofa import SlicerSofaLogic

from slicer import vtkMRMLModelNode
from slicer import vtkMRMLMarkupsROINode
from slicer import vtkMRMLMarkupsLineNode
from slicer import vtkMRMLSequenceNode
from slicer import vtkMRMLSequenceBrowserNode

# Constants
SOFA_DATA_URL = 'https://github.com/rafaelpalomar/SlicerSofaTestingData/releases/download/'

class MultiMaterialSimulation(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class"""

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = _("Multi-Material Simulation")
        self.parent.categories = [translate("qSlicerAbstractCoreModule", "Examples")]
        self.parent.dependencies = []
        self.parent.contributors = [
            "Rafael Palomar (Oslo University Hospital/NTNU, Norway)",
            "Nazim Haouchine (Harvard/BWH, USA)",
            "Paul Baksic (INRIA, France)",
            "Steve Pieper (Isomics, Inc., USA)",
            "Andras Lasso (Queen's University, Canada)"
        ]
        self.parent.helpText = _("""This is an example module to use the SOFA framework to do simple multi-material simulation based on surface meshes, ROI selections and a single force vector""")
        self.parent.acknowledgementText = _("""""")

        slicer.app.connect("startupCompleted()", registerSampleData)

def registerSampleData():
    """Add data sets to Sample Data module."""
    import SampleData
    iconsPath = os.path.join(os.path.dirname(__file__), "Resources/Icons")

    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        category='SOFA',
        sampleName='HeartDeviceJoint',
        thumbnailFileName=os.path.join(iconsPath, 'HeartDeviceJoint.png'),
        uris=SOFA_DATA_URL + 'SHA256/6d0cecb9f1e8dd48c6bd458ae2ca29aa592c219ff9190220dc80adb507e33676',
        fileNames='HeartDeviceJoint.vtk',
        checksums='SHA256:6d0cecb9f1e8dd48c6bd458ae2ca29aa592c219ff9190220dc80adb507e33676',
        nodeNames='HeartDeviceJoint',
        loadFileType='ModelFile'
    )

@parameterNodeWrapper
class MultiMaterialSimulationParameterNode:
    """The parameters needed by the module."""
    simulationModelNode: vtkMRMLModelNode
    fixedROI: vtkMRMLMarkupsROINode
    movingROI: vtkMRMLMarkupsROINode
    sequenceNode: vtkMRMLSequenceNode
    sequenceBrowserNode: vtkMRMLSequenceBrowserNode
    forceVector: vtkMRMLMarkupsLineNode
    forceMagnitude: float
    dt: float
    totalSteps: int
    currentStep: int
    simulationRunning: bool

    def getROI(self, ROIType: str) -> np.array:
        """Get the Region of Interest (ROI) bounds."""
        roi = self.fixedROI if ROIType == 'Fixed' else self.movingROI if ROIType == 'Moving' else None
        if roi is None:
            raise ValueError('ROIType must be either \'Fixed\' or \'Moving\'')

        center = [0] * 3
        roi.GetCenter(center)
        size = roi.GetSize()

        R_min, R_max = center[0] - size[0] / 2, center[0] + size[0] / 2
        A_min, A_max = center[1] - size[1] / 2, center[1] + size[1] / 2
        S_min, S_max = center[2] - size[2] / 2, center[2] + size[2] / 2

        return np.array([R_min, A_min, S_min, R_max, A_max, S_max]) * 0.001

    def getForceVector(self) -> np.array:
        """Get the normalized force vector."""
        if self.forceVector is None:
            return [0.0] * 3

        p1 = self.forceVector.GetNthControlPointPosition(0)
        p2 = self.forceVector.GetNthControlPointPosition(1)
        force_vector = np.array([p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2]])
        magnitude = np.linalg.norm(force_vector)
        normalized_force_vector = force_vector / magnitude if magnitude != 0 else force_vector

        return normalized_force_vector * self.forceMagnitude

class MultiMaterialSimulationWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    """Uses ScriptedLoadableModuleWidget base class"""

    def __init__(self, parent=None) -> None:
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)
        self.logic = None
        self.parameterNode = None
        self.parameterNodeGuiTag = None
        self.timer = qt.QTimer(parent)
        self.timer.timeout.connect(self.simulationStep)

    def setup(self) -> None:
        ScriptedLoadableModuleWidget.setup(self)

        uiWidget = slicer.util.loadUI(self.resourcePath("UI/MultiMaterialSimulation.ui"))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)

        self.logic = MultiMaterialSimulationLogic()
        uiWidget.setMRMLScene(slicer.mrmlScene)

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
        self.timer.stop()
        self.logic.stopSimulation()
        self.logic.clean()
        self.removeObservers()

    def enter(self) -> None:
        self.initializeParameterNode()

    def exit(self) -> None:
        if self.parameterNode:
            self.parameterNode.disconnectGui(self.parameterNodeGuiTag)
            self.parameterNodeGuiTag = None

    def onSceneStartClose(self, caller, event) -> None:
        self.setParameterNode(None)

    def onSceneEndClose(self, caller, event) -> None:
        if self.parent.isEntered:
            self.initializeParameterNode()

    def initializeParameterNode(self) -> None:
        if self.logic:
            self.setParameterNode(self.logic.getParameterNode())
            self.logic.resetParameterNode()

    def setParameterNode(self, inputParameterNode: Optional[MultiMaterialSimulationParameterNode]) -> None:
        if self.parameterNode:
            self.parameterNode.disconnectGui(self.parameterNodeGuiTag)
        self.parameterNode = inputParameterNode
        if self.parameterNode:
            self.parameterNodeGuiTag = self.parameterNode.connectGui(self.ui)

    def updateSimulationGUI(self, caller, event) -> None:
        self.ui.startSimulationPushButton.setEnabled(not self.logic.isSimulationRunning and
                                                     self.logic.getParameterNode().simulationModelNode is not None)
        self.ui.stopSimulationPushButton.setEnabled(self.logic.isSimulationRunning)

        self.ui.addFixedROIPushButton.setEnabled(self.parameterNode is not None)
        self.ui.addMovingROIPushButton.setEnabled(self.parameterNode is not None)
        self.ui.addForceVectorPushButton.setEnabled(self.parameterNode is not None)

    def startSimulation(self) -> None:
        self.logic.dt = self.ui.dtSpinBox.value
        self.logic.totalSteps = self.ui.totalStepsSpinBox.value
        self.logic.currentStep = self.ui.currentStepSpinBox.value
        self.logic.startSimulation()
        self.timer.start(0)  # This timer drives the simulation updates

    def stopSimulation(self) -> None:
        self.timer.stop()
        self.logic.stopSimulation()

    def simulationStep(self) -> None:
        self.logic.simulationStep(self.parameterNode)

class MultiMaterialSimulationLogic(SlicerSofaLogic):
    """This class implements all the actual computation done by your module."""

    def __init__(self) -> None:
        super().__init__()
        self.connectionStatus = 0
        self.fixedBoxROI = None
        self.movingBoxROI = None
        self.mouseInteractor = None

    def updateMRML(self, parameterNode) -> None:
        meshPointsArray = self.mechanicalObject.position.array() * 1000
        modelPointsArray = slicer.util.arrayFromModelPoints(parameterNode.simulationModelNode)
        modelPointsArray[:] = meshPointsArray
        slicer.util.arrayFromModelPointsModified(parameterNode.simulationModelNode)

    def getParameterNode(self) -> MultiMaterialSimulationParameterNode:
        return MultiMaterialSimulationParameterNode(super().getParameterNode())

    def resetParameterNode(self) -> None:
        if self.getParameterNode():
            self.getParameterNode().simulationModelNode = None
            self.getParameterNode().boundaryROI = None
            self.getParameterNode().sequenceNode = None
            self.getParameterNode().sequenceBrowserNode = None
            self.getParameterNode().dt = 0.001
            self.getParameterNode().currentStep = 0
            self.getParameterNode().totalSteps = -1

    def startSimulation(self) -> None:
        parameterNode = self.getParameterNode()
        sequenceNode = parameterNode.sequenceNode
        browserNode = parameterNode.sequenceBrowserNode
        simulationModelNode = parameterNode.simulationModelNode

        if None not in [sequenceNode, browserNode, simulationModelNode]:
            browserNode.AddSynchronizedSequenceNodeID(sequenceNode.GetID())
            browserNode.AddProxyNode(simulationModelNode, sequenceNode, False)
            browserNode.SetRecording(sequenceNode, True)
            browserNode.SetRecordingActive(True)

        super().startSimulation(parameterNode)
        self._simulationRunning = True
        parameterNode.Modified()

    def stopSimulation(self) -> None:
        super().stopSimulation()
        self._simulationRunning = False
        browserNode = self.getParameterNode().sequenceBrowserNode
        if browserNode is not None:
            browserNode.SetRecordingActive(False)
        self.getParameterNode().Modified()

    def onSimulationModelNodeModified(self, caller, event) -> None:
        simulationModelNode = self.getParameterNode().simulationModelNode
        if simulationModelNode.GetUnstructuredGrid() is not None:
            simulationModelNode.GetUnstructuredGrid().SetPoints(caller.GetPolyData().GetPoints())
        elif simulationModelNode.GetPolyData() is not None:
            simulationModelNode.GetPolyData().SetPoints(caller.GetPolyData().GetPoints())

    def addFixedROI(self) -> None:
        roiNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsROINode")
        mesh, bounds = self._getMeshAndBounds()
        if mesh is not None:
            self._setROICenterAndSize(roiNode, bounds)
        self.getParameterNode().fixedROI = roiNode

    def addMovingROI(self) -> None:
        roiNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsROINode")
        mesh, bounds = self._getMeshAndBounds()
        if mesh is not None:
            self._setROICenterAndSize(roiNode, bounds)
        self.getParameterNode().movingROI = roiNode

    def addForceVector(self) -> None:
        forceVector = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLMarkupsLineNode', 'Force')
        if forceVector is not None:
            forceVector.CreateDefaultDisplayNodes()
            forceVector.GetMeasurement('length').EnabledOff()
            mesh, bounds = self._getMeshAndBounds()
            if mesh is not None:
                self._setForceVectorPoints(forceVector, bounds)
        self.getParameterNode().forceVector = forceVector

    def addDeviceTransform(self) -> None:
        transformNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLTransformNode", "Device Transform")
        transformNode.CreateDefaultDisplayNodes()
        self.getParameterNode().DeviceTransformNode = transformNode

    def addRecordingSequence(self) -> None:
        browserNode = self.getParameterNode().sequenceBrowserNode
        modelNode = self.getParameterNode().simulationModelNode

        if browserNode is None:
            browserNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLSequenceBrowserNode', "SOFA Simulation")
            browserNode.SetPlaybackActive(False)
            browserNode.SetRecordingActive(False)
            self.getParameterNode().sequenceBrowserNode = browserNode

        sequenceNode = slicer.vtkMRMLSequenceNode()
        if modelNode is not None:
            sequenceNode.SetName(modelNode.GetName() + "-Sequence")
        slicer.mrmlScene.AddNode(sequenceNode)
        self.getParameterNode().sequenceNode = sequenceNode

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
        container = inputNode.addObject('TriangleSetTopologyContainer', name='Container')
        container.position = slicer.util.arrayFromModelPoints(parameterNode.simulationModelNode)*0.001
        container.triangle = slicer.util.arrayFromModelPolyIds(parameterNode.simulationModelNode).reshape(-1,4)[:,1:]

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

    def _getMeshAndBounds(self):
        """Helper function to get mesh and bounds from simulation model node."""
        simulationModelNode = self.getParameterNode().simulationModelNode
        if simulationModelNode is not None:
            if simulationModelNode.GetUnstructuredGrid() is not None:
                return simulationModelNode.GetUnstructuredGrid(), simulationModelNode.GetUnstructuredGrid().GetBounds()
            elif simulationModelNode.GetPolyData() is not None:
                return simulationModelNode.GetPolyData(), simulationModelNode.GetPolyData().GetBounds()
        return None, None

    def _setROICenterAndSize(self, roiNode, bounds):
        """Helper function to set ROI center and size."""
        center = [(bounds[0] + bounds[1]) / 2.0, (bounds[2] + bounds[3]) / 2.0, (bounds[4] + bounds[5]) / 2.0]
        size = [abs(bounds[1] - bounds[0]) / 2.0, abs(bounds[3] - bounds[2]) / 2.0, abs(bounds[5] - bounds[4]) / 2.0]
        roiNode.SetXYZ(center)
        roiNode.SetRadiusXYZ(size[0], size[1], size[2])

    def _setForceVectorPoints(self, forceVector, bounds):
        """Helper function to set force vector points."""
        center = [(bounds[0] + bounds[1]) / 2.0, (bounds[2] + bounds[3]) / 2.0, (bounds[4] + bounds[5]) / 2.0]
        startPoint = [center[0], bounds[2], center[2]]
        endPoint = [center[0], bounds[3], center[2]]
        vectorLength = endPoint[1] - startPoint[1]
        midPoint = startPoint[1] + vectorLength / 2.0
        startPoint[1] = midPoint - vectorLength / 2.0
        endPoint[1] = midPoint + vectorLength / 2.0
        forceVector.AddControlPoint(vtk.vtkVector3d(startPoint))
        forceVector.AddControlPoint(vtk.vtkVector3d(endPoint))

class MultiMaterialSimulationTest(ScriptedLoadableModuleTest):
    """This is the test case for your scripted module."""

    def setUp(self):
        slicer.mrmlScene.Clear()

    def runTest(self):
        self.delayDisplay("Starting test_multi_material_simulation")
        self.test_multi_material_simulation()
        self.delayDisplay('Test test_multi_material_simulation passed')

    def compareModels(self, referenceModelNode, testModelNode) -> bool:
        distance_filter = vtk.vtkDistancePolyDataFilter()
        distance_filter.SetInputData(0, referenceModelNode.GetPolyData())
        distance_filter.SetInputData(1, testModelNode.GetPolyData())
        distance_filter.Update()

        distance_array = vtk.util.numpy_support.vtk_to_numpy(distance_filter.GetOutput().GetPointData().GetScalars('Distance'))
        mean_distance, max_distance, std_distance = distance_array.mean(), distance_array.max(), distance_array.std()

        mean_threshold_pass, mean_threshold_fail = 0.5, 1.0
        max_threshold_pass, max_threshold_fail = 2.0, 3.0
        std_threshold_pass, std_threshold_fail = 0.2, 0.5

        mean_status = "Pass" if mean_distance < mean_threshold_pass else "Warning" if mean_distance < mean_threshold_fail else "Fail"
        max_status = "Pass" if max_distance < max_threshold_pass else "Warning" if max_distance < max_threshold_fail else "Fail"
        std_status = "Pass" if std_distance < std_threshold_pass else "Warning" if std_distance < std_threshold_fail else "Fail"

        overall_status = "Pass" if mean_status == "Pass" and max_status == "Pass" and std_status == "Pass" else "Warning" if "Warning" in [mean_status, max_status, std_status] else "Fail"

        print(f'Mean Distance: {mean_distance:.3f} mm - Status: {mean_status}')
        print(f'Max Distance: {max_distance:.3f} mm - Status: {max_status}')
        print(f'Standard Deviation: {std_distance:.3f} mm - Status: {std_status}')
        print(f'Overall Status: {overall_status}')

        return overall_status == "Pass"

    def test_multi_material_simulation(self):
        import SampleData

        layoutManager = slicer.app.layoutManager()
        layoutManager.setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutOneUp3DView)

        self.setUp()
        simulationLogic = MultiMaterialSimulationLogic()

        self.delayDisplay('Loading Testing Data')
        simulationModelNode = SampleData.downloadSample("HeartDeviceJoint")
        deformedModelDataSource = SampleData.SampleDataSource(
            sampleName='HeartDeviceJointDeformed',
            uris=SOFA_DATA_URL + 'SHA256/17cfdce795b0df95049f8fe4f5c6923fdaa3db304e1dfd7e6276e5e7c6a2497e',
            fileNames='HeartDeviceJointDeformed.vtk',
            checksums='SHA256:17cfdce795b0df95049f8fe4f5c6923fdaa3db304e1dfd7e6276e5e7c6a2497e',
            nodeNames='HeartDeviceJointDeformed',
            loadFileType='ModelFile'
        )
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
        parameterNode = simulationLogic.getParameterNode()
        parameterNode.simulationModelNode = simulationModelNode
        parameterNode.fixedROI = fixedROINode
        parameterNode.movingROI = movingROINode
        parameterNode.forceVector = forceLineNode
        parameterNode.forceMagnitude = 0.5
        parameterNode.dt = 0.001
        parameterNode.currentStep = 0
        parameterNode.totalSteps = 100
        simulationLogic.totalSteps = parameterNode.totalSteps
        simulationLogic.currentStep = parameterNode.currentStep

        self.delayDisplay('Starting simulation')
        view = slicer.app.layoutManager().threeDWidget(0).threeDView()
        simulationLogic.startSimulation()
        for _ in range(parameterNode.totalSteps):
            simulationLogic.simulationStep(parameterNode)
            view.forceRender()
        simulationLogic.stopSimulation()
        simulationLogic.clean()

        if not self.compareModels(deformedModelNode, simulationModelNode):
            raise Exception("Model comparison failed")