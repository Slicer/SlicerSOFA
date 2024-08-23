###################################################################################
# MIT License
#
# Copyright (c) 2024 Oslo University Hospital, Norway. All Rights Reserved.
# Copyright (c) 2024 NTNU, Norway. All Rights Reserved.
# Copyright (c) 2024 INRIA, France. All Rights Reserved.
# Copyright (c) 2004 Brigham and Women's Hospital (BWH). All Rights Reserved.
# Copyright (c) 2024 Isomics, Inc., USA. All Rights Reserved.
# Copyright (c) 2024 Queen's University, Canada. All Rights Reserved.
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

def addGridTransformFromArray(narray, name="Grid Transform"):
    """Create a new grid transform node from content of a numpy array and add it to the scene.

    Voxels values are deep-copied, therefore if the numpy array
    is modified after calling this method, voxel values in the volume node will not change.
    :param narray: numpy array containing grid vectors.
    Must be [slices, rows, columns, 3]
    :param name: grid transform node name
    """
    import slicer

    if len(narray.shape) != 4 or narray.shape[3] != 3:
        raise RuntimeError("Need vector volume numpy array for grid transform")
    nodeClassName = "vtkMRMLGridTransformNode"
    gridNode = slicer.mrmlScene.AddNewNodeByClass(nodeClassName, name)
    gridNode.CreateDefaultDisplayNodes()
    displacementGrid = gridNode.GetTransformFromParent().GetDisplacementGrid()
    arrayShape = narray.shape
    displacementGrid.SetDimensions(arrayShape[2], arrayShape[1], arrayShape[0])
    scalarType = vtk.util.numpy_support.get_vtk_array_type(narray.dtype)
    displacementGrid.AllocateScalars(scalarType, 3)
    displacementArray = slicer.util.arrayFromGridTransform(gridNode)
    displacementArray[:] = narray
    slicer.util.arrayFromGridTransformModified(gridNode)
    return gridNode

class SparseGridSimulation(ScriptedLoadableModule):
    """Main class for sparse grid Simulation module.

    This class uses the ScriptedLoadableModule base class.
    """

    def __init__(self, parent):
        """Initialize the SparseGridSimulation module."""
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = _("sparse grid Simulation")
        self.parent.categories = [translate("qSlicerAbstractCoreModule", "Examples")]
        self.parent.dependencies = []
        self.parent.contributors = [
            "Rafael Palomar (Oslo University Hospital/NTNU, Norway)",
            "Nazim Haouchine (Harvard/BWH, USA)",
            "Paul Baksic (INRIA, France)",
            "Steve Pieper (Isomics, Inc., USA)",
            "Andras Lasso (Queen's University, Canada)"
        ]
        self.parent.helpText = _("""This is an example module to use the SOFA framework to do simple sparse grid simulation based on triangular meshes (instead of tetrahedral), ROI selections for depiction of fixed/moving points, and a single force vector""")
        self.parent.acknowledgementText = _("""""")

        # Register sample data after the application startup is complete
        slicer.app.connect("startupCompleted()", registerSampleData)

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

@parameterNodeWrapper
class SparseGridSimulationParameterNode:
    """Defines the parameters needed by the SparseGridSimulation module."""
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
        """Get the bounds of the specified ROI (Region of Interest).

        Args:
            ROIType (str): The type of ROI ('Fixed' or 'Moving').

        Returns:
            np.array: The bounds of the specified ROI in meters (converted from mm).
        """
        roi = self.fixedROI if ROIType == 'Fixed' else self.movingROI if ROIType == 'Moving' else None
        if roi is None:
            raise ValueError('ROIType must be either \'Fixed\' or \'Moving\'')

        center = [0] * 3
        roi.GetCenter(center)
        size = roi.GetSize()

        R_min, R_max = center[0] - size[0] / 2, center[0] + size[0] / 2
        A_min, A_max = center[1] - size[1] / 2, center[1] + size[1] / 2
        S_min, S_max = center[2] - size[2] / 2, center[2] + size[2] / 2

        # Convert the bounds from millimeters to meters (SI units)
        return np.array([R_min, A_min, S_min, R_max, A_max, S_max]) * 0.001

    def getForceVector(self) -> np.array:
        """Get the total normalized force vector.

        Returns:
            np.array: TotalForce expressed as force vector + rotation quaternion.
                      The rotation quaternion is ignored (unit) here.
        """
        if self.forceVector is None:
            return [0.0]*3

        p1 = self.forceVector.GetNthControlPointPosition(0)
        p2 = self.forceVector.GetNthControlPointPosition(1)
        force_vector = np.array([p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2]])
        magnitude = np.linalg.norm(force_vector)
        normalized_force_vector = force_vector / magnitude if magnitude != 0 else force_vector

        return normalized_force_vector * magnitude

class SparseGridSimulationWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    """User interface class for the SparseGridSimulation module.

    This class uses the ScriptedLoadableModuleWidget base class.
    """

    def __init__(self, parent=None) -> None:
        """Initialize the SparseGridSimulation widget."""
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)
        self.logic = None
        self.parameterNode = None
        self.parameterNodeGuiTag = None
        self.timer = qt.QTimer(parent)
        self.timer.timeout.connect(self.simulationStep)

    def setup(self) -> None:
        """Setup the UI components and logic for the SparseGridSimulation module."""
        ScriptedLoadableModuleWidget.setup(self)

        # Load the UI from the .ui file
        uiWidget = slicer.util.loadUI(self.resourcePath("UI/SparseGridSimulation.ui"))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)

        # Create the logic class
        self.logic = SparseGridSimulationLogic()
        uiWidget.setMRMLScene(slicer.mrmlScene)

        # Add observers to update parameter node when the scene is closed
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)

        # Connect UI buttons to their respective functions
        self.ui.startSimulationPushButton.connect("clicked()", self.startSimulation)
        self.ui.stopSimulationPushButton.connect("clicked()", self.stopSimulation)
        self.ui.addFixedROIPushButton.connect("clicked()", self.logic.addFixedROI)
        self.ui.addMovingROIPushButton.connect("clicked()", self.logic.addMovingROI)
        self.ui.addForceVectorPushButton.connect("clicked()", self.logic.addForceVector)
        self.ui.addRecordingSequencePushButton.connect("clicked()", self.logic.addRecordingSequence)

        # Initialize the parameter node
        self.initializeParameterNode()
        self.logic.getParameterNode().AddObserver(vtk.vtkCommand.ModifiedEvent, self.updateSimulationGUI)

    def cleanup(self) -> None:
        """Cleanup function called when the application closes and the module widget is destroyed."""
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

    def onSceneStartClose(self, caller, event) -> None:
        """Called just before the scene is closed."""
        self.setParameterNode(None)

    def onSceneEndClose(self, caller, event) -> None:
        """Called just after the scene is closed."""
        if self.parent.isEntered:
            self.initializeParameterNode()

    def initializeParameterNode(self) -> None:
        """Initialize the parameter node and reset its values."""
        if self.logic:
            self.setParameterNode(self.logic.getParameterNode())
            self.logic.resetParameterNode()

    def setParameterNode(self, inputParameterNode: Optional[SparseGridSimulationParameterNode]) -> None:
        """Set and observe the parameter node.

        Args:
            inputParameterNode (Optional[SparseGridSimulationParameterNode]): The parameter node to set.
        """
        if self.parameterNode:
            self.parameterNode.disconnectGui(self.parameterNodeGuiTag)
        self.parameterNode = inputParameterNode
        if self.parameterNode:
            self.parameterNodeGuiTag = self.parameterNode.connectGui(self.ui)

    def updateSimulationGUI(self, caller, event) -> None:
        """Update the GUI elements according to the state of the parameter node."""
        self.ui.startSimulationPushButton.setEnabled(not self.logic.isSimulationRunning and
                                                     self.logic.getParameterNode().simulationModelNode is not None)
        self.ui.stopSimulationPushButton.setEnabled(self.logic.isSimulationRunning)

        self.ui.addFixedROIPushButton.setEnabled(self.parameterNode is not None)
        self.ui.addMovingROIPushButton.setEnabled(self.parameterNode is not None)
        self.ui.addForceVectorPushButton.setEnabled(self.parameterNode is not None)

    def startSimulation(self) -> None:
        """Start the simulation and initiate the timer for updates."""
        self.logic.dt = self.ui.dtSpinBox.value
        self.logic.totalSteps = self.ui.totalStepsSpinBox.value
        self.logic.currentStep = self.ui.currentStepSpinBox.value
        self.logic.startSimulation()
        self.timer.start(0)  # This timer drives the simulation updates

    def stopSimulation(self) -> None:
        """Stop the simulation and the update timer."""
        self.timer.stop()
        self.logic.stopSimulation()

    def simulationStep(self) -> None:
        """Perform a single step of the simulation."""
        self.logic.simulationStep(self.parameterNode)

class SparseGridSimulationLogic(SlicerSofaLogic):
    """Logic class for the SparseGridSimulation module.

    This class implements all the computations and data manipulations required by the module.
    """

    def __init__(self) -> None:
        """Initialize the logic class."""
        super().__init__()

        self.femModelNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLModelNode')
        self.femModelNode.SetAndObserveMesh(self._initFEMUnstructuredGrid())
        self.femModelNode.CreateDefaultDisplayNodes()
        self.probeDimension = 10
        self.probeGrid = None
        self.probeFilter = None
        self.displacementGridNode = None
        self.femTopology = None

        self._createGridTransformPipeline()

        # Initialize MarkupsROI and ScalarVolume for debugging
        self.debugROINode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsROINode", "ProbingBoundsROI")
        self.debugVolumeNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", "ProbingSpaceVolume")

    def _createGridTransformPipeline(self) -> None:
        # Initialize probe grid and filter
        self.probeGrid = vtk.vtkImageData()
        self.probeGrid.SetDimensions(self.probeDimension, self.probeDimension, self.probeDimension)
        self.probeGrid.AllocateScalars(vtk.VTK_DOUBLE, 1)
        self.probeGrid.SetOrigin(0, 0, 0)
        self.probeGrid.SetSpacing(1, 1, 1)
        self.probeFilter = vtk.vtkProbeFilter()
        self.probeFilter.SetInputData(self.probeGrid)
        self.probeFilter.SetSourceData(self.femModelNode.GetUnstructuredGrid())
        self.probeFilter.SetPassPointArrays(True)

        # Create the grid transform node if it doesn't exist
        if not self.displacementGridNode:
            self.displacementGridNode = addGridTransformFromArray(np.zeros((self.probeDimension, self.probeDimension, self.probeDimension, 3)), name="Displacement")

    def _initFEMUnstructuredGrid(self) -> vtk.vtkUnstructuredGrid:
        points = vtk.vtkPoints()
        unstructuredGrid = vtk.vtkUnstructuredGrid()
        unstructuredGrid.SetPoints(points)
        displacementVTKArray = vtk.vtkFloatArray()
        displacementVTKArray.SetNumberOfComponents(3)
        displacementVTKArray.SetName('Displacement')
        unstructuredGrid.GetPointData().AddArray(displacementVTKArray)
        return unstructuredGrid

    def updateMRML(self, parameterNode) -> None:
        """Update the MRML model node with the new mesh points from the simulation."""
        # Update surface points
        surfacePointsArray = self.surfaceMechanicalObject.position.array() * 1000  # Convert from meters to millimeters
        surfaceModelPointsArray = slicer.util.arrayFromModelPoints(parameterNode.simulationModelNode)
        surfaceModelPointsArray[:] = surfacePointsArray
        slicer.util.arrayFromModelPointsModified(parameterNode.simulationModelNode)

        #Update Grid Transform
        # Since the sparse grid was created by SOFA, we need to update our femModelNode and the associated Displacement array
        self.femModelNode.GetUnstructuredGrid().GetPoints().SetNumberOfPoints(int(self.femMechanicalObject.position.size/3))
        femModelPointsArray = slicer.util.arrayFromModelPoints(self.femModelNode)
        femModelPointsArray[:] = self.femMechanicalObject.position.array() * 1000
        self.femModelNode.GetUnstructuredGrid().GetPointData().GetArray("Displacement").SetNumberOfTuples(int(self.femMechanicalObject.position.size/3))
        displacementArray = slicer.util.arrayFromModelPointData(self.femModelNode, "Displacement")
        displacementArray[:] = (self.femMechanicalObject.position - self.femMechanicalObject.rest_position)*1000
        self._add_hexahedral_cells_to_unstructured_grid(self.femTopology.hexahedra.array(), self.femModelNode.GetUnstructuredGrid())
        slicer.util.arrayFromModelPointsModified(self.femModelNode)

        # Update the geometry of the probing image, which need to match the sparse grid created by SOFA
        femGridBounds = [0] * 6
        self.femModelNode.GetRASBounds(femGridBounds)
        self.probeGrid.SetOrigin(femGridBounds[0], femGridBounds[2], femGridBounds[4])
        probeSize = (abs(femGridBounds[1] - femGridBounds[0]),
                     abs(femGridBounds[3] - femGridBounds[2]),
                     abs(femGridBounds[5] - femGridBounds[4]))
        self.probeGrid.SetSpacing(probeSize[0] / self.probeDimension,
                                  probeSize[1] / self.probeDimension,
                                  probeSize[2] / self.probeDimension)
        self.probeGrid.AllocateScalars(vtk.VTK_DOUBLE, 1)
        self.probeGrid.Modified()
        self.probeFilter.Update()

        probeGrid = self.probeFilter.GetOutputDataObject(0)
        probeVTKArray = probeGrid.GetPointData().GetArray("Displacement")
        probeArray = vtk.util.numpy_support.vtk_to_numpy(probeVTKArray)
        probeArrayShape = (self.probeDimension,self.probeDimension,self.probeDimension,3)
        probeArray = probeArray.reshape(probeArrayShape)
        gridArray = slicer.util.arrayFromGridTransform(self.displacementGridNode)
        gridArray[:] = -1. * probeArray
        slicer.util.arrayFromGridTransformModified(self.displacementGridNode)
        self.displacementGrid = self.displacementGridNode.GetTransformFromParent().GetDisplacementGrid()
        self.displacementGrid.SetOrigin(probeGrid.GetOrigin())
        self.displacementGrid.SetSpacing(probeGrid.GetSpacing())

        # # Update the debugging ROI node with the probe bounds
        # self._updateDebugROINode(self.probeGrid.GetBounds())

        # # Update the debugging volume node with the displacement values
        # self._updateDebugVolumeNode(probeGrid)

    def _add_hexahedral_cells_to_unstructured_grid(self, cell_connectivity, unstructured_grid):
        """
        Add hexahedral cells to an existing vtkUnstructuredGrid.

        Parameters:
        - cell_connectivity: List of lists where each inner list represents the point indices for a hexahedron cell.
        - unstructured_grid: An existing vtkUnstructuredGrid to which the hexahedral cells will be added.

        Returns:
        - vtkUnstructuredGrid: The unstructured grid with added hexahedral cells.
        """
        # Ensure the unstructured grid is a valid vtkUnstructuredGrid instance
        if not isinstance(unstructured_grid, vtk.vtkUnstructuredGrid):
            raise ValueError("The provided unstructured grid must be an instance of vtkUnstructuredGrid.")

        # Create a vtkCellArray to store the cells
        cell_array = vtk.vtkCellArray()

        # Iterate over the cell connectivity to create hexahedral cells
        for cell in cell_connectivity:
            if len(cell) != 8:
                raise ValueError("Each hexahedral cell should have exactly 8 points.")

            hexahedron = vtk.vtkHexahedron()

            for i in range(8):
                hexahedron.GetPointIds().SetId(i, cell[i])

            cell_array.InsertNextCell(hexahedron)

        # Set the cells (hexahedrons) into the unstructured grid
        unstructured_grid.SetCells(vtk.VTK_HEXAHEDRON, cell_array)

        return unstructured_grid


    def _updateDebugROINode(self, bounds) -> None:
            """Update the ROI node with the bounds of the probed data."""
            center = [(bounds[0] + bounds[1]) / 2.0, (bounds[2] + bounds[3]) / 2.0, (bounds[4] + bounds[5]) / 2.0]
            size = [abs(bounds[1] - bounds[0]), abs(bounds[3] - bounds[2]), abs(bounds[5] - bounds[4])]
            self.debugROINode.SetXYZ(center)
            self.debugROINode.SetRadiusXYZ(size[0] / 2, size[1] / 2, size[2] / 2)

    def _updateDebugVolumeNode(self, probeImage) -> None:
        """Update the scalar volume node with the probed displacement values."""
        dimensions = probeImage.GetDimensions()
        origin = probeImage.GetOrigin()
        spacing = probeImage.GetSpacing()

        # Allocate and set vtkImageData
        imageData = vtk.vtkImageData()
        imageData.SetDimensions(dimensions)
        imageData.SetOrigin(origin)
        imageData.SetSpacing(spacing)
        imageData.AllocateScalars(vtk.VTK_DOUBLE, 1)  # Use double to match displacement data

        # Get the displacement data from the probe filter output
        probeArray = vtk.util.numpy_support.vtk_to_numpy(probeImage.GetPointData().GetArray('Displacement'))

        # Reshape and extract the magnitude of the displacement vectors
        probeArray = np.linalg.norm(probeArray.reshape(-1, 3), axis=1)
        probeArray = probeArray.reshape(dimensions)

        # Set the probed displacement magnitudes to the scalar volume
        imageDataArray = vtk.util.numpy_support.vtk_to_numpy(imageData.GetPointData().GetScalars())
        imageDataArray[:] = probeArray.ravel()

        # Set the vtkImageData to the scalar volume node
        self.debugVolumeNode.SetAndObserveImageData(imageData)
        self.debugVolumeNode.SetOrigin(imageData.GetOrigin())
        self.debugVolumeNode.SetSpacing(imageData.GetSpacing())
        self.debugVolumeNode.CreateDefaultDisplayNodes()
        self.debugVolumeNode.GetDisplayNode().SetAutoWindowLevel(False)
        self.debugVolumeNode.GetDisplayNode().SetWindowLevel(probeArray.max(), probeArray.mean())
        self.debugVolumeNode.GetDisplayNode().SetAndObserveColorNodeID("vtkMRMLColorTableNodeGrey")

    def simulationStep(self, parameterNode) -> None:
        self.forceVector.totalForce = parameterNode.getForceVector()
        super().simulationStep(parameterNode)
        self.updateMRML(parameterNode)


    def getParameterNode(self) -> SparseGridSimulationParameterNode:
        """Get the parameter node for the simulation.

        Returns:
            SparseGridSimulationParameterNode: The parameter node for the simulation.
        """
        return SparseGridSimulationParameterNode(super().getParameterNode())

    def resetParameterNode(self) -> None:
        """Reset the values of the parameter node to their defaults."""
        if self.getParameterNode():
            self.getParameterNode().simulationModelNode = None
            self.getParameterNode().boundaryROI = None
            self.getParameterNode().sequenceNode = None
            self.getParameterNode().sequenceBrowserNode = None
            self.getParameterNode().dt = 0.001
            self.getParameterNode().currentStep = 0
            self.getParameterNode().totalSteps = -1

    def startSimulation(self) -> None:
        """Start the simulation."""
        parameterNode = self.getParameterNode()
        sequenceNode = parameterNode.sequenceNode
        browserNode = parameterNode.sequenceBrowserNode
        simulationModelNode = parameterNode.simulationModelNode

        # Setup and synchronize the sequence browser node
        if None not in [sequenceNode, browserNode, simulationModelNode]:
            browserNode.AddSynchronizedSequenceNodeID(sequenceNode.GetID())
            browserNode.AddProxyNode(simulationModelNode, sequenceNode, False)
            browserNode.SetRecording(sequenceNode, True)
            browserNode.SetRecordingActive(True)

        super().startSimulation(parameterNode)
        self._simulationRunning = True
        parameterNode.Modified()

    def stopSimulation(self) -> None:
        """Stop the simulation."""
        super().stopSimulation()
        self._simulationRunning = False
        browserNode = self.getParameterNode().sequenceBrowserNode
        if browserNode is not None:
            browserNode.SetRecordingActive(False)
        self.getParameterNode().Modified()

    def simulationStep(self, parameterNode) -> None:
        self.forceVector.totalForce = parameterNode.getForceVector()
        super().simulationStep(parameterNode)

    def onSimulationModelNodeModified(self, caller, event) -> None:
        """Update the model node when it is modified during the simulation.

        Args:
            caller: The caller object triggering the event.
            event: The event triggered.
        """
        simulationModelNode = self.getParameterNode().simulationModelNode
        if simulationModelNode.GetUnstructuredGrid() is not None:
            simulationModelNode.GetUnstructuredGrid().SetPoints(caller.GetPolyData().GetPoints())
        elif simulationModelNode.GetPolyData() is not None:
            simulationModelNode.GetPolyData().SetPoints(caller.GetPolyData().GetPoints())

    def addFixedROI(self) -> None:
        """Add a fixed Region of Interest (ROI) to the scene."""
        roiNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsROINode")
        mesh, bounds = self._getMeshAndBounds()
        if mesh is not None:
            self._setROICenterAndSize(roiNode, bounds)
        self.getParameterNode().fixedROI = roiNode

    def addMovingROI(self) -> None:
        """Add a moving Region of Interest (ROI) to the scene."""
        roiNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsROINode")
        mesh, bounds = self._getMeshAndBounds()
        if mesh is not None:
            self._setROICenterAndSize(roiNode, bounds)
        self.getParameterNode().movingROI = roiNode

    def addForceVector(self) -> None:
        """Add a force vector to the scene."""
        forceVector = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLMarkupsLineNode', 'Force')
        if forceVector is not None:
            forceVector.CreateDefaultDisplayNodes()
            forceVector.GetMeasurement('length').EnabledOff()
            mesh, bounds = self._getMeshAndBounds()
            if mesh is not None:
                self._setForceVectorPoints(forceVector, bounds)
        self.getParameterNode().forceVector = forceVector

    def addDeviceTransform(self) -> None:
        """Add a device transform node to the scene."""
        transformNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLTransformNode", "Device Transform")
        transformNode.CreateDefaultDisplayNodes()
        self.getParameterNode().DeviceTransformNode = transformNode

    def addRecordingSequence(self) -> None:
        """Add a recording sequence to the scene."""
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
        """Create the SOFA simulation scene.

        Args:
            parameterNode (SparseGridSimulationParameterNode): The parameter node containing the simulation data.

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

        rootNode.dt = parameterNode.dt
        rootNode.gravity = [0, 0, 0]

        rootNode.addObject('DefaultAnimationLoop', parallelODESolving=True)
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
        self.femTopology = fem.addObject('SparseGridTopology', n=[20, 20, 20], position="@../InputSurfaceNode/Container.position")
        fem.addObject('EulerImplicitSolver', rayleighStiffness=0.1, rayleighMass=0.1)
        fem.addObject('CGLinearSolver', iterations=100, tolerance=1e-5, threshold=1e-5)
        self.femMechanicalObject = fem.addObject('MechanicalObject', name='MO')
        slicer.modules.fmo = self.femMechanicalObject
        fem.addObject('UniformMass', totalMass=0.5)
        fem.addObject('ParallelHexahedronFEMForceField', name="FEMForce", youngModulus=50000000, poissonRatio=0.40, method="large")

        surf = fem.addChild('Surf')
        surf.addObject('MeshTopology', position="@../../InputSurfaceNode/Container.position")
        self.surfaceMechanicalObject = surf.addObject('MechanicalObject', position="@../../InputSurfaceNode/Container.position")
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
        self.forceVector = fem.addObject('ConstantForceField', name='ForceField', indices='@boxForce.indices')
        self.forceVector.totalForce = parameterNode.getForceVector()

        return rootNode

    def _getMeshAndBounds(self):
        """Helper function to get mesh and bounds from the simulation model node.

        Returns:
            Tuple: A tuple containing the mesh and its bounds.
        """
        simulationModelNode = self.getParameterNode().simulationModelNode
        if simulationModelNode is not None:
            if simulationModelNode.GetUnstructuredGrid() is not None:
                return simulationModelNode.GetUnstructuredGrid(), simulationModelNode.GetUnstructuredGrid().GetBounds()
            elif simulationModelNode.GetPolyData() is not None:
                return simulationModelNode.GetPolyData(), simulationModelNode.GetPolyData().GetBounds()
        return None, None

    def _setROICenterAndSize(self, roiNode, bounds):
        """Helper function to set the center and size of an ROI node.

        Args:
            roiNode: The ROI node to set.
            bounds: The bounds of the mesh.
        """
        center = [(bounds[0] + bounds[1]) / 2.0, (bounds[2] + bounds[3]) / 2.0, (bounds[4] + bounds[5]) / 2.0]
        size = [abs(bounds[1] - bounds[0]) / 2.0, abs(bounds[3] - bounds[2]) / 2.0, abs(bounds[5] - bounds[4]) / 2.0]
        roiNode.SetXYZ(center)
        roiNode.SetRadiusXYZ(size[0], size[1], size[2])

    def _setForceVectorPoints(self, forceVector, bounds):
        """Helper function to set the points of a force vector.

        Args:
            forceVector: The force vector to set.
            bounds: The bounds of the mesh.
        """
        center = [(bounds[0] + bounds[1]) / 2.0, (bounds[2] + bounds[3]) / 2.0, (bounds[4] + bounds[5]) / 2.0]
        startPoint = [center[0], bounds[2], center[2]]
        endPoint = [center[0], bounds[3], center[2]]
        vectorLength = endPoint[1] - startPoint[1]
        midPoint = startPoint[1] + vectorLength / 2.0
        startPoint[1] = midPoint - vectorLength / 2.0
        endPoint[1] = midPoint + vectorLength / 2.0
        forceVector.AddControlPoint(vtk.vtkVector3d(startPoint))
        forceVector.AddControlPoint(vtk.vtkVector3d(endPoint))

class SparseGridSimulationTest(ScriptedLoadableModuleTest):
    """This is the test case for your scripted module.

    This class uses the ScriptedLoadableModuleTest base class.
    """

    def setUp(self):
        """Reset the state by clearing the MRML scene."""
        slicer.mrmlScene.Clear()

    def runTest(self):
        """Run the tests for the SparseGridSimulation module."""
        self.delayDisplay("Starting test_spine_deformation_simulation")
        self.test_spine_deformation_simulation()
        self.delayDisplay('Test test_spine_deformation_simulation passed')

    def compareModels(self, referenceModelNode, testModelNode) -> bool:
        """Compare two model nodes to evaluate the simulation accuracy.

        Args:
            referenceModelNode: The reference model node.
            testModelNode: The test model node.

        Returns:
            bool: True if the comparison passes, False otherwise.
        """
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

    def test_spine_deformation_simulation(self):
        """Test the sparse grid simulation using predefined data."""
        import SampleData

        self.setUp()
        logic = SparseGridSimulationLogic()

        self.delayDisplay('Loading Testing Data')
        SampleData.downloadSample('Spine')
        simulationModelNode = slicer.util.getNode('SpineSurfaceModel_Dec')
        deformedModelDataSource = SampleData.SampleDataSource(
            sampleName='SpineSurfaceModel_Dec_deformed',
            uris=SOFA_DATA_URL + 'SHA256/417e57d5e634519a9d23677d1bb205d907c5afe0e659da79d118766fd3743bdf',
            fileNames='SpineSurfaceModel_Dec_deformed.vtk',
            checksums='SHA256:417e57d5e634519a9d23677d1bb205d907c5afe0e659da79d118766fd3743bdf',
            nodeNames='SpineSurfaceModel_Dec_deformed',
            loadFileType='ModelFile'
        )
        sampleDataLogic = SampleData.SampleDataLogic()
        deformedModelNode = sampleDataLogic.downloadFromSource(deformedModelDataSource)[0]

        layoutManager = slicer.app.layoutManager()
        layoutManager.setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutOneUp3DView)

        self.delayDisplay('Creating fixed ROI selection')
        fixedROINode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLMarkupsROINode', 'FixedROI')
        fixedROINode.SetSize([108.1689567565918, 134.68839263916016, 73.236760323082])
        fixedROINode.SetCenter([-8.383378982543945, 152.76014709472656, 84.74845886230469])

        self.delayDisplay('Creating moving ROI selection')
        movingROINode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLMarkupsROINode', 'MovingROI')
        movingROINode.SetSize([108.1689567565918, 134.68839263916016, 113.71309709627243])
        movingROINode.SetCenter([-8.383378982543945, 152.76014709472656, 335.7271728515625])

        self.delayDisplay('Creating force vector')
        forceLineNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLMarkupsLineNode', 'Force')
        forceLineNode.AddControlPoint([-8.383174999999998, 85.42806199999998, 223.870506])
        forceLineNode.AddControlPoint([-8.383174999999998, 220.092743, 223.870506])

        self.delayDisplay('Setting simulation parameters')
        parameterNode = logic.getParameterNode()
        parameterNode.simulationModelNode = simulationModelNode
        parameterNode.fixedROI = fixedROINode
        parameterNode.movingROI = movingROINode
        parameterNode.forceVector = forceLineNode
        parameterNode.forceMagnitude = 10000
        parameterNode.dt = 0.001
        parameterNode.currentStep = 0
        parameterNode.totalSteps = 100
        logic.totalSteps = parameterNode.totalSteps
        logic.currentStep = parameterNode.currentStep

        self.delayDisplay('Starting simulation')
        view = slicer.app.layoutManager().threeDWidget(0).threeDView()
        logic.startSimulation()
        for _ in range(parameterNode.totalSteps):
            logic.simulationStep(parameterNode)
            view.forceRender()
        logic.stopSimulation()
        logic.clean()

        if not self.compareModels(deformedModelNode, simulationModelNode):
            raise Exception("Model comparison failed")
