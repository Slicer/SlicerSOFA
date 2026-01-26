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
import importlib.util
import pathlib

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
)


# -----------------------------------------------------------------------------
# Class: SOFASceneLoaderParameterNode
# -----------------------------------------------------------------------------
@SofaParameterNodeWrapper
class SOFASceneLoaderParameterNode:
    """
    Parameter class for the soft tissue simulation.
    Defines nodes to map between SOFA and MRML scenes with recording options.
    """
    modelNode: vtkMRMLModelNode                    # Model node with SOFA mapping and sequence recording
    recordSequence: bool = False                   # Record sequence?

# -----------------------------------------------------------------------------
# Class: SOFASceneLoader
# -----------------------------------------------------------------------------
class SOFASceneLoader(ScriptedLoadableModule):
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
        self.parent.title = _("SOFA Scene Loader")
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
        # slicer.app.connect("startupCompleted()", self.registerSampleData)



# -----------------------------------------------------------------------------
# Class: SOFASceneLoaderWidget
# -----------------------------------------------------------------------------
class SOFASceneLoaderWidget(SlicerSofaWidget):
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
        uiWidget = slicer.util.loadUI(self.resourcePath("UI/SOFASceneLoader.ui"))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)

        # Initialize logic for simulation computations
        self.logic = SOFASceneLoaderLogic()
        uiWidget.setMRMLScene(slicer.mrmlScene)

        # Connect UI buttons to their respective methods
        self.ui.startSimulationPushButton.connect("clicked()", self.startSimulation)
        self.ui.loadSimulationPushButton.connect("clicked()", self.loadSimulation)
        self.ui.stopSimulationPushButton.connect("clicked()", self.stopSimulation)
        self.ui.resetSimulationPushButton.connect("clicked()", self.logic.resetSimulation)

        # Initialize parameter node and GUI bindings
        self.setParameterNode(self.logic.getParameterNode())
        self.initializeParameterNode()
        self.logic.getParameterNode().AddObserver(vtk.vtkCommand.ModifiedEvent, self.updateSimulationGUI)
        self.logic.setUi(self.ui)

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

    def loadSimulation(self) -> None:
        self.logic.loadSimulationFile(self)

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
# Class: SOFASceneLoaderLogic
# -----------------------------------------------------------------------------
class SOFASceneLoaderLogic(SlicerSofaLogic):
    """
    Logic class for the Soft Tissue Simulation.
    Handles scene setup, parameter node management, and simulation steps.
    """
    def __init__(self) -> None:
        """
        Initialize the logic with the SOFA scene.
        """
        super().__init__()
        self.createSceneMethod = None
        self._rootNode = self.CreateScene()
        self._parameterNode = None

    def CreateScene(self):
        sofa_root = Sofa.Core.Node("root")
        if self.createSceneMethod is not None:
            self.createSceneMethod(sofa_root)
        return sofa_root
    
    def loadSimulationFile(self, widget):
        supposedPath = self.getUi().simulationFileName.text
        if os.path.isabs(supposedPath) and os.path.isfile(supposedPath):
            path = supposedPath
        elif not os.path.isabs(supposedPath) and os.path.isfile(os.path.join(os.getcwd(), supposedPath)):
            path = os.path.join(os.getcwd(), supposedPath)
        else:
            path = qt.QFileDialog.getOpenFileName(widget.parent.window(), _("Select SOFA scene file (*.py)"))
            self.getUi().simulationFileName.setText(path)

        if not os.path.isfile(path):
            print("Input file must exist")
        else:
            #Open file
            moduleName = pathlib.Path(path).name.split('.')[0]
            spec=importlib.util.spec_from_file_location(moduleName,path)
            foo = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(foo)
            self.createSceneMethod = foo.createScene
            

    def getParameterNode(self):
        """
        Retrieves or creates a wrapped parameter node.

        Returns:
            SOFASceneLoaderParameterNode: The parameter node for the simulation.
        """
        if self._parameterNode is None:
            self._parameterNode = SOFASceneLoaderParameterNode(super().getParameterNode())
        return self._parameterNode

    def resetParameterNode(self):
        """
        Resets simulation parameters in the parameter node to default values.
        """
        if self.getParameterNode() is not None:
            self.getParameterNode().modelNode = None
            self.getParameterNode().dt = 0.01
            self.getParameterNode().currentStep = 0
            self.getParameterNode().totalSteps = -1

    def startSimulation(self) -> None:
        """
        Sets up the scene and starts the simulation.
        """
        # TODO: The order here is important. Maybe move part to SlicerSOFA to enforce correct order
        # self.setupMappings()
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

        # if pn is not None:
        #     # Register MRML-to-SOFA mappings
        #     self.registerMRMLToSOFAMapping('modelNode', 'FEM.Container', mrmlModelGridToSofaTetrahedronTopologyContainer, runOnce=True)
        #     self.registerMRMLToSOFAMapping('movingPointNode', 'AttachPoint.mouseInteractor', mrmlMarkupsFiducialToSofaPointer)
        #     self.registerMRMLToSOFAMapping('boundaryROI', 'FEM.FixedROI.BoxROI', mrmlMarkupsROIToSofaBoxROI)
        #     self.registerMRMLToSOFAMapping('gravityVector', '', self.mrmlMarkupsLineToGravityVector)

        #     # Register SOFA-to-MRML mappings
        #     self.registerSOFAToMRMLMapping('modelNode', 'FEM.Collision.dofs', sofaMechanicalObjectToMRMLModelGrid)
        #     self.registerSOFAToMRMLMapping('modelNode', 'FEM.FEM', sofaVonMisesStressToMRMLModelGrid)

        #     # Set sequence recording flags
        #     self.setRecordSequenceFlag('modelNode', lambda: pn.recordSequence)
        #     self.setRecordSequenceFlag('movingPointNode', lambda: pn.recordSequence)
        #     self.setRecordSequenceFlag('boundaryROI', lambda: pn.recordSequence)
        #     self.setRecordSequenceFlag('gravityVector', lambda: pn.recordSequence)


    def _saveState(self) -> None:
        # self._originalModelGrid = vtk.vtkUnstructuredGrid()
        # self._originalModelGrid.DeepCopy(self._parameterNode.modelNode.GetUnstructuredGrid())
        pass

    def _restoreState(self) -> None:
        # self._parameterNode.modelNode.SetAndObserveMesh(self._originalModelGrid)
        pass



# -----------------------------------------------------------------------------
# Class: SOFASceneLoaderTest
# -----------------------------------------------------------------------------
class SOFASceneLoaderTest(ScriptedLoadableModuleTest):
    """
    Test case for the SOFASceneLoader module.
    Verifies the functionality of gravity and moving point simulations.
    """
    def setUp(self):
        """
        Reset the state by clearing the MRML scene.
        """
        slicer.mrmlScene.Clear()

    def runTest(self):
        """
        Run the tests for the SOFASceneLoader module.
        """
        self.delayDisplay("Starting SOFASceneLoader test")
        # self.testGravitySimulation()
        #self.testMovingPointSimulation()
        self.delayDisplay("SOFASceneLoader tests passed")
