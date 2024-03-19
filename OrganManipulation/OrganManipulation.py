import logging
import os
from typing import Annotated, Optional
import qt
import vtk
import random
import time
import uuid

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

from slicer import vtkMRMLModelNode
from slicer import vtkMRMLIGTLConnectorNode

#
# OrganManipulation
#


class OrganManipulation(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = _("Ogan Manipulation")
        self.parent.categories = [translate("qSlicerAbstractCoreModule", "Simulation.SOFA")]
        self.parent.dependencies = []  # TODO: add here list of module names that this module requires
        self.parent.contributors = ["Rafael Palomar (Oslo University Hospital), Paul Baksic (INRIA), Steve Pieper (Isomics, inc.), Andras Lasso (Queen's University), Sam Horvath (Kitware, inc.)"]
        # TODO: update with short description of the module and a link to online module documentation
        # _() function marks text as translatable to other languages
        self.parent.helpText = _("""""")
        # TODO: replace with organization, grant and thanks
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

    slicerSOFADataURL= 'https://github.com/rafaelpalomar/SlicerSOFATestingData/releases/download/'

    # To ensure that the source code repository remains small (can be downloaded and installed quickly)
    # it is recommended to store data sets that are larger than a few MB in a Github release.

    # Right lung low poly tetrahedral mesh dataset
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        category='SOFA',
        sampleName='RightLungLowTetra',
        thumbnailFileName=os.path.join(iconsPath, 'RightLungLowTetra.png'),
        uris=slicerSOFADataURL+ 'SHA256/a35ce6ca2ae565fe039010eca3bb23f5ef5f5de518b1c10257f12cb7ead05c5d',
        fileNames='RightLungLowTetra.vtk',
        checksums='SHA256:a35ce6ca2ae565fe039010eca3bb23f5ef5f5de518b1c10257f12cb7ead05c5d',
        nodeNames='RightLung',
        loadFileType='ModelFile'
    )

#
# OrganManipulationParameterNode
#

@parameterNodeWrapper
class OrganManipulationParameterNode:
    """
    The parameters needed by module.
    """
    modelNode: vtkMRMLModelNode
    modelNodeFileName: str
    serverPort: int

#
# OrganManipulationWidget
#

class OrganManipulationWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    """Uses ScriptedLoadableModuleWidget base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent=None) -> None:
        """Called when the user opens the module the first time and the widget is initialized."""
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)  # needed for parameter node observation
        self.logic = None
        self._parameterNode = None
        self._parameterNodeGuiTag = None

        # These two variables are part of a workaround to solve a race condition
        # between the parameter node update and the gui update (e.g., (1) when a dynamic
        # property is defined in QT Designer and (2) a manual QT connect is defined and (3)
        # the corresponding slot function needs an updated parameter node). This workaround
        # uses a timer to make the slot function go last in the order of events.
        self._timer = qt.QTimer()
        self._timeout = False

    def setup(self) -> None:
        """Called when the user opens the module the first time and the widget is initialized."""
        ScriptedLoadableModuleWidget.setup(self)

        # Load widget from .ui file (created by Qt Designer).
        # Additional widgets can be instantiated manually and added to self.layout.
        uiWidget = slicer.util.loadUI(self.resourcePath("UI/OrganManipulation.ui"))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)

        # Set scene in MRML widgets. Make sure that in Qt designer the top-level qMRMLWidget's
        # "mrmlSceneChanged(vtkMRMLScene*)" signal in is connected to each MRML widget's.
        # "setMRMLScene(vtkMRMLScene*)" slot.
        uiWidget.setMRMLScene(slicer.mrmlScene)

        # Create logic class. Logic implements all computations that should be possible to run
        # in batch mode, without a graphical user interface.
        self.logic = OrganManipulationLogic()

        # Connections

        # These connections ensure that we update parameter node when scene is closed
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)

        self.ui.startSimulationPushButton.connect("clicked()", self.onStartSimulation)

        # Make sure parameter node is initialized (needed for module reload)
        self.initializeParameterNode()

    def cleanup(self) -> None:
        """Called when the application closes and the module widget is destroyed."""
        self.removeObservers()

    def enter(self) -> None:
        """Called each time the user opens this module."""
        # Make sure parameter node exists and observed
        self.initializeParameterNode()

    def exit(self) -> None:
        """Called each time the user opens a different module."""
        # Do not react to parameter node changes (GUI will be updated when the user enters into the module)
        if self._parameterNode:
            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
            self._parameterNodeGuiTag = None

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
        """Ensure parameter node exists and observed."""
        # Parameter node stores all user choices in parameter values, node selections, etc.
        # so that when the scene is saved and reloaded, these settings are restored.

        self.setParameterNode(self.logic.getParameterNode())
        self._parameterNode.modelNode = None
        self._parameterNode.serverPort = 0
        self._parameterNode.modelNodeFileName = slicer.app.temporaryPath + '/' + str(uuid.uuid4()) + ".vtk"

        # # Select default input nodes if nothing is selected yet to save a few clicks for the user
        if not self._parameterNode.modelNode:
            firstModelNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLModelNode")
            if firstModelNode:
                self._parameterNode.modelNode = firstModelNode

    def setParameterNode(self, inputParameterNode: Optional[OrganManipulationParameterNode]) -> None:
        """
        Set and observe parameter node.
        Observation is needed because when the parameter node is changed then the GUI must be updated immediately.
        """
        if self._parameterNode:
            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
        self._parameterNode = inputParameterNode
        if self._parameterNode:
            # Note: in the .ui file, a Qt dynamic property called "SlicerParameterName" is set on each
            # ui element that needs connection.
            self._parameterNodeGuiTag = self._parameterNode.connectGui(self.ui)

    def onStartSimulation(self):
       self.logic.startSimulation()


#
# OrganManipulationLogic
#


class OrganManipulationLogic(ScriptedLoadableModuleLogic):
    """This class should implement all the actual
    computation done by your module.  The interface
    should be such that other python code can import
    this class and make use of the functionality without
    requiring an instance of the Widget.
    Uses ScriptedLoadableModuleLogic base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    # SOFAIGTL_PORT = 18944
    # SIMULATION_TPL_FILENAME = "OrganManipulation.pyscn.tpl"
    # SIMULATION_FILENAME = "OrganManipulation.pyscn"

    def __init__(self) -> None:
        """Called when the logic class is instantiated. Can be used for initializing member variables."""
        ScriptedLoadableModuleLogic.__init__(self)
        self.connectionStatus = 0
        self._parameterNode = self.getParameterNode()
        self._controller = None
        self.igtlConnectorNode = None
        self.SOFAReceiverModelNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLModelNode')
        self.SOFAReceiverModelNode.SetName('SOFAMesh')
        self.SOFAReceiverModelNode.AddObserver(slicer.vtkMRMLModelNode.PolyDataModifiedEvent, self.onModelNodeModified)

    def getParameterNode(self):
        return OrganManipulationParameterNode(super().getParameterNode())

    def startServer(self, port=0, retries=3) -> int:
        """
        Attempts to start an OpenIGTLink server on a specified port or, if the port is 0, on a random available port.

    This function tries to start an OpenIGTLink server using a vtkMRMLIGTLConnectorNode. If the specified port is 0,
        the function will choose a random port between 1025 and 65535 for each attempt. It will make up to 'retries' attempts
        to start the server. After each unsuccessful attempt (if the server did not start), it waits for 1 second before
        trying again. The process of starting the server includes creating a new vtkMRMLIGTLConnectorNode in the scene,
        setting its type to server with the chosen port, and attempting to start it. If the port is not provided retry attempts will be
        made on another random port

    Parameters:
        - port (int, optional): The port number on which to start the OpenIGTLink server. If 0, a random port is used. Defaults to 0.
        - retries (int, optional): The number of attempts to make if the server fails to start on the first try. Defaults to 3.

    Returns:
        int: The port number on which the OpenIGTLink server was successfully started, or 0 if the server could not be
         started after the specified number of retries.
        """
        self.igtlConnectorNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLIGTLConnectorNode")
        status = 0
        attempt = 0

        while status == 0 and attempt < retries:
            _port = random.randint(1025, 65535) if port == 0 else port
            logging.debug("Starting OIGTLink server on port " + str(_port) + " (attempt " + str(attempt) + ")")
            self.igtlConnectorNode.SetTypeServer(_port)
            status = self.igtlConnectorNode.Start()
            attempt=+1
            time.sleep(1)

        if status != 0:
            logging.debug("OIGTLink server started on port " + str(_port) + " (attempt " + str(attempt) + ")")
            return _port
        else:
            logging.debug("OIGTLink failed to start on port " + str(_port) + " (attempt )" + str(attempt) + ")")
            return 0

    def startSimulation(self) -> None:
        import Simulations.SOFASimulationSingle as single
        self._parameterNode.serverPort = self.startServer()
        self.igtlConnectorNode.RegisterIncomingMRMLNode(self.SOFAReceiverModelNode)
        if self._parameterNode.modelNode is not None and self._parameterNode.serverPort != 0:
            slicer.util.saveNode(self._parameterNode.modelNode, self._parameterNode.modelNodeFileName)
            if self._controller is None:
                self._controller = single.SimulationController(self._parameterNode)
            self._controller.start()

    def onModelNodeModified(self, caller, event):
        if self._parameterNode.modelNode.GetUnstructuredGrid() is not None:
            self._parameterNode.modelNode.GetUnstructuredGrid().SetPoints(caller.GetPolyData().GetPoints())
        elif self._parameterNode.modelNode.GetPolyData() is not None:
            self._parameterNode.modelNode.GetPolyData().SetPoints(caller.GetPolyData().GetPoints())

#
# OrganManipulationTest
#


class OrganManipulationTest(ScriptedLoadableModuleTest):
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
        pass
