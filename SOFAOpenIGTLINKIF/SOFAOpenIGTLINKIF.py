import logging
import os
import qt
from typing import Annotated, Optional
import pathlib

import vtk

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
# SOFAOpenIGTLINKIF
#

class SOFAOpenIGTLINKIF(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = _("SOFAOpenIGTLINKIF")
        # TODO: set categories (folders where the module shows up in the module selector)
        self.parent.categories = [translate("qSlicerAbstractCoreModule", "Simulation")]
        self.parent.dependencies = ["OpenIGTLinkIF"]
        self.parent.contributors = ["Rafael Palomar (Oslo University Hospital), Paul Baksic (INRIA), Steve Pieper (Isomics, inc.), Andras Lasso (Queen's University), Sam Horvath (Kitware, inc.)"]
        # TODO: update with short description of t
        # _() function marks text as translatable to other languages
        self.parent.helpText = _("""
This extension receives and render mesh data from a SOFA client using OpenIGTLink
See more information in <a href="https://github.com/RafaelPalomar/Slicer-Sofa">module documentation</a>.
""")
        # TODO: replace with organization, grant and thanks
        self.parent.acknowledgementText = _("""
This file was originally developed by Jean-Christophe Fillion-Robin, Kitware Inc., Andras Lasso, PerkLab,
and Steve Pieper, Isomics, Inc. and was partially funded by NIH grant 3P41RR013218-12S1.
""")

        # Additional initialization step after application startup is complete
        slicer.app.connect("startupCompleted()", registerSampleData)

#
# Register sample data sets in Sample Data module
#

def registerSampleData():
    """Add data sets to Sample Data module."""
    # It is always recommended to provide sample data for users to make it easy to try the module,
    # but if no sample data is available then this method (and associated startupCompeted signal connection) can be removed.

    import SampleData

    iconsPath = os.path.join(os.path.dirname(__file__), "Resources/Icons")

    # To ensure that the source code repository remains small (can be downloaded and installed quickly)
    # it is recommended to store data sets that are larger than a few MB in a Github release.

    # SOFAOpenIGTLINKIF1
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        # Category and sample name displayed in Sample Data module
        category="SOFAOpenIGTLINKIF",
        sampleName="SOFAOpenIGTLINKIF1",
        # Thumbnail should have size of approximately 260x280 pixels and stored in Resources/Icons folder.
        # It can be created by Screen Capture module, "Capture all views" option enabled, "Number of images" set to "Single".
        thumbnailFileName=os.path.join(iconsPath, "SOFAOpenIGTLINKIF1.png"),
        # Download URL and target file name
        uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95",
        fileNames="SOFAOpenIGTLINKIF1.nrrd",
        # Checksum to ensure file integrity. Can be computed by this command:
        #  import hashlib; print(hashlib.sha256(open(filename, "rb").read()).hexdigest())
        checksums="SHA256:998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95",
        # This node name will be used when the data set is loaded
        nodeNames="SOFAOpenIGTLINKIF1",
    )

    # SOFAOpenIGTLINKIF2
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        # Category and sample name displayed in Sample Data module
        category="SOFAOpenIGTLINKIF",
        sampleName="SOFAOpenIGTLINKIF2",
        thumbnailFileName=os.path.join(iconsPath, "SOFAOpenIGTLINKIF2.png"),
        # Download URL and target file name
        uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97",
        fileNames="SOFAOpenIGTLINKIF2.nrrd",
        checksums="SHA256:1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97",
        # This node name will be used when the data set is loaded
        nodeNames="SOFAOpenIGTLINKIF2",
    )


#
# SOFAOpenIGTLINKIFParameterNode
#


@parameterNodeWrapper
class SOFAOpenIGTLINKIFParameterNode:
    """
    The parameters needed by module.
    """
    modelNode: vtkMRMLModelNode
    igtlConnectorNode: vtkMRMLIGTLConnectorNode
    savingDirectory: pathlib.Path
    serverUp: bool

#
# SOFAOpenIGTLINKIFWidget
#


class SOFAOpenIGTLINKIFWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
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
        uiWidget = slicer.util.loadUI(self.resourcePath("UI/SOFAOpenIGTLINKIF.ui"))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)

        # Set scene in MRML widgets. Make sure that in Qt designer the top-level qMRMLWidget's
        # "mrmlSceneChanged(vtkMRMLScene*)" signal in is connected to each MRML widget's.
        # "setMRMLScene(vtkMRMLScene*)" slot.
        uiWidget.setMRMLScene(slicer.mrmlScene)

        # Create logic class. Logic implements all computations that should be possible to run
        # in batch mode, without a graphical user interface.
        self.logic = SOFAOpenIGTLINKIFLogic()

        # Connections

        # These connections ensure that we update parameter node when scene is closed
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)

        ### SAVE SECTION

        self.ui.SOFAMRMLModelNodeComboBox.connect('currentNodeChanged(vtkMRMLNode*)', self.onModelNodeComboBoxChanged)
        self.ui.saveModelButton.connect('clicked(bool)', self.onSaveModelButtonPushed)

        #### OpenIGTLink Connection SECTION
        self.ui.serverActiveCheckBox.connect("clicked()", self.onActivateOpenIGTLinkConnectionClicked)

        # Make sure parameter node is initialized (needed for module reload)
        self.initializeParameterNode()
        self.initializeGUI() # This is an addition to avoid initializing parameter node before connections

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

        # Select default input nodes if nothing is selected yet to save a few clicks for the user
        if not self._parameterNode.modelNode:
            firstModelNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLModelNode")
            if firstModelNode:
                self._parameterNode.modelNode = firstModelNode

    def setParameterNode(self, inputParameterNode: Optional[SOFAOpenIGTLINKIFParameterNode]) -> None:
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

    def initializeGUI(self):
        """
            initailize the save directory using settings
        """
        self.ui.SOFADataDirectoryButton.directory = self.logic.getDefaultSOFASavingDirectory()

    def onActivateOpenIGTLinkConnectionClicked(self):
        """
        Run processing when user clicks on the OpenIGTLink checkbox.
        """
        if self._timeout == False:
            self._timeout = True
            self._timer.singleShot(0, self.onActivateOpenIGTLinkConnectionClicked)
            return

        parameterNode = self.logic.getParameterNode()

        if parameterNode.serverUp is True:
            self.logic.StartOIGTLConnection() # Start connection
        else:
            self.logic.StopOIGTLConnection() # Stop connection

        if self.logic.getConnectionStatus() == 1:
            self.ui.OIGTLconnectionLabel.text = "OpenIGTLink server - ACTIVE" # Update displaying text
        else:
            self.ui.OIGTLconnectionLabel.text = "OpenIGTLink server - INACTIVE" # Update displaying text

        self._timeout = False

    def onModelNodeComboBoxChanged(self):
        """
        Runs when the SOFA model combo box is changed
        """
        self.ui.saveModelButton.setEnabled(self._parameterNode.modelNode is not None)

    def onSaveModelButtonPushed(self):
        """
        Runs processing when user clicks "Save Data" button.
        """
        # with slicer.util.tryWithErrorDisplay("Failed to compute results.", waitCursor=True):
        #     # Compute output
        #     save_folder_path = self.logic.SaveData()
        #     self.ui.filesSavedLabel.setText("All files have been saved in:\n" + save_folder_path)
        slicer.util.saveNode(self._parameterNode.modelNode, self._parameterNode.savingDirectory.as_posix()+'/mesh.vtk')

# SOFAOpenIGTLINKIFLogic
#

class SOFAOpenIGTLINKIFLogic(ScriptedLoadableModuleLogic):
    """This class should implement all the actual
    computation done by your module.  The interface
    should be such that other python code can import
    this class and make use of the functionality without
    requiring an instance of the Widget.
    Uses ScriptedLoadableModuleLogic base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    SOFAIGTL_PORT = 18944

    def __init__(self) -> None:
        """Called when the logic class is instantiated. Can be used for initializing member variables."""
        ScriptedLoadableModuleLogic.__init__(self)
        self.connectionStatus = 0
        self._parameterNode = self.getParameterNode()

    def getParameterNode(self):
        return SOFAOpenIGTLINKIFParameterNode(super().getParameterNode())

    def getDefaultSOFASavingDirectory(self) -> str:
        """Returns the default SOFA Saving Directory. Current directory in this implementation."""
        return os.getcwd()

    def getPort(self) -> int:
        return SOFAOpenIGTLINKIFLogic.SOFAIGTL_PORT

    def getConnectionStatus(self) -> int:
        return self.connectionStatus

    def StartOIGTLConnection(self) -> None:
        """
        Starts OIGTL connection.
        """
        logging.debug("StartOIGTLConnection")
        parameterNode = self.getParameterNode()

        if not parameterNode.igtlConnectorNode:
             parameterNode.igtlConnectorNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLIGTLConnectorNode")
             parameterNode.igtlConnectorNode.SetName('SOFAIGTLConnector')
             SOFAReceiverModelNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLModelNode')
             SOFAReceiverModelNode.SetName('SOFAMesh')
             parameterNode.igtlConnectorNode.RegisterIncomingMRMLNode(SOFAReceiverModelNode)
             parameterNode.igtlConnectorNode.SetTypeServer(self.getPort())
             SOFAReceiverModelNode.AddObserver(slicer.vtkMRMLModelNode.PolyDataModifiedEvent, self.onModelNodeModified)

        # Check connection status
        if self.connectionStatus == 0:
            parameterNode.igtlConnectorNode.Start()
            logging.debug('Connection Successful')
            self.connectionStatus = 1
        else:
            logging.debug('ERROR: Unable to activate server')

    def StopOIGTLConnection(self) -> None:
        """
        Stops OIGTL connection.
        """
        logging.debug("StopOIGTLConnection")
        parameterNode = self.getParameterNode()
        if parameterNode.igtlConnectorNode is not None and self.connectionStatus == 1:
            parameterNode.igtlConnectorNode.Stop()
            self.connectionStatus = 0

    def onModelNodeModified(self, caller, event):
        if self._parameterNode.modelNode.GetUnstructuredGrid() is not None:
            self._parameterNode.modelNode.GetUnstructuredGrid().SetPoints(caller.GetPolyData().GetPoints())
        elif self._parameterNode.modelNode.GetPolyData() is not None:
            self._parameterNode.modelNode.GetPolyData().SetPoints(caller.GetPolyData().GetPoints())


#
# SOFAOpenIGTLINKIFTest
#

class SOFAOpenIGTLINKIFTest(ScriptedLoadableModuleTest):
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
        self.test_SOFAOpenIGTLINKIF1()

    def test_SOFAOpenIGTLINKIF1(self):
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

        # Test the module logic

        logic = SOFAOpenIGTLINKIFLogic()

        self.delayDisplay("Test passed")
