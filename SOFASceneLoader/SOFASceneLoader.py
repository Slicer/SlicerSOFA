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
from vtk.util.numpy_support import numpy_to_vtk
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
    sofaMechanicalObjectToMRMLModelGrid,
    sofaVonMisesStressToMRMLModelGrid,
    sofaTetrahedronTopologyToMRMLModelGrid,
    sofaMeshTopologyToMRMLModelGrid,
    sofaEdgeTopologyToMRMLModelGrid,
    sofaTriangleTopologyToMRMLModelGrid,
    sofaOglModelToMRMLModelGrid
)


def _oglModelToMRMLModelGrid(modelNode, sofaNode):
    """
    Map an OglModel to a vtkMRMLModelNode: geometry, topology and material
    colour through the shared mapping, plus this module's texture support.

    Texture handling lives here rather than in SlicerSofaUtils.Mappings
    because this module's automatic mapping is its only consumer: it is the
    one place that maps SOFA components to MRML nodes without the module
    author choosing the mapping.
    """
    sofaOglModelToMRMLModelGrid(modelNode, sofaNode)
    _applyOglModelTexture(modelNode, sofaNode)


def _applyOglModelTexture(modelNode, sofaNode):
    """
    Transfer an OglModel's texture to a vtkMRMLModelNode -- once per node.

    The component's texcoords become the mesh's TCoords array and its
    texturename is read into a hidden vector volume node wired to the display
    node's texture connection, with a white base colour so the material
    diffuse does not tint the image.  Scenes whose visual meshes carry no
    texture coordinates, or whose texture file is missing or unreadable by
    VTK, are left with the material colour the shared mapping applied.

    The texture volume is reused across mapping passes and scene reloads
    instead of accumulating hidden volumes.

    Args:
        modelNode (vtkMRMLModelNode): MRML model node to texture.
        sofaNode: SOFA OglModel component.

    Returns:
        bool: True when the model node is textured after this call.
    """
    grid = modelNode.GetUnstructuredGrid()
    if grid is None or grid.GetPointData() is None:
        return False
    if grid.GetPointData().GetTCoords() is not None:
        return True

    texcoords = sofaNode.texcoords.array()
    if texcoords is None or len(texcoords) == 0 or grid.GetNumberOfPoints() != len(texcoords):
        return False

    displayNode = modelNode.GetDisplayNode()
    texturePath = str(sofaNode.texturename.value)
    if displayNode is None or not texturePath or not os.path.isfile(texturePath):
        return False

    reader = vtk.vtkImageReader2Factory.CreateImageReader2(texturePath)
    if reader is None:
        return False
    reader.SetFileName(texturePath)
    reader.Update()

    tcoordsArray = numpy_to_vtk(num_array=texcoords, deep=True, array_type=vtk.VTK_FLOAT)
    tcoordsArray.SetName('TCoords')
    grid.GetPointData().SetTCoords(tcoordsArray)

    # Reuse this model's texture volume if it already exists, so reloading or
    # resetting a scene does not accumulate hidden volumes.  Looked up by name
    # rather than made a singleton: singletons survive vtkMRMLScene::Clear, so
    # a texture volume would outlive the scene that needed it.
    textureName = modelNode.GetName() + '_Texture'
    textureNode = slicer.mrmlScene.GetFirstNode(textureName, 'vtkMRMLVectorVolumeNode')
    if textureNode is None:
        textureNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLVectorVolumeNode', textureName)
    textureNode.SetAndObserveImageData(reader.GetOutput())
    textureNode.SetHideFromEditors(True)
    displayNode.SetTextureImageDataConnection(textureNode.GetImageDataConnection())
    displayNode.SetColor(1.0, 1.0, 1.0)
    return True


SOFA2MRML_dict = {
"MechanicalObject" : sofaMechanicalObjectToMRMLModelGrid,
"TetrahedralCorotationalFEMForceField" : sofaVonMisesStressToMRMLModelGrid,
"TetrahedronFEMForceField" : sofaVonMisesStressToMRMLModelGrid,
"MeshTopology" : sofaMeshTopologyToMRMLModelGrid,
"EdgeSetTopologyContainer" : sofaEdgeTopologyToMRMLModelGrid,
"TriangleSetTopologyContainer" : sofaTriangleTopologyToMRMLModelGrid,
"TetrahedronSetTopologyContainer" : sofaTetrahedronTopologyToMRMLModelGrid,
"OglModel" : _oglModelToMRMLModelGrid
}

# -----------------------------------------------------------------------------
# Class: SOFANodeWrapper
# -----------------------------------------------------------------------------
class SOFANodeWrapper:
    """
    Wrapper class for SOFA nodes that intercepts addObject calls and recursively wraps children.
    
    This wrapper allows triggering an internal callback function whenever addObject is called,
    while maintaining all other functionality of the original SOFA node.
    """
    def __init__(self, sofa_node, logic, path=""):
        """
        Initialize the wrapper with a SOFA node and optional path.
        
        Args:
            sofa_node: The original SOFA node to wrap
            path: The current path in the SOFA tree (e.g., "first.second")
        """
        self._sofa_node = sofa_node
        self._path = path
        self._logic = logic
    

    def getInternalSofaNode(self):
        return self._sofa_node
    
    def _getPath(self):
        """
        Get the current path of this node in the SOFA tree.
        
        Returns:
            str: The current path (e.g., "first.second")
        """
        return self._path
    
    def addObject(self, obj, *args, **kwargs):
        """
        Intercept addObject calls to trigger the internal callback function.
        
        Args:
            obj: The object being added to the SOFA node
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments
            
        Returns:
            The result of the original addObject call
        """
        sofaObj = self._sofa_node.addObject(obj, *args, **kwargs)

        # Call the internal callback function with the object being added and current path
        if obj in SOFA2MRML_dict:
            mrmlID = self._getPath().replace('.', '_')
            # If no mrml node in the parameter node
            if not hasattr(self._logic.getParameterNode(), mrmlID ) : 
                # First find out if one with correct ID already exist in mrml scene
                existingNode = slicer.mrmlScene.GetNodeByID(mrmlID)
                if existingNode:
                    modelNode = existingNode
                else:
                    # Create new node if it doesn't exist
                    modelNode = vtkMRMLModelNode()
                    modelNode.SetSingletonTag(mrmlID)
                    modelNode.SetName(mrmlID)
                    slicer.mrmlScene.AddNode(modelNode)
                    modelNode.CreateDefaultDisplayNodes()
                    modelNode.GetDisplayNode().SetVisibility(True)

                setattr(self._logic.getParameterNode(), mrmlID, modelNode)
            else:
                # Case 2: Parameter node already has the attribute
                # Get the referenced node from parameter node
                modelNode = getattr(self._logic.getParameterNode(), mrmlID)
                # If it is not in the scene then add it
                if not slicer.mrmlScene.GetNodeByID(f"vtkMRMLModelNode{mrmlID}"):
                    slicer.mrmlScene.AddNode(modelNode)

            self._logic.registerSOFAToMRMLMapping(mrmlID, f"{self._getPath()}.{sofaObj.getName()}", SOFA2MRML_dict[obj])


        # Delegate to the original node's addObject method
        return sofaObj
    
    def addChild(self, child_name, *args, **kwargs):
        """
        Override addChild to recursively wrap any new children.
        
        Args:
            child_name: Name of the child node to create
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments
            
        Returns:
            A wrapped version of the newly created child node
        """
        # Create the child using the original node's method
        child_node = self._sofa_node.addChild(child_name, *args, **kwargs)
        
        # Build the new path for the child node
        if self._path != "":
            child_path = f"{self._path}.{child_name}"
        else:
            child_path = child_name
        
        # Return a wrapped version of the child node with the updated path
        return SOFANodeWrapper(child_node, self._logic, child_path)
    
    def __getattr__(self, name):
        """
        Delegate all other method calls to the original SOFA node.
        
        This ensures that the wrapper maintains full compatibility with
        the original SOFA node interface.
        
        Args:
            name: Name of the method or attribute to access
            
        Returns:
            The method or attribute from the original SOFA node
        """
        return getattr(self._sofa_node, name)
    
    def __setattr__(self, key, value):
        if key in ["_sofa_node", "_path", "_logic"]:
            self.__dict__[key] = value
        self.__dict__["_sofa_node"].__setattr__(key, value)


# -----------------------------------------------------------------------------
# Class: SOFASceneLoaderParameterNode
# -----------------------------------------------------------------------------
@SofaParameterNodeWrapper
class SOFASceneLoaderParameterNode:
    """
    Parameter class for the soft tissue simulation.
    Defines nodes to map between SOFA and MRML scenes with recording options.
    """
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
        self.ui.startSimulationPushButton.setEnabled(not parameterNode.isSimulationRunning)
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
        self._parameterNode = None
        # Placeholder root so the logic is queryable before a file is loaded
        self._rootNode = self.createScene(self._parameterNode)


    def createScene(self, parameterNode) -> Sofa.Core.Node:
        """
        Build the SOFA scene by re-executing the loaded scene file's
        createScene() against a wrapped root node (the wrapper registers the
        automatic SOFA-to-MRML mappings).  Called by SlicerSofaLogic.setupScene.
        """
        sofa_root = Sofa.Core.Node("root")
        wrapped_root = SOFANodeWrapper(sofa_root, self)

        if self.createSceneMethod is not None:
            #This is for path handling when the scene refers to models using relative path
            cwd = os.getcwd()
            os.chdir(self.createSceneFilePath)
            self.createSceneMethod(wrapped_root)
            os.chdir(cwd)

        self.getParameterNode().Modified()

        # Return the wrapped root node
        return wrapped_root.getInternalSofaNode()
    
    def loadSimulationFile(self, widget):
        supposedPath = self.getUi().ui.simulationFileName.text
        if os.path.isabs(supposedPath) and os.path.isfile(supposedPath):
            path = supposedPath
        elif not os.path.isabs(supposedPath) and os.path.isfile(os.path.join(os.getcwd(), supposedPath)):
            path = os.path.join(os.getcwd(), supposedPath)
        else:
            path = qt.QFileDialog.getOpenFileName(widget.parent.window(),"Select SOFA scene file", os.getcwd() , "Python Files (*.py);;All Files (*)")
            self.getUi().ui.simulationFileName.setText(path)

        if not os.path.isfile(path):
            print("Input file must exist")
        else:
            #Open file
            
            moduleName = pathlib.Path(path).name.split('.')[0]
            spec=importlib.util.spec_from_file_location(moduleName,path)
            foo = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(foo)
            self.createSceneMethod = foo.createScene
            self.createSceneFilePath = pathlib.Path(path).parent

            self._resetMappings()
            self.setupScene(self.getParameterNode())
            self.__updateMRML__()
            self._parameterNode.currentStep = 0
            slicer.app.layoutManager().resetThreeDViews()


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
            self.getParameterNode().dt = 0.01
            self.getParameterNode().currentStep = 0
            self.getParameterNode().totalSteps = 0

    def startSimulation(self) -> None:
        """
        Sets up the scene and starts the simulation.
        """
        # TODO: The order here is important. Maybe move part to SlicerSOFA to enforce correct order
        # self.setupMappings()
        # 
        self._parameterNode.isSimulationRunning = True
        self.setupSequenceRecording()
        self.onSimulationStarted()
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
        #Mappings are setup in the sofa node wrapper during the scene creation

    def _resetMappings(self) -> None:
        self.sofaMappings = []
        self.mrmlMappings = []

    def _saveState(self) -> None:
        pass

    def _restoreState(self) -> None:

        self.stopSimulation()
        self._resetMappings()
        if self.createSceneMethod is not None:
            self.setupScene(self.getParameterNode())
        self._parameterNode.currentStep = 0
        self.updateSimulationProgress()
        self.__updateMRML__()
    




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


    def test_sofa_node_wrapper(self):
        """SOFANodeWrapper tracks paths and registers one mapping per object.

        NOTE: SOFA components live in plugins that must be loaded explicitly.
        Without a RequiredPlugin for Sofa.Component.StateContainer the
        MechanicalObject type is never registered and addObject fails with
        "Object type MechanicalObject<> was not created".
        """
        logic = SOFASceneLoaderLogic()
        logic.getParameterNode()

        sofa_root = Sofa.Core.Node("root")
        test_root = SOFANodeWrapper(sofa_root, logic)
        test_root.addObject("RequiredPlugin", name="Sofa.Component.StateContainer")

        child_node = test_root.addChild("child_node")
        second_child_node = child_node.addChild("second_child_node")

        sofaObj = child_node.addObject("MechanicalObject", name="toto")
        self.assertIsNotNone(sofaObj, "MechanicalObject was not created")

        # Path tracking.  _getPath() is the wrapper's own accessor; a bare
        # getPath() would fall through __getattr__ to the wrapped SOFA node.
        self.assertEqual(test_root._getPath(), "")
        self.assertEqual(child_node._getPath(), "child_node")
        self.assertEqual(second_child_node._getPath(), "child_node.second_child_node")

        # The object was added to child_node, so the flattened MRML id is
        # "child_node" -- not "child_node_second_child_node".
        parameterNode = logic.getParameterNode()
        self.assertTrue(hasattr(parameterNode, "child_node"),
                        "Parameter node should hold a model node named child_node")

        self.assertEqual(len(logic.sofaMappings), 1,
                         f"Expected exactly one mapping, got {len(logic.sofaMappings)}")

        fieldName, sofaPath, mappingFunction, runOnce = logic.sofaMappings[0]
        self.assertEqual(fieldName, "child_node")
        self.assertEqual(sofaPath, "child_node.toto")
        self.assertIs(mappingFunction, SOFA2MRML_dict["MechanicalObject"])

    def test_sofa_node_wrapper_flattens_nested_paths(self):
        """A mapping registered on a nested node flattens dots to underscores.

        This is what the original test intended to assert: an object added to
        child_node.second_child_node maps onto the MRML field
        child_node_second_child_node.
        """
        logic = SOFASceneLoaderLogic()
        logic.getParameterNode()

        sofa_root = Sofa.Core.Node("root")
        test_root = SOFANodeWrapper(sofa_root, logic)
        test_root.addObject("RequiredPlugin", name="Sofa.Component.StateContainer")

        second_child_node = test_root.addChild("child_node").addChild("second_child_node")
        second_child_node.addObject("MechanicalObject", name="dofs")

        self.assertEqual(len(logic.sofaMappings), 1)
        fieldName, sofaPath, _, _ = logic.sofaMappings[0]
        self.assertEqual(fieldName, "child_node_second_child_node")
        self.assertEqual(sofaPath, "child_node.second_child_node.dofs")
        self.assertTrue(hasattr(logic.getParameterNode(), "child_node_second_child_node"))

    def test_oglModelTextureIsAppliedOnceAndReusesItsVolume(self):
        """Texture coordinates and image transfer once; the volume is reused.

        A real OglModel cannot be built here -- it initialises OpenGL state
        and hangs without a GL context -- so the mapping is driven with a
        stand-in exposing only the three data fields it reads.  The texture
        image is generated into a temporary file so the test is hermetic.
        """
        import tempfile

        class FakeData:
            def __init__(self, value):
                self.value = value

            def array(self):
                return self.value

        class FakeOglModel:
            def __init__(self, texcoords, texturename):
                self.texcoords = FakeData(texcoords)
                self.texturename = FakeData(texturename)

        # A 4x4 RGB image on disk, read back through vtkImageReader2Factory.
        source = vtk.vtkImageCanvasSource2D()
        source.SetScalarTypeToUnsignedChar()
        source.SetNumberOfScalarComponents(3)
        source.SetExtent(0, 3, 0, 3, 0, 0)
        source.SetDrawColor(255, 128, 0)
        source.FillBox(0, 3, 0, 3)
        source.Update()
        texturePath = os.path.join(tempfile.mkdtemp(), "texture.png")
        writer = vtk.vtkPNGWriter()
        writer.SetFileName(texturePath)
        writer.SetInputData(source.GetOutput())
        writer.Write()

        # A single triangle, so three points and three texture coordinates.
        grid = vtk.vtkUnstructuredGrid()
        points = vtk.vtkPoints()
        for point in ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)):
            points.InsertNextPoint(*point)
        grid.SetPoints(points)
        triangle = vtk.vtkTriangle()
        for i in range(3):
            triangle.GetPointIds().SetId(i, i)
        grid.InsertNextCell(triangle.GetCellType(), triangle.GetPointIds())

        modelNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", "Textured")
        modelNode.SetAndObserveMesh(grid)
        modelNode.CreateDefaultDisplayNodes()

        oglModel = FakeOglModel([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], texturePath)

        self.assertTrue(_applyOglModelTexture(modelNode, oglModel))
        tcoords = modelNode.GetUnstructuredGrid().GetPointData().GetTCoords()
        self.assertIsNotNone(tcoords)
        self.assertEqual(tcoords.GetNumberOfTuples(), 3)
        self.assertIsNotNone(modelNode.GetDisplayNode().GetTextureImageDataConnection())
        volumeCount = len(slicer.util.getNodesByClass("vtkMRMLVectorVolumeNode"))
        self.assertEqual(volumeCount, 1)

        # A further mapping pass must not add a second texture volume.
        self.assertTrue(_applyOglModelTexture(modelNode, oglModel))
        self.assertEqual(len(slicer.util.getNodesByClass("vtkMRMLVectorVolumeNode")), volumeCount)

    def test_oglModelWithoutTextureCoordinatesIsLeftAlone(self):
        """A visual model with no texcoords keeps the material colour."""

        class FakeData:
            def __init__(self, value):
                self.value = value

            def array(self):
                return self.value

        class FakeOglModel:
            texcoords = FakeData([])
            texturename = FakeData("")

        grid = vtk.vtkUnstructuredGrid()
        points = vtk.vtkPoints()
        points.InsertNextPoint(0.0, 0.0, 0.0)
        grid.SetPoints(points)
        modelNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", "Plain")
        modelNode.SetAndObserveMesh(grid)
        modelNode.CreateDefaultDisplayNodes()

        self.assertFalse(_applyOglModelTexture(modelNode, FakeOglModel()))
        self.assertIsNone(modelNode.GetUnstructuredGrid().GetPointData().GetTCoords())
        self.assertEqual(len(slicer.util.getNodesByClass("vtkMRMLVectorVolumeNode")), 0)

    def runTest(self):
        """
        Run the tests for the SOFASceneLoader module.
        """
        self.delayDisplay("Starting SOFASceneLoader test")
        self.setUp()
        self.test_sofa_node_wrapper()
        self.setUp()
        self.test_sofa_node_wrapper_flattens_nested_paths()
        self.setUp()
        self.test_oglModelTextureIsAppliedOnceAndReusesItsVolume()
        self.setUp()
        self.test_oglModelWithoutTextureCoordinatesIsLeftAlone()
        #self.testMovingPointSimulation()
        self.delayDisplay("SOFASceneLoader tests passed")
