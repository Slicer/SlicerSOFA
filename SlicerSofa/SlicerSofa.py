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
from dataclasses import dataclass, field
import inspect
from typing import get_type_hints
from enum import Enum

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

from slicer import vtkMRMLScalarVolumeNode

from SofaEnvironment import *

# -----------------------------------------------------------------------------
# Decorator: RunOnce
# -----------------------------------------------------------------------------
def RunOnce(func):
    """
    Decorator that marks a function to be executed only once.

    Attributes:
        runOnce (bool): Flag indicating the function should run only once.
    """
    func.runOnce = True
    return func

# -----------------------------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------------------------
def arrayFromMarkupsROIPoints(roiNode):
    """
    Utility function to return RAS (Right-Anterior-Superior) boundaries from a vtkMRMLMarkupsROINode.

    Args:
        roiNode (vtkMRMLMarkupsROINode): The ROI node from which to extract boundaries.

    Returns:
        list: A list containing [R_min, A_min, S_min, R_max, A_max, S_max].
              Returns [0.0, 0.0, 0.0, 0.0, 0.0, 0.0] if roiNode is None.
    """
    if roiNode is None:
        return [0.0] * 6

    center = [0] * 3
    roiNode.GetCenter(center)
    size = roiNode.GetSize()

    # Calculate min and max RAS bounds
    R_min = center[0] - size[0] / 2
    R_max = center[0] + size[0] / 2
    A_min = center[1] - size[1] / 2
    A_max = center[1] + size[1] / 2
    S_min = center[2] - size[2] / 2
    S_max = center[2] + size[2] / 2

    return [R_min, A_min, S_min, R_max, A_max, S_max]

def arrayVectorFromMarkupsLinePoints(lineNode):
    """
    Utility function to return the vector from a vtkMRMLMarkupsLineNode.

    Args:
        lineNode (vtkMRMLMarkupsLineNode): The line node from which to extract the vector.

    Returns:
        list: A list containing the vector components [x, y, z].
              Returns [0.0, 0.0, 0.0] if lineNode is None.
    """
    if lineNode is None:
        return [0.0] * 3

    # Calculate vector direction and normalize
    p1 = lineNode.GetNthControlPointPosition(0)
    p2 = lineNode.GetNthControlPointPosition(1)
    return [p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2]]

# -----------------------------------------------------------------------------
# Class: NodeMapper
# -----------------------------------------------------------------------------
class NodeMapper:
    """
    Class responsible for defining a mapping between a node in the MRML scene and a node in SOFA.
    It supports mappings for data flow between SOFA and MRML, including optional sequence recording.
    """

    def __init__(self, nodeName=None, sofaMapping=None, mrmlMapping=None, recordSequence=None):
        """
        Initializes the NodeMapper with optional SOFA and MRML mapping functions.

        Args:
            nodeName (str, optional): The name of the node in the SOFA scene. Defaults to None.
            sofaMapping (callable, optional): Function to map data to SOFA. Defaults to None.
            mrmlMapping (callable, optional): Function to map data to MRML. Defaults to None.
            recordSequence (bool, optional): Whether to record this node in a sequence. Defaults to False.

        Raises:
            ValueError: If sofaMapping or mrmlMapping is provided but not callable.
        """
        # Validate that mappings are callable, or raise an error
        if sofaMapping is not None and not callable(sofaMapping):
            raise ValueError("sofaMapping is not callable")
        if mrmlMapping is not None and not callable(mrmlMapping):
            raise ValueError("mrmlMapping is not callable")
        if recordSequence is not None and not callable(recordSequence):
            raise ValueError("recordSequence is not callable")

        # Initialization of attributes
        self.nodeName = nodeName               # Name of the node in the SOFA scene
        self.type = None                       # Placeholder for datatype (inferred dynamically)
        self.sofaMapping = sofaMapping         # Function to transfer data to SOFA
        self.mrmlMapping = mrmlMapping         # Function to transfer data to MRML
        self.recordSequence = recordSequence   # Whether to record this node in a sequence
        self.sofaMappingHasRun = False         # Status if sofaMapping has already been executed
        self.mrmlMappingHasRun = False         # Status if mrmlMapping has already been executed

    def __get__(self, instance, owner):
        """
        Descriptor method for accessing the value of the node.

        Args:
            instance: The instance accessing the descriptor.
            owner: The owner class.

        Returns:
            The current value of the node.
        """
        return self.value

    def __set__(self, instance, value):
        """
        Sets the value of the node if it is a string.

        Args:
            instance: The instance setting the value.
            value (str): The value to set.

        Raises:
            ValueError: If the value is not a string.
        """
        if not isinstance(value, str):
            raise ValueError(f"Expected type {str}, got {type(value)}")
        self.value = value

    def infer_type(self, ownerClass, fieldName):
        """
        Infers the data type of the field based on the annotations in the owner class.

        Args:
            ownerClass (class): The class owning this NodeMapper.
            fieldName (str): The name of the field.
        """
        type_hints = get_type_hints(ownerClass)
        self.type = type_hints.get(fieldName)

# -----------------------------------------------------------------------------
# Decorator: SofaParameterNodeWrapper
# -----------------------------------------------------------------------------
def SofaParameterNodeWrapper(cls):
    """
    Decorator function that wraps a class with additional SOFA-specific attributes and type checking,
    using `parameterNodeWrapper` and dataclasses.

    Args:
        cls (class): The class to wrap.

    Returns:
        class: The wrapped class with additional SOFA-specific functionality.
    """
    sofaNodeMappers = {}  # Store instances of NodeMapper for the class

    def __checkAndCreate__(cls, var, expectedType, defaultValue):
        """
        Internal function to check if a variable is of the expected type; if not, it sets a default value.

        Args:
            cls (class): The class being checked.
            var (str): The variable name.
            expectedType (type): The expected type of the variable.
            defaultValue: The default value to set if the variable is not present.

        Raises:
            TypeError: If the existing variable is not of the expected type.
        """
        if hasattr(cls, var):
            current_value = getattr(cls, var)
            if not isinstance(current_value, expectedType):
                raise TypeError(f"sofaParameterNodeWrapper: {var} should be of type {expectedType.__name__}, "
                                f"got {type(current_value).__name__}")
        else:
            # Sets attribute and creates a type annotation if not present
            setattr(cls, var, defaultValue)
            if not hasattr(cls, '__annotations__'):
                cls.__annotations__ = {}
            cls.__annotations__[var] = expectedType

    # Sets default simulation control parameters in the class
    __checkAndCreate__(cls, 'dt', float, 0.01)
    __checkAndCreate__(cls, 'totalSteps', int, -1)
    __checkAndCreate__(cls, 'currentStep', int, 0)
    __checkAndCreate__(cls, 'isSimulationRunning', bool, False)
    __checkAndCreate__(cls, 'sofaParameterNodeWrapped', bool, True)

    # Infers types for SOFA node parameters based on NodeMapper instances and stores them
    for fieldName, sofa_node in cls.__dict__.items():
        if isinstance(sofa_node, NodeMapper):
            sofa_node.infer_type(cls, fieldName)
            sofaNodeMappers[fieldName] = sofa_node

    # Wrap the class using `parameterNodeWrapper` and `dataclass`
    wrapped_cls = parameterNodeWrapper(cls)
    wrapped_cls = dataclass(wrapped_cls)

    # Attach the sofaNodeMappers dictionary to the final wrapped class
    setattr(wrapped_cls, '__sofaNodeMappers__', sofaNodeMappers)

    return wrapped_cls

# -----------------------------------------------------------------------------
# Class: SlicerSofa
# -----------------------------------------------------------------------------
class SlicerSofa(ScriptedLoadableModule):
    """
    Base class for the Slicer module using ScriptedLoadableModule, configuring metadata.
    """
    def __init__(self, parent):
        """
        Initialize the module with metadata.

        Args:
            parent (ScriptedLoadableModuleWidget): The parent object.
        """
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = _("Slicer Sofa")  # Module title
        self.parent.categories = [translate("qSlicerAbstractCoreModule", "")]  # Module category
        self.parent.dependencies = []  # Module dependencies
        self.parent.contributors = [
            "Rafael Palomar (Oslo University Hospital / NTNU, Norway)",
            "Paul Baksic (INRIA, France)",
            "Steve Pieper (Isomics, Inc., USA)",
            "Andras Lasso (Queen's University, Canada)",
            "Sam Horvath (Kitware, Inc., USA)",
            "Jean Christophe Fillion-Robin (Kitware, Inc., USA)"
        ]
        self.parent.helpText = _("""
This module supports SOFA simulations. See documentation <a href="https://github.com/RafaelPalomar/Slicer-SOFA">here</a>.
""")
        self.parent.acknowledgementText = _("""
Funded by Oslo University Hospital
""")
        parent.hidden = True  # Hides this module from the Slicer module list

# -----------------------------------------------------------------------------
# Class: SlicerSofaLogic
# -----------------------------------------------------------------------------
class SlicerSofaLogic(ScriptedLoadableModuleLogic):
    """
    Implements the computation logic for the Slicer SOFA module, handling the simulation lifecycle.
    """

    def __init__(self, createSceneFunction=None) -> None:
        """
        Initialize the logic with an optional scene creation function.

        Args:
            createSceneFunction (callable, optional): Function to create the SOFA scene. Defaults to None.
        """
        super().__init__()
        self._createSceneFunction = createSceneFunction
        self._sceneUp = False
        self._rootNode = None
        self._parameterNode = None

    def __checkParameterNode__(self, parameterNode):
        """
        Ensures the provided parameterNode has the expected SOFA wrapping.

        Args:
            parameterNode: The parameter node to check.

        Raises:
            ValueError: If parameterNode is None or not properly wrapped.
        """
        if parameterNode is None:
            raise ValueError("parameterNode can't be None")
        if not getattr(parameterNode, 'sofaParameterNodeWrapped', False):
            raise ValueError("parameterNode is not a valid parameterNode wrapped by the sofaParameterNodeWrapper")

    def setupScene(self, parameterNode, rootNode):
        """
        Initializes the SOFA simulation scene with parameter and root nodes.

        Args:
            parameterNode: The parameter node containing simulation parameters.
            rootNode (Sofa.Core.Node): The root node of the SOFA scene.

        Raises:
            ValueError: If rootNode is not a valid Sofa.Core.Node.
        """
        if self._sceneUp:
            return
        self.__checkParameterNode__(parameterNode)
        self._parameterNode = parameterNode
        if not isinstance(rootNode, Sofa.Core.Node):
            raise ValueError("rootNode is not a valid Sofa.Core.Node root node")
        self._rootNode = rootNode

        self.__updateSofa__()
        Sofa.Simulation.init(self._rootNode)
        self._sceneUp = True

    def startSimulation(self) -> None:
        """
        Starts the simulation by setting up the scene, resetting run-once flags,
        initializing sequence recording, and marking the simulation as running.
        """
        if self._parameterNode is None:
            raise ValueError("Parameter node has not been initialized.")
        if self._rootNode is None:
            if self._createSceneFunction is None:
                raise ValueError("No scene creation function provided.")
            self._rootNode = self._createSceneFunction()
        self.resetRunOnceFlags()
        self.setupScene(self._parameterNode, self._rootNode)
        self._parameterNode.currentStep = 0
        self._parameterNode.isSimulationRunning = True
        self.setupSequenceRecording()
        self._parameterNode.Modified()
        self.onSimulationStarted()

    def stopSimulation(self) -> None:
        """
        Stops the simulation and halts sequence recording.
        """
        if self._sceneUp:
            self._parameterNode.isSimulationRunning = False
            self.stopSequenceRecording()
            self.onSimulationStopped()

    def onSimulationStarted(self):
        """
        Hook for module-specific logic when the simulation starts.
        Can be overridden in subclasses.
        """
        pass

    def onSimulationStopped(self):
        """
        Hook for module-specific logic when the simulation stops.
        Can be overridden in subclasses.
        """
        pass

    def simulationStep(self) -> None:
        """
        Executes a single simulation step, updating the simulation and MRML scenes.
        """
        if not self._sceneUp or not self._parameterNode.isSimulationRunning:
            return

        self.__updateSofa__()

        if self._parameterNode.currentStep < self._parameterNode.totalSteps or self._parameterNode.totalSteps < 0:
            Sofa.Simulation.animate(self._rootNode, self._parameterNode.dt)
            self._parameterNode.currentStep += 1
        else:
            self._parameterNode.isSimulationRunning = False

        self.__updateMRML__()

    def resetRunOnceFlags(self):
        """
        Resets the 'runOnce' flags for all NodeMappers to allow their mapping functions to execute again.
        """
        if self._parameterNode is None:
            raise ValueError("Parameter node has not been initialized.")
        sofaNodeMappers = self._parameterNode.__class__.__sofaNodeMappers__
        for sofaNodeMapper in sofaNodeMappers.values():
            sofaNodeMapper.sofaMappingHasRun = False
            sofaNodeMapper.mrmlMappingHasRun = False

    def initializeParameterNode(self):
        """
        Initializes the parameter node by retrieving it and resetting its parameters.
        """
        if self._parameterNode is None:
            self._parameterNode = self.getParameterNode()
        self.resetParameterNode()

    def resetParameterNode(self):
        """
        Resets simulation parameters in the parameter node to default values.
        """
        if self._parameterNode:
            self._parameterNode.currentStep = 0
            self._parameterNode.isSimulationRunning = False
            # Reset other parameters as needed
            self.onParameterNodeReset()

    def onParameterNodeReset(self):
        """
        Hook for module-specific parameter node reset logic.
        Can be overridden in subclasses.
        """
        pass

    def clean(self) -> None:
        """
        Cleans up the simulation resources by unloading the SOFA root node.
        """
        if self._sceneUp and self._rootNode:
            Sofa.Simulation.unload(self._rootNode)
            self._rootNode = None
            self._sceneUp = False
            self._parameterNode.isSimulationRunning = False

    @property
    def rootNode(self):
        """
        Property to access the SOFA root node.

        Returns:
            Sofa.Core.Node: The root node of the SOFA scene.
        """
        return self._rootNode

    def setupSequenceRecording(self):
        """
        Sets up sequence recording for any nodes specified with recordSequence=True.
        """
        sofaNodeMappers = self._parameterNode.__class__.__sofaNodeMappers__

        for fieldName, sofaNodeMapper in sofaNodeMappers.items():
            if sofaNodeMapper.recordSequence(self._parameterNode):
                mrmlNode = getattr(self._parameterNode, fieldName)
                if mrmlNode is not None:
                    sequenceNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLSequenceNode', mrmlNode.GetName() + "_Sequence")
                    browserNode = self._getOrCreateSequenceBrowserNode()
                    browserNode.AddSynchronizedSequenceNodeID(sequenceNode.GetID())
                    browserNode.AddProxyNode(mrmlNode, sequenceNode, False)
                    browserNode.SetRecording(sequenceNode, True)

        if hasattr(self, '_sequenceBrowserNode') and self._sequenceBrowserNode:
            self._sequenceBrowserNode.SetRecordingActive(True)

    def stopSequenceRecording(self):
        """
        Stops any active sequence recording if a sequence browser node exists.
        """
        if hasattr(self, '_sequenceBrowserNode') and self._sequenceBrowserNode is not None:
            self._sequenceBrowserNode.SetRecordingActive(False)

    def _getOrCreateSequenceBrowserNode(self):
        """
        Retrieves or creates a sequence browser node, used for managing node recording.

        Returns:
            vtkMRMLSequenceBrowserNode: The sequence browser node.
        """
        if not hasattr(self, '_sequenceBrowserNode') or self._sequenceBrowserNode is None:
            self._sequenceBrowserNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLSequenceBrowserNode', "SOFA Simulation Browser")
            self._sequenceBrowserNode.SetPlaybackActive(False)
        return self._sequenceBrowserNode

    def __updateSofa__(self) -> None:
        """
        Updates SOFA nodes based on the sofaMapping function in each NodeMapper.
        """
        sofaNodeMappers = self._parameterNode.__class__.__sofaNodeMappers__

        for fieldName, sofaNodeMapper in sofaNodeMappers.items():
            sofaMapping = sofaNodeMapper.sofaMapping
            name = sofaNodeMapper.nodeName

            if sofaMapping:
                runOnce = getattr(sofaMapping, 'runOnce', False)
                if runOnce and sofaNodeMapper.sofaMappingHasRun:
                    continue  # Skip if already run
                node = self._rootNode[name] if name else self._rootNode
                sofaMapping(self._parameterNode, node)
                if runOnce:
                    sofaNodeMapper.sofaMappingHasRun = True

    def __updateMRML__(self) -> None:
        """
        Updates MRML nodes based on the mrmlMapping function in each NodeMapper.
        """
        sofaNodeMappers = self._parameterNode.__class__.__sofaNodeMappers__

        for fieldName, sofaNodeMapper in sofaNodeMappers.items():
            mrmlMapping = sofaNodeMapper.mrmlMapping
            name = sofaNodeMapper.nodeName

            if mrmlMapping:
                runOnce = getattr(mrmlMapping, 'runOnce', False)
                if runOnce and sofaNodeMapper.mrmlMappingHasRun:
                    continue  # Skip if already run
                node = self._rootNode[name] if name else self._rootNode
                mrmlMapping(self._parameterNode, node)
                if runOnce:
                    sofaNodeMapper.mrmlMappingHasRun = True
