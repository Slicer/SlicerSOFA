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

# Decorator that allows a function to be marked as "run once"
def RunOnce(func):
    func.runOnce = True
    return func

#
# NodeMapper class
#
class NodeMapper:
    """
    Class responsible for defining a mapping between a node in the MRML scene and a node in SOFA.
    It supports mappings for data flow between SOFA and MRML, including optional sequence recording.
    """

    def __init__(self, nodeName=None, sofaMapping=None, mrmlMapping=None, recordSequence=False):
        # Validate that mappings are callable, or raise an error
        if sofaMapping is not None and not callable(sofaMapping):
            raise ValueError("sofaMapping is not callable")
        if mrmlMapping is not None and not callable(mrmlMapping):
            raise ValueError("mrmlMapping is not callable")

        # Initialization of attributes
        self.nodeName = nodeName               # Name of the node in the SOFA scene
        self.type = None                       # Placeholder for datatype (inferred dynamically)
        self.sofaMapping = sofaMapping         # Function to transfer data to SOFA
        self.mrmlMapping = mrmlMapping         # Function to transfer data to MRML
        self.recordSequence = recordSequence   # Whether to record this node in a sequence
        self.sofaMappingHasRun = False         # Status if sofaMapping has already been executed
        self.mrmlMappingHasRun = False         # Status if mrmlMapping has already been executed

    def __get__(self, instance, owner):
        # Descriptor method for accessing the value of the node
        return self.value

    def __set__(self, instance, value):
        # Sets the value if it is a string, otherwise raises an error
        if not isinstance(value, str):
            raise ValueError(f"Expected type {str}, got {type(value)}")
        self.value = value

    def infer_type(self, ownerClass, fieldName):
        """
        Infers the data type of the field based on the annotations in the owner class.
        """
        type_hints = get_type_hints(ownerClass)
        self.type = type_hints.get(fieldName)

#
# SofaParameterNodeWrapper decorator
#
def SofaParameterNodeWrapper(cls):
    """
    Decorator function that wraps a class with additional SOFA-specific attributes and type checking,
    using `parameterNodeWrapper` and dataclasses.
    """

    sofaNodeMappers = {}  # Store instances of NodeMapper for the class

    # Internal function to check if a variable is of the expected type; if not, it sets a default value
    def __checkAndCreate__(cls, var, expectedType, defaultValue):
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

#
# SlicerSofa
#

class SlicerSofa(ScriptedLoadableModule):
    """
    Base class for the Slicer module using ScriptedLoadableModule, configuring metadata.
    """
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = _("Slicer Sofa")
        self.parent.categories = [translate("qSlicerAbstractCoreModule", "")]
        self.parent.dependencies = []
        self.parent.contributors = [
            "Rafael Palomar (Oslo University Hospital / NTNU, Norway), Paul Baksic (INRIA, France), "
            "Steve Pieper (Isomics, Inc., USA), Andras Lasso (Queen's University, Canada), Sam Horvath (Kitware, Inc., USA), "
            "Jean Christophe Fillion-Robin (Kitware, Inc., USA)"
        ]
        self.parent.helpText = _("""
This module supports SOFA simulations. See documentation <a href="https://github.com/RafaelPalomar/Slicer-SOFA">here</a>.
""")
        self.parent.acknowledgementText = _("""
Funded by Oslo University Hospital
""")
        parent.hidden = True  # Hides this module from the Slicer module list

#
# SlicerSofaLogic
#

class SlicerSofaLogic(ScriptedLoadableModuleLogic):
    """
    Implements the computation logic for the Slicer SOFA module, handling the simulation lifecycle.
    """

    def __init__(self) -> None:
        super().__init__()
        self._sceneUp = False
        self._rootNode = None
        self._parameterNode = None

    # Ensures the provided parameterNode has the expected SOFA wrapping
    def __checkParameterNode__(self, parameterNode):
        if parameterNode is None:
            raise ValueError("parameterNode can't be None")
        if not getattr(parameterNode, 'sofaParameterNodeWrapped') or parameterNode.sofaParameterNodeWrapped is not True:
            raise ValueError("parameterNode is not a valid parameterNode wrapped by the sofaParameterNodeWrapper")

    def setupScene(self, parameterNode, rootNode):
        """
        Initializes the SOFA simulation scene with parameter and root nodes.
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
        Starts the SOFA simulation, including sequence recording if specified.
        """
        self._parameterNode.currentStep = 0
        self._parameterNode.isSimulationRunning = True
        self.setupSequenceRecording()  # Set up recording if applicable

    def stopSimulation(self) -> None:
        """
        Stops the simulation and halts sequence recording.
        """
        if self._sceneUp:
            self._parameterNode.isSimulationRunning = False
            self.stopSequenceRecording()

    def simulationStep(self) -> None:
        """
        Executes a single simulation step, updating the simulation and MRML scenes.
        """
        if not self._sceneUp or not self._parameterNode.isSimulationRunning:
            slicer.util.errorDisplay("Can't advance the simulation forward. Simulation is not running.")
            return

        self.__updateSofa__()

        if self._parameterNode.currentStep < self._parameterNode.totalSteps:
            Sofa.Simulation.animate(self._rootNode, self._parameterNode.dt)
            self._parameterNode.currentStep += 1
        elif self._parameterNode.totalSteps < 0:
            Sofa.Simulation.animate(self._rootNode, self._parameterNode.dt)
        else:
            self._parameterNode.isSimulationRunning = False

        self.__updateMRML__()

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
        return self._rootNode

    def setupSequenceRecording(self):
        """
        Sets up sequence recording for any nodes specified with recordSequence=True.
        """
        sofaNodeMappers = self._parameterNode.__class__.__sofaNodeMappers__

        for fieldName, sofaNodeMapper in sofaNodeMappers.items():
            if sofaNodeMapper.recordSequence:
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
