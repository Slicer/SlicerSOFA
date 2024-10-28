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

def run_once(func):
    func.run_once = True
    return func

#
# NodeMapper field class
#
class NodeMapper:

    def __init__(self, nodeName=None, sofaMapping=None, mrmlMapping=None, recordSequence=False):
        if sofaMapping is not None and not callable(sofaMapping):
            raise ValueError("sofaMapping is not callable")
        if mrmlMapping is not None and not callable(mrmlMapping):
            raise ValueError("mrmlMapping is not callable")

        self.nodeName = nodeName               # Name given in the SOFA scene
        self.type = None                       # Placeholder for datatype
        self.sofaMapping = sofaMapping         # Function used to move data to SOFA
        self.mrmlMapping = mrmlMapping         # Function used to move data to MRML
        self.recordSequence = recordSequence   # Flag indicating if the node should be recorded
        self.sofaMapping_has_run = False       # Whether the mapper runs only one time
        self.mrmlMapping_has_run = False       # Whether the mapper runs only one time

    def __get__(self, instance, owner):
        return self.value

    def __set__(self, instance, value):
        if not isinstance(value, str):
            raise ValueError(f"Expected type {str}, got {type(value)}")
        self.value = value

    def infer_type(self, owner_class, field_name):
        # Get the type hints (annotations) of the owner class
        type_hints = get_type_hints(owner_class)
        # Store the type associated with the field_name
        self.type = type_hints.get(field_name)

#
# sofaParameterNodeWrapper decorator
#

def sofaParameterNodeWrapper(cls):
    # Temporary dictionary to store NodeMapper instances
    sofa_node_mappers = {}

    def __checkAndCreate__(cls, var, expectedType, defaultValue):
        if hasattr(cls, var):
            current_value = getattr(cls, var)
            if not isinstance(current_value, expectedType):
                raise TypeError(f"sofaParameterNodeWrapper: {var} should be of type {expectedType.__name__}, "
                                f"got {type(current_value).__name__}")
        else:
            setattr(cls, var, defaultValue)
            # Ensure __annotations__ exists
            if not hasattr(cls, '__annotations__'):
                cls.__annotations__ = {}
            # Add the type annotation for the new attribute
            cls.__annotations__[var] = expectedType

    # Add simulation control variables with default values
    __checkAndCreate__(cls, 'dt', float, 0.01)
    __checkAndCreate__(cls, 'totalSteps', int, -1)
    __checkAndCreate__(cls, 'currentStep', int, 0)
    __checkAndCreate__(cls, 'isSimulationRunning', bool, False)
    __checkAndCreate__(cls, 'sofaParameterNodeWrapped', bool, True)

    # Infer types for SofaNode parameters and store them
    for field_name, sofa_node in cls.__dict__.items():
        if isinstance(sofa_node, NodeMapper):
            # Infer the type dynamically
            sofa_node.infer_type(cls, field_name)
            # Store the NodeMapper instance
            sofa_node_mappers[field_name] = sofa_node

    # Now, wrap the class using parameterNodeWrapper and dataclass
    wrapped_cls = parameterNodeWrapper(cls)
    wrapped_cls = dataclass(wrapped_cls)

    # Assign the __sofa_node_mappers__ to the final class
    setattr(wrapped_cls, '__sofa_node_mappers__', sofa_node_mappers)

    return wrapped_cls

#
# SlicerSofa
#

class SlicerSofa(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = _("Slicer Sofa")
        self.parent.categories = [translate("qSlicerAbstractCoreModule", "")]
        self.parent.dependencies = []
        self.parent.contributors = [ "Rafael Palomar (Oslo University Hospital, Norway), Paul Baksic (INRIA, France), Steve Pieper (Isomics, Inc., USA), Andras Lasso (Queen's University, Canada), Sam Horvath (Kitware, Inc., USA), Jean Christophe Fillion-Robin (Kitware, Inc., USA)"]
        self.parent.helpText = _("""
This is a support module to enable simulations using the SOFA framework
See more information in <a href="https://github.com/RafaelPalomar/Slicer-SOFA">module documentation</a>.
""")
        self.parent.acknowledgementText = _("""
This project has been funded by Oslo University Hospital
""")

        # Hide module, so that it only shows up in the Liver module, and not as a separate module
        parent.hidden = True

#
# SofaLogic
#

class SlicerSofaLogic(ScriptedLoadableModuleLogic):
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

        self._sceneUp = False
        self._rootNode = None
        self._parameterNode = None

    def __checkParameterNode__(self, parameterNode):
        if parameterNode is None:
            raise ValueError("parameterNode can't be None")
        if not getattr(parameterNode, 'sofaParameterNodeWrapped') or parameterNode.sofaParameterNodeWrapped is not True:
            raise ValueError("parameterNode is not a valid parameterNode wrapped by the sofaParameterNodeWrapper")

    def setupScene(self, parameterNode, rootNode):
        """Setup the simulation scene."""

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
        """Start the simulation."""
        self._parameterNode.currentStep = 0
        self._parameterNode.isSimulationRunning = True

        # Automatically set up sequence recording
        self.setupSequenceRecording()

    def stopSimulation(self) -> None:
        """Stop the simulation."""
        if self._sceneUp is True:
            self._parameterNode.isSimulationRunning = False
            self.stopSequenceRecording()

    def simulationStep(self) -> None:
        """Perform a single simulation step."""

        if self._sceneUp is False or self._parameterNode.isSimulationRunning is False:
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
        """Cleans up the simulation"""
        if self._sceneUp is True and self._rootNode:
            Sofa.Simulation.unload(self._rootNode)
            self._rootNode = None
            self._sceneUp = False
            self._parameterNode.isSimulationRunning = False

    @property
    def rootNode(self):
        return self._rootNode

    def setupSequenceRecording(self):
        """Sets up sequence recording for nodes that have recordSequence=True."""
        sofa_node_mappers = self._parameterNode.__class__.__sofa_node_mappers__

        for field_name, sofa_node_mapper in sofa_node_mappers.items():
            if sofa_node_mapper.recordSequence:
                mrmlNode = getattr(self._parameterNode, field_name)
                if mrmlNode is not None:
                    # Create a sequence node for recording
                    sequenceNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLSequenceNode', mrmlNode.GetName() + "_Sequence")
                    # Store the sequence node in the mapper for future reference (optional)
                    # sofa_node_mapper.sequenceNode = sequenceNode
                    # Get or create a sequence browser node
                    browserNode = self._getOrCreateSequenceBrowserNode()
                    # Add the sequence node to the browser
                    browserNode.AddSynchronizedSequenceNodeID(sequenceNode.GetID())
                    # Add the proxy node
                    browserNode.AddProxyNode(mrmlNode, sequenceNode, False)
                    # Set recording
                    browserNode.SetRecording(sequenceNode, True)
        # Start recording
        if hasattr(self, '_sequenceBrowserNode') and self._sequenceBrowserNode:
            self._sequenceBrowserNode.SetRecordingActive(True)

    def stopSequenceRecording(self):
        if hasattr(self, '_sequenceBrowserNode') and self._sequenceBrowserNode is not None:
            self._sequenceBrowserNode.SetRecordingActive(False)

    def _getOrCreateSequenceBrowserNode(self):
        if not hasattr(self, '_sequenceBrowserNode') or self._sequenceBrowserNode is None:
            # Create a new sequence browser node
            self._sequenceBrowserNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLSequenceBrowserNode', "SOFA Simulation Browser")
            self._sequenceBrowserNode.SetPlaybackActive(False)
        return self._sequenceBrowserNode

    def __updateSofa__(self) -> None:
        sofa_node_mappers = self._parameterNode.__class__.__sofa_node_mappers__

        for field_name, sofa_node_mapper in sofa_node_mappers.items():
            sofaMapping = sofa_node_mapper.sofaMapping
            name = sofa_node_mapper.nodeName

            if sofaMapping is not None:
                run_once = getattr(sofaMapping, 'run_once', False)
                if run_once and sofa_node_mapper.sofaMapping_has_run:
                    continue  # Skip execution
                node = self._rootNode[name] if name else self._rootNode
                sofaMapping(self._parameterNode, node)
                if run_once:
                    sofa_node_mapper.sofaMapping_has_run = True

    def __updateMRML__(self) -> None:
        sofa_node_mappers = self._parameterNode.__class__.__sofa_node_mappers__

        for field_name, sofa_node_mapper in sofa_node_mappers.items():
            mrmlMapping = sofa_node_mapper.mrmlMapping
            name = sofa_node_mapper.nodeName

            if mrmlMapping is not None:
                run_once = getattr(mrmlMapping, 'run_once', False)
                if run_once and sofa_node_mapper.mrmlMapping_has_run:
                    continue  # Skip execution
                node = self._rootNode[name] if name else self._rootNode
                mrmlMapping(self._parameterNode, node)
                if run_once:
                    sofa_node_mapper.mrmlMapping_has_run = True
