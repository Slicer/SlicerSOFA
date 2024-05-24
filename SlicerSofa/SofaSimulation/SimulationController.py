from qt import QObject, QTimer
from abc import abstractmethod

from SofaSimulation import *

class SimulationController(QObject):

    def __init__(self, parameterNode=None, parent=None):

        super(SimulationController, self).__init__(parent)

        self.parameterNode = parameterNode

        self._timer = QTimer(self)
        self._timer.timeout.connect(self.step)
        self.parameterNode = parameterNode
        self._sceneUp = False
        self._stopSignal = False
        self.rootNode = None
        self._currentStep = 0

    def setupScene(self):
        self.rootNode = self.createScene(self.parameterNode)
        Sofa.Simulation.init(self.rootNode)
        self._sceneUp = True

    def start(self):
        if self._sceneUp is not True:
            self.setupScene()
        else:
            Sofa.Simulation.reset(self.rootNode)
        self._stopSignal = False
        self._currentStep = 0
        self._timer.start(0)

    def stop(self):
        if self._sceneUp is True:
            self._timer.stop()

    def step(self) -> None:

        self.updateParameters()

        if self._currentStep < self.parameterNode.totalSteps and not self._stopSignal:
            Sofa.Simulation.animate(self.rootNode, self.rootNode.dt.value)
            self._currentStep += 1
        elif self.parameterNode.totalSteps < 0:
            Sofa.Simulation.animate(self.rootNode, self.rootNode.dt.value)
        else:
            self._timer.stop()  # Stop the timer after completing the simulation steps

        self.updateScene()

    def updateParameters(self) -> None:
        self.rootNode.dt.value = self.parameterNode.dt
        self.rootNode.gravity = self.parameterNode.getGravityVector()

    @abstractmethod
    def updateScene(self) -> None:
        pass

    @abstractmethod
    def createScene(self, parameterNode) -> Sofa.Core.Node:
        pass
