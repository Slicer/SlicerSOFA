from qt import QObject, QTimer, Signal
from abc import abstractmethod
import Sofa.Simulation

class SimulationController(QObject):
    # Define start and stop signals
    simulationStart = Signal()  # Signal emitted when the simulation starts
    simulationStop = Signal()   # Signal emitted when the simulation stops

    def __init__(self, parameterNode=None, parent=None):
        super(SimulationController, self).__init__(parent)

        self.parameterNode = parameterNode
        self._timer = QTimer(self)
        self._timer.timeout.connect(self.step)
        self.parameterNode = parameterNode
        self._sceneUp = False
        self.rootNode = None
        self._currentStep = 0

    def setupScene(self):
        """Setup the simulation scene."""
        self.rootNode = self.createScene(self.parameterNode)
        Sofa.Simulation.init(self.rootNode)
        self._sceneUp = True

    def start(self):
        """Start the simulation."""
        self._currentStep = 0
        self._timer.start(0)
        self.simulationStart.emit()  # Emit the start signal

    def stop(self):
        """Stop the simulation."""
        if self._sceneUp is True:
            self._timer.stop()
            self.simulationStop.emit()  # Emit the stop signal

    def step(self) -> None:
        """Perform a single simulation step."""

        if self._sceneUp is False:
            return

        self.updateParameters()

        if self._currentStep < self.parameterNode.totalSteps:
            Sofa.Simulation.animate(self.rootNode, self.parameterNode.dt)
            self._currentStep += 1
        elif self.parameterNode.totalSteps < 0:
            Sofa.Simulation.animate(self.rootNode, self.parameterNode.dt)
        else:
            self._timer.stop()          # Stop the timer after completing the simulation steps
            self.simulationStop.emit()  # Emit the stop signal

        self.updateScene()

    def clean(self) -> None:
        """Cleans up the simulation"""
        if self._sceneUp is True:
            Sofa.Simulation.unload(self.rootNode)
            self.rootNode = None
            self._sceneUp = False

    @abstractmethod
    def updateParameters(self) -> None:
        """Update simulation parameters."""
        pass

    @abstractmethod
    def updateScene(self) -> None:
        """Update the simulation scene. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def createScene(self, parameterNode) -> Sofa.Core.Node:
        """Create the simulation scene. Must be implemented by subclasses."""
        pass
