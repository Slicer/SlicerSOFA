import os
from qt import QRunnable, QTimer, QThreadPool, QObject

os.environ['SOFA_ROOT'] = '/home/rafael/src/Slicer-SOFA/Release/inner-build/lib/Slicer-5.7/qt-loadable-modules'

import slicer
import Sofa
import SofaRuntime

def createScene(rootNode):
    from stlib3.scene import MainHeader, ContactHeader
    from stlib3.solver import DefaultSolver
    from stlib3.physics.deformable import ElasticMaterialObject
    from stlib3.physics.rigid import Floor
    from splib3.numerics import Vec3

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
        "SofaIGTLink"
    ], dt=0.01, gravity=[9810, 0, 0])

    rootNode.addObject('VisualStyle', displayFlags='showVisualModels showForceFields')
    rootNode.addObject('BackgroundSetting', color=[0.8, 0.8, 0.8, 1])
    rootNode.addObject('DefaultAnimationLoop', name="FreeMotionAnimationLoop", parallelODESolving=True)
    rootNode.addObject('iGTLinkClient', name="iGTLClient", sender=True, hostname="127.0.0.1", port=18944)

    meshNode = rootNode.addChild('Mesh')
    meshNode.addObject('EulerImplicitSolver', firstOrder=False, rayleighMass=0.1, rayleighStiffness=0.1)
    meshNode.addObject('SparseLDLSolver', name="precond", template="CompressedRowSparseMatrixd", parallelInverseProduct=True)
    meshNode.addObject('MeshVTKLoader', name="loader", filename="/tmp/RightLung.vtk")
    meshNode.addObject('TetrahedronSetTopologyContainer', name="Container", src="@loader")
    meshNode.addObject('TetrahedronSetTopologyModifier', name="Modifier")
    meshNode.addObject('MechanicalObject', name="mstate", template="Vec3f")
    meshNode.addObject('TetrahedronFEMForceField', name="FEM", youngModulus=1.5, poissonRatio=0.45, method="large")
    meshNode.addObject('MeshMatrixMass', totalMass=1)
    meshNode.addObject('iGTLinkPolyDataMessage', name="SOFAMesh", iGTLink="@../iGTLClient",
                       points="@mstate.position",
                       enableIndices=False,
                       enableEdges=False,
                       enableTriangles=False,
                       enableTetra=False,
                       enableHexa=False)

    collisionNode = meshNode.addChild('Collision')
    collisionNode.addObject('TriangleSetTopologyContainer', name="Container")
    collisionNode.addObject('TriangleSetTopologyModifier', name="Modifier")
    collisionNode.addObject('Tetra2TriangleTopologicalMapping', input="@../Container", output="@Container")

class SimulationRunnable(QRunnable, QObject):
    def __init__(self):
        QRunnable.__init__(self)
        QObject.__init__(self)
        self.setAutoDelete(False)

    def run(self):
        # Since this runs in a separate thread, avoid direct GUI operations here
        self.root = Sofa.Core.Node()
        createScene(self.root)
        Sofa.Simulation.init(self.root)

        # Use a QTimer to periodically update the simulation
        # Since QTimer needs to run in the context of a Qt event loop, it should be
        # connected and started in the main thread. Consider using signals or
        # moving the QTimer to the main thread if necessary.

        # This placeholder demonstrates a simulation loop without real-time updates
        for _ in range(100):  # Simulate 100 steps as an example
            Sofa.Simulation.animate(self.root, self.root.dt.value)
            # In a real application, consider using signals to update the GUI or handle the simulation state

class SimulationController(QObject):
    def __init__(self):
        super(SimulationController, self).__init__()
        self.simulationTask = SimulationRunnable()
        self.threadPool = QThreadPool.globalInstance()

    def startSimulation(self):
        # Start the simulation runnable in the thread pool
        self.threadPool.start(self.simulationTask)

def startSimulation():
    controller = SimulationController()
    slicer.app.connect("aboutToQuit()", controller.threadPool.waitForDone)  # Ensure threads finish on app close
    controller.startSimulation()
