"""SOFA-logo demo scene for the SOFA Scene Loader module.

A deformable FEM beam carrying the SOFA logo (logo head and S/O/F/A letters
as OglModel visual surfaces) falls under gravity onto a static cube obstacle.
Each OglModel declares a material color; the logo head is SOFA orange.

Contributed by Paul Baksic (INRIA).  Mesh files in mesh/ ship with the scene;
the .mtl files referenced by the OBJ headers are intentionally absent -- the
colors come from the OglModel components, not from OBJ materials.
"""

import os


def _logoTextureCoordinates(objPath):
    """Planar texture coordinates for the logo face.

    The logo meshes were generated from SOFA_LOGO.svg (bakpaul/TestScenes),
    so the face plane of the OBJ maps affinely onto the image: the image
    column follows mesh x and the image row follows mesh z (verified by
    scoring every axis candidate against the silhouette boundary).  VTK's PNG
    reader stores the file bottom-up, so v = 1 - row.  Computed from the raw
    OBJ vertices, in file order, so the coordinates line up with the loader's
    position output regardless of the loader transform.
    """
    points = []
    with open(objPath) as f:
        for line in f:
            if line.startswith('v '):
                fields = line.split()
                points.append((float(fields[1]), float(fields[3])))
    xs = [x for x, _ in points]
    zs = [z for _, z in points]
    xMin, xExtent = min(xs), max(xs) - min(xs)
    zMin, zExtent = min(zs), max(zs) - min(zs)
    return [[(x - xMin) / xExtent, 1.0 - (z - zMin) / zExtent] for x, z in points]


def createScene(root_node):
    root_node.name = "root"
    root_node.dt = 0.005
    root_node.gravity = [0, 0, -9.81]

    plugins = root_node.addChild('plugins')

    plugins.addObject('RequiredPlugin', name="MultiThreading")
    plugins.addObject('RequiredPlugin', name="Sofa.Component.AnimationLoop")
    plugins.addObject('RequiredPlugin', name="Sofa.Component.Collision.Detection.Algorithm")
    plugins.addObject('RequiredPlugin', name="Sofa.Component.Collision.Detection.Intersection")
    plugins.addObject('RequiredPlugin', name="Sofa.Component.Collision.Geometry")
    plugins.addObject('RequiredPlugin', name="Sofa.Component.Collision.Response.Contact")
    plugins.addObject('RequiredPlugin', name="Sofa.Component.Constraint.Lagrangian.Correction")
    plugins.addObject('RequiredPlugin', name="Sofa.Component.Constraint.Lagrangian.Solver")
    plugins.addObject('RequiredPlugin', name="Sofa.Component.IO.Mesh")
    plugins.addObject('RequiredPlugin', name="Sofa.Component.LinearSolver.Direct")
    plugins.addObject('RequiredPlugin', name="Sofa.Component.Mapping.Linear")
    plugins.addObject('RequiredPlugin', name="Sofa.Component.Mass")
    plugins.addObject('RequiredPlugin', name="Sofa.Component.ODESolver.Backward")
    plugins.addObject('RequiredPlugin', name="Sofa.Component.SolidMechanics.FEM.Elastic")
    plugins.addObject('RequiredPlugin', name="Sofa.Component.StateContainer")
    plugins.addObject('RequiredPlugin', name="Sofa.Component.Topology.Container.Dynamic")
    plugins.addObject('RequiredPlugin', name="Sofa.Component.Topology.Container.Grid")
    plugins.addObject('RequiredPlugin', name="Sofa.Component.Topology.Mapping")
    plugins.addObject('RequiredPlugin', name="Sofa.Component.Visual")
    plugins.addObject('RequiredPlugin', name="Sofa.GL.Component.Rendering3D")
    plugins.addObject('RequiredPlugin', name="Sofa.GUI.Component")

    root_node.addObject('VisualStyle', displayFlags="showVisual")
    root_node.addObject('ConstraintAttachButtonSetting')
    root_node.addObject('FreeMotionAnimationLoop', computeBoundingBox="false")
    root_node.addObject('BlockGaussSeidelConstraintSolver', maxIterations="50", tolerance="1.0e-6")
    root_node.addObject('CollisionPipeline', name="Pipeline")
    root_node.addObject('ParallelBruteForceBroadPhase', name="BroadPhase")
    root_node.addObject('ParallelBVHNarrowPhase', name="NarrowPhase")
    root_node.addObject('CollisionResponse', name="ContactManager", response="FrictionContactConstraint", responseParams="mu=0.3")
    root_node.addObject('NewProximityIntersection', name="Intersection", alarmDistance="0.02", contactDistance="0.002")
    beam_domain_from_grid_topology = root_node.addChild('BeamDomainFromGridTopology')

    beam_domain_from_grid_topology.addObject('RegularGridTopology', name="HexaTop", n="15 3 6", min="-0.25 0.02 0.5", max="0.25 0.08 0.72")
    tetra_topology = beam_domain_from_grid_topology.addChild('TetraTopology')

    tetra_topology.addObject('TetrahedronSetTopologyContainer', name="Container", position="@HexaTop.position")
    tetra_topology.addObject('TetrahedronSetTopologyModifier', name="Modifier")
    tetra_topology.addObject('Hexa2TetraTopologicalMapping', input="@HexaTop", output="@Container", swapping="true")

    f_e_mechanical_model = root_node.addChild('FE-MechanicalModel')

    f_e_mechanical_model.addObject('EulerImplicitSolver')
    f_e_mechanical_model.addObject('SparseLDLSolver', name="ldl", template="CompressedRowSparseMatrixMat3x3", parallelInverseProduct="true")
    f_e_mechanical_model.addObject('TetrahedronSetTopologyContainer', name="Container", position="@../BeamDomainFromGridTopology/HexaTop.position", tetrahedra="@../BeamDomainFromGridTopology/TetraTopology/Container.tetrahedra")
    f_e_mechanical_model.addObject('TetrahedronSetTopologyModifier', name="Modifier")
    f_e_mechanical_model.addObject('MechanicalObject', name="mstate", template="Vec3d", src="@Container")
    f_e_mechanical_model.addObject('TetrahedronFEMForceField', name="forceField", listening="true", youngModulus="2e4", poissonRatio="0.45", method="large", computeVonMisesStress="2")
    f_e_mechanical_model.addObject('MeshMatrixMass', totalMass="1.2")
    surface = f_e_mechanical_model.addChild('Surface')

    surface.addObject('TriangleSetTopologyContainer', name="Container")
    surface.addObject('TriangleSetTopologyModifier', name="Modifier")
    surface.addObject('Tetra2TriangleTopologicalMapping', input="@../Container", output="@Container", flipNormals="false")
    surface.addObject('MechanicalObject', name="dofs", rest_position="@../mstate.rest_position")
    surface.addObject('TriangleCollisionModel', name="Collision", contactDistance="0.001", color="0.94117647058824 0.93725490196078 0.89411764705882")
    surface.addObject('IdentityMapping', name="SurfaceMapping")

    visu_logo = f_e_mechanical_model.addChild('VisuLogo')

    sceneDir = os.path.dirname(os.path.abspath(__file__))
    visu_logo.addObject('MeshOBJLoader', name="SurfaceLoader", filename="mesh/LogoVisu.obj", scale3d="0.015 0.015 0.015", translation="-0.25 0.05 0.5", rotation="180 0 0")
    visu_logo.addObject('OglModel', name="VisualModel", color="0.7 .35 0 1.0", position="@SurfaceLoader.position", triangles="@SurfaceLoader.triangles",
                        texcoords=_logoTextureCoordinates(os.path.join(sceneDir, "mesh", "LogoVisu.obj")),
                        texturename=os.path.join(sceneDir, "textures", "SOFA_LOGO.png"))
    visu_logo.addObject('BarycentricMapping', name="MappingVisu", input="@../mstate", output="@VisualModel", isMechanical="false")

    visu_s = f_e_mechanical_model.addChild('VisuS')

    visu_s.addObject('MeshOBJLoader', name="SurfaceLoader", filename="mesh/SVisu.obj", scale3d="0.015 0.015 0.015", translation="-0.25 0.05 0.5", rotation="180 0 0")
    visu_s.addObject('OglModel', name="VisualModel", color="0.7 0.7 0.7 1", position="@SurfaceLoader.position", triangles="@SurfaceLoader.triangles")
    visu_s.addObject('BarycentricMapping', name="MappingVisu", input="@../mstate", output="@VisualModel", isMechanical="false")

    visu_o = f_e_mechanical_model.addChild('VisuO')

    visu_o.addObject('MeshOBJLoader', name="SurfaceLoader", filename="mesh/O.obj", scale3d="0.015 0.015 0.015", translation="-0.25 0.05 0.5", rotation="180 0 0")
    visu_o.addObject('OglModel', name="VisualModel", color="0.7 0.7 0.7 1", position="@SurfaceLoader.position", triangles="@SurfaceLoader.triangles")
    visu_o.addObject('BarycentricMapping', name="MappingVisu", input="@../mstate", output="@VisualModel", isMechanical="false")

    visu_f = f_e_mechanical_model.addChild('VisuF')

    visu_f.addObject('MeshOBJLoader', name="SurfaceLoader", filename="mesh/FVisu.obj", scale3d="0.015 0.015 0.015", translation="-0.25 0.05 0.5", rotation="180 0 0")
    visu_f.addObject('OglModel', name="VisualModel", color="0.7 0.7 0.7 1", position="@SurfaceLoader.position", triangles="@SurfaceLoader.triangles")
    visu_f.addObject('BarycentricMapping', name="MappingVisu", input="@../mstate", output="@VisualModel", isMechanical="false")

    visu_a = f_e_mechanical_model.addChild('VisuA')

    visu_a.addObject('MeshOBJLoader', name="SurfaceLoader", filename="mesh/AVisu.obj", scale3d="0.015 0.015 0.015", translation="-0.25 0.05 0.5", rotation="180 0 0")
    visu_a.addObject('OglModel', name="VisualModel", color="0.7 0.7 0.7 1", position="@SurfaceLoader.position", triangles="@SurfaceLoader.triangles")
    visu_a.addObject('BarycentricMapping', name="MappingVisu", input="@../mstate", output="@VisualModel", isMechanical="false")

    f_e_mechanical_model.addObject('LinearSolverConstraintCorrection', linearSolver="@ldl")

    cube = root_node.addChild('Cube')

    cube.addObject('VisualStyle', displayFlags="showCollisionModels")
    cube.addObject('TriangleSetTopologyContainer', name="CubeTopo", position="-0.05 0 0   -0.05 0.1 0  0.05 0.1 0   0.05 0 0                                                 -0.05 0 -0.1   -0.05 0.1 -0.1  0.05 0.1 -0.1   0.05 0 -0.1", triangles="0 2 1  0 3 2                                                  0 1 5  0 5 4                                                  0 4 7  0 7 3                                                  1 2 6  1 6 5                                                  3 7 6  3 6 2                                                  4 5 6  4 6 7")
    cube.addObject('MechanicalObject', template="Vec3")
    cube.addObject('TriangleCollisionModel', name="CubeCM", contactDistance="0.001", moving="0", simulated="0")

