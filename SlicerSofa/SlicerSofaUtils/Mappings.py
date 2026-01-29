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

import slicer
import vtk
import numpy as np
from vtk.util.numpy_support import numpy_to_vtk,vtk_to_numpy
from slicer.parameterNodeWrapper import parameterPack
from slicer import vtkMRMLGridTransformNode

# -----------------------------------------------------------------------------
# Mapping functions MRML->Sofa
# -----------------------------------------------------------------------------
#
def mrmlModelPolyToSofaTriangleTopologyContainer(modelNode, sofaNode) -> None:
    """
    This is mapping function that will transfer geometry (points) and
    topology (cells) from a vtkPolyData stored in a vtkMRMLModel node
    to a Sofa TriangleSetTopologyContainer.

    Attributes:
        modelNode (ParameterNode): Parameter node for the mapping
        sofaNode (str): Sofa node nodeNodePath
    """

    if modelNode is None:
        raise ValueError("modelNode can't be None")
    if sofaNode is None:
        raise ValueError("modelNode can't be None")

    # Update SOFA node with tetrahedra and positions
    with sofaNode.triangles.writeable() as topology:
        topology[:] = slicer.util.arrayFromModelPolyIds(modelNode).reshape(-1,4)[:,1:4]
    with sofaNode.position.writeable() as geometry:
        geometry[:] = slicer.util.arrayFromModelPoints(modelNode)

def mrmlModelGridToSofaTetrahedronTopologyContainer(modelNode, sofaNode) -> None:
    """
    This is mapping function that will transfer geometry (points) and
    topology (cells) from a vtkUnstructuredGrid stored in a vtkMRMLModel node
    to a Sofa TetrahedronSetTopologyContainer.

    Attributes:
        modelNode (ParameterNode): Parameter node for the mapping
        sofaNode (str): Sofa node nodeNodePath
    """

    if modelNode is None:
        raise ValueError("modelNode can't be None")
    if sofaNode is None:
        raise ValueError("modelNode can't be None")

    # Update SOFA node with tetrahedra and positions
    with sofaNode.tetrahedra.writeable() as topology:
        topology[:] = arrayFromModelGridCells(modelNode)
    with sofaNode.position.writeable() as geometry:
        geometry[:] = slicer.util.arrayFromModelPoints(modelNode)


def mrmlMarkupsFiducialToSofaPointer(fiducialNode, sofaNode) -> None:
    """
    This is mapping function that will transfer a 3D fiducial point
    to a pointer interactor in Sofa

    Attributes:
        fiducialNode (ParameterNode): Parameter node for the mapping
        sofaNode (str): Sofa node sofaNode
    """

    if fiducialNode is None:
        raise ValueError("modelNode can't be None")
    if sofaNode is None:
        raise ValueError("modelNode can't be None")

    # Set the SOFA node position based on the first control point of the fiducial node
    with sofaNode.position.writeable() as geometry:
        geometry[:] = slicer.util.arrayFromMarkupsControlPoints(fiducialNode)

def mrmlMarkupsROIToSofaBoxROI(roiNode, sofaNode):
    """
    Maps a vtkMRMLMarkupsROINode to a SOFA Box ROI.

    Args:
        roiNode (vtkMRMLMarkupsROINode): MRML ROI node.
        sofaNode: SOFA node representing the target Box ROI.
    """
    if roiNode is None:
        raise ValueError("modelNode can't be None")
    if sofaNode is None:
        raise ValueError("modelNode can't be None")

    with sofaNode.box.writeable() as box:
        box[:] = arrayFromMarkupsROIPoints(roiNode)

# -----------------------------------------------------------------------------
# Mapping functions Sofa->MRML
# -----------------------------------------------------------------------------

def sofaMechanicalObjectToMRMLModelPoly(modelNode, sofaNode):
    """
    Maps geometry from a SOFA MechanicalObject to a vtkPolyData stored
    in a vtkMRMLModelNode.

    Args:
        sofaNode: SOFA MechanicalObject node.
        modelNode (vtkMRMLModelNode): MRML model node to store the geometry.
    """
    if modelNode is None:
        raise ValueError("modelNode can't be None")
    if sofaNode is None:
        raise ValueError("modelNode can't be None")

    if modelNode.GetPolyData() is None:
        polyData = vtk.vtkPolyData()
        modelNode.SetAndObservePolyData(polyData)

    surfacePointsArray = sofaNode.position.array()
    surfaceModelPointsArray = slicer.util.arrayFromModelPoints(modelNode)
    surfaceModelPointsArray[:] = surfacePointsArray
    slicer.util.arrayFromModelPointsModified(modelNode)



def sofaMechanicalObjectToMRMLModelPoly(modelNode, sofaNode):
    """
    Maps geometry from a SOFA MechanicalObject to a vtkPolyData stored
    in a vtkMRMLModelNode.

    Args:
        sofaNode: SOFA MechanicalObject node.
        modelNode (vtkMRMLModelNode): MRML model node to store the geometry.
    """
    if modelNode is None:
        raise ValueError("modelNode can't be None")
    if sofaNode is None:
        raise ValueError("modelNode can't be None")

    if modelNode.GetPolyData() is None:
        polyData = vtk.vtkPolyData()
        modelNode.SetAndObservePolyData(polyData)

    surfacePointsArray = sofaNode.position.array()
    surfaceModelPointsArray = slicer.util.arrayFromModelPoints(modelNode)
    surfaceModelPointsArray[:] = surfacePointsArray
    slicer.util.arrayFromModelPointsModified(modelNode)



def sofaMechanicalObjectToMRMLModelGrid(modelNode, sofaNode):
    """
    Maps geometry from a SOFA MechanicalObject to a vtkUnstructuredGrid stored
    in a vtkMRMLModelNode.

    Args:
        sofaNode: SOFA MechanicalObject node.
        modelNode (vtkMRMLModelNode): MRML model node to store the geometry.
    """
    if modelNode is None:
        raise ValueError("modelNode can't be None")
    if sofaNode is None:
        raise ValueError("modelNode can't be None")

    positionArray = sofaNode.position.array()
    if positionArray.shape[1] >= 3:
        # Extract only the first 3 columns (x, y, z)
        points3D = positionArray[:, :3]
    else:
        points3D = positionArray

    points = numpy_to_vtk(num_array=points3D, deep=True, array_type=vtk.VTK_FLOAT)
    
    vtkPoints = vtk.vtkPoints()
    vtkPoints.SetData(points)

    if modelNode.GetUnstructuredGrid() is None:
        unstructuredGrid = vtk.vtkUnstructuredGrid()
        modelNode.SetAndObserveMesh(unstructuredGrid)

    modelNode.GetUnstructuredGrid().SetPoints(vtkPoints)
    modelNode.Modified()

def sofaSparseGridTopologyToMRMLModelGrid(modelNode, sofaNode):
    """
    Maps topology from a SOFA SparseGridTopology to a vtkUnstructuredGrid
    stored in a vtkMRMLModelNode.

    Args:
        sofaNode: SOFA SparseGridTopology node.
        modelNode (vtkMRMLModelNode): MRML model node to store the topology.
    """
    if modelNode is None:
        raise ValueError("modelNode can't be None")
    if sofaNode is None:
        raise ValueError("modelNode can't be None")

    cellArray = vtk.vtkCellArray()
    for cell in sofaNode.hexahedra.array():
        hexahedron = vtk.vtkHexahedron()
        for i, pointId in enumerate(cell):
            hexahedron.GetPointIds().SetId(i, pointId)
        cellArray.InsertNextCell(hexahedron)

    if modelNode.GetUnstructuredGrid() is None:
        unstructuredGrid = vtk.vtkUnstructuredGrid()
        modelNode.SetAndObserveMesh(unstructuredGrid)

    modelNode.GetUnstructuredGrid().SetCells(vtk.VTK_HEXAHEDRON, cellArray)
    modelNode.Modified()

def sofaMeshTopologyToMRMLModelGrid(modelNode, sofaNode):
    """
    Maps topology from a SOFA TetrahedronSetTopologyContainer to a vtkUnstructuredGrid
    stored in a vtkMRMLModelNode.

    Args:
        sofaNode: SOFA TetrahedronSetTopologyContainer node.
        modelNode (vtkMRMLModelNode): MRML model node to store the topology.
    """
    if modelNode is None:
        raise ValueError("modelNode can't be None")
    if sofaNode is None:
        raise ValueError("modelNode can't be None")
    
    if len(sofaNode.tetrahedra.array()) != 0:
        sofaTetrahedronTopologyToMRMLModelGrid(modelNode, sofaNode)
    elif len(sofaNode.triangles.array()) != 0:
        sofaTriangleTopologyToMRMLModelGrid(modelNode, sofaNode)
    elif len(sofaNode.edges.array()) != 0:
        sofaEdgeTopologyToMRMLModelGrid(modelNode, sofaNode)
    else :
        print(f"No topology element found in {sofaNode}")


def sofaOglModelToMRMLModelGrid(modelNode, sofaNode):
    """
    Maps topology from a SOFA TetrahedronSetTopologyContainer to a vtkUnstructuredGrid
    stored in a vtkMRMLModelNode.

    Args:
        sofaNode: SOFA TetrahedronSetTopologyContainer node.
        modelNode (vtkMRMLModelNode): MRML model node to store the topology.
    """
    if modelNode is None:
        raise ValueError("modelNode can't be None")
    if sofaNode is None:
        raise ValueError("modelNode can't be None")
    

    if len(sofaNode.triangles.array()) != 0:
        sofaTriangleTopologyToMRMLModelGrid(modelNode, sofaNode)
    elif len(sofaNode.edges.array()) != 0:
        sofaEdgeTopologyToMRMLModelGrid(modelNode, sofaNode)
    else :
        print(f"No topology element found in {sofaNode}")
        
    sofaMechanicalObjectToMRMLModelGrid(modelNode, sofaNode)
        
    modelNode.Modified()

    # TODO use the material data in the ogl model to get the color and apply it to the mrml node
    # color = sofaNode.color.array()
    # modelNode.GetDisplayNode().SetColor(color[0],color[1],color[2])


def sofaEdgeTopologyToMRMLModelGrid(modelNode, sofaNode):
    """
    Maps topology from a SOFA EdgeSetTopologyContainer to a vtkUnstructuredGrid
    stored in a vtkMRMLModelNode.

    Args:
        sofaNode: SOFA EdgeSetTopologyContainer node.
        modelNode (vtkMRMLModelNode): MRML model node to store the topology.
    """
    if modelNode is None:
        raise ValueError("modelNode can't be None")
    if sofaNode is None:
        raise ValueError("sofaNode can't be None")

    cellArray = vtk.vtkCellArray()
    for cell in sofaNode.edges.array():
        edge = vtk.vtkLine()
        for i, pointId in enumerate(cell):
            edge.GetPointIds().SetId(i, pointId)
        cellArray.InsertNextCell(edge)

    if modelNode.GetUnstructuredGrid() is None:
        unstructuredGrid = vtk.vtkUnstructuredGrid()
        modelNode.SetAndObserveMesh(unstructuredGrid)

    modelNode.GetUnstructuredGrid().SetCells(vtk.VTK_LINE, cellArray)
    modelNode.Modified()

def sofaTriangleTopologyToMRMLModelGrid(modelNode, sofaNode):
    """
    Maps topology from a SOFA TriangleSetTopologyContainer to a vtkUnstructuredGrid
    stored in a vtkMRMLModelNode.

    Args:
        sofaNode: SOFA TriangleSetTopologyContainer node.
        modelNode (vtkMRMLModelNode): MRML model node to store the topology.
    """
    if modelNode is None:
        raise ValueError("modelNode can't be None")
    if sofaNode is None:
        raise ValueError("sofaNode can't be None")

    cellArray = vtk.vtkCellArray()
    for cell in sofaNode.triangles.array():
        triangle = vtk.vtkTriangle()
        for i, pointId in enumerate(cell):
            triangle.GetPointIds().SetId(i, pointId)
        cellArray.InsertNextCell(triangle)

    if modelNode.GetUnstructuredGrid() is None:
        unstructuredGrid = vtk.vtkUnstructuredGrid()
        modelNode.SetAndObserveMesh(unstructuredGrid)

    modelNode.GetUnstructuredGrid().SetCells(vtk.VTK_TRIANGLE, cellArray)
    modelNode.Modified()

def sofaTetrahedronTopologyToMRMLModelGrid(modelNode, sofaNode):
    """
    Maps topology from a SOFA TetrahedronSetTopologyContainer to a vtkUnstructuredGrid
    stored in a vtkMRMLModelNode.

    Args:
        sofaNode: SOFA TetrahedronSetTopologyContainer node.
        modelNode (vtkMRMLModelNode): MRML model node to store the topology.
    """
    if modelNode is None:
        raise ValueError("modelNode can't be None")
    if sofaNode is None:
        raise ValueError("modelNode can't be None")

    cellArray = vtk.vtkCellArray()
    for cell in sofaNode.tetrahedra.array():
        tetrahedron = vtk.vtkTetra()
        for i, pointId in enumerate(cell):
            tetrahedron.GetPointIds().SetId(i, pointId)
        cellArray.InsertNextCell(tetrahedron)

    if modelNode.GetUnstructuredGrid() is None:
        unstructuredGrid = vtk.vtkUnstructuredGrid()
        modelNode.SetAndObserveMesh(unstructuredGrid)

    modelNode.GetUnstructuredGrid().SetCells(vtk.VTK_TETRA, cellArray)
    modelNode.Modified()


def sofaVonMisesStressToMRMLModelGrid(modelNode, sofaNode):
    """
    Maps von Mises stress data from a SOFA node to a vtkUnstructuredGrid
    stored in a vtkMRMLModelNode.

    Args:
        sofaNode: SOFA node containing von Mises stress data.
        modelNode (vtkMRMLModelNode): MRML model node to store the stress data.
    """
    if modelNode is None:
        raise ValueError("modelNode can't be None")
    if sofaNode is None:
        raise ValueError("modelNode can't be None")

    if modelNode.GetUnstructuredGrid() is None:
        unstructuredGrid = vtk.vtkUnstructuredGrid()
        modelNode.SetAndObserveMesh(unstructuredGrid)


    unstructuredGrid = modelNode.GetUnstructuredGrid()
    if not unstructuredGrid:
        raise ValueError("Unstructured grid associated to modelNode can't be none")

    # Retrieve or initialize the von Mises stress array in the MRML model node
    stressArray = unstructuredGrid.GetPointData().GetArray("VonMisesStress")
    if stressArray is None:
        # Create a stress array if it doesn't exist
        stressArray = vtk.vtkFloatArray()
        stressArray.SetName("VonMisesStress")
        unstructuredGrid.GetPointData().AddArray(stressArray)

    vonMisesStresses = sofaNode.vonMisesPerNode.array()
    numberOfPoints = unstructuredGrid.GetNumberOfPoints()
    stressArray.SetNumberOfValues(numberOfPoints)

    if len(vonMisesStresses) == numberOfPoints:
        for i in range(len(vonMisesStresses)):
            stressArray.SetValue(i, vonMisesStresses[i])
    else:
        stressArray.Fill(0.0)

    displayNode = modelNode.GetDisplayNode()

    if displayNode:
        colorNode = slicer.util.getNode('ColdToHotRainbow')
        displayNode.SetActiveScalarName(stressArray.GetName())  # Set your scalar field name here
        displayNode.SetAndObserveColorNodeID(colorNode.GetID())
        # Set the scalar visibility and range
        displayNode.SetAutoScalarRange(False)  # Disable auto range
        displayNode.SetScalarRange(np.min(vonMisesStresses), np.max(vonMisesStresses))  # Set your desired range
        displayNode.Modified() 

    # Notify MRML about changes to the array
    unstructuredGrid.GetPointData().Modified()
    modelNode.Modified()


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

    # # Calculate direction vector and normalize
    controlPoints = slicer.util.arrayFromMarkupsControlPoints(lineNode)
    vector = controlPoints[1]-controlPoints[0]
    norm = np.linalg.norm(vector)
    if norm == 0:
       return vector  # Return the original vector if norm is zero to avoid division by zero
    return vector / norm

def arrayFromModelGridCells(modelNode):

    if modelNode is None:
        raise ValueError("modelNode can't be None")

    unstructuredGrid = modelNode.GetUnstructuredGrid()
    if not unstructuredGrid:
        raise ValueError("Unstructured grid associated to modelNode can't be none")

    cellsArray = np.array(unstructuredGrid.GetCells().GetData())
    cellConnectivity= cellsArray.reshape(-1,5)[:,1:5]

    return cellConnectivity
