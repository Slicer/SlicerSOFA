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
from vtk.util.numpy_support import numpy_to_vtk
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
    sofaNode.triangle = slicer.util.arrayFromModelPolyIds(modelNode).reshape(-1,4)[:,1:]
    sofaNode.position = slicer.util.arrayFromModelPoints(modelNode)


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

    unstructuredGrid = modelNode.GetUnstructuredGrid()
    if not unstructuredGrid:
        raise ValueError("Unstructured grid associated to modelNode can't be none")

    # Retrieve unstructured grid data from the model node
    points = unstructuredGrid.GetPoints()
    numPoints = points.GetNumberOfPoints()

    # Convert VTK points to a list for SOFA node
    pointCoords = [points.GetPoint(i) for i in range(numPoints)]

    # Parse cell data (tetrahedra connectivity)
    cells = unstructuredGrid.GetCells()
    cellArray = vtk.util.numpy_support.vtk_to_numpy(cells.GetData())
    cellConnectivity = []
    idx = 0
    for i in range(unstructuredGrid.GetNumberOfCells()):
        numPoints = cellArray[idx]
        cellConnectivity.append(cellArray[idx + 1:idx + 1 + numPoints].tolist())
        idx += numPoints + 1

    # Update SOFA node with tetrahedra and positions
    sofaNode.tetrahedra = cellConnectivity
    sofaNode.position = pointCoords


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
    sofaNode.position = [list(fiducialNode.GetNthControlPointPosition(0)) * 3]

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

    sofaNode.box = [arrayFromMarkupsROIPoints(roiNode)]

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

    points = numpy_to_vtk(num_array=sofaNode.position.array(), deep=True, array_type=vtk.VTK_FLOAT)
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

    modelNode.GetUnstructuredGrid().SetCells(vtk.VTK_HEXAHEDRON, cellArray)

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

    unstructuredGrid = modelNode.GetUnstructuredGrid()
    if not unstructuredGrid:
        raise ValueError("Unstructured grid associated to modelNode can't be none")

    # Retrieve or initialize the von Mises stress array in the MRML model node
    stressArray = unstructuredGrid.GetCellData().GetArray("VonMisesStress")
    if stressArray is None:
        # Create a stress array if it doesn't exist
        stressArray = vtk.vtkFloatArray()
        stressArray.SetName("VonMisesStress")
        unstructuredGrid.GetCellData().AddArray(stressArray)

    # Resize and populate the stress array
    vonMisesStresses = sofaNode.vonMisesPerElement.array()
    stressArray.SetNumberOfValues(len(vonMisesStresses))
    stressArray.SetVoidArray(vonMisesStresses, len(vonMisesStresses), 1)

    # Notify MRML about changes to the array
    unstructuredGrid.GetCellData().Modified()
    modelNode.Modified()

    # Update the display node's scalar range
    displayNode = modelNode.GetDisplayNode()
    if displayNode:
        displayNode.UpdateScalarRange()

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



