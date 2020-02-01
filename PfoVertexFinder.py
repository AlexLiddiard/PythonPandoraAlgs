import numpy as np
import BaseConfig
import PCAnalysis as pca

def InteractionVertex2D(pfoVertex, pfoXCoords, pfoZCoords, parentPfoVertex, parentPfoXCoords, parentPfoZCoords, pfoHits, parentPfoHits):
    if pfoHits < 2 or parentPfoHits < 2:
        return
    pfoEigenVector = FindPfoEigenVector2D(pfoXCoords, pfoZCoords, pfoVertex)
    parentPfoEigenVector = FindPfoEigenVector2D(parentPfoXCoords, parentPfoZCoords, parentPfoVertex)
    matrix = np.transpose([pfoEigenVector, -parentPfoEigenVector])
    vertexDifference = parentPfoVertex - pfoVertex
    try:
        solutions = np.linalg.solve(matrix, vertexDifference)
        intersection = pfoVertex + solutions[0]*pfoEigenVector
        return intersection
    except:
        return

def FindPfoEigenVector2D(pfoXCoords, pfoZCoords, pfoVertex):
    eigenvectors = pca.Pca((pfoXCoords, pfoZCoords), pfoVertex)[1]
    return eigenvectors[:,-1]

def FindPfoEigenVector3D(pfoXCoords, pfoYCoords, pfoZCoords, pfoVertex):
    eigenvectors = pca.Pca((pfoXCoords, pfoYCoords, pfoZCoords), pfoVertex)[1]
    return eigenvectors[:,-1]

def InteractionVertex3D(pfoVertex, pfoXCoords, pfoYCoords, pfoZCoords, parentPfoVertex, parentPfoXCoords, parentPfoYCoords, parentPfoZCoords, pfoHits, parentPfoHits):
    if pfoHits < 2 or parentPfoHits < 2:
        return
    pfoEigenVector = FindPfoEigenVector3D(pfoXCoords, pfoYCoords, pfoZCoords, pfoVertex)
    parentPfoEigenVector = FindPfoEigenVector3D(parentPfoXCoords, parentPfoYCoords, parentPfoZCoords, parentPfoVertex)
    perpendicularVector = np.cross(pfoEigenVector, parentPfoEigenVector)
    matrix = np.transpose([pfoEigenVector, perpendicularVector, -parentPfoEigenVector])
    vertexDifference = parentPfoVertex - pfoVertex
    try:
        solutions = np.linalg.solve(matrix, vertexDifference)
        intersection = solutions[0]*pfoEigenVector + pfoVertex + solutions[1]*perpendicularVector/2
        return intersection
    except:
        return