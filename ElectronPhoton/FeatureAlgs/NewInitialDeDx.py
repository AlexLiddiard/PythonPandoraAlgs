import math as m
import numpy as np
import PCAnalysis as pca

# Gets all 3D hits within a 4 cm radius of the vertex, and returns the PCA eigenvector which has the largest corresponding eigenvalue.
def GetInitialHits(vertex, xCoords, yCoords, zCoords, sphereRadius = 4):
    if len(xCoords) == 0:
        return m.nan
    xCoordsNew = xCoords - vertex[0]
    yCoordsNew = yCoords - vertex[1]
    zCoordsNew = zCoords - vertex[2]
    magnitudes = np.sqrt(np.square(xCoordsNew) + np.square(yCoordsNew) + np.square(zCoordsNew))
    order = magnitudes.argsort()
    orderedXCoords = xCoords[order]
    orderedYCoords = yCoords[order]
    orderedZCoords = zCoords[order]

    i = np.argmax(magnitudes > sphereRadius)
    initialHitsX = orderedXCoords[:i]
    initialHitsY = orderedYCoords[:i]
    initialHitsZ = orderedZCoords[:i]
    
    return pca.Pca((initialHitsX, initialHitsY, initialHitsZ), intercept=vertex)[1][:,-1]

# Project the 3D PCA eigenvector into the corresponding 2D view.
def ProjectEigenvector(eigenvector, view):
    if view == 'U':
        eigenvector = [eigenvector[1], 0.5 * eigenvector[2] - 0.8660254 * eigenvector[1]]
        return eigenvector
    if view == 'V':
        eigenvector = [eigenvector[0], 0.5 * eigenvector[2] + 0.8660254 * eigenvector[1]]
        return eigenvector
    if view == 'W':
        eigenvector = [eigenvector[0], eigenvector[1]]
        return eigenvector

# Get hits and charge in 1x4 rectangle in 2D view with edge on vertex and pointing in eigenvector direction.
def GetHitsAndChargesInRectangle(eigenvector, rectangleVertex, driftCoords, wireCoords, chargeArray, rectangleWidth=1, rectangleLength=4):
