import math as m
import numpy as np
import PCAnalysis as pca

# Gets all 3D hits within a 4 cm radius of the vertex, and returns the PCA eigenvector which has the largest corresponding eigenvalue.
def GetInitialDirection(vertex, xCoords, yCoords, zCoords, sphereRadius = 4):
    if len(xCoords) <= 1:
        return
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

    if len(initialHitsX) <= 1:
        return
    
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
def GetHitsInRectangle(driftCoords, wireCoords, rectangleTopLeft, rectangleWidth=1, rectangleLength=4):
    filt = (
        (driftCoords >= rectangleTopLeft[0]) & 
        (driftCoords <= rectangleTopLeft[0] + rectangleLength) &
        (wireCoords >= rectangleTopLeft[1] - rectangleWidth) &
        (wireCoords <= rectangleTopLeft[1])
    )
    return filt

def GetLongitudinalError(driftCoordErrors, wireCoordErrors, longDirection):
    return driftCoordErrors * wireCoordErrors / np.sqrt((np.square(wireCoordErrors * longDirection[0]) + np.square(driftCoordErrors * longDirection[1])))

def GetDeDx(xCoords3D, yCoords3D, zCoords3D, vertex3D, view2D, driftCoords2D, driftCoordErrors2D, wireCoords2D, wireCoordError2D, vertex2D, charges2D, sphereRadius, rectangleWidth, rectangleLength):
    showerLongDirection3D = GetInitialDirection(vertex3D, xCoords3D, yCoords3D, zCoords3D, sphereRadius)
    if showerLongDirection3D is None:
        return m.nan
    showerLongDirection2D = ProjectEigenvector(showerLongDirection3D, view2D)
    scaleFactor = np.linalg.norm(showerLongDirection2D) / np.linalg.norm(showerLongDirection3D)
    showerLongDirection2Dnormed = np.divide(showerLongDirection2D, np.linalg.norm(showerLongDirection2D))
    basisVectors = np.column_stack((showerLongDirection2D, [showerLongDirection2D[1], -showerLongDirection2D[0]]))
    pca.ChangeCoordBasis(basisVectors, (driftCoords2D, wireCoords2D), normed=True, translation=vertex2D)
    filt = GetHitsInRectangle(driftCoords2D, wireCoords2D, (0, rectangleWidth/2), rectangleWidth, rectangleLength * scaleFactor)
    driftCoordErrorsInrectangle = driftCoordErrors2D[filt]
    chargesInRectangle = charges2D[filt]
    longitudinalErrors = GetLongitudinalError(driftCoordErrorsInrectangle, wireCoordError2D, showerLongDirection2Dnormed)
    dedxs = chargesInRectangle / longitudinalErrors
    dedx = np.median(dedxs) * scaleFactor
    return dedx
    
def GetFeatures(pfo, calculateViews, sphereRadius=4, rectangleWidth=1, rectangleLength=4):
    featureDict = {}
    dedx = m.nan
    if calculateViews["U"]:
        if pfo.ValidVertex():
            dedx = GetDeDx(pfo.xCoord3D, pfo.yCoord3D, pfo.zCoord3D, pfo.vertex3D, "U", pfo.driftCoordU, pfo.driftCoordErrU, pfo.wireCoordU, 0.3, pfo.vertexU, pfo.energyU, sphereRadius, rectangleWidth, rectangleLength)
        featureDict.update({ "dedxU": dedx })
    if calculateViews["V"]:
        if pfo.ValidVertex():
            dedx = GetDeDx(pfo.xCoord3D, pfo.yCoord3D, pfo.zCoord3D, pfo.vertex3D, "V", pfo.driftCoordV, pfo.driftCoordErrV, pfo.wireCoordV, 0.3, pfo.vertexV, pfo.energyV, sphereRadius, rectangleWidth, rectangleLength)
        featureDict.update({ "dedxV": dedx })
    if calculateViews["W"]:
        if pfo.ValidVertex():
            dedx = GetDeDx(pfo.xCoord3D, pfo.yCoord3D, pfo.zCoord3D, pfo.vertex3D, "W", pfo.driftCoordW, pfo.driftCoordErrW, pfo.wireCoordW, 0.3, pfo.vertexW, pfo.energyW, sphereRadius, rectangleWidth, rectangleLength)
        featureDict.update({ "dedxW": dedx })
    return featureDict