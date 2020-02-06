import math as m
import numpy as np
import PCAnalysis as pca
import matplotlib.pyplot as plt

# Gets all 3D hits within a 4 cm radius of the vertex, and returns the PCA eigenvector which has the largest corresponding eigenvalue.
def GetInitialDirection(xCoords, yCoords, zCoords, vertex, sphereRadius = 4):
    if len(xCoords) <= 1:
        return
    filt = GetHitsInRadius((xCoords, yCoords, zCoords), vertex, sphereRadius)
    if filt.sum() <= 1:
        return
    return pca.Pca((xCoords[filt], yCoords[filt], zCoords[filt]), intercept=vertex)[1][:,-1]

# Project the 3D PCA eigenvector into the corresponding 2D view.
def ProjectEigenvector(eigenvector, view):
    if view == 'U':
        eigenvector = [eigenvector[0], 0.5 * eigenvector[2] - 0.8660254 * eigenvector[1]]
        return eigenvector
    if view == 'V':
        eigenvector = [eigenvector[0], 0.5 * eigenvector[2] + 0.8660254 * eigenvector[1]]
        return eigenvector
    if view == 'W':
        eigenvector = [eigenvector[0], eigenvector[2]]
        return eigenvector


def GetHitsInRadius(coordSets, centre, radius=4):
    centre = np.reshape(centre, (-1, 1))
    return np.linalg.norm(coordSets - centre, axis=0) <= radius

# Get hits and charge in rectangle in 2D view
def Get2dHitsInRectangle(xCoords, yCoords, rectangleTopLeft=(0, 0.5), rectangleWidth=1, rectangleLength=4):
    filt = (
        (xCoords >= rectangleTopLeft[0]) & 
        (xCoords <= rectangleTopLeft[0] + rectangleLength) &
        (yCoords >= rectangleTopLeft[1] - rectangleWidth) &
        (yCoords <= rectangleTopLeft[1])
    )
    return filt

def GetLongitudinalError(driftCoordErrors, wireCoordErrors, longDirection):
    return driftCoordErrors * wireCoordErrors / np.sqrt((np.square(wireCoordErrors * longDirection[0]) + np.square(driftCoordErrors * longDirection[1])))

def GetDeDx(xCoords3D, yCoords3D, zCoords3D, vertex3D, view2D, driftCoords2D, driftCoordErrors2D, wireCoords2D, wireCoordError2D, vertex2D, charges2D, sphereRadius, rectangleWidth, rectangleLength):
    showerLongDirection3D = GetInitialDirection(xCoords3D, yCoords3D, zCoords3D, vertex3D, sphereRadius)
    if showerLongDirection3D is None:
        return m.nan
    showerLongDirection2D = ProjectEigenvector(showerLongDirection3D, view2D)
    showerLongDirection2Dmag = np.linalg.norm(showerLongDirection2D)
    if showerLongDirection2Dmag == 0:
        return m.nan
    showerLongDirection2Dnormed = np.divide(showerLongDirection2D, showerLongDirection2Dmag)
    basisVectors = np.row_stack((showerLongDirection2Dnormed, [showerLongDirection2Dnormed[1], -showerLongDirection2Dnormed[0]]))
    reducedCoords = pca.ChangeCoordBasis((driftCoords2D, wireCoords2D), basisVectors, normed=True, preTranslation=-vertex2D)
    filt = Get2dHitsInRectangle(reducedCoords[0], reducedCoords[1], (0, rectangleWidth/2), rectangleWidth, rectangleLength * showerLongDirection2Dmag)
    if filt.sum() == 0:
        return np.nan
    driftCoordErrorsInrectangle = driftCoordErrors2D[filt]
    chargesInRectangle = charges2D[filt]
    lCoordErrors = GetLongitudinalError(driftCoordErrorsInrectangle, wireCoordError2D, showerLongDirection2Dnormed)
    dedxs1 = chargesInRectangle / lCoordErrors
    dedx1 = np.median(dedxs1) * showerLongDirection2Dmag

    #dedxs2 = chargesInRectangle / wireCoordError
    #dedx2 = np.median(dedxs2) * abs(showerLongDirection2D[1])
    return dedx1
    
def GetFeatures(pfo, calculateViews, sphereRadius=5, rectangleWidth=1, rectangleLength=5):
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