import AlgorithmConfig as cfg
import math as m
import numpy as np
import PCAnalysis as pca
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from UpRootFileReader import ProjectVector
from PfoVertexing import CalculateShower3DVertex

def axisEqual3D(ax):
    extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
    sz = extents[:,1] - extents[:,0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize/2
    for ctr, dim in zip(centers, 'xyz'):
        getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)

# Gets all 3D hits within a 4 cm radius of the vertex, and returns the PCA eigenvector which has the largest corresponding eigenvalue.
def GetInitialDirection(coordSets, vertex, radius = 4):
    coordSets = np.array(coordSets)
    if len(coordSets[0]) <= 1:
        return
    filt = GetHitsInRadius(coordSets=coordSets, centre=vertex, radius=radius)
    if filt.sum() <= 1:
        return
    return pca.Pca(coordSets=coordSets[:,filt], intercept=vertex)[1][:,-1]

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

def GetDeDx(xCoords3D, yCoords3D, zCoords3D, view2D, driftCoords2D, driftCoordErrors2D, wireCoords2D, wireCoordError2D, charges2D, sphereRadius, rectangleWidth, rectangleLength, vertex3D):
    vertex2D = ProjectVector(vertex3D, view2D)

    showerLongDirection3D = GetInitialDirection((xCoords3D, yCoords3D, zCoords3D), vertex3D, sphereRadius)
    if showerLongDirection3D is None:
        return m.nan
    showerLongDirection2D = ProjectVector(showerLongDirection3D, view2D)
    showerLongDirection2Dmag = np.linalg.norm(showerLongDirection2D)
    if showerLongDirection2Dmag == 0:
        return m.nan
    showerLongDirection2Dnormed = np.divide(showerLongDirection2D, showerLongDirection2Dmag)
    basisVectors = np.column_stack((showerLongDirection2Dnormed, [showerLongDirection2Dnormed[1], -showerLongDirection2Dnormed[0]]))
    reducedCoords = pca.ChangeCoordBasis((driftCoords2D, wireCoords2D), basisVectors, normed=True, preTranslation=-vertex2D)
    filt = Get2dHitsInRectangle(reducedCoords[0], reducedCoords[1], (0, rectangleWidth/2), rectangleWidth, rectangleLength * showerLongDirection2Dmag)
    #plt.scatter(driftCoords2D, wireCoords2D)
    #plt.scatter(driftCoords2D[filt], wireCoords2D[filt])
    #plt.plot([vertex2D[0], vertex2D[0] + showerLongDirection2Dnormed[0]], [vertex2D[1], vertex2D[1] + showerLongDirection2Dnormed[1]])
    #plt.scatter(vertex2D[0], vertex2D[1])
    #plt.axes().set_aspect('equal')
    #plt.show()def axisEqual3D(ax):
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



def GetFeatures(pfo, calculateViews):
    featureDict = {}
    dedx = m.nan

    vertex3D = pfo.vertex3D
    if cfg.newInitialDeDx["calcVertex"]:
        calculatedVertex = CalculateShower3DVertex(pfo.xCoord3D, pfo.yCoord3D, pfo.zCoord3D, cfg.vertexCalculation["initialLength"], cfg.vertexCalculation["outlierFraction"])
        if calculatedVertex is not None:
            vertex3D = calculatedVertex
    
    if calculateViews["U"]:
        if pfo.ValidVertex():
            dedx = GetDeDx(pfo.xCoord3D, pfo.yCoord3D, pfo.zCoord3D, "U", pfo.driftCoordU, pfo.driftCoordErrU, pfo.wireCoordU, 0.3, pfo.energyU, cfg.newInitialDeDx["initialDirectionRadius"], cfg.newInitialDeDx["rectangleWidth"], cfg.newInitialDeDx["rectangleLength"], vertex3D)
        featureDict.update({ "dedxU": dedx })
    if calculateViews["V"]:
        if pfo.ValidVertex():
            dedx = GetDeDx(pfo.xCoord3D, pfo.yCoord3D, pfo.zCoord3D, "V", pfo.driftCoordV, pfo.driftCoordErrV, pfo.wireCoordV, 0.3, pfo.energyV, cfg.newInitialDeDx["initialDirectionRadius"], cfg.newInitialDeDx["rectangleWidth"], cfg.newInitialDeDx["rectangleLength"], vertex3D)
        featureDict.update({ "dedxV": dedx })
    if calculateViews["W"]:
        if pfo.ValidVertex():
            dedx = GetDeDx(pfo.xCoord3D, pfo.yCoord3D, pfo.zCoord3D, "W", pfo.driftCoordW, pfo.driftCoordErrW, pfo.wireCoordW, 0.3, pfo.energyW, cfg.newInitialDeDx["initialDirectionRadius"], cfg.newInitialDeDx["rectangleWidth"], cfg.newInitialDeDx["rectangleLength"], vertex3D)
        featureDict.update({ "dedxW": dedx })
    return featureDict